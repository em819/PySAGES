# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

from jax import jit
from jax import numpy as np
from jax_md import dataclasses
import numpy as nnp

from pysages.backends.core import SamplingContext
from pysages.backends.snapshot import (
    Box,
    HelperMethods,
    Snapshot,
    SnapshotMethods,
    build_data_querier,
)
from pysages.typing import Callable, NamedTuple
from pysages.utils import check_device_array, copy
import jax.lax
import jax.tree_util

# PySAGES forces `jax_enable_x64` on (pysages/__init__.py), so the default float is
# float64 -- the dtype every sampling method initializes its state with via bare
# `np.zeros(...)`. When the jax-md integrator runs in float32 (e.g. so3lr_dev's
# `precision: "float32"`), data a method recomputes from the snapshot (e.g. ABF's
# `Wp`) comes back float32 and flips the `fori_loop` carry dtype. We promote to this
# canonical float at the snapshot *query* layer (`build_snapshot_methods`), i.e. only
# in the view handed to methods, so method math stays float64 while the stored
# `Snapshot` remains a faithful float32 mirror of the integrator. That matters because
# the host engine reconstructs its (float32) state from the returned snapshot; promoting
# the snapshot itself would leak float64 back across that boundary. For a genuine
# float64 run every cast is a no-op.
_DEFAULT_FLOAT = np.zeros(()).dtype


def _as_default_float(x):
    return x.astype(_DEFAULT_FLOAT) if np.issubdtype(x.dtype, np.floating) else x


def _match_float_dtypes(reference, target):
    """Cast every floating-point leaf of ``target`` back to the dtype of the
    corresponding leaf in ``reference``.

    PySAGES forces ``jax_enable_x64`` on (pysages/__init__.py). With x64 enabled,
    any explicit float64 constant the host integrator mixes into its MD step
    (e.g. a ``jnp.array(...)`` that defaults to float64, masses, dt) silently
    promotes the float32 integrator arrays to float64 -- even though the same
    integrator stays in float32 when run standalone (x64 off). That promotion
    breaks jax-md's ``fori_loop``, whose carry must be dtype-stable across the
    body. We therefore restore the reference (= integrator) float dtype on every
    carried array at the end of each step, so a float32 run carries float32
    end-to-end no matter which sampling method or integrator constants are in
    play. This is method-independent and a no-op for a genuine float64 run.
    """
    def cast_like(ref, val):
        if (hasattr(val, "dtype") and hasattr(ref, "dtype")
                and np.issubdtype(val.dtype, np.floating)):
            return val.astype(ref.dtype)
        return val

    return jax.tree_util.tree_map(cast_like, reference, target)


class Sampler:
    def __init__(self, method_bundle, context_state, callback: Callable):
        initial_snapshot, initialize, method_update = method_bundle
        self.state = initialize()
        self.callback = callback
        self.context_state = context_state
        self.snapshot = initial_snapshot
        self.update = method_update

    def restore(self, prev_snapshot):
        self.snapshot = prev_snapshot

    def take_snapshot(self):
        return copy(self.snapshot)


def take_snapshot(state, box, dt):
    dims = box.shape[0]
    positions = state.position
    forces = getattr(state, "force", state.position)
    ids = np.arange(len(positions))
    velocities = getattr(state, "velocity", state.position)
    masses = state.mass.reshape(-1, 1)
    vel_mass = (velocities, masses)
    origin = tuple(0.0 for _ in range(dims))

    # Support NVT (.chain), NPT (.thermostat), and NVE (no chain)
    chain = getattr(state, 'chain', None) or getattr(state, 'thermostat', None)
    chain_data = vars(chain) if chain else None

    # Capture barostat state for NPT ensembles
    barostat = getattr(state, 'barostat', None)
    barostat_data = vars(barostat) if barostat else None

    # Capture NPT box deformation variables
    npt_box_data = None
    if hasattr(state, 'reference_box'):
        npt_box_data = {
            'reference_box': state.reference_box,
            'box_position': state.box_position,
            'box_momentum': state.box_momentum,
            'box_mass': state.box_mass,
        }

    check_device_array(positions)  # currently, we only support `DeviceArray`s

    return Snapshot(positions, vel_mass, forces, ids, None, Box(box, origin), dt,
                    chain_data=chain_data, barostat_data=barostat_data,
                    npt_box_data=npt_box_data)


def update_snapshot(snapshot, state, box=None):
    _, masses = snapshot.vel_mass
    positions = state.position
    vel_mass = (state.velocity, masses)
    forces = state.force

    # Support NVT (.chain), NPT (.thermostat), and NVE (no chain)
    chain = getattr(state, 'chain', None) or getattr(state, 'thermostat', None)
    chain_data = vars(chain) if chain else None

    # Capture barostat state for NPT ensembles
    barostat = getattr(state, 'barostat', None)
    barostat_data = vars(barostat) if barostat else None

    # Capture NPT box deformation variables
    npt_box_data = None
    if hasattr(state, 'reference_box'):
        npt_box_data = {
            'reference_box': state.reference_box,
            'box_position': state.box_position,
            'box_momentum': state.box_momentum,
            'box_mass': state.box_mass,
        }

    replacements = dict(
        positions=positions,
        vel_mass=vel_mass,
        forces=forces,
        chain_data=chain_data,
        barostat_data=barostat_data,
        npt_box_data=npt_box_data,
    )

    # Update box if provided (needed for NPT where box evolves)
    if box is not None:
        dims = box.shape[0]
        origin = tuple(0.0 for _ in range(dims))
        replacements['box'] = Box(box, origin)

    return snapshot._replace(**replacements)


def build_snapshot_methods(context, sampling_method):
    def indices(snapshot):
        return snapshot.ids

    def masses(snapshot):
        _, M = snapshot.vel_mass
        return _as_default_float(M)

    def positions(snapshot):
        # Promote to the canonical float for the method view (see `_as_default_float`)
        return _as_default_float(snapshot.positions)

    def momenta(snapshot):
        V, M = snapshot.vel_mass
        return _as_default_float((V * M).flatten())

    return SnapshotMethods(positions, indices, jit(momenta), masses)


def build_helpers(context, sampling_method):
    def dimensionality():
        return context.box.shape[0]

    snapshot_methods = build_snapshot_methods(context, sampling_method)
    flags = sampling_method.snapshot_flags
    helpers = HelperMethods(build_data_querier(snapshot_methods, flags), dimensionality)

    return helpers

jax_fn_container = {'is_defined': False, 'run_fn': None}

def build_runner(context, sampler, jit_compile=True):
    step_fn = context.step_fn
    dt = context.dt

    if not jax_fn_container['is_defined']:
        jax_fn_container['is_defined'] = True

        def _step(sampling_context_state, snapshot, sampler_state):
            # Remember the integrator's float dtype (float32 for a float32 run)
            # before stepping, so we can restore it across the whole carry at the
            # end -- the host MD step can silently promote it to float64 under
            # PySAGES' forced x64 mode (see `_match_float_dtypes`).
            ref_context_state = sampling_context_state
            ref_snapshot = snapshot

            sampling_context_state = step_fn(sampling_context_state)  # jax_md simulation step
            context_state = sampling_context_state.state
            # Extract box from extras (needed for NPT where box evolves each step)
            extras = sampling_context_state.extras
            box_from_extras = extras.get("box") if extras else None
            snapshot = update_snapshot(snapshot, context_state, box=box_from_extras)
            sampler_state = sampler.update(snapshot, sampler_state)  # pysages update
            if sampler_state.bias is not None:  # bias the simulation
                context_state = sampling_context_state.state
                # Missing 2nd-half momentum kick from the bias: jax-md's
                # velocity_verlet overwrites state.force with the unbiased
                # force after the position update, so without this kick the
                # bias contributes only the next step's 1st half-kick — i.e.,
                # 0.5*dt per MD step instead of dt. Adding 0.5*dt*bias here
                # plus the leading-half kick on the next step (via state.force
                # below) restores a full dt kick per step.
                # The bias is float64 (PySAGES methods compute in x64); cast it
                # down to the integrator dtype so a float32 run keeps its momentum
                # and force in float32 (no-op for a float64 run).
                bias = sampler_state.bias.astype(context_state.momentum.dtype)
                new_momentum = context_state.momentum + 0.5 * dt * bias
                biased_forces = context_state.force + bias
                context_state = dataclasses.replace(
                    context_state, momentum=new_momentum, force=biased_forces,
                )
                sampling_context_state = sampling_context_state._replace(state=context_state)

            # Restore the integrator float dtype across the entire carry so the
            # `fori_loop`/`scan` carry stays dtype-stable regardless of any
            # float64 promotion inside the host MD step or the method bias.
            sampling_context_state = _match_float_dtypes(ref_context_state, sampling_context_state)
            snapshot = _match_float_dtypes(ref_snapshot, snapshot)
            return sampling_context_state, snapshot, sampler_state

        step = jit(_step) if jit_compile else _step



        def _run_body(i, input_states_and_snapshots):
            context_state, snapshot, sampler_state, cv_arr = input_states_and_snapshots
            context_state, snapshot, sampler_state = step(context_state, snapshot, sampler_state)

            if sampler.callback:
                sampler.callback(snapshot, sampler_state, i)

            cv_arr = cv_arr.at[i].set(sampler_state.xi[0])
            return (context_state, snapshot, sampler_state, cv_arr)

        run_body = jit(_run_body) if jit_compile else _run_body
    

        if jit_compile:
            jax_fn_container['run_fn'] = run_body
        else:
            jax_fn_container['run_fn'] = step

    if jit_compile:
        def run(timesteps):
            # TODO: Allow to optionally batch timesteps with `lax.fori_loop`
            cv_per_step_arr = np.zeros((timesteps, len(sampler.state.xi[0]))) 
            sampler.context_state, sampler.snapshot, sampler.state, cv_per_step_arr = jax.block_until_ready( 
                    jax.lax.fori_loop(0, timesteps, jax_fn_container['run_fn'], (sampler.context_state, sampler.snapshot, sampler.state, cv_per_step_arr))
                )

            with open('cv_vs_timestep.txt', 'a') as cvf:
                nnp.savetxt(cvf, cv_per_step_arr)
    else:
        def run(timesteps):
            # TODO: Allow to optionally batch timesteps with `lax.fori_loop`
            for i in range(timesteps):
                context_state, snapshot, state = jax_fn_container['run_fn'](
                    sampler.context_state, sampler.snapshot, sampler.state
                )
                sampler.context_state = context_state
                sampler.snapshot = snapshot
                sampler.state = state
                if sampler.callback:
                    sampler.callback(sampler.snapshot, sampler.state, i)

    #return run

    return run




class View(NamedTuple):
    synchronize: Callable


def bind(sampling_context: SamplingContext, callback: Callable, **kwargs):
    context = sampling_context.context
    sampling_method = sampling_context.method
    context_state = context.init_fn(**kwargs)
    snapshot = take_snapshot(context_state.state, context.box, context.dt)
    helpers = build_helpers(context, sampling_method)
    method_bundle = sampling_method.build(snapshot, helpers)
    sampler = Sampler(method_bundle, context_state, callback)
    sampling_context.view = View((lambda: None))
    sampling_context.run = build_runner(
        context, sampler, jit_compile=kwargs.get("jit_compile", True)
    )
    return sampler
