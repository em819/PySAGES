# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg
import jax
from pysages.colvars.core import AxisCV, TwoPointCV, multicomponent, FourPointCV
import logging


def barycenter(positions):
    """
    Returns the geometric center, or centroid, of a group of points in space.

    Parameters
    ----------
    positions : jax.Array
        Array containing the positions of the points for which to compute the barycenter.

    Returns
    -------
    barycenter : jax.Array
        3D array with the barycenter coordinates.
    """
    return np.sum(positions, axis=0) / positions.shape[0]


def weighted_barycenter(positions, weights):
    """
    Returns the center of a group of points in space weighted by arbitrary weights.

    Parameters
    ----------
    positions : jax.Array
        Array containing the positions of the points for which to compute the barycenter.
    weights : jax.Array
        Array containing the weights to be used when computing the barycenter.

    Returns
    -------
    weighted_barycenter : jax.Array
        3D array with the weighted barycenter coordinates.
    """
    group_length = positions.shape[0]
    center = np.zeros(3)
    # TODO: Replace by `np.sum` and `vmap`  # pylint:disable=fixme
    for i in range(group_length):
        w, p = weights[i], positions[i]
        center += w * p
    return center


class Component(AxisCV):
    """
    Use a specific Cartesian component of the center of mass of the group of atom selected
    via the indices.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices. From each group the barycenter is calculated.
    axis: int
       Cartesian coordinate axis component `0` (X), `1` (Y), `2` (Z) that is requested as CV.
    """

    @property
    def function(self):
        return lambda rs: barycenter(rs)[self.axis]


class RelativeComponent(AxisCV):
    """
    Cartesian-axis component of one group's barycenter measured *relative to* a
    second (reference) group's barycenter::

        xi = barycenter(group_1)[axis] - barycenter(group_2)[axis]

    Unlike :class:`Component` (an absolute lab-frame coordinate), this CV is
    invariant under a rigid translation of the whole system. Consequently an
    enhanced-sampling bias acting on it exerts *no net force* on the center of
    mass: the CV gradient is ``+1/N_1`` on the first group's atoms and
    ``-1/N_2`` on the second group's atoms, which sums to zero. This removes the
    center-of-mass drift that a bias on an absolute-coordinate CV (such as
    :class:`Component`) produces under ABF / Metadynamics.

    Typical use (interfacial transfer PMF): ``group_1`` = the solute,
    ``group_2`` = the whole system (or all solvent), ``axis = 2``. The CV is then
    the solute's z position relative to the system centroid, which itself
    barely moves because the solute is a small fraction of the total mass.

    Parameters
    ----------
    indices: list[tuple(int)], list[list[int]]
        Exactly two groups. The first is the moving (solute) group, the second
        is the reference group. Each group is a nested list / tuple / range of
        atom indices (so both groups are passed as nested sequences, e.g.
        ``[[0, 1, 2], [3, 4, ..., N-1]]``).
    axis: int
        Cartesian component requested as CV: ``0`` (X), ``1`` (Y), ``2`` (Z).

    Notes
    -----
    Uses the unweighted geometric centroid (:func:`barycenter`), matching
    :class:`Component`. No periodic unwrapping is applied: the jax-md backend
    returns the stored (effectively wrapped) coordinates as-is — unlike the
    HOOMD/LAMMPS backends, it ignores ``requires_box_unwrapping``. This is fine
    here because (i) the plain mean is the *correct* reference for a group that
    spans the box (a minimum-image barycenter is ill-defined for a box-spanning
    group), and (ii) the zero-sum-gradient / COM-conservation property is exact
    regardless of wrapping. The two PBC hazards are a *compact* group straddling
    a periodic seam (its plain-mean centroid becomes meaningless) and the
    centroid--centroid separation crossing L/2 (the CV jumps by L). Keep compact
    groups away from the seam and the sampled separation below L/2 — e.g. via
    soft walls — or use a PBC-aware variant. Pick a reference group
    large/symmetric enough that its centroid is stable (the full system is the
    safe default); a small reference group would couple the CV to its own
    fluctuations.
    """

    def __init__(self, indices, axis):
        super().__init__(indices, axis, group_length=2)

    @property
    def function(self):
        axis = self.axis
        return lambda r1, r2: barycenter(r1)[axis] - barycenter(r2)[axis]


class RelativeComponentPBC(AxisCV):
    """
    PBC-robust version of :class:`RelativeComponent`::

        xi = minimum_image( barycenter_pbc(grp1)[axis] - barycenter(grp2)[axis] )

    along a single axis under orthorhombic PBC. Compared with the plain
    :class:`RelativeComponent`, this variant is correct and continuous even when
    the moving group straddles a periodic seam or when the centroid--centroid
    separation approaches L/2:

    - ``grp1`` (the compact, moving group, e.g. the solute) is made whole with
      the anchor-relative minimum-image barycenter (:func:`barycenter_pbc`), so
      its centroid is meaningful even if its atoms wrap across a boundary.
    - ``grp2`` (the reference) uses the plain unweighted mean
      (:func:`barycenter`). This is intentional: a minimum-image barycenter is
      ill-defined for a group that *legitimately spans the box* (e.g. the whole
      system / all solvent), so the reference must NOT be folded.
    - The centroid--centroid separation is wrapped to ``(-L/2, L/2]`` so the CV
      never jumps by L when the moving group crosses the seam.

    The translation-invariance / zero-net-COM-force property of
    :class:`RelativeComponent` is preserved: ``round`` has zero gradient a.e., so
    the CV gradient is still ``+1/N_1`` on grp1 and ``-1/N_2`` on grp2 and sums
    to zero.

    Parameters
    ----------
    indices: list[tuple(int)], list[list[int]]
        Exactly two groups: ``grp1`` (moving/compact) then ``grp2`` (reference).
    axis: int
        Cartesian component: ``0`` (X), ``1`` (Y), ``2`` (Z).
    box: scalar, shape-(3,) array, or shape-(3,3) diagonal matrix
        Orthorhombic box edge lengths. Fixed at construction — assumes an
        invariant box (NVT / NVE).

    Notes
    -----
    ``grp1`` must be compact (intra-group atomic separations < L/2) for the
    anchor-relative wholeness to be valid — true for any molecular solute.
    ``grp2`` is treated as the box-spanning reference; do not pass a compact
    group that can itself straddle the seam as ``grp2``.
    """

    def __init__(self, indices, axis, box):
        super().__init__(indices, axis, group_length=2)
        self.box = np.asarray(box, dtype=float)
        self.requires_box_unwrapping = False

    @property
    def function(self):
        box = self.box
        axis = self.axis
        L = _box_edges(box)[axis]

        def f(r1, r2):
            d = barycenter_pbc(r1, box)[axis] - barycenter(r2)[axis]
            return d - L * np.round(d / L)

        return f


class Distance(TwoPointCV):
    """
    Use the distance of atom groups selected via the indices as collective variable.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Select atom groups via indices. (2 Groups required)
    """

    @property
    def function(self):
        if len(self.groups) == 0:
            return distance
        return lambda r1, r2: distance(barycenter(r1), barycenter(r2))


def distance(r1, r2):
    """
    Returns the distance between two points in space or
    between the barycenters of two groups of points in space.

    Parameters
    ----------
    r1: jax.Array
        Array containing the position in space of the first point or group of points.
    r2: jax.Array
        Array containing the position in space of the second point or group of points.

    Returns
    -------
    distance: float
        Distance between the two points.
    """

    return linalg.norm(r1 - r2)


def _box_edges(box):
    """Normalize `box` to a 1D array of orthorhombic edge lengths."""
    return box if box.ndim <= 1 else np.diag(box)


def distance_pbc(r1, r2, box):
    """
    Minimum-image distance between two points under orthorhombic PBC.

    Parameters
    ----------
    r1, r2: jax.Array
        Positions (or group barycenters) in space.
    box: jax.Array
        Scalar, shape-(3,) array, or shape-(3,3) diagonal matrix giving the
        box edge lengths. Triclinic boxes are not handled by this helper.
    """
    L = _box_edges(box)
    d = r1 - r2
    d = d - L * np.round(d / L)
    return linalg.norm(d)


def barycenter_pbc(positions, box):
    """
    PBC-aware geometric center for a group of points.

    Displacements of every atom in the group are measured relative to
    ``positions[0]`` under the minimum-image convention, averaged, then the
    anchor is added back. Correct whenever every atom in the group lies
    within half a box from the first atom — a safe assumption for molecular
    groups (bond lengths ≪ L/2). For groups whose atoms span more than L/2
    a more sophisticated algorithm would be required.
    """
    L = _box_edges(box)
    anchor = positions[0]
    disps = positions - anchor
    disps = disps - L * np.round(disps / L)
    return anchor + disps.mean(axis=0)


class DifferenceOfDistances(FourPointCV):
    def __init__(self, indices):
        super().__init__(indices)

    @property
    def function(self):
        if len(self.groups) == 0:
            return lambda p1, p2, p3, p4: distance(p1, p2) - distance(p3, p4)
        return lambda p1, p2, p3, p4: (
            distance(barycenter(p1), barycenter(p2))
            - distance(barycenter(p3), barycenter(p4))
        )


class DistancePBC(TwoPointCV):
    """
    Minimum-image distance between two atoms (or group barycenters) under
    orthorhombic periodic boundary conditions.

    Safe to use with HarmonicBias / CVRestraints across box boundaries,
    provided the restrained range stays well below L/2 on every axis. Group
    barycenters are computed with minimum-image reference to the first atom
    of each group, so groups that span a boundary are handled correctly
    as long as intra-group atomic distances stay below L/2 (typical for
    bonded molecular groups).

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Two atom indices, or two groups of atoms.
    box: scalar, shape-(3,) array, or shape-(3,3) diagonal matrix
       Orthorhombic box edge lengths. Fixed at construction — assumes an
       invariant box (NVT / NVE).
    """

    def __init__(self, indices, box):
        super().__init__(indices)
        self.box = np.asarray(box, dtype=float)
        self.requires_box_unwrapping = False

    @property
    def function(self):
        box = self.box
        if len(self.groups) == 0:
            return lambda r1, r2: distance_pbc(r1, r2, box)
        return lambda r1, r2: distance_pbc(
            barycenter_pbc(r1, box), barycenter_pbc(r2, box), box
        )


class DifferenceOfDistancesPBC(FourPointCV):
    """
    d(p1, p2) - d(p3, p4) using minimum-image distances under orthorhombic
    PBC. Multi-atom group barycenters use the same PBC-aware algorithm as
    :class:`DistancePBC`.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Four atom indices (or four groups).
    box: scalar, shape-(3,) array, or shape-(3,3) diagonal matrix
       Orthorhombic box edge lengths (fixed at construction).
    """

    def __init__(self, indices, box):
        super().__init__(indices)
        self.box = np.asarray(box, dtype=float)
        self.requires_box_unwrapping = False

    @property
    def function(self):
        box = self.box
        if len(self.groups) == 0:
            return lambda p1, p2, p3, p4: (
                distance_pbc(p1, p2, box) - distance_pbc(p3, p4, box)
            )
        return lambda p1, p2, p3, p4: (
            distance_pbc(barycenter_pbc(p1, box), barycenter_pbc(p2, box), box)
            - distance_pbc(barycenter_pbc(p3, box), barycenter_pbc(p4, box), box)
        )


class MeanOfDistances(FourPointCV):
    """
    Mean of two distances, (d(p1, p2) + d(p3, p4)) / 2.

    Analogous to :class:`DifferenceOfDistances` but symmetric: useful as a
    reaction-progress coordinate where two bonds form/break together (e.g. the
    two forming C-C bonds of a Diels-Alder reaction). Returns the *mean* rather
    than the raw sum so it stays in single-bond-length units, on the same scale
    as the companion difference CV. Multi-atom groups use geometric-center
    (barycenter) distances.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Four atom indices (or four groups).
    """

    def __init__(self, indices):
        super().__init__(indices)

    @property
    def function(self):
        if len(self.groups) == 0:
            return lambda p1, p2, p3, p4: 0.5 * (distance(p1, p2) + distance(p3, p4))
        return lambda p1, p2, p3, p4: 0.5 * (
            distance(barycenter(p1), barycenter(p2))
            + distance(barycenter(p3), barycenter(p4))
        )


class MeanOfDistancesPBC(FourPointCV):
    """
    Mean of two distances, (d(p1, p2) + d(p3, p4)) / 2, using minimum-image
    distances under orthorhombic PBC. Multi-atom group barycenters use the same
    PBC-aware algorithm as :class:`DistancePBC`. See :class:`MeanOfDistances`
    for the (non-PBC) rationale.

    Parameters
    ----------
    indices: list[int], list[tuple(int)]
       Four atom indices (or four groups).
    box: scalar, shape-(3,) array, or shape-(3,3) diagonal matrix
       Orthorhombic box edge lengths (fixed at construction).
    """

    def __init__(self, indices, box):
        super().__init__(indices)
        self.box = np.asarray(box, dtype=float)
        self.requires_box_unwrapping = False

    @property
    def function(self):
        box = self.box
        if len(self.groups) == 0:
            return lambda p1, p2, p3, p4: 0.5 * (
                distance_pbc(p1, p2, box) + distance_pbc(p3, p4, box)
            )
        return lambda p1, p2, p3, p4: 0.5 * (
            distance_pbc(barycenter_pbc(p1, box), barycenter_pbc(p2, box), box)
            + distance_pbc(barycenter_pbc(p3, box), barycenter_pbc(p4, box), box)
        )


@multicomponent
class Displacement(TwoPointCV):
    """
    Relative displacement between two points in space.

    Parameters
    ----------
    indices: Union[list[int], list[tuple(int)]]
        Indices of the reference atoms (two groups are required).
    """

    @property
    def function(self):
        if len(self.groups) == 0:
            return displacement
        return lambda r1, r2: displacement(barycenter(r1), barycenter(r2))


def displacement(r1, r2):
    """
    Displacement between two points in space or
    between the barycenters of two groups of points in space.

    Parameters
    ----------
    r1: jax.Array
        Array containing the position in space of the first point or group of points.
    r2: jax.Array
        Array containing the position in space of the second point or group of points.

    Returns
    -------
    displacement: jax.Array
        Displacement between the two points.
    """

    return r2 - r1

class CoordinationNumber(TwoPointCV):

    #Covalent radii Cordero from Mendeleev package
    radii_table_default = {
            1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 
            6: 0.73, 7: 0.71, 8: 0.66, 9: 0.57, 10: 0.58, 
            11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07, 
            16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 
            21: 1.7, 22: 1.6, 23: 1.53, 24: 1.39, 25: 1.5, 
            26: 1.42, 27: 1.38, 28: 1.24, 29: 1.32, 30: 1.22, 
            31: 1.22, 32: 1.2, 33: 1.19, 34: 1.2, 35: 1.2, 
            36: 1.16, 37: 2.2, 38: 1.95, 39: 1.9, 40: 1.75, 
            41: 1.64, 42: 1.54, 43: 1.47, 44: 1.46, 45: 1.42, 
            46: 1.39, 47: 1.45, 48: 1.44, 49: 1.42, 50: 1.39, 
            51: 1.39, 52: 1.38, 53: 1.39, 54: 1.4, 55: 2.44, 
            56: 2.15, 57: 2.07, 58: 2.04, 59: 2.03, 60: 2.01, 
            61: 1.99, 62: 1.98, 63: 1.98, 64: 1.96, 65: 1.94, 
            66: 1.92, 67: 1.92, 68: 1.89, 69: 1.9, 70: 1.87, 
            71: 1.87, 72: 1.75, 73: 1.7, 74: 1.62, 75: 1.51, 
            76: 1.44, 77: 1.41, 78: 1.36, 79: 1.36, 80: 1.32, 
            81: 1.45, 82: 1.46, 83: 1.48, 84: 1.4, 85: 1.5, 
            86: 1.5, 87: 2.6, 88: 2.21, 89: 2.15, 90: 2.06, 
            91: 2.0, 92: 1.96, 93: 1.9, 94: 1.87, 95: 1.8, 96: 1.69}
    def __init__(
        self,
        indices,
        nbrs,
        species,
        box,
        species_nn=None,
        cn_exponents=None,
        indices_nn=None,
        switching_function='stable',
    ):
        """
        Coordination Number Collective Variable.

        Parameters
        ----------
        indices : list of tuples
            Two groups: indices[0] = all atom indices (neighbor pool),
            indices[1] = center atoms (A1) whose CN is computed.
        nbrs : jax_md neighbor list
            Edge-list format neighbor list.
        species : array-like
            Element (atomic number) of every particle.
        box : array-like
            Simulation box lengths (3D vector, [0,0,0] for non-periodic).
        species_nn : int, optional
            If set, only count neighbors of this element type.
        cn_exponents : tuple of (int, int), optional
            Exponents (n, m) for switching function. Default (6, 12).
        indices_nn : list or array-like, optional
            Explicit set of atom indices (A2) to consider as coordination
            partners. When None (default), all atoms are candidates.
            When specified, only atoms in indices_nn are counted as neighbors.
            Works as an AND filter with species_nn if both are set.
        switching_function : {'stable', 'legacy'}, optional
            Which form of the rational switching function to use. Default 'stable'.

            - 'stable' uses the algebraic identity
                  1 - r^m = (1 - r^n) * sum_{k=0}^{K-1} r^(k*n),  K = m/n
              so that (1 - r^n)/(1 - r^m) = 1 / sum_{k=0}^{K-1} r^(k*n). The 0/0
              at r = 1 is cancelled before differentiation, giving a smooth,
              finite jax.grad everywhere. Requires m to be an integer multiple
              of n.
            - 'legacy' uses the naive form (1 - r^n) / (1 - r^m + 1e-30). The
              forward value is stable, but jax.grad returns NaN for r within a
              few ulps of 1, which can silently break biased MD (see morpholine
              decomposition incident, 2026-04-21). Kept for reproducing older
              results.
        """
        super().__init__(indices)
        self.nbrs = nbrs
        self.species = species
        self.box = box if np.any(box) else None
        self.species_nn = species_nn
        self.indices_cn = np.array(list(indices[1]))

        # Pre-compute boolean mask for neighbor index filtering (A2).
        # Shape (N,) where N = number of atoms. nn_mask[j] is True if atom j
        # is a valid neighbor candidate. O(1) lookup, JAX-friendly.
        N = len(species)
        if indices_nn is not None:
            nn_indices_list = list(indices_nn)
            if len(nn_indices_list) > 0:
                nn_indices_array = np.array(nn_indices_list, dtype=np.int32)
                self.nn_mask = np.zeros(N, dtype=np.bool_).at[nn_indices_array].set(True)
            else:
                self.nn_mask = np.zeros(N, dtype=np.bool_)
        else:
            self.nn_mask = None

        self.r0_table_species = self.extract_r0_for_species(self.radii_table_default, self.species)
        self.r0_lookup_table, self.element_to_index, self.element_ids_array, self.particle_to_lookup_idx = self.create_r0_lookup_table(self.r0_table_species, self.species)
        self.number_nn_max = np.minimum(int(self.determine_max_neighbors(self.nbrs, self.indices_cn) * 4.0), len(self.species))
        self.num_exp, self.denom_exp = cn_exponents if cn_exponents is not None else (6, 12)

        if switching_function not in ('stable', 'legacy'):
            raise ValueError(
                f"switching_function={switching_function!r} must be 'stable' or 'legacy'."
            )
        # The 'stable' form is only valid when m is an integer multiple of n (so the
        # (1 - r^n) factor can be cancelled analytically). All standard CN exponent
        # pairs in the literature satisfy this ((6,12), (6,18), (6,30), (8,16), ...).
        if switching_function == 'stable' and self.denom_exp % self.num_exp != 0:
            raise ValueError(
                f"cn_exponents=(n={self.num_exp}, m={self.denom_exp}) must satisfy "
                "m % n == 0 for switching_function='stable'. Either pick exponents with "
                "m a multiple of n, or pass switching_function='legacy' (not recommended; "
                "its jax.grad is NaN near r = r0)."
            )
        self.switching_function = switching_function


    def determine_max_neighbors(self, edge_list_obj, particle_idxs):
        edge_list = edge_list_obj.idx
        def determine_num_neighbors(i, num_neighbor_list):
            #num_neighbor_list[i] = np.size(np.where(edge_list[0] == particle_idxs[i], edge_list[1]))
            n_n = np.sum(edge_list[1] == particle_idxs[i])
            num_neighbor_list = num_neighbor_list.at[i].set(n_n)
            return num_neighbor_list

        n = len(particle_idxs)
        num_neighbors = np.zeros(n, dtype=np.int32)
        num_neighbors = jax.lax.fori_loop(0, n, determine_num_neighbors, num_neighbors)
        #Return the biggest number of neighbors
        return np.max(num_neighbors)


    def extract_r0_for_species(self, radii_table_default, species):
        unique_elements = np.unique(species)
        i, j = np.triu_indices(len(unique_elements), k=0)
        pairs = np.stack([unique_elements[i], unique_elements[j]], axis=1)

        return {(int(p[0]), int(p[1])): (radii_table_default[int(p[0])] + radii_table_default[int(p[1])]) * 1.6 for p in pairs}

    def create_r0_lookup_table(self, r0_table_species, species):
        unique_elements = sorted(set(elem for pair in r0_table_species.keys() for elem in pair))
        # Create mapping from element_id to array index in r0_table
        element_to_index = {elem: i for i, elem in enumerate(unique_elements)}
        element_keys = np.array(list(element_to_index.keys()))
        lookup_indices = np.array(list(element_to_index.values()))

        def get_lookup_index(element_id):
            matches = element_keys == element_id
            return np.where(np.any(matches), lookup_indices[np.argmax(matches)], 0)

        n_elements = len(unique_elements)
        r0_table = np.ones((n_elements, n_elements))

        for (elem1, elem2), r0_val in r0_table_species.items():
            i,j = element_to_index[elem1], element_to_index[elem2]
            r0_table = r0_table.at[i, j].set(r0_val)
            r0_table = r0_table.at[j, i].set(r0_val)

        element_ids_array = np.array(unique_elements)
        index_lookup = np.full(np.max(element_ids_array)+1, -1, dtype=np.int32).at[element_ids_array].set(np.arange(0,element_ids_array.size))
        #return r0_table, element_to_index, element_ids_array
        return r0_table, element_to_index, element_ids_array, index_lookup 

    @property
    def function(self):
        # r_cn is passed by PySAGES core (_build) due to TwoPointCV group splitting,
        # but we use self.indices_cn to index into r_all directly because
        # calculate_coordination_number needs full-system positions for the neighbor list.
        return lambda r_all, r_cn: calculate_coordination_number(
                self.nbrs,
                self.indices_cn,
                r_all,
                self.number_nn_max,
                self.r0_lookup_table,
                self.particle_to_lookup_idx,
                self.species,
                self.species_nn,
                self.box,
                self.num_exp,
                self.denom_exp,
                self.nn_mask,
                self.switching_function,
                )

def calculate_coordination_number(edge_list_obj, indices_cn, all_positions,
                                   max_neighbors, r0_dict, element_to_local_index,
                                   all_species, species_nn, box, num_exp, denom_exp,
                                   nn_mask, switching_function='stable'):
    """
    Compute coordination number using a smooth switching function.

    Parameters
    ----------
    nn_mask : jax.Array or None
        Boolean array of shape (N,). If not None, only atoms j where
        nn_mask[j] == True are considered as coordination partners.
    """
    n = len(indices_cn)
    N = all_positions.shape[0]

    # Update neighborlist; only pass box= for periodic systems because
    # space.free() does not accept the box keyword
    if box is not None:
        edge_list_obj = edge_list_obj.update(all_positions, neighbor=edge_list_obj.idx, box=box)
    else:
        edge_list_obj = edge_list_obj.update(all_positions, neighbor=edge_list_obj.idx)
    edge_list = edge_list_obj.idx

    def get_all_neighbors():
        all_neighbors = np.full((n, max_neighbors), -1, dtype=np.int32)

        def add_neighbor(i, state):
            neighbors, counts = state
            particle_idx = indices_cn[i]

            # Collect all unique neighbors for particle i
            unique_targets_to_particle_i = np.unique(
                np.where(edge_list[1] == particle_idx, edge_list[0], -1),
                size=max_neighbors, fill_value=-1)

            mask = (unique_targets_to_particle_i >= 0) & (unique_targets_to_particle_i != particle_idx)

            # Apply neighbor index filter (A2) if specified
            if nn_mask is not None:
                safe_idx = np.where(unique_targets_to_particle_i >= 0,
                                    unique_targets_to_particle_i, 0)
                mask = mask & nn_mask[safe_idx]

            # Apply species filter if specified
            if species_nn is not None:
                final_mask = (all_species[unique_targets_to_particle_i] == species_nn) & mask
            else:
                final_mask = mask

            filtered_targets = np.where(final_mask, unique_targets_to_particle_i, -1)
            valid_count = np.sum(filtered_targets >= 0)

            neighbors = neighbors.at[i].set(filtered_targets)
            counts = counts.at[i].set(valid_count)

            return neighbors, counts

        init_state = (all_neighbors, np.zeros(n, dtype=np.int32))
        neighbors, counts = jax.lax.fori_loop(0, n, add_neighbor, init_state)

        return neighbors, counts

    all_neighbor_indices, all_neighbor_counts = get_all_neighbors()

    # Compute distances vectorized across all particles
    particle_positions = all_positions[indices_cn]  # Shape: (n, 3)

    valid_mask = all_neighbor_indices >= 0
    safe_neighbor_indices = np.where(valid_mask, all_neighbor_indices, 0)

    neighbor_positions = all_positions[safe_neighbor_indices]  # Shape: (n, max_neighbors, 3)
    diff = neighbor_positions - particle_positions[:, None, :]  # Shape: (n, max_neighbors, 3)

    # Apply minimal-image-convention for periodic systems (assumes orthorhombic box)
    if box is not None:
        half_box = box / 2.0
        diff_mic = np.remainder(diff + half_box, box) - half_box
    else:
        diff_mic = diff

    distances = np.linalg.norm(diff_mic, axis=2)  # Shape: (n, max_neighbors)

    def normalize_distances(distances, r0_table, element_ids_center, element_ids_neighbors):
        reference_distances = r0_table[element_ids_center[:, None], element_ids_neighbors]
        return distances / reference_distances

    element_ids_center = element_to_local_index[all_species[indices_cn]]
    element_ids_neighbors = element_to_local_index[all_species[safe_neighbor_indices]]
    normalized_distances = normalize_distances(distances, r0_dict, element_ids_center, element_ids_neighbors)

    if switching_function == 'stable':
        # Replace padding slots with a safe finite r_norm so both the forward value and
        # the autodiff gradient stay finite everywhere. Writing NaN for padding and
        # relying on np.where to filter it out is not gradient-safe: JAX's where VJP
        # multiplies the unselected branch's cotangent by 0, but 0 * NaN = NaN, so
        # padding entries contaminate the Jacobian.
        safe_r_norm = np.where(valid_mask, normalized_distances, 2.0)

        # Gradient-stable rational switching function using the identity
        #   1 - r^m = (1 - r^n) * sum_{k=0}^{K-1} r^(k*n),   K = m / n
        # so that (1 - r^n) / (1 - r^m) = 1 / sum_{k=0}^{K-1} r^(k*n). The cancellation
        # is done analytically before differentiation, so jax.grad gives a smooth and
        # finite derivative even at r = 1 (where the naive form's autodiff is NaN).
        K = denom_exp // num_exp
        r_n = safe_r_norm ** num_exp
        denom = np.ones_like(r_n)
        r_k = np.ones_like(r_n)
        for _ in range(K - 1):
            r_k = r_k * r_n
            denom = denom + r_k
        cn_per_neighbor = 1.0 / denom

        # valid_mask is a fixed neighbor-list padding mask independent of positions, so
        # multiplying through it zeros out padding contributions in both the forward
        # pass and the Jacobian without any NaN flow.
        cn_value = np.sum(cn_per_neighbor * valid_mask)
    else:
        # 'legacy' branch: original PLUMED-style rational form. Forward values are fine,
        # but jax.grad returns NaN when any r_norm is within ~1e-8 of 1.0, which can
        # silently blow up biased simulations (morpholine decomposition, 2026-04-21).
        # Retained for reproducing older results only.
        masked_distances = np.where(valid_mask, distances, np.nan)
        normalized_distances = normalize_distances(
            masked_distances, r0_dict, element_ids_center, element_ids_neighbors
        )
        mask = ~np.isnan(normalized_distances)
        cn_terms = np.where(
            mask,
            (1.0 - normalized_distances**num_exp)
            / (1.0 - normalized_distances**denom_exp + 1e-30),
            0.0,
        )
        cn_value = np.sum(cn_terms)

    return cn_value
