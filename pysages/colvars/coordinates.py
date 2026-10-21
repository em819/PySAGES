# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg
import jax
from pysages.colvars.core import AxisCV, CollectiveVariable, TwoPointCV, multicomponent, FourPointCV
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
        r0_multiplier=None,
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
        r0_multiplier : float, optional — factor applied to the sum of covalent radii, 
                        r0 = r0_multiplier * (r_i + r_j). Default 1.6.
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

        #self.r0_table_species = self.extract_r0_for_species(self.radii_table_default, self.species)
        r0_multiplier = 1.6 if r0_multiplier is None else float(r0_multiplier)
        if not r0_multiplier > 0.0:
            raise ValueError(f"r0_multiplier={r0_multiplier!r} must be positive.")
        self.r0_table_species = self.extract_r0_for_species(self.radii_table_default,self.species, r0_multiplier)

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


    def extract_r0_for_species(self, radii_table_default, species, multiplier=1.6):
        unique_elements = np.unique(species)
        i, j = np.triu_indices(len(unique_elements), k=0)
        pairs = np.stack([unique_elements[i], unique_elements[j]], axis=1)

        return {(int(p[0]), int(p[1])): (radii_table_default[int(p[0])] + radii_table_default[int(p[1])]) * multiplier for p in pairs}

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

    # Padding/filtered slots have safe_neighbor_indices == 0; when the center atom
    # is itself index 0 these become self-pairs with a zero displacement, and
    # np.linalg.norm has a 0/0 = NaN gradient there. Masking after the norm cannot
    # remove it (the where() VJP multiplies the dead branch by 0, and 0 * NaN = NaN),
    # so make invalid displacements non-zero *before* the norm. valid_mask is a
    # position-independent padding mask, so this is a no-op on both the value
    # (zeroed by * valid_mask below) and the gradient of every valid slot.
    diff_mic = np.where(valid_mask[:, :, None], diff_mic, 1.0)
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


class ChargeSeparation(CollectiveVariable):
    """
    Signed charge-separation distance between an acid/base and the excess proton
    (H3O+) or proton hole (OH-) in the solvent, after Xu, Liu, Malosso, Loche and
    Ceriotti, arXiv:2610.13265 (Eqs. 7-9).

    Every hydrogen j is shared among the proton-owner sites i (heavy atoms) with
    Gaussian weights, the formal charge of each water oxygen is the number of
    hydrogens it owns minus two, and the CV sums charge times distance to the
    reference point over the water oxygens:

        n_{j->i} = exp(-|r_j - r_i|^2 / 2 sigma^2) / sum_k exp(-|r_j - r_k|^2 / 2 sigma^2)
        q_w      = sum_j n_{j->w} - n_ref_water
        s        = sum_w q_w |mic(r_w - r_ref)|

    s ~ 0 for the undissociated solute, s ~ +d after an acidic dissociation (H3O+
    at distance d) and s ~ -d after a basic one (OH- at distance d); during proton
    transfer and Grotthuss hops s changes continuously.

    Differences to the paper: the hydrogens are shared among all heavy atoms passed
    as owner sites (not only O), so that C-H and N-H hydrogens of a peptide stay on
    their own atom instead of being assigned to a nearby water; all pairs are
    evaluated (no neighbor-list cutoff; the weights beyond the paper's 3.5 A cutoff
    are < 1e-30); r_ref is the centroid of one or more reference atoms. Distances use
    the minimum-image convention of the orthorhombic box given at construction, so
    the CV is only valid at constant volume.

    Parameters
    ----------
    indices : list of three groups
        [hydrogen atoms, owner sites (heavy atoms), reference atoms].
    box : array-like
        Orthorhombic box lengths (3,).
    water_sites : array-like of int
        Positions of the water oxygens within the owner-site group.
    sigma : float, optional
        Width of the Gaussian assignment (Angstrom). Default 0.3 (paper value).
    n_ref_water : float, optional
        Hydrogens owned by a neutral water oxygen. Default 2.
    pair_guard : bool, optional
        Guard against water autoionisation (default False: the CV of the paper).
        A single acidic or basic dissociation creates one charged water (H3O+ or
        OH-); a solvent that holds a hydronium and a hydroxide at the same time has
        autoionised. With the guard such configurations are mapped onto s = 0, the
        value of the undissociated solute:

            P = sum_w S((n_w - n_hydronium) / guard_width)     (hydronium character)
            M = sum_w S((n_hydroxide - n_w) / guard_width)     (hydroxide character)
            s_guard = [1 - S(P M / guard_x0)] s

        with S the C2 smoothstep of :class:`IonSiteDistance` (0 below 0, 1 above 1).
        Waters with n_hydroxide <= n_w <= n_hydronium contribute exactly zero to P and
        M, so P M and its gradient are exactly zero for neutral water, one H3O+ (also
        when its proton is shared with a neighbour), one OH- and the undissociated
        solute: there s_guard = s with the same gradient. A full pair moves s_guard
        to the undissociated value, so a bias that flattens the free energy along s
        opposes its formation by the free energy of the dissociated state relative to
        the undissociated one. Partial pairs are not suppressed: with the defaults a
        water-water proton transfer still shifts s by up to ~1 A before the guard
        switches on. Configurations with a pair enter the statistics at s ~ 0.

        Not recommended with ABF. The factor multiplies s, so while a pair forms the
        gradient grows with s and the CV becomes very stiff: in an Asp solution with
        the H3O+ 7.7 A from the site and a water at 1.60 H, |grad s| rose from 15 to
        430 and the curvature along the gradient from -150 to 7400 1/A. ABF applies
        the bin mean force times this gradient, which can exceed what a 0.5 fs step
        integrates stably. For P M >= guard_x0 the CV is exactly flat (zero
        gradient), and the ABF force estimate diverges as 1 / |grad s| at the edge
        of that region, which lies at s ~ 0.
    n_hydronium, n_hydroxide : float, optional
        H counts above / below which a water starts to count as hydronium /
        hydroxide. Default n_ref_water + 0.1 / n_ref_water - 0.1. At sigma = 0.3 A in a
        1410-water solution of an Asp peptide, waters stayed above 1.91 H in
        undissociated frames, but one dropped to 1.60 H in 1 of 1602 frames with an
        H3O+ (spontaneous partial autoionisation), which switches the guard on.
        Waters above 2.1 H shared the excess proton of the H3O+.
    guard_width : float, optional
        Width (H count units) over which the hydronium / hydroxide share goes from 0
        to 1. Default 0.8, so a full H3O+ (3 H) or OH- (1 H) has share 1.
    guard_x0 : float, optional
        Value of P M at which s_guard reaches 0. Default 0.5.
    charge_sites : array-like of int, optional
        Positions within the owner-site group whose charge enters s, with their
        neutral H counts in `n_ref_sites`:

            s = sum_i (n_i - n_ref_sites_i) |mic(r_i - r_ref)|,   i in charge_sites

        Default None: the water oxygens with n_ref_water (the CV above). Including
        the solute's other proton-owner sites (e.g. amide N, carbonyl O; not the
        reference atoms) keeps s at the distance of the charge when a solvent ion
        takes a proton from or gives one to such a site, instead of reading s ~ 0
        (undissociated). With all solute sites neutral the extra terms are tiny: the
        sites hold their neutral H count to ~1e-6 in prepared structures and to ~1e-4
        in thermal frames (the Gaussian share of a hydrogen-bonded H, which the
        water-only CV leaves as a fractional charge on the donor water), so s changes
        by ~1e-4 A. The pair guard still looks at the waters only.
    n_ref_sites : array-like of float, optional
        Neutral H counts of the `charge_sites` (same length; required with them).
    """

    def __init__(self, indices, box, water_sites, sigma=0.3, n_ref_water=2.0,
                 pair_guard=False, n_hydronium=None, n_hydroxide=None, guard_width=0.8,
                 guard_x0=0.5, charge_sites=None, n_ref_sites=None):
        super().__init__(indices, 3)
        self.box = np.asarray(box, dtype=float)
        self.water_sites = np.asarray(water_sites, dtype=np.int32)
        self.sigma = float(sigma)
        self.n_ref_water = float(n_ref_water)
        self.requires_box_unwrapping = False
        self.pair_guard = bool(pair_guard)
        self.n_hydronium = float(self.n_ref_water + 0.1 if n_hydronium is None else n_hydronium)
        self.n_hydroxide = float(self.n_ref_water - 0.1 if n_hydroxide is None else n_hydroxide)
        self.guard_width = float(guard_width)
        self.guard_x0 = float(guard_x0)
        if self.pair_guard:
            if not (self.n_hydroxide < self.n_ref_water < self.n_hydronium):
                raise ValueError("pair_guard: need n_hydroxide < n_ref_water < n_hydronium.")
            if not (self.guard_width > 0.0 and self.guard_x0 > 0.0):
                raise ValueError("pair_guard: guard_width and guard_x0 must be positive.")
            if (self.n_hydronium + self.guard_width > self.n_ref_water + 1.0 + 1e-12
                    or self.n_hydroxide - self.guard_width < self.n_ref_water - 1.0 - 1e-12):
                raise ValueError("pair_guard: a full H3O+ / OH- must have share 1 "
                                 "(n_hydronium + guard_width <= n_ref_water + 1 and "
                                 "n_hydroxide - guard_width >= n_ref_water - 1).")
        if (charge_sites is None) != (n_ref_sites is None):
            raise ValueError("charge_sites and n_ref_sites must be given together.")
        self.charge_sites = self.n_ref_sites = None
        if charge_sites is not None:
            sites = np.asarray(charge_sites, dtype=np.int32).ravel()
            n_ref = np.asarray(n_ref_sites, dtype=float).ravel()
            n_owner = len(self.groups[1])
            if len(n_ref) != len(sites):
                raise ValueError("charge_sites and n_ref_sites must have the same length.")
            if (len(sites) == 0 or len(np.unique(sites)) != len(sites)
                    or int(sites.min()) < 0 or int(sites.max()) >= n_owner):
                raise ValueError("charge_sites must be distinct positions within the "
                                 "owner-site group.")
            self.charge_sites = sites
            self.n_ref_sites = n_ref

    @property
    def guard(self):
        """Guard parameters for :func:`charge_separation` (None without the guard)."""
        if not self.pair_guard:
            return None
        return (self.n_hydronium, self.n_hydroxide, self.guard_width, self.guard_x0)

    @property
    def function(self):
        guard = self.guard
        # getattr: CVs pickled before charge_sites existed carry the waters only
        charge_sites = getattr(self, "charge_sites", None)
        n_ref_sites = getattr(self, "n_ref_sites", None)
        return lambda r_h, r_sites, r_ref: charge_separation(
            r_h, r_sites, r_ref, self.box, self.water_sites, self.sigma,
            self.n_ref_water, guard, charge_sites, n_ref_sites)[0]


def charge_pair_indicator(n_water, n_hydronium, n_hydroxide, guard_width):
    """
    Hydronium character P, hydroxide character M and their product P M (> 0 only
    when the solvent holds a hydronium and a hydroxide at the same time, i.e. has
    autoionised) from the H counts of the water oxygens.
    """
    P = np.sum(_smoothstep((n_water - n_hydronium) / guard_width))
    M = np.sum(_smoothstep((n_hydroxide - n_water) / guard_width))
    return P, M, P * M


def charge_separation(r_h, r_sites, r_ref, box, water_sites, sigma, n_ref_water,
                      guard=None, charge_sites=None, n_ref_sites=None):
    """
    Value of :class:`ChargeSeparation` and the number of hydrogens owned by every
    owner site (returned for monitoring, e.g. protonation of other solute sites or
    additional H3O+/OH- pairs in the solvent). `guard` = (n_hydronium, n_hydroxide,
    guard_width, guard_x0) applies the autoionisation guard of
    :class:`ChargeSeparation`; None (default) gives the plain CV. `charge_sites` /
    `n_ref_sites`: the sites whose charge enters s and their neutral H counts
    (default None: the water sites with n_ref_water).
    """
    def mic(d):
        return d - box * np.round(d / box)

    if charge_sites is None:
        charge_sites, n_ref_sites = water_sites, n_ref_water

    # Gaussian sharing of every hydrogen among all owner sites (rows sum to one)
    d2 = np.sum(mic(r_h[:, None, :] - r_sites[None, :, :]) ** 2, axis=-1)
    n_owned = np.sum(jax.nn.softmax(-d2 / (2.0 * sigma**2), axis=1), axis=0)
    q_water = n_owned[charge_sites] - n_ref_sites

    # reference point: centroid of the reference atoms, made whole around the first one
    r0 = r_ref[0] + np.mean(mic(r_ref - r_ref[0]), axis=0)
    d_w = np.sqrt(np.sum(mic(r_sites[charge_sites] - r0) ** 2, axis=-1) + 1e-12)

    s = np.sum(q_water * d_w)
    if guard is not None:
        n_hydronium, n_hydroxide, guard_width, guard_x0 = guard
        _, _, x = charge_pair_indicator(n_owned[water_sites], n_hydronium, n_hydroxide,
                                        guard_width)
        s = (1.0 - _smoothstep(x / guard_x0)) * s
    return s, n_owned


class IonSiteDistance(CollectiveVariable):
    """
    Distance between an ionizable site and the water molecule that carries the excess
    proton (H3O+, ``mode='acid'``) or the proton hole (OH-, ``mode='base'``). Meant to
    be biased together with a :class:`CoordinationNumber` of the same site: the
    coordination number drives the proton transfer, this CV the separation of the
    ion pair. Adapted from the "Hydrodist" CV of Park et al. (JACS 2006, 128, 11318)
    and Guo, Albesa and Wexler (JPCB 2023, 127, 8446).

        n_i   = sum_{j in H} w_{j->i}                  (H owned by water oxygen i)
        w_{j->k} = exp(-|r_j - r_k|^2 / 2 sigma^2) / sum_l exp(-|r_j - r_l|^2 / 2 sigma^2)
        e_i   = S((x_i + a) / 2a),   x_i = n_i - n0 (acid) or n0 - n_i (base)
        E     = sum_i e_i                              (number of ions, 0 ... 1)
        d_ref = -tau ln sum_i exp(-d_i / tau)          (soft distance to the nearest water)
        w_0   = ramp(1 - E - b)                        (weight of the nearest-water term)
        xi    = [sum_i e_i d_i + w_0 d_ref] / [E + w_0]

    with d_i the distance of water oxygen i to the site (averaged over the site atoms,
    e.g. both carboxylate O). Every hydrogen j is shared among all owner atoms k (all
    heavy atoms, including the site and the water oxygens) with Gaussian weights, as in
    :class:`ChargeSeparation`. e_i is the ion share of water i: S(t) = t^3 (10 - 15 t +
    6 t^2) is the C2 smoothstep on [0, 1] (0 below, 1 above), so e_i = 0 for
    x_i <= -a and e_i = 1 for x_i >= a. ramp(y) is a C2-smooth version of max(y, 0)
    that is exactly 0 for y <= -b and exactly y for y >= b, with b = 0.05.

    Behaviour:

    - Neutral water owns 2.00 H, H3O+ 3.00 and OH- 1.00 (sigma = 0.25 A: a strong
      hydrogen bond adds < 0.05). With n0 = 2.5, a = 0.4 (acid) every neutral water
      has exactly zero ion share and gradient, and an H3O+ has share 1.
    - While no H3O+/OH- exists (e.g. the proton is still on the site), E = 0 and
      xi = d_ref, the distance to the nearest water, i.e. the water that would receive
      the proton. The CV therefore has an O(1) gradient everywhere and stays continuous
      when the ion forms on that water. It is never flat on purpose: ABF estimates the
      generalized force from W p with W = pinv(J^T), which grows as 1 / |grad xi|, so a
      CV that is constant in part of the space passes through arbitrarily small
      gradients at the edge of that part and produces unbounded force estimates
      (``use_pinv`` only handles an exactly zero gradient).
    - Once a full ion exists (E = 1) the nearest-water term has exactly zero weight and
      xi is the ion's distance. When the proton hops between two waters they share
      n_A + n_B = 5 H (acid), so e_A + e_B = 1 because S(t) + S(1 - t) = 1: xi =
      e_A d_A + e_B d_B moves smoothly from one distance to the other (over ~0.04 A of
      proton motion for a 2.45 A hop at sigma = 0.25 A, i.e. several MD steps), without
      the nearest-water term cutting in. A larger sigma makes hops smoother, but a
      neutral water must stay below n0 - a.
    - In the no-ion state the bias acts on the waters around the site; ABF slowly
      explores configurations in which the nearest water is pushed away from the site.
      This only costs sampling time.
    - The site's own H count does not enter, which keeps the CV distinguishable from
      the coordination number during the transfer.
    - A base mode tracks OH- (n0 = 1.5). Prefer simulating basic residues in their
      protonated form (acid mode).

    The owner group must contain every heavy atom that can carry a hydrogen (in
    practice all heavy atoms); otherwise e.g. the H of a C-H group is assigned to the
    nearest water oxygen. Distances use the minimum-image convention of the
    orthorhombic box given at construction, so the CV is only valid at constant
    volume. All hydrogen - owner pairs are evaluated (no neighbor list).

    Parameters
    ----------
    indices : list of three groups
        [site atoms, owner atoms (all heavy atoms, incl. site and water O), hydrogen
        atoms]. The site atoms must also be owners.
    box : array-like
        Orthorhombic box: scalar, shape-(3,) edge lengths or shape-(3, 3) diagonal.
    water_sites : array-like of int
        Positions of the water oxygens within the owner group.
    mode : {'acid', 'base'}
        'acid' tracks the water with the most H (H3O+), 'base' the fewest (OH-).
    n0 : float, optional
        Water H count at which the ion share is 1/2. Default 2.5 (acid) / 1.5 (base).
    a : float, optional
        Half-width of the ion-share step (H count units): share 0 below n0 - a, 1
        above n0 + a. Default 0.4. Neutral water (2 H) must have share 0 and the ion
        (3 H in acid, 1 H in base mode) share 1.
    sigma : float, optional
        Width of the Gaussian H assignment (Angstrom). Default 0.25.
    tau : float, optional
        Softness of the nearest-water distance d_ref (Angstrom). Default 0.2.
    """

    def __init__(self, indices, box, water_sites, mode="acid", n0=None, a=0.4,
                 sigma=0.25, tau=0.2):
        super().__init__(indices, 3)
        if mode not in ("acid", "base"):
            raise ValueError(f"mode={mode!r} must be 'acid' or 'base'.")
        sign = 1.0 if mode == "acid" else -1.0
        n0 = float((2.5 if mode == "acid" else 1.5) if n0 is None else n0)
        if not a > 0.0:
            raise ValueError(f"a={a!r} must be positive.")
        if not (sigma > 0.0 and tau > 0.0):
            raise ValueError("sigma and tau must be positive.")
        if not sign * (2.0 - n0) <= -a:
            raise ValueError(f"n0={n0}, a={a}: neutral water (2 H) must have zero ion share "
                             f"({'n0 - a >= 2' if mode == 'acid' else 'n0 + a <= 2'}).")
        if not sign * ((3.0 if mode == "acid" else 1.0) - n0) >= a:
            ion = "H3O+ (3 H)" if mode == "acid" else "OH- (1 H)"
            raise ValueError(f"n0={n0}, a={a}: the {ion} must have ion share 1 "
                             f"({'n0 + a <= 3' if mode == 'acid' else 'n0 - a >= 1'}).")
        atoms = [self.indices[g].tolist() for g in self.groups]
        water_sites = [int(i) for i in water_sites]
        if len(water_sites) == 0 or len(set(water_sites)) != len(water_sites):
            raise ValueError("water_sites must be non-empty and unique.")
        if min(water_sites) < 0 or max(water_sites) >= len(atoms[1]):
            raise ValueError("water_sites must index the owner group.")
        if not set(atoms[0]) <= set(atoms[1]):
            raise ValueError("The site atoms must also be owner atoms (so that the site "
                             "keeps its own hydrogens).")
        if set(atoms[1][i] for i in water_sites) & set(atoms[0]):
            raise ValueError("The site atoms must not be water oxygens.")
        self.box = _box_edges(np.asarray(box, dtype=float))
        self.water_sites = np.asarray(water_sites, dtype=np.int32)
        self.mode = mode
        self.sign = sign
        self.n0 = n0
        self.a = float(a)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.requires_box_unwrapping = False

    @property
    def function(self):
        return lambda r_site, r_own, r_h: ion_site_distance(
            r_site, r_own, r_h, self.box, self.water_sites, self.sign, self.n0, self.a,
            self.sigma, self.tau)[0]


def _smooth_ramp(x, a):
    """C2 ramp: 0 for x <= -a, x for x >= a and L (v^3 - v^4 / 2) in between
    (v = (x + a) / L, L = 2a)."""
    L = 2.0 * a
    v = np.clip((x + a) / L, 0.0, 1.0)
    return np.where(x >= a, x, L * v**3 * (1.0 - 0.5 * v))


def _smoothstep(t):
    """C2 smoothstep: 0 for t <= 0, 1 for t >= 1 and t^3 (10 - 15 t + 6 t^2) in between;
    S(t) + S(1 - t) = 1."""
    t = np.clip(t, 0.0, 1.0)
    return t**3 * (10.0 - 15.0 * t + 6.0 * t**2)


# Half-width of the ramp that switches the nearest-water term off as the ion forms
_ION_SITE_DISTANCE_B = 0.05


def ion_site_distance(r_site, r_own, r_h, box, water_sites, sign, n0, a, sigma, tau):
    """
    Value of :class:`IonSiteDistance` and its intermediates for monitoring: the H count
    of every water oxygen, the normalized weights of the waters, the normalized weight
    of the nearest-water term and the nearest-water distance d_ref.
    """
    def mic(d):
        return d - box * np.round(d / box)

    # H owned by every water oxygen (Gaussian sharing among all owners, rows sum to one)
    d2_h = np.sum(mic(r_h[:, None, :] - r_own[None, :, :]) ** 2, axis=-1)
    n_w = np.sum(jax.nn.softmax(-d2_h / (2.0 * sigma**2), axis=1), axis=0)[water_sites]

    # ion shares; neutral waters get exactly zero, a full ion exactly one
    e = _smoothstep((sign * (n_w - n0) + a) / (2.0 * a))
    b = _ION_SITE_DISTANCE_B
    w0 = _smooth_ramp(1.0 - np.sum(e) - b, b)
    norm = np.sum(e) + w0

    d2_site = np.sum(mic(r_own[water_sites][:, None, :] - r_site[None, :, :]) ** 2, axis=-1)
    d_w = np.mean(np.sqrt(d2_site + 1e-12), axis=1)
    d_ref = -tau * jax.nn.logsumexp(-d_w / tau)
    xi = (np.sum(e * d_w) + w0 * d_ref) / norm
    return xi, n_w, e / norm, w0 / norm, d_ref
