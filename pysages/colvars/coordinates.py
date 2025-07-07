# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg
import jax
from pysages.colvars.core import AxisCV, TwoPointCV, multicomponent


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

jax_cn_fn_container = {
        'is_defined': False, 
        'run_cn_fn': None, 
        'r0_table_species' : None, 
        'r0_lookup_table' : None}

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
        indices, #list of tuples of indices list, first tuple includes all indices, second tuple those whose CN should be computed
        nbrs, #jax-md neighborlist (edge-list)
        species, #element of every single particle (list)
        species_nn=None, #Optional, species for which CN should be computed (if none, element-independent CN is computed)
    ):
        #super().__init__(indices, group_length=2)
        super().__init__(indices)
        self.nbrs = nbrs
        self.species = species
        self.species_nn = species_nn
        self.indices_cn = np.array(list(indices[1]))
        #Construct subset of total radii_table for the species involved. Multiply covalent bond lengths by 1.6 to obtain critical/nearly broken lengths.
        #self.r0_table_species = {p: (radii_table_default[p[0]] + radii_table_default[p[1]]) * 1.6 for p in list(combinations(list(set(self.species))))}
        self.r0_table_species = self.extract_r0_for_species(self.radii_table_default, self.species)
        #self.indices_nn = self.nbrs.idx[1][np.where(np.isin(self.nbrs.idx[0], self.indices[1]))]
        self.r0_lookup_table, self.element_to_index, self.element_ids_array, self.particle_to_lookup_idx = self.create_r0_lookup_table(self.r0_table_species, self.species)
        #print(f'indices_nn : {self.indices_nn}')
        #Estimate max number of neighbors from the current neighborlist; multiply by 1.5 to account for potential fluctuations 
        self.number_nn_max = np.minimum( int(len(self.nbrs.idx[1][np.where(np.isin(self.nbrs.idx[0], indices[1][0]))]) * 1.5), len(species) )
        print(f'Indices : {self.indices}')

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

        #return r0_table, element_to_index, element_ids_array
        return r0_table, element_to_index, element_ids_array, jax.vmap(get_lookup_index)(species) 

    @property
    def function(self):
        #return lambda r_all, r_cn: calculate_coordination_number(r_all, r_cn, self.nbrs, self.species, self.indices, self.number_nn)
        return lambda r_all, r_cn: calculate_coordination_number(self.nbrs.idx, self.indices_cn, r_all, self.number_nn_max, self.r0_table_species, self.species, self.species_nn)

def calculate_coordination_number(edge_list, indices_cn, all_positions, max_neighbors, r0_dict, all_species, species_nn):

    n = len(indices_cn)
    N = all_positions.shape[0]
    num_edges = edge_list.shape[1]
    def get_neighbors_for_particle(particle_idx):
        is_source = edge_list[0] == particle_idx

        all_neighbors = np.where(is_source, edge_list[1], N)

        sorted_neighbors = np.sort(all_neighbors)
        work_neighbors = sorted_neighbors[:max_neighbors]

        is_different = np.concatenate([
            np.array([True]),  # First element is always kept
            np.diff(work_neighbors) > 0  # Keep if different from previous
        ])

        is_valid = work_neighbors < N
        #Only select those neighbors that have correct element
        is_correct_species = True if species_nn == None else all_species[work_neighbors] == species_nn
        keep_mask = is_different & is_valid & is_correct_species
        final_neighbors = np.full(max_neighbors, -1)

        def place_neighbor(i, state):
            final_neighbors, placed_count = state

            # Check if we should place this neighbor
            should_place = keep_mask[i] & (placed_count < max_neighbors)
            neighbor_to_place = work_neighbors[i]

            # Update the array at the current placed_count position
            final_neighbors = np.where(
                should_place & (np.arange(max_neighbors) == placed_count),
                neighbor_to_place,
                final_neighbors
            )

            # Update count
            placed_count = placed_count + should_place.astype(np.int32)

            return final_neighbors, placed_count

        init_state = (final_neighbors, np.int32(0))
        final_neighbors, actual_count = jax.lax.fori_loop(
            0, max_neighbors, place_neighbor, init_state
        )

        return final_neighbors, actual_count

    def compute_normalized_distances():
        pass

    def compute_coordination_number(all_distances, r0_lookup=None, species_nn=None):
       #First compute the normalized distances |Ri - Rj|/r_0
       #normalized_distances = compute_normalized_distances(all_distances, r0_lookup, species_nn)

       #For now, use global r0 value of 3A
       normalized_distances = all_distances / 1.7
       #Apply CN formula from colvars package with n=6 and m=12
       cn = np.nansum( (1. - (normalized_distances)**6)/(1- (normalized_distances)**12) )
       return cn


    def compute_distances_for_particle(i):
        particle_idx = indices_cn[i]
        particle_pos = all_positions[particle_idx]

        neighbor_idxs, neighbor_count = get_neighbors_for_particle(particle_idx)

        valid_mask = neighbor_idxs >= 0
        safe_neighbor_idxs = np.where(valid_mask, neighbor_idxs, 0)
        neighbor_positions = all_positions[safe_neighbor_idxs]

        diff = neighbor_positions - particle_pos[None, :]
        distances = np.linalg.norm(diff, axis=1)
        #masked_distances = np.where(valid_mask, distances, np.inf)
        masked_distances = np.where(valid_mask, distances, np.nan)

        return masked_distances, neighbor_idxs, neighbor_count


    all_distances, all_neighbor_indices, all_neighbor_counts = jax.vmap(compute_distances_for_particle)(np.arange(n))
    return compute_coordination_number(all_distances)
