# SPDX-License-Identifier: MIT
# See LICENSE.md and CONTRIBUTORS.md at https://github.com/SSAGESLabs/PySAGES

"""
Collective Variables that are computed from the Cartesian coordinates.
"""

from jax import numpy as np
from jax.numpy import linalg
import jax
from pysages.colvars.core import AxisCV, TwoPointCV, multicomponent
import jax.debug as jdb
import time
import logging

#logging.basicConfig(level=logging.DEBUG)
#jax.config.update('jax_log_compiles', True)


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
        species, #element of every single particle (list),
        box, #Simulation box lengths (assumed constant)
        species_nn=None, #Optional, species for which CN should be computed (if none, element-independent CN is computed)
        cn_exponents=None, #Optional, tuple of exponents n and m for CN switching function definition
    ):
        #super().__init__(indices, group_length=2)
        super().__init__(indices)
        self.nbrs = nbrs
        self.species = species
        self.box=box
        self.species_nn = species_nn
        self.indices_cn = np.array(list(indices[1]))
        #jdb.print('indices_cn = {indices_cn}', indices_cn=self.indices_cn)
        #Construct subset of total radii_table for the species involved. Multiply covalent bond lengths by 1.6 to obtain critical/nearly broken lengths.
        #self.r0_table_species = {p: (radii_table_default[p[0]] + radii_table_default[p[1]]) * 1.6 for p in list(combinations(list(set(self.species))))}
        self.r0_table_species = self.extract_r0_for_species(self.radii_table_default, self.species)
        #self.indices_nn = self.nbrs.idx[1][np.where(np.isin(self.nbrs.idx[0], self.indices[1]))]
        self.r0_lookup_table, self.element_to_index, self.element_ids_array, self.particle_to_lookup_idx = self.create_r0_lookup_table(self.r0_table_species, self.species)
        #Estimate max number of neighbors from the current neighborlist; multiply by 1.5 to account for potential fluctuations 
        #self.number_nn_max = np.minimum( int(len(self.nbrs.idx[1][np.where(np.isin(self.nbrs.idx[0], indices[1][0]))]) * 1.5), len(species) )
        self.number_nn_max = np.minimum( int(self.determine_max_neighbors(self.nbrs, self.indices_cn) * 4.0), len(self.species) )
        self.num_exp, self.denom_exp = cn_exponents if cn_exponents is not None else (6,12)
        #jdb.print("n={n}",n=self.num_exp)
        #jdb.print("m={m}", m=self.denom_exp)


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
        #return lambda r_all, r_cn: calculate_coordination_number(r_all, r_cn, self.nbrs, self.species, self.indices, self.number_nn)
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
                self.denom_exp
                )

def calculate_coordination_number(edge_list_obj, indices_cn, all_positions, max_neighbors, r0_dict, element_to_local_index,all_species, species_nn, box, num_exp, denom_exp):
#def calculate_coordination_number(edge_list, indices_cn, all_positions, max_neighbors, r0_dict, all_species, species_nn):
    n = len(indices_cn)
    N = all_positions.shape[0]

    #Update neighborlist
    edge_list_obj = edge_list_obj.update(all_positions, neighbor=edge_list_obj.idx, box=box)
    edge_list = edge_list_obj.idx

    def get_all_neighbors():
        # Create boolean mask for edges originating from our particles
        #particle_mask = np.isin(edge_list[0], indices_cn)

        #unique_sources = indices_cn
        #unique_targets = np.unique(np.where(particle_mask, edge_list[1], N), size=max_neighbors, fill_value=N)

        all_neighbors = np.full((n, max_neighbors), -1, dtype=np.int32)

        def add_neighbor(i, state):
            neighbors, counts = state
            particle_idx = indices_cn[i]

            #Collect all unique neighbors for particle i
            unique_targets_to_particle_i = np.unique(np.where(edge_list[1] == particle_idx, edge_list[0], -1), size=max_neighbors, fill_value=-1)

            mask = (unique_targets_to_particle_i >= 0) & (unique_targets_to_particle_i != particle_idx)
            if species_nn is not None:
                #If we only want the CN for specific neighboring elements (e.g hydrogen)

                final_mask = all_species[unique_targets_to_particle_i] == species_nn & mask
            else:
                final_mask = mask

            filtered_targets = np.where(final_mask, unique_targets_to_particle_i, -1)

            # Count valid neighbors
            valid_count = np.sum(filtered_targets >= 0)

            neighbors = neighbors.at[i].set(filtered_targets)
            counts = counts.at[i].set(valid_count)

            return neighbors, counts

        init_state = (all_neighbors, np.zeros(n, dtype=np.int32))
        neighbors, counts = jax.lax.fori_loop(0, n, add_neighbor, init_state)

        return neighbors, counts

    # Get all neighbors
    all_neighbor_indices, all_neighbor_counts = get_all_neighbors()

    # Compute distances vectorized across all particles
    particle_positions = all_positions[indices_cn]  # Shape: (n, 3)

    # Handle invalid neighbors safely
    valid_mask = all_neighbor_indices >= 0
    safe_neighbor_indices = np.where(valid_mask, all_neighbor_indices, 0)

    # Get neighbor positions - shape: (n, max_neighbors, 3)
    neighbor_positions = all_positions[safe_neighbor_indices]

    # Compute differences - broadcasting over neighbor dimension
    diff = neighbor_positions - particle_positions[:, None, :]  # Shape: (n, max_neighbors, 3)

    #Apply minimal-image-convention (if non-PBC system, box should be [0.,0.,0.], assumes orthorhombic box)
    half_box = box/2.0
    diff_mic = np.remainder(diff + half_box, box) - half_box
    # Compute distances
    distances = np.linalg.norm(diff_mic, axis=2)  # Shape: (n, max_neighbors)

    # Mask invalid distances
    masked_distances = np.where(valid_mask, distances, np.nan)

    def normalize_distances(distances, r0_table, element_ids_center, element_ids_neighbors):

        reference_distances = r0_table[element_ids_center[:,None], element_ids_neighbors]
        #jdb.print('Reference distances : {reference_distances}', reference_distances=reference_distances)
        return distances/reference_distances

    # Compute coordination numbers
    #normalized_distances = masked_distances / 1.7
    element_ids_center = element_to_local_index[all_species[indices_cn]]
    element_ids_neighbors = element_to_local_index[all_species[safe_neighbor_indices]]
    normalized_distances = normalize_distances(masked_distances, r0_dict, element_ids_center, element_ids_neighbors)
    mask = ~np.isnan(normalized_distances)
    #numerator = 1.0 - normalized_distances**6
    #denominator = 1.0 - normalized_distances**12

    cn_terms = np.where(mask, (1.0 - normalized_distances**num_exp)/(1.0 - normalized_distances**denom_exp), 0.0)
    cn_value = np.sum(cn_terms)

    #jdb.print('CN terms : {cn_terms}', cn_terms=cn_terms)
    #jdb.print('CN : {cn_value}', cn_value=cn_value)
    #jdb.print('NH2 coordinates : {pos}', pos=all_positions[np.array([1567, 1568, 1569])])
    #jdb.print('all_neighbor_indices : {all_neighbor_indices}', all_neighbor_indices=all_species[safe_neighbor_indices])
    #jdb.print('Masked distances : {masked_distances}', masked_distances=normalized_distances)
    #jdb.breakpoint()
    return cn_value
