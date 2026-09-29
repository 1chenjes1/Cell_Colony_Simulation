import numpy as np
import itertools
import networkx as nx
from networkx.algorithms import isomorphism
from classes.interaction import Interaction

def is_only_self_regulation(mat):
    off_diag = mat.copy()
    np.fill_diagonal(off_diag, 0)
    return np.all(off_diag == 0)


def find_node_mapping(G1, G2):
    em = isomorphism.categorical_edge_match("label", 0)
    GM = isomorphism.DiGraphMatcher(G1, G2, edge_match=em)
    if GM.is_isomorphic():
        return next(GM.isomorphisms_iter())  # dict: {node_in_G1: node_in_G2}
    return None

def relabel_results(df, mapping):
    """
    Reorder columns of df according to node mapping, 
    without changing column names to Species objects.
    """
    df = df.copy()
    reverse_mapping = {v: k for k, v in mapping.items()}
    N = len(mapping)

    # Reorder the first N columns (assumed to correspond to species/parameters)
    first_N_cols = df.columns[:N]
    new_order = [reverse_mapping[i] for i in range(N)]
    df[first_N_cols] = df[first_N_cols].iloc[:, new_order].values

    # Reorder cluster_means (if present)
    if "cluster_means" in df.columns:
        def reorder_cm(cm):
            if isinstance(cm, list) or isinstance(cm, np.ndarray):
                arr = np.array(cm)
                if arr.ndim == 2 and arr.shape[1] == N:
                    return arr[:, new_order].tolist()
                elif arr.ndim == 1 and len(arr) == N:
                    return arr[new_order].tolist()
            return cm
        df["cluster_means"] = df["cluster_means"].apply(reorder_cm)

    return df

def generate_simulate_unique_interactions(species_list,param_grid):
    # TODO: uses inter.simulate, idk if this is compatible with new classes (env, singal, cell)
    N = len(species_list)
    possible_values = [-1, 0, 1]
    
    results = []
    unique_graphs = []
    unique_matrices = []

    for combo in itertools.product(possible_values, repeat=N*N):
        mat = np.array(combo).reshape((N, N))

        if is_only_self_regulation(mat):
            continue 

        # Convert to graph
        G = matrix_to_graph(mat)

        inter = Interaction(mat)
        inter.graph = G

        found_iso = False

        for (G_unique, inter_unique) in unique_graphs:
            mapping = find_node_mapping(G, G_unique)
            if mapping is not None:
                # Isomorphic = reuse canonical simulation and relabel
                inter.result = relabel_results(inter_unique.result, mapping)
                found_iso = True
                break
        if not found_iso: # == Unique graph
            inter.result = inter.simulate(species_list, param_grid, 1)
            unique_graphs.append((G, inter))
            unique_matrices.append(mat)

        results.append(inter)

    return unique_matrices, results

def matrix_to_graph(mat):
    N = mat.shape[0]
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i)
    for i in range(N):
        for j in range(N):
            if mat[i, j] != 0:
                G.add_edge(i, j, label=mat[i, j])
    return G