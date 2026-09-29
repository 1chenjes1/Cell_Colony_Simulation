
import numpy as np
import networkx as nx
from networkx.algorithms import isomorphism
import torch
from torch_geometric.data import Data

import graph_isomorphism as giu


def build_fixed_blocks(n_genes):
    # every gene inhibits every other gene
    core_block = -np.ones((n_genes, n_genes), dtype=int)
    np.fill_diagonal(core_block, 0)

    # gene i produces signal i (identity matrix)
    g_to_s_block = np.eye(n_genes, dtype=int)

    return core_block, g_to_s_block


def build_interaction_matrix(mat):
    """
    Full network:

            genes        signals
          0 ... n-1    n ... 2n-1

    genes | core_block | g_to_s_block
    sig   |    mat     |    zeros
    """

    n = mat.shape[0]

    core_block, g_to_s_block = build_fixed_blocks(n)

    top = np.hstack((core_block, g_to_s_block))
    bottom = np.hstack((mat, np.zeros((n, n), dtype=int)))

    return np.vstack((top, bottom))

def is_valid_matrix(mat):
    """
    mat is ONLY the signal -> gene block.

    Rows = signals source
    Columns = genes target
    """

    # 1. Must have at least one signal-gene interaction (edge)
    if np.sum(mat != 0) == 0:
        return False, None
    
    # 2. Must only have one outgoing signal
    # outgoing_connections = np.sum(mat != 0, axis=1)  # count non-zero entries per row
    # if np.any(outgoing_connections > 1):
    #     return False, None

    # Build full network
    full_mat = build_interaction_matrix(mat)

    G = giu.matrix_to_graph(full_mat)

    # 3. Full network must be weakly connected (no disconnected subgraphs)
    if not nx.is_weakly_connected(G):
        return False, None
    
    return True, G

def generate_unique_signalling_interactions(n_genes):
    possible_values = [-1, 0, 1]

    unique_graphs = []
    unique_matrices = []

    hash_buckets = {}
    
    total = 3 ** (n_genes * n_genes)

    print("Generating all possible networks and filtering for graph ismorphism...")

    for combo in tqdm(itertools.product(possible_values, repeat=n_genes*n_genes), total=total):

        # signal -> gene block
        mat = np.array(combo).reshape((n_genes, n_genes))

        # is_valid_matrix takes s_to_g block and returns entire full matrix
        valid, G = is_valid_matrix(mat)

        if not valid:
            continue

        h = nx.weisfeiler_lehman_graph_hash(G, edge_attr='label')

        found_iso = False
        if h in hash_buckets:
            em = isomorphism.categorical_edge_match("label", 0)
            for idx in hash_buckets[h]:
                GM = isomorphism.DiGraphMatcher(G, unique_graphs[idx], edge_match=em)
                if GM.is_isomorphic():
                    found_iso = True
                    break

        if not found_iso:
            idx = len(unique_graphs)
            unique_graphs.append(G)
            unique_matrices.append(mat.copy())
            hash_buckets.setdefault(h, []).append(idx)

    return unique_matrices

def adjacency_to_pyg(mat, topo_id):
    """
    Convert adjacency matrix into PyTorch Geometric graph format.
    
    mat:
        adjacency matrix (genes x genes)
        values represent interaction type (+1 activation, -1 inhibition)
    """

    mat = torch.tensor(mat, dtype=torch.float)

    # edge_index: [2, num_edges]
    edge_index = (mat != 0).nonzero(as_tuple=False).T

    # edge weights (+1/-1)
    edge_attr = mat[mat != 0].view(-1, 1)

    # Node features
    # simple identity encoding of genes
    n_nodes = mat.shape[0]
    x = torch.eye(n_nodes)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        topology_id=torch.tensor([topo_id])
    )

    return data

