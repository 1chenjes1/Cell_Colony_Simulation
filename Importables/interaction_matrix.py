# Importables/interaction_matrix.py
"""
Build interaction matrix to govern gene to gene, gene to signal, and signal to signal interactions in each colony.
Assume signals do not affect one another.
"""
import numpy as np

def build_interaction_matrix(g_to_g, g_to_s, s_to_s):
    """
    Full network:

            genes        signals
          0 ... n-1    n ... 2n-1

    genes |   g_to_g   |   g_to_s
    sig   |   s_to_s   |   zeros
    """

    n = g_to_g.shape[0]

    top = np.hstack((g_to_g, g_to_s))
    bottom = np.hstack((s_to_s, np.zeros((n, n), dtype=int)))

    return np.vstack((top, bottom))
