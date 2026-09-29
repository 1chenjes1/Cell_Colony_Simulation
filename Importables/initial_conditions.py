# Importables/initial_conditions.py
"""
Latin-Hypercube sampling of cell initial conditions (gene expression
states) for the signal sweep.
"""

import numpy as np
from pyDOE import lhs


def generate_y0s(genes, n_cells, num_ics=10):
    """
    Generate `num_ics` independent Latin-Hypercube-sampled initial
    conditions, each covering all `n_cells` cells' gene expression state.

    Each gene's sampled range is [0, k + 1], where k is that gene's
    current production rate (`gene.k.value`) — i.e. samples span from no
    expression up to (roughly) one unit above the gene's max steady-state
    output.

    Args:
        genes (list[Gene]): genes whose initial values are being sampled.
        n_cells (int): number of cells per initial-condition draw.
        num_ics (int): number of independent initial-condition draws
            (e.g. one per simulated "colony").

    Returns:
        np.ndarray: shape (num_ics, n_cells, n_genes), where
        y0_matrix[ic, i, :] is cell i's initial gene expression vector
        for IC draw `ic`.
    """
    n_genes = len(genes)
    total_samples = num_ics * n_cells

    lower_bounds = np.zeros(n_genes)
    upper_bounds = np.array([g.k.value + 1 for g in genes])

    lhs_samples = lhs(n_genes, samples=total_samples)
    y0_samples = lower_bounds + lhs_samples * (upper_bounds - lower_bounds)
    y0_samples = np.round(y0_samples, decimals=1)

    # Reshape to (num_ics, n_cells, n_genes)
    y0_matrix = y0_samples.reshape(num_ics, n_cells, n_genes)
    return y0_matrix


def generate_y0s_one_pattern(genes, n_cells, num_ics=10,
                              target_gene=0,
                              high_value=6,
                              low_value=0.5,
                              radius=0.5):
    """
    Generate initial conditions clustered around a single fixed
    expression pattern (one gene "on" at `high_value`, all others "off"
    at `low_value`), with small uniform jitter — useful for testing
    whether a topology preserves or breaks a specific starting pattern,
    rather than sampling broadly like `generate_y0s`.

    Args:
        genes (list[Gene]): genes whose initial values are being sampled.
        n_cells (int): number of cells per initial-condition draw.
        num_ics (int): number of independent initial-condition draws.
        target_gene (int): index of the gene to set "on" (`high_value`).
        high_value (float): expression level for the "on" gene.
        low_value (float): expression level for all other genes.
        radius (float): half-width of the uniform jitter added to every
            gene's value.

    Returns:
        np.ndarray: shape (num_ics, n_cells, n_genes), same layout as
        `generate_y0s`.
    """
    n_genes = len(genes)
    total_samples = num_ics * n_cells

    # Create center vector
    center = np.full(n_genes, low_value)
    center[target_gene] = high_value

    # Small uniform perturbation
    perturb = np.random.uniform(
        low=-radius,
        high=radius,
        size=(total_samples, n_genes)
    )

    y0_samples = center + perturb

    # Prevent negatives if needed
    y0_samples = np.clip(y0_samples, 0, None)

    return y0_samples.reshape(num_ics, n_cells, n_genes)