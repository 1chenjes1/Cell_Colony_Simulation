# Importables/param_sweep.py
"""
Latin-Hypercube parameter sampling, plus running a full parameter +
initial-condition sweep over a single fixed GRN topology — parallelized
across parameter sets.
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
from pyDOE import lhs
from tqdm import tqdm
import anndata as ad

from classes.cell import Cell
from classes.env import Environment

from .initial_conditions import generate_y0s


def create_param_grid(param_list, n_samples=5):
    """
    Draw `n_samples` Latin-Hypercube parameter sets spanning each Rate's
    [lower_bound, upper_bound] range.

    Args:
        param_list (list[Rate]): the Rate parameters to sample.
        n_samples (int): number of parameter sets to draw. Note: if a
            Rate's lower_bound == upper_bound (a fixed parameter, e.g.
            k1 in this project), every sample takes that single value.

    Returns:
        list[dict]: one dict per sample, mapping each Rate's `name` to
        its sampled value for that sample.
    """
    param_names = [k.name for k in param_list]
    n_params = len(param_names)
    lower_bounds = np.array([k.lower_bound for k in param_list])
    upper_bounds = np.array([k.upper_bound for k in param_list])

    lhs_samples = lhs(n_params, n_samples)
    scaled_samples = lower_bounds + lhs_samples * (upper_bounds - lower_bounds)
    param_grid = [
        {param_names[i]: scaled_samples[sample_idx, i] for i in range(n_params)}
        for sample_idx in range(n_samples)
    ]

    return param_grid


def _run_param_set_task(task):
    """
    Worker function for `run_param_sweep`'s process pool: sets one
    parameter set's values, simulates all `num_ics` colonies for it, and
    returns their combined trajectory rows.

    Must remain a plain module-level function (not a closure/lambda) so
    it can be pickled for `multiprocessing.Pool.map`. Each worker process
    gets its own pickled copy of `param_list`/`gene_list`/`signal_list`,
    so mutating the Rate objects here (`k.set_value(...)`) is safe and
    does not leak across parameter sets running in other processes.

    Args:
        task (tuple): (param_idx, param_set, param_list, gene_list,
            interaction_matrix, signal_list, n_cells, num_ics, T_final,
            euler_step_size) — see `run_param_sweep` for what each is.

    Returns:
        pd.DataFrame: this parameter set's trajectory rows across all
        `num_ics` colonies, tagged with `param_set_id`, `sample_id`,
        `initial_condition`, and each parameter's sampled value.
    """
    (param_idx, param_set, param_list, gene_list, interaction_matrix,
     signal_list, n_cells, num_ics, T_final, euler_step_size) = task

    # --- 1. Set parameter values (this process's own copy of param_list) ---
    for k in param_list:
        k.set_value(param_set[k.name])

    y0_matrix = generate_y0s(gene_list, n_cells, num_ics)

    histories = []
    for ic in range(y0_matrix.shape[0]):
        # --- 2. Set up environment with sampled y0s ---
        y0_list = [y0_matrix[ic, i, :] for i in range(n_cells)]
        cells = [Cell(gene_list, y0_list[i], interaction_matrix, i) for i in range(n_cells)]

        env = Environment(n_cells, gene_list, interaction_matrix, signal_list, T_final, euler_step_size, noise_width=0.2)
        env.cells = cells

        # --- 3. Run simulation ---
        env.run()
        df = env.get_full_history()

        # --- 4. Tag metadata for later analysis ---
        df["initial_condition"] = ic
        df["param_set_id"] = param_idx
        # Unique across the whole sweep without needing a shared counter
        # across processes (a plain incrementing `run_counter` wouldn't
        # be meaningful once parameter sets run in parallel).
        df["sample_id"] = f"{param_idx}_{ic}"
        for param_name, param_value in param_set.items():
            df[param_name] = param_value

        histories.append(df)

    return pd.concat(histories, ignore_index=False)


def big_df_to_anndata(big_df, gene_list):
    """
    Convert a combined trajectory dataframe (as returned by
    `run_param_sweep`) into an AnnData object: one gene-expression matrix
    (`X`) plus a metadata table (`obs`) with one row per (cell, timepoint,
    colony) observation.
 
    Args:
        big_df (pd.DataFrame): combined trajectory dataframe. Must
            contain one column per gene plus whatever metadata columns
            you want kept (time, sample_id, cell_id, signal_*, swept
            parameter values, etc.) — every non-gene column becomes an
            `obs` column.
        gene_list (list[Gene]): the genes to pull into `X`; each
            `Gene.name` must match a column in `big_df` exactly.
 
    Returns:
        anndata.AnnData: `X` = gene expression matrix (float32), `obs` =
        every other column (index reset to a plain string range so
        AnnData doesn't complain about a non-unique/non-string index),
        `var` = gene names.
    """
    gene_names = [g.name for g in gene_list]
    
    obs_cols = [c for c in big_df.columns if c not in gene_names]
 
    X = big_df[gene_names].to_numpy(dtype=np.float32)
    obs_df = big_df[obs_cols].reset_index(drop=True)
    obs_df.index = obs_df.index.astype(str)
 
    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=pd.DataFrame(index=gene_names),
    )
 
    return adata


def run_param_sweep(
    param_grid, param_list, n_cells, num_ics,
    gene_list, interaction_matrix, signal_list,
    T_final, euler_step_size,
    save_path=None, n_processes=None,
):
    """
    For a single fixed `interaction_matrix` (topology), simulate every
    parameter set in `param_grid` in parallel (one process per parameter
    set), each with `num_ics` independently sampled colonies of
    `n_cells` cells, and return the combined trajectory history.

    Args:
        param_grid (list[dict]): parameter sets from `create_param_grid`
            (or any list of {rate_name: value} dicts).
        param_list (list[Rate]): the Rate objects to update for each
            parameter set (shared, mutable — referenced by `gene_list`/
            `signal_list`; each worker process gets its own pickled copy,
            see `_run_param_set_task`).
        n_cells (int): cells per colony.
        num_ics (int): number of colonies (independent initial-condition
            draws) per parameter set.
        gene_list (list[Gene]): genes shared by every cell.
        interaction_matrix (np.ndarray): the fixed GRN + signaling
            interaction matrix for this topology (e.g. from
            `sweep.topology.build_interaction_matrix` — must be the FULL
            matrix, not just the signal->gene block; see note below).
        signal_list (list[Signal]): signals shared across the colony
            (pass `[]` or `None` for no signaling).
        T_final (float): simulation end time.
        euler_step_size (float): integration time step.
        save_path (str, None): Path to save the combined trajectory
            history as a CSV. If None, no file is saved.
        n_processes (int, None): number of worker processes. Defaults to
            `min(cpu_count(), len(param_grid))`.

    Returns:
        anndata.AnnData: combined trajectory across every parameter set
        and colony (see `big_df_to_anndata`) — `obs` includes `sample_id`,
        `param_set_id`, `initial_condition`, and each parameter's sampled
        value.
    """
    if not param_grid:
        return pd.DataFrame()

    tasks = [
        (param_idx, param_set, param_list, gene_list, interaction_matrix,
         signal_list, n_cells, num_ics, T_final, euler_step_size)
        for param_idx, param_set in enumerate(param_grid)
    ]

    n_processes = n_processes or min(mp.cpu_count(), len(tasks))
    with mp.Pool(processes=n_processes) as pool:
        all_histories = list(tqdm(
            pool.imap_unordered(_run_param_set_task, tasks), 
            total=len(tasks),
            desc="Running parameter sweep",
        ))

    big_df = pd.concat(all_histories, ignore_index=False)
    big_df.fillna(0, inplace=True)

    adata = big_df_to_anndata(big_df, gene_list)

    # --- Save to file ---
    if save_path is not None:
        adata.write_h5ad(save_path, index=False)
        print(f"Saved full simulation history to {save_path}")

    return adata