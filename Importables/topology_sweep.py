import pandas as pd
import multiprocessing as mp
import numpy as np

from .unique_signalling_matrices import build_interaction_matrix
from .param_sweep import run_param_sweep, create_param_grid


def _run_topology_task(task):
    topo_id, mat, param_grid, param_list, n_cells, n_colonies, gene_list, signal_list, T_final, euler_step_size = task

    # Build matrix for this topology
    M = build_interaction_matrix(mat)

    # Run parameter sweep
    timeseries_df = run_param_sweep(
        param_grid, 
        param_list, 
        n_cells, 
        n_colonies,
        gene_list, 
        M, 
        topo_id,
        signal_list, 
        T_final, 
        euler_step_size
    )

    # Tag metadata
    timeseries_df["topology_id"] = topo_id
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            timeseries_df[f"g{i+1}_to_g{j+1}"] = mat[i, j]

    return timeseries_df


def run_signal_sweep(param_list, n_cells, n_colonies, gene_list, signal_list, T_final, euler_step_size, signalling_matrices):
    print("Starting signal sweep...")
    
    param_grid = create_param_grid(param_list, n_samples=1)

    # Build tasks
    tasks = []

    for topo_id, mat in enumerate(signalling_matrices):

        tasks.append((
            topo_id,
            mat,
            param_grid, param_list,
            n_cells, n_colonies, gene_list, signal_list,
            T_final, euler_step_size
        ))

    if not tasks:
        return pd.DataFrame()

    # Run only the signal sweep in parallel. Unique-topology filtering happens before this.
    n_processes = min(mp.cpu_count(), len(tasks))
    with mp.Pool(processes=n_processes) as pool:
        timeseries_all = pool.map(_run_topology_task, tasks)

    # Unpack results
    timeseries_df_all = pd.concat(timeseries_all, ignore_index=True)
    timeseries_df_all.fillna(0, inplace=True)

    return timeseries_df_all

def _compact_obs_df(df, obs_cols, signal_cols):
    """Downcast saved metadata only; simulation data/parameters stay untouched."""
    obs_df = df[obs_cols + signal_cols].copy()

    for col in ["topology_id", "param_set_id", "sample_id", "cell_id", "k2_id"]:
        obs_df[col] = obs_df[col].astype("category")

    for col in ["time", "k1", "k2", *signal_cols]:
        obs_df[col] = obs_df[col].astype(np.float16)

    return obs_df

def _compact_topology_params(df, topology_param_cols):
    topology_params = df.groupby("topology_id", observed=True)[topology_param_cols].first()
    return topology_params.astype(np.int8)