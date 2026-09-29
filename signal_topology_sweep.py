from classes.species import Gene, Signal
from classes.rate import Rate

from Importables.unique_signalling_matrices import build_interaction_matrix, generate_unique_signalling_interactions, adjacency_to_pyg
from Importables.topology_sweep import run_signal_sweep, _compact_obs_df, _compact_topology_params

import argparse
import os
import anndata as ad
import numpy as np
import pandas as pd
import torch 


def parse_args():
    parser = argparse.ArgumentParser(description="Run GRN + signaling network sweep.")
    parser.add_argument(
        "--n_genes", type=int, default=3,
        help="Number of genes (and signals) in the system."
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Filename prefix for saved outputs. Defaults to '{n_genes}Gene' if not given."
    )
    parser.add_argument(
        "--k2", type=float, default=1.0,
        help="k2 rate value."
    )
    parser.add_argument(
        "--n_cells", type=int, default=10,
        help="Number of cells in a colony"
    )
    parser.add_argument(
        "--n_colonies", type=int, default=10,
        help="Number of colonies to simulate"
    )
    parser.add_argument(
        "--T_final", type=int, default=250,
        help="Number of timesteps"
    )
    parser.add_argument(
        "--euler_step_size", type=float, default=0.2,
        help="Step size for Euler's method"
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    n_genes = args.n_genes
    prefix = args.prefix if args.prefix is not None else f"{n_genes}Gene"
    k2_values = [args.k2]
    n_cells = args.n_cells
    n_colonies = args.n_colonies
    T_final = args.T_final
    euler_step_size = args.euler_step_size


    all_adatas = []

    # ── Parameters & objects defined once, outside the loop ──────────────────
    k1 = Rate('k1', 6, 6)
    k2 = Rate('k2', k2_values[0], k2_values[0])  # initial value; updated each iteration
    param_list = [k1, k2]

    gene_list = [
        Gene(f"g{i+1}", k1, 0)
        for i in range(n_genes)
    ]

    signal_list = [
        Signal(f"s{i+1}", k2, 0)
        for i in range(n_genes)
    ]

    signalling_matrices = generate_unique_signalling_interactions(n_genes)
    print(f"unique topologies before loop filter: {len(signalling_matrices)}")


    # subsample to 10k if there are more than that
    if len(signalling_matrices) > 10000:
        rng = np.random.default_rng(seed=42)  # seed for reproducibility
        idx = rng.choice(len(signalling_matrices), size=10000, replace=False)
        signalling_matrices = [signalling_matrices[i] for i in idx]
    
    print(f"unique topologies after subsampling: {len(signalling_matrices)}")

    pyg_graphs = [
        adjacency_to_pyg(mat, topo_id)
        for topo_id, mat in enumerate(signalling_matrices)
    ]

    gene_cols = [g.name for g in gene_list]

    signal_cols = [
        f"signal_{s.name}"
        for s in signal_list
    ]
    obs_cols = ["topology_id", "time", "param_set_id", "sample_id", "cell_id", "k2_id", "k1", "k2"]
    # ─────────────────────────────────────────────────────────────────────────

    for i, k2_v in enumerate(k2_values):
        print(f"Starting simulations for k2 = {k2_v}")

        # Just mutate the rate value — Signal objects already hold a reference to k2
        k2.value = k2_v

        timeseries_df_all = run_signal_sweep(param_list, n_cells, n_colonies, gene_list, signal_list, T_final, euler_step_size, signalling_matrices)

        time_steps = [0, 250]
        sub_df = timeseries_df_all[np.isclose(timeseries_df_all['time'].values[:, None], time_steps).any(axis=1)].copy()
        sub_df["k2"] = k2_v
        sub_df["k2_id"] = i
        sub_df["k1"] = 6

        topology_param_cols = [c for c in sub_df.columns if "_to_" in c]

        X = sub_df[gene_cols].to_numpy(dtype=np.float16, copy=True)
        obs_df = _compact_obs_df(sub_df, obs_cols, signal_cols)
        adata_timeseries = ad.AnnData(
            X=X,
            obs=obs_df,
            var=pd.DataFrame(index=gene_cols),
        )

        adata_timeseries.uns["topology_params"] = _compact_topology_params(sub_df, topology_param_cols)

        all_adatas.append(adata_timeseries)

    for i, adata in enumerate(all_adatas):
        adata.obs_names = [f"{i}_{idx}" for idx in adata.obs_names]

    adata_all = ad.concat(all_adatas, join='outer')
    adata_all.uns['k2_values']       = np.asarray(k2_values, dtype=np.float16)
    adata_all.uns['topology_params'] = all_adatas[0].uns['topology_params']

    outdir = f"/bilbo_team/lab-members/jessica-chen/network_topo/Simulation_Results/{prefix}"
    os.makedirs(outdir, exist_ok=True)
    adata_all.write(
        f"{outdir}/Simulation_Results.h5ad",
        compression="gzip"
    )

    torch.save(
        pyg_graphs,
        f"{outdir}/Graphs.pt"
    )
    print("Done!")