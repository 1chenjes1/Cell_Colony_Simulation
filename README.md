# CellColonySim

A stochastic simulator for multicellular gene regulatory networks (GRNs)
with intercellular signaling. Each cell runs its own noisy gene
expression dynamics (Euler-Maruyama); cells in a colony communicate
through a shared, deterministically-integrated (RK4) pool of secreted
signals.

<img src="/figures/System.png" width="500">

On top of the core simulator, the project supports two ways of exploring
signaling topologies:

1. **One topology you choose** — pick the signal→gene wiring yourself and
   sweep it across parameter sets (`Importables/param_sweep.py`).
2. **Every unique topology, automatically** — exhaustively enumerate and
   de-duplicate (by graph isomorphism) every possible signal→gene wiring
   for a given number of genes, then simulate all of them and export the
   topologies themselves as graphs for downstream GNN work
   (`Importables/unique_signalling_matrices.py` +
   `Importables/topology_sweep.py`, driven by `main.py`).

Run `signal_topology_sweep.py` as follows:
```bash
python signal_topology_sweep.py \
  --n_genes 3 \
  --n_cells 10 \
  --n_colonies 10 \
  --k2 1.0 \
  --T_final 250 \
  --euler_step_size 0.2
```