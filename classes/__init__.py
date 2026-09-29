"""
classes package

Core simulation classes for the gene-regulatory-network (GRN) / cell-signaling
model:

- species.py     : Gene / Signal species definitions
- rate.py        : bounded kinetic rate parameter
- cell.py        : single-cell GRN state + stochastic (Euler-Maruyama) update
- signalpool.py  : shared extracellular signal pool + deterministic (RK4) update
- env.py         : Environment orchestrating cells + signal pool over time
- interaction.py : wrapper around an interaction matrix (work in progress)
- system.py      : DEPRECATED legacy ODE system, not used in the signal sweep
"""