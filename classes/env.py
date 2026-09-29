# classes/env.py
"""
Defines Environment, which orchestrates a population of Cells sharing a
single SignalPool over time, and provides utilities for exporting the
combined simulation history to a tidy DataFrame.
"""
 
import numpy as np
from classes.cell import Cell
from classes.signalpool import SignalPool, NullSignalPool
import pandas as pd

class Environment:
    """
    Couples a population of `n_cells` Cells to a shared SignalPool and
    advances them together in time.
 
    Note: `self.cells` starts as an empty list — it is *not* populated by
    `__init__`. The caller is expected to construct and append `Cell`
    instances after creating the Environment, e.g.:
 
        env = Environment(n_cells, gene_list, interaction_matrix,
                           signals, T_final, euler_step_size)
        for i in range(n_cells):
            env.cells.append(Cell(gene_list, y0[i], interaction_matrix, cell_id=i))
 
    Attributes:
        n_cells (int): Number of cells expected in the population.
        gene_list (list[Gene]): Genes shared by every cell (topology).
        interaction_matrix (np.ndarray): Shared GRN + signaling
            interaction matrix (see `Cell` for its layout).
        T_final (float): Simulation end time.
        euler_step_size (float): Integration time step `dt`.
        time (float): Current simulation time.
        noise_width (float): Magnitude of stochastic noise applied to
            each cell's update (see `Cell.update`).
        cells (list[Cell]): The cell population (populated externally,
            see note above).
        signal_pool (SignalPool | NullSignalPool): Shared signal pool.
    """

    def __init__(self, n_cells, gene_list, interaction_matrix, signals, T_final, euler_step_size, noise_width=0.1, pool_override=None):
        self.n_cells = n_cells
        self.gene_list = gene_list
        self.interaction_matrix = interaction_matrix 
        self.T_final = T_final
        self.euler_step_size = euler_step_size
        self.time = 0.0
        self.noise_width = noise_width

        self.cells = []
        self.signal_pool = self.make_signal_pool(signals, pool_override=pool_override)


    def make_signal_pool(self, signals, pool_override=None):
        """
        Choose which SignalPool implementation to use.
 
        Returns `pool_override` if one is given (e.g. a `FixedSignalPool`
        for debugging). Otherwise returns a `NullSignalPool` if there are
        no signals defined or only a single cell (no population-level
        signaling to speak of), else a normal `SignalPool`.
        """

        if pool_override is not None:
            return pool_override
        if signals is None or len(signals) == 0 or self.n_cells == 1:
            return NullSignalPool()
        else:
            return SignalPool(signals)

    def update(self):
        """
        Advance the whole environment by one time step:
 
          1. Average each cell's instantaneous secretion into the signal
             pool (`Cell.get_secretion`), averaged over `n_cells`.
          2. Advance the signal pool by one RK4 step using that averaged
             secretion, and record its new state.
          3. Advance every cell by one Euler-Maruyama step using the
             *updated* signal pool, and record each cell's new state.
        """

        # 1 - Get cell secretion -> signal concentration
        secretion = sum(cell.get_secretion(self.signal_pool) 
                        for cell in self.cells)/ self.n_cells
        
        # 2 - Update signal pool
        self.signal_pool.update(secretion, self.euler_step_size)
        self.signal_pool.record_state(self.time)

        # 3 - Update cells using new signal conc
        for cell in self.cells:
            cell.update(self.signal_pool, self.euler_step_size, self.noise_width)
            cell.record_state(self.time)

    def run(self):
        """
        Advance the environment in fixed steps of `euler_step_size`
        until `time` reaches `T_final`.
        """

        while self.time < self.T_final:
            self.time += self.euler_step_size
            self.update()

    def get_full_history(self):
        """
        Assemble the full simulation history (all cells + the signal
        pool) into a single tidy DataFrame.
 
        Args:
            colony_id: identifier for this particular cell colony /
                simulation replicate, used to build `sample_id`.
 
        Returns:
            pd.DataFrame: one row per (cell, timepoint), with columns for
            each gene, each signal (prefixed "signal_"), `time`,
            `sample_id` (= f"{topology_id}_{colony_id}") and `cell_id`.
        """
        
        # Collect all cell histories
        dfs = []
        for i, cell in enumerate(self.cells):
            df_cell = cell.get_history_df().copy()
            df_cell['cell_id'] = i
            dfs.append(df_cell)
        
        # Concatenate cell dataframes along columns
        df_full = pd.concat(dfs, axis=0).reset_index()
        
        # Collect signal history and set time as index
        df_signals = self.signal_pool.get_history_df()
        df_signals = df_signals.add_prefix("signal_").reset_index() 
        
        # Join signals with cell data on time index
        df_full = df_full.merge(df_signals, on='time', how="left")
        
        return df_full
