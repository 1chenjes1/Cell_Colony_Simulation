# classes/cell.py
"""
Defines Cell, a single cell carrying its own gene regulatory network (GRN)
state. Each cell:
 
  - holds a state vector `y` (one concentration per gene),
  - reads/writes to a shared SignalPool for intercellular communication,
  - is advanced in time with a stochastic Euler-Maruyama integrator.
 
Gene regulation follows a Hill-type production model:
    production = (activators) / (1 + activators + repressors)
with Hill coefficient n=2 for gene-gene regulation and n=1 for
signal-gene regulation (see `system()` below).
"""

from ast import Return

import numpy as np
import pandas as pd

class Cell:
    """
    A single cell with its own gene regulatory network (GRN) state.
 
    The cell's genes interact with each other (via `interaction_matrix`)
    and, optionally, with extracellular signals held in a shared
    `SignalPool`: the cell both reads signal concentrations (to regulate
    its own genes) and secretes into the pool (via `get_secretion`).
 
    Attributes:
        genes (list[Gene]): The genes tracked by this cell, in the same
            order as the first `n_genes` rows/columns of
            `interaction_matrix`.
        y (np.ndarray): Current gene expression state, one value per gene.
        interaction_matrix (np.ndarray): Square matrix of size
            (n_genes + n_signals) x (n_genes + n_signals) encoding signed
            regulatory interactions. Layout (block form):
                [ gene->gene    gene->signal  ]
                [ signal->gene  signal->signal]
            Only the gene->gene, gene->signal and signal->gene blocks are
            used; a positive entry is an activating interaction, negative
            is repressing, and zero means "no interaction".
        cell_id: Identifier for this cell (used e.g. when building
            multi-cell history dataframes).
        history (dict): Recorded {"time": [...], "values": [...]} state
            trajectory, appended to on every `record_state` call.
    """

    def __init__(self, genes, y0, interaction_matrix, cell_id):
        self.genes = genes
        self.y = y0
        self.interaction_matrix = interaction_matrix
        self.cell_id = cell_id
        self.history = {"time": [], "values": []}

        self.record_state(time=0.0)
    
    def get_gene_values(self):
        # Return a copy of the current gene expression state vector
        return self.y.copy()
    
    def get_gene_names(self):
         # Return the ordered list of gene names for this cell
         return [g.name for g in self.genes]

    def record_state(self, time):
        # Append the current state (and the given time) to `history
        self.history["time"].append(time)
        self.history["values"].append(self.y.copy())

    def get_history_df(self, decimals=2):
        """
        Build a DataFrame of this cell's recorded trajectory.
 
        Returns:
            pd.DataFrame: rows indexed by time, one column per gene
        """
        gene_names = self.get_gene_names()
        df = pd.DataFrame(self.history["values"], columns=gene_names)

        df.index = self.history["time"]
        df.index.name = "time" 
        df = df.round(decimals)
        return df
    
    def get_secretion(self, signal_pool):
        """
        Compute this cell's instantaneous secretion rate into each signal
        in `signal_pool`.
 
        For each signal, genes that regulate it (via the gene->signal
        block of `interaction_matrix`) contribute additively to an
        activator pool `A` or repressor pool `R` (Hill coefficient n=1,
        i.e. linear in gene expression), and secretion is the Hill-type
        fraction A / (1 + A + R).
 
        Args:
            signal_pool: The SignalPool (or NullSignalPool) this cell
                secretes into.
 
        Returns:
            np.ndarray: secretion rate for each signal in
            `signal_pool.signals`, same length as `signal_pool.y`.
        """
        y = self.y

        # signal secretion initialization
        secretion = np.zeros_like(signal_pool.y)

        # genes -> signals matrix
        n_genes = len(self.genes)
        gene_to_signal = self.interaction_matrix[:n_genes, n_genes:] # n_genes x n_signals (regulatory x target)

        # Build ODE for secretion
        for si, _ in enumerate(signal_pool.signals):
            A = 0
            R = 0
            for gi, _ in enumerate(self.genes):
                interaction = gene_to_signal[gi, si]
                if interaction == 0:
                        continue
                
                y_gi = y[gi]

                if interaction > 0:
                    A += y_gi
                elif interaction < 0:
                    R += y_gi
                else:
                    raise ValueError("unknown interaction")
                
            secretion[si] = (A)/(1+A+R) 
        
        return secretion

    def system(self, signal_pool):
        """
        Compute dY/dt for every gene in this cell, given the current
        signal concentrations in `signal_pool`.
 
        For each gene, activator/repressor contributions are pooled from
        two sources into a *single* combined A (activators) / R
        (repressors):
          - other genes, via the gene->gene block of
            `interaction_matrix`, entering as y^2 (Hill coefficient n=2);
          - extracellular signals, via the signal->gene block, entering
            linearly as signal_pool.y (Hill coefficient n=1).
 
        Production then follows one Hill-type fraction:
            prod = A / (1 + A + R)   if A > 0
                 = 1 / (1 + R)       otherwise
        and each gene decays with unit rate:
            dY = k * prod + basal - y
 
        Args:
            signal_pool: The SignalPool (or NullSignalPool) providing
                current extracellular signal concentrations.
 
        Returns:
            np.ndarray: dY/dt for each gene, in `self.genes` order.
        """

        y = self.y.flatten()
        dY = np.zeros_like(y)

        n_genes = len(self.genes)
        grn = self.interaction_matrix[:n_genes, :n_genes]
        signal_to_gene = (
            self.interaction_matrix[n_genes:, :n_genes]
            if len(signal_pool.signals) > 0 else None
        )
        
        # Build ODE for each gene
        for ti, t in enumerate(self.genes):
            A = 0.0
            R = 0.0

            # Gene regulators (Hill coefficient n=2)
            for ri in range(n_genes):
                interaction = grn[ri, ti]
                if interaction == 0:
                    continue
                val = y[ri] ** 2
                if interaction > 0:
                    A += val
                else:
                    R += val

            # Signal regulators (Hill coefficient n=1)
            if signal_to_gene is not None:
                for si in range(len(signal_pool.signals)):
                    interaction = signal_to_gene[si, ti]
                    if interaction == 0:
                        continue
                    val = signal_pool.y[si]
                    if interaction > 0:
                        A += val
                    else:
                        R += val

            # Unified production term
            prod = A / (1 + A + R) if A > 0 else 1 / (1 + R)

            dY[ti] = t.k.value * prod + t.basal - y[ti]

        return dY
    
    def update(self, signal_pool, euler_step_size, noise_width):
        """
        Advance this cell's gene expression state by one time step using
        the Euler-Maruyama scheme for the SDE:
 
            dY = system(Y) * dt + noise_width * dW,   dW ~ N(0, dt)
 
        i.e.:
            Y_{n+1} = Y_n + system(Y_n) * dt + noise_width * sqrt(dt) * N(0, 1)
 
        Negative concentrations are clipped to zero after the step.
 
        Args:
            signal_pool: The current SignalPool, used to evaluate
                `system()`.
            euler_step_size (float): Integration time step `dt`.
            noise_width (float): Magnitude of the additive Wiener noise
                term.
        """
        
        dt = euler_step_size
        y0 = self.y.copy()

        # dY
        dy = self.system(signal_pool)

        # stochastic diffusion
        dW = np.sqrt(dt) * np.random.normal(size=len(y0))

        # Euler–Maruyama step
        self.y = y0 + dy * dt + noise_width * dW

        # prevent negative concentrations
        self.y[self.y < 0] = 0

    def print_odes(self, signal_pool):
        """
        Print the ODE governing each gene, exactly as implemented
        in `system()`.
 
        Gene regulators enter with Hill coefficient n=2 (name^2), signal
        regulators enter linearly (n=1), and — matching `system()` — all
        activator/repressor terms (gene- and signal-derived alike) are
        combined into a *single* Hill-type production fraction:
 
            prod = (activators) / (1 + activators + repressors)

        """
        y = self.y.flatten()
 
        n_genes = len(self.genes)
        grn = self.interaction_matrix[:n_genes, :n_genes]
        signal_to_gene = self.interaction_matrix[n_genes:, :n_genes]
 
        for ti, t in enumerate(self.genes):
 
            k = t.k.value
            basal = t.basal
 
            activators, repressors = [], []
 
            # Gene-gene interactions (Hill coefficient n=2)
            for ri, r in enumerate(self.genes):
                interaction = grn[ri, ti]
                if interaction == 0:
                    continue
 
                if interaction > 0:
                    activators.append(f"{r.name}^2")
                else:
                    repressors.append(f"{r.name}^2")
 
            # Signal-gene interactions (Hill coefficient n=1)
            if len(signal_pool.signals) > 0:
                for si, s in enumerate(signal_pool.signals):
                    if s.k.value != 0:
                        interaction = signal_to_gene[si, ti]
                        if interaction == 0:
                            continue
 
                        if interaction > 0:
                            activators.append(s.name)
                        else:
                            repressors.append(s.name)
 
            # Build combined expression (single Hill fraction, as in system())
            act = " + ".join(activators)
            rep = " + ".join(repressors)
 
            prod_expr = (
                f"({act}) / (1 + {act} + {rep})"
                if activators else
                f"1 / (1 + {rep})"
            )
 
            eq = f"d{t.name}/dt = {k} * {prod_expr} + {basal} - {t.name}"
            print(eq)