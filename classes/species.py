# classes/species.py
"""
Defines the two kinds of biochemical "species" used to build a gene
regulatory network (GRN) model:
 
- Gene:   an intracellular species, local to a single cell.
- Signal: an extracellular species, shared across all cells in an
          Environment via a SignalPool.
 
Both are thin wrappers around a `Rate` (the species' production rate `k`)
and a basal expression level, tagged with a `role` string ("local" vs
"global") that downstream code (Cell, SignalPool, System) uses to decide
how the species is simulated.
"""

class Species:
    """
    Base class for a biochemical species in the model.
 
    Attributes:
        name (str): Species identifier, used for labeling history/plots
            and for building the interaction matrix.
        role (str): "local" for genes (simulated per-cell) or "global"
            for signals (simulated once, shared across cells).
        k (Rate): The species' production/synthesis rate parameter.
        basal (float): Basal (regulator-independent) expression level.
    """

    def __init__(self, name, role, k, basal):
        self.name = name
        self.role = role
        self.k = k
        self.basal = basal

class Gene(Species):
    """An intracellular gene, local to a single Cell (role="local")."""
    def __init__(self, name, k, basal):
        super().__init__(name, role="local", k=k, basal=basal)

class Signal(Species):
    """
    An extracellular signaling molecule, shared across cells via a
    SignalPool (role="global").
    """
    def __init__(self, name, k, basal):
        super().__init__(name, role="global", k=k, basal=basal)
