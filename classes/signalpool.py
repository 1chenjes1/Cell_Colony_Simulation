# classes/signalpool.py
"""
Defines SignalPool and its variants, representing the shared extracellular
signal(s) that all cells in an Environment read from and secrete into.
 
Unlike per-cell gene expression (advanced stochastically, see
`Cell.update`), the signal pool is advanced deterministically with a
classic 4th-order Runge-Kutta (RK4) integrator, since it represents a
well-mixed, population-averaged quantity rather than a single noisy cell.
"""

import numpy as np
import pandas as pd

class SignalPool:
    """
    A well-mixed pool of extracellular signal(s), shared by all cells in
    an Environment.
 
    Each signal `s` follows a simple linear production/decay ODE driven
    by population-averaged cell secretion:
        d(signal)/dt = k_s * secretion_s - signal_s + basal_s
 
    Attributes:
        signals (list[Signal]): The signals tracked by this pool.
        y (np.ndarray): Current concentration of each signal.
        history (dict): Recorded {"time": [...], "values": [...]}
            trajectory, appended to on every `record_state` call.
    """

    def __init__(self, signals):
        self.signals = signals
        self.y = np.zeros(len(signals)).flatten()
        self.history = {"time": [], "values": []}

        self.record_state(time=0.0)

    def get_signal_values(self):
        """
        Return a copy of the current signal concentration vector
        """
        return self.y.copy()
    
    def get_signal_names(self):
         """
         Return list of signal names
         """
         return [s.name for s in self.signals]

    def record_state(self, time):
        """
        Append the current state (and the given time) to `history`
        """
        self.history["time"].append(time)
        self.history["values"].append(self.y.copy())

    def get_history_df(self, decimals=2):
        """
        Build a DataFrame of the recorded signal trajectory.
 
        Returns:
            pd.DataFrame: rows indexed by time, one column per signal,
            values rounded to `decimals` decimal places.
        """

        signal_names = self.get_signal_names()
        df = pd.DataFrame(self.history["values"], columns=signal_names)
        df.index = self.history["time"]
        df.index.name = "time"
        df = df.round(decimals)  # Round all values to the specified number of decimals
        return df
    
    def system(self, secretion):
        """
        Compute d(signal)/dt for every signal, given the current
        population-averaged cell `secretion` into each signal.
 
        Args:
            secretion (np.ndarray): secretion rate for each signal
                (same order as `self.signals`), as produced by
                `Environment.update` (population-averaged
                `Cell.get_secretion`).
 
        Returns:
            np.ndarray: dY/dt for each signal.
        """

        y = self.y
        dy = np.zeros_like(y)
        for si, s in enumerate(self.signals):
            k = s.k.value
            basal = s.basal
        
            dy[si] = k * secretion[si] - y[si] + basal

        return dy.flatten()
    
    def system_from_state(self, y_state, secretion):
        """
        Same as `system`, but evaluated at an arbitrary state `y_state`
        rather than `self.y`. Used for the intermediate RK4 stages
        (k2, k3, k4), which must be evaluated at perturbed states without
        mutating `self.y`.
        """
        dy = np.zeros_like(y_state)
        for si, s in enumerate(self.signals):
            k = s.k.value
            basal = s.basal
            dy[si] = k * secretion[si] - y_state[si] + basal
        return dy

    def update(self, secretion, euler_step_size):
        """
        Advance the signal pool by one time step using classic 4th-order
        Runge-Kutta (RK4) integration of `system`, holding `secretion`
        fixed over the step. Negative concentrations are clipped to zero
        after the step.
 
        Args:
            secretion (np.ndarray): population-averaged cell secretion
                into each signal over this step (held constant across
                the RK4 sub-stages).
            euler_step_size (float): integration time step `h`.
        """

        y0 = self.y.copy()
        h = euler_step_size

        # k1
        k1 = self.system(secretion)

        # k2 (use temporary state)
        y_temp = y0 + 0.5 * h * k1
        k2 = self.system_from_state(y_temp, secretion)

        # k3
        y_temp = y0 + 0.5 * h * k2
        k3 = self.system_from_state(y_temp, secretion)

        # k4
        y_temp = y0 + h * k3
        k4 = self.system_from_state(y_temp, secretion)

        # combine
        self.y = y0 + (h/6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # prevent negatives
        self.y[self.y < 0] = 0

class NullSignalPool(SignalPool):
    """
    A dummy signal pool for when no extracellular signaling is used
    (e.g. a single-cell simulation, or no Signal species defined).
 
    Every operation is a no-op / returns an empty result, so `Cell` and
    `Environment` code can treat "no signaling" and "signaling" cases
    uniformly without branching.
    """
    
    def __init__(self):
        self.signals = []
        self.y = np.zeros(0)
        self.history = {"time": [], "values": []}
    
    def get_signal_values(self):
        return np.zeros(0)
    
    def get_signal_names(self):
        return []
    
    def record_state(self, time):
        pass
    
    def update(self, secretion, euler_step_size):
        # do nothing (no signals to update)
        pass
    
    def system(self, secretion):
        return np.zeros(0)
    
    def get_history_df(self):
        # Return empty DataFrame with a 'time' column so merging doesn't fail
        df = pd.DataFrame(columns=[])
        df.index.name = "time"
        return df
    

class FixedSignalPool(SignalPool):
    """
    A SignalPool where signals are fixed (for debugging / sanity checking).
    Inherits from SignalPool but overrides update/system to freeze values.
    """
    def __init__(self, signals, fixed_values):
        super().__init__(signals)
        self.y = np.array(fixed_values).flatten()
        self.fixed = True 

    def system(self, secretion):
        """
        Override system dynamics: return zero, signals are fixed.
        """
        return np.zeros_like(self.y)

    def update(self, secretion, euler_step_size):
        """
        Override update: do nothing, signals remain fixed.
        """
        pass
 