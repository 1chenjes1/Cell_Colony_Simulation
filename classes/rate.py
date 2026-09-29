# classes/rate.py
"""
Defines Rate, a small container for a bounded kinetic parameter
"""
 
class Rate:
    """
    A bounded kinetic rate parameter.
 
    Rate bundles a single scalar parameter (e.g. a gene's production rate
    or a signal's synthesis rate) together with the [lower_bound,
    upper_bound] range it is allowed to take. The bounds are used
    elsewhere (e.g. during Latin Hypercube parameter sampling, see
    `pydoe.lhs` usage in the wider project) to draw candidate values;
    `value` holds whatever value is currently assigned to the parameter
    for a given simulation run.
 
    Attributes:
        name (str): Human-readable identifier for the rate (e.g. "k_GeneA").
        value (float): Current value of the rate. Defaults to 0 until
            `set_value` is called.
        lower_bound (float): Minimum allowed value for sampling.
        upper_bound (float): Maximum allowed value for sampling.
    """
    
    def __init__(self, name, lower_bound, upper_bound):
        self.name = name
        self.value = 0
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def set_value(self, value):
        self.value = value