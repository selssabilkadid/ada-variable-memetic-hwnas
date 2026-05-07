"""Discrete test problems for NSGA-II."""

import numpy as np

class OneZeroMax:
    """
    A simple discrete bi-objective problem.
    
    Objectives:
    1. Minimize the sum of variables.
    2. Minimize the negative sum of variables.
    """

    def __init__(self, n_var: int = 30, max_val: int = 1):
        self.n_var = n_var
        self.n_obj = 2
        self.max_val = max_val
        
        # Defines how many choices each gene has
        self.num_choices = np.full(n_var, max_val + 1)

    def evaluate(self, x: np.ndarray):
        """
        Calculate objectives.
        Returns the objectives array.
        """
        sum_x = np.sum(x)
        
        f1 = sum_x          # Minimize sum
        f2 = -sum_x         # Minimize negative sum (Maximize sum)
        
        # Return the array
        return np.array([f1, f2])