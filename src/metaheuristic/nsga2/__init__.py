"""
NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm.
"""

from .core.nsga2 import NSGA2
from .core.individual import Individual

__version__ = "1.0.0"
__all__ = ["NSGA2", "Individual"]