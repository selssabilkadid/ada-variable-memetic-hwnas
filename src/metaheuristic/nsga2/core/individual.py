from __future__ import annotations
import numpy as np
from typing import Optional


class Individual:
    """
    A single architecture solution in the NSGA-II population.

    Attributes:
        decision_vars: the architecture encoding.
        objectives: objective function values.
        rank: nondomination rank (0 = Pareto front 1).
        crowding_distance: crowding distance within its front.
    """

    __slots__ = (
        "decision_vars",
        "objectives",
        "constraints",
        "rank",
        "crowding_distance",
        "mutation_type",
    )

    def __init__(
        self,
        decision_vars: np.ndarray,
        n_objectives: int,
    ) -> None:
        
        # Initialize variables
        self.decision_vars = np.asarray(decision_vars, dtype=np.int32)
        self.objectives: Optional[np.ndarray] = None
        self.rank: int = 0
        self.crowding_distance: float = 0.0
        self.mutation_type: int = -1   # -1=unset

        # Allocate an array for objectives
        if n_objectives > 0:
            self.objectives = np.full(n_objectives, np.inf)

    def evaluate(self, problem) -> None:
        """
        Evaluate objectives using the given problem.

        Args:
            problem: A callable with ``evaluate(x)`` method that returns objectives.
        """
        obj = problem.evaluate(self.decision_vars)
        self.objectives = np.asarray(obj, dtype=np.float64)

    def __repr__(self) -> str:
        """String representation for debugging."""
        obj_str = (
            f"{self.objectives}" if self.objectives is not None else "unevaluated"
        )
        return (
            f"Individual(rank={self.rank}, cd={self.crowding_distance:.4f}, "
            f"obj={obj_str})"
        )

    def copy(self) -> "Individual":
        """Return a copy of the individual."""
        new = Individual.__new__(Individual)
        new.decision_vars = self.decision_vars.copy()
        new.objectives = self.objectives.copy() if self.objectives is not None else None
        new.rank = self.rank
        new.crowding_distance = self.crowding_distance
        new.mutation_type = self.mutation_type
        return new