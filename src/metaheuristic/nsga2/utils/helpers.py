"""Helper utilities for post-processing NSGA-II results."""

from __future__ import annotations
import numpy as np


def nondominated_filter(objectives: np.ndarray) -> np.ndarray:
    """Return the indices of nondominated solutions from an objective array.

    A solution i is nondominated if no other solution j has objectives
    all <= i's and at least one strictly <.

    Args:
        objectives: (N, M) array of objective values.

    Returns:
        Boolean mask array of length N, True for nondominated solutions.
    """
    n = objectives.shape[0]
    is_nondominated = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_nondominated[i]:
            continue
        for j in range(n):
            if i == j or not is_nondominated[j]:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                is_nondominated[i] = False
                break

    return is_nondominated
