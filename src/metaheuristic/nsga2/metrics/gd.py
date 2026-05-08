from __future__ import annotations
import numpy as np

def generational_distance(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """GD = sqrt(mean(min_d_i^2)) — avg distance from obtained solutions to true front.

    Args:
        obtained_front: (N, M) array of obtained objectives.
        true_front: (P, M) array of true Pareto-optimal objectives.
    Returns: 
        GD value.
    """
    if len(obtained_front) == 0:
        return float("inf")

    # Pairwise distances: (N, P)
    diffs = obtained_front[:, np.newaxis, :] - true_front[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Nearest true-front point per obtained solution
    min_distances = np.min(distances, axis=1)

    return float(np.sqrt(np.mean(min_distances ** 2)))