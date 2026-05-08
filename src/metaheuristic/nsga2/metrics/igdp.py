from __future__ import annotations
import numpy as np

def inverted_generational_distance_plus(
    obtained_front: np.ndarray, true_front: np.ndarray
) -> float:
    """IGD+ = mean(min_d_j+) — only penalises worse dimensions.

    Args:
        obtained_front: (N, M) array of obtained objectives.
        true_front: (P, M) array of true Pareto-optimal objectives.
    Returns: 
        IGD+ value.
    """
    if len(true_front) == 0 or len(obtained_front) == 0:
        return float("inf")

    # Pairwise differences
    diffs = true_front[:, np.newaxis, :] - obtained_front[np.newaxis, :, :]

    # Zero out dimensions where obtained is better (negative diff)
    dominated_diffs = np.maximum(diffs, 0.0)

    distances = np.sqrt(np.sum(dominated_diffs ** 2, axis=2))
    min_distances = np.min(distances, axis=1)

    return float(np.mean(min_distances))