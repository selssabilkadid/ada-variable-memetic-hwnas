from __future__ import annotations
import numpy as np

def spread(obtained_front: np.ndarray, true_front: np.ndarray) -> float:
    """Delta = (df + dl + sum|d_i - d_bar|) / (df + dl + (N-1)*d_bar).

    Args:
        obtained_front: (N, 2) array of obtained objectives.
        true_front: (P, 2) array of true Pareto-optimal objectives.
    Returns: 
        Spread value.
    """
    n = len(obtained_front)

    if n < 2:
        return float("inf")

    # Boundary distances: obtained extremes vs true extremes
    df = np.linalg.norm(
        obtained_front[np.argmin(obtained_front[:, 0])] -
        true_front[np.argmin(true_front[:, 0])]
    )
    dl = np.linalg.norm(
        obtained_front[np.argmin(obtained_front[:, 1])] -
        true_front[np.argmin(true_front[:, 1])]
    )

    # Consecutive distances along the obtained front
    sorted_front = obtained_front[np.argsort(obtained_front[:, 0])]
    diffs = np.diff(sorted_front, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))

    if len(distances) == 0:
        return float("inf")

    d_bar = np.mean(distances)
    denominator = df + dl + (n - 1) * d_bar

    if denominator == 0:
        return float("inf")

    return float((df + dl + np.sum(np.abs(distances - d_bar))) / denominator)