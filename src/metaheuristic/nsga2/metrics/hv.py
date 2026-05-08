from __future__ import annotations
import numpy as np

def hypervolume(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Compute 2D hypervolume by summing rectangles from each point to the reference point.

    Args:
        front: (N, 2) array of objectives (minimisation).
        ref_point: (2,) reference point — must dominate all solutions.
    Returns: 
        Hypervolume value.
    """
    if len(front) == 0:
        return 0.0

    front = np.asarray(front, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)

    # Drop points worse than reference
    mask = np.all(front <= ref_point, axis=1)
    front = front[mask]

    if len(front) == 0:
        return 0.0

    return _hv_2d(front, ref_point)


def _hv_2d(front: np.ndarray, ref_point: np.ndarray) -> float:
    """Sort points by the first objective and sum the area each point contributes."""
    sorted_front = front[np.argsort(front[:, 0])]

    hv = 0.0
    prev_y = ref_point[1]

    for i in range(len(sorted_front)):
        x_i, y_i = sorted_front[i]

        width = max(ref_point[0] - x_i, 0.0)
        height = max(prev_y - y_i, 0.0)

        hv += width * height
        prev_y = min(prev_y, y_i)

    return hv