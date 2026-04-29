from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# get_hashKey
# ---------------------------------------------------------------------------

def get_hashKey(X, problem_name: str) -> str:
    """
    Produce a deterministic string key for architecture X.

    For NASBench201, X is a list/array of 6 op-codes.
    For NASBench101, X is a binary adjacency+ops vector of length 22.
    We simply join the elements as a string — collision-free for these
    finite discrete spaces.
    """
    return problem_name + "_" + "_".join(str(int(v)) for v in X)


# ---------------------------------------------------------------------------
# find_all_neighbors
# ---------------------------------------------------------------------------

def find_all_neighbors(X, distance: int = 1, **kwargs) -> list:
    """
    Return all architectures reachable from X by flipping `distance` positions.

    For NASBench201: each position can take any value in available_ops.
    For NASBench101: positions are either ops or edges with separate allowed sets.

    This matches the exhaustive neighbourhood used in PSI_processor's
    improving_with_zc_predictor() path.
    """
    problem = kwargs.get("problem")
    X = list(X)
    n = len(X)

    if distance == 1:
        return _neighbors_distance_1(X, problem)
    else:
        # distance > 1: all combinations of `distance` simultaneous flips
        neighbors = []
        _neighbors_recursive(X, problem, distance, 0, neighbors)
        return neighbors


def _neighbors_distance_1(X, problem) -> list:
    neighbors = []
    problem_name = getattr(problem, "name", "NASBench201")

    if problem_name == "NASBench101":
        IDX_OPS  = getattr(problem, "IDX_OPS",  [])
        OPS      = getattr(problem, "OPS",      [])
        EDGES    = getattr(problem, "EDGES",    [])
        for idx in range(len(X)):
            if idx == 0 or idx == len(X) - 1:
                continue   # input/output nodes fixed
            allowed = OPS[:] if idx in IDX_OPS else EDGES[:]
            allowed = [v for v in allowed if v != X[idx]]
            for val in allowed:
                nb = X[:]
                nb[idx] = val
                neighbors.append(nb)
    else:
        # NASBench201
        available_ops = getattr(problem, "available_ops", list(range(5)))
        for idx in range(len(X)):
            for op in available_ops:
                if op != X[idx]:
                    nb = X[:]
                    nb[idx] = op
                    neighbors.append(nb)
    return neighbors


def _neighbors_recursive(X, problem, remaining, start_idx, result):
    """Generate all neighbours with `remaining` flips starting from start_idx."""
    if remaining == 0:
        result.append(X[:])
        return
    available_ops = getattr(problem, "available_ops", list(range(5)))
    for idx in range(start_idx, len(X)):
        for op in available_ops:
            if op != X[idx]:
                old = X[idx]
                X[idx] = op
                _neighbors_recursive(X, problem, remaining - 1, idx + 1, result)
                X[idx] = old


# ---------------------------------------------------------------------------
# find_the_better
# ---------------------------------------------------------------------------

def find_the_better(F_a, F_b, position: int | None = None) -> int:
    """
    Compare two objective vectors F_a and F_b (both minimisation).

    Parameters
    ----------
    position : None  → full Pareto dominance comparison
               0     → compare only objective 0 (best_f0 mode)
               1     → compare only objective 1 (best_f1 mode)

    Returns
    -------
     0  : F_a is better (dominates, or lower on the selected objective)
     1  : F_b is better
    -1  : neither dominates the other (non-dominated)
    """
    F_a = np.asarray(F_a, dtype=float)
    F_b = np.asarray(F_b, dtype=float)

    if position is not None:
        # Single-objective comparison on the chosen axis
        if F_a[position] < F_b[position]:
            return 0
        elif F_b[position] < F_a[position]:
            return 1
        else:
            return -1

    # Full Pareto dominance
    a_leq_b = np.all(F_a <= F_b)
    b_leq_a = np.all(F_b <= F_a)
    a_lt_b  = np.any(F_a <  F_b)
    b_lt_a  = np.any(F_b <  F_a)

    if a_leq_b and a_lt_b:
        return 0   # a dominates b
    if b_leq_a and b_lt_a:
        return 1   # b dominates a
    return -1      # non-dominated


# ---------------------------------------------------------------------------
# get_idx_non_dominated_front
# ---------------------------------------------------------------------------

def get_idx_non_dominated_front(F) -> np.ndarray:
    """
    Return indices of the non-dominated subset of objective matrix F.
    F has shape (n, n_obj), minimisation on all objectives.
    """
    if len(F) == 0:
        return np.array([], dtype=int)

    F = np.asarray(F, dtype=float)
    n = len(F)
    is_nd = np.ones(n, dtype=bool)

    for i in range(n):
        if not is_nd[i]:
            continue
        for j in range(n):
            if i == j or not is_nd[j]:
                continue
            # If j dominates i, mark i as dominated
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_nd[i] = False
                break

    return np.where(is_nd)[0]


# ---------------------------------------------------------------------------
# check_above_or_below
# ---------------------------------------------------------------------------

def check_above_or_below(
    considering_pt: np.ndarray,
    remaining_pt_1: np.ndarray,
    remaining_pt_2: np.ndarray,
) -> int:
    """
    Determine whether `considering_pt` lies above (-1) or below (1)
    the line connecting remaining_pt_1 and remaining_pt_2.

    Used by PSI_processor to identify knee-point candidates: a point
    above the line between its neighbours is concave (a knee).

    Returns
    -------
    -1  : above the line (concave / knee candidate)
     1  : below or on the line (convex)
    """
    p1, p2, p = remaining_pt_1, remaining_pt_2, considering_pt

    # Cross product of (p2 - p1) × (p - p1): positive = left/above, negative = right/below
    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    # In a minimisation Pareto front sorted by f0 ascending:
    # points that are concave (knee) sit ABOVE the chord → cross product > 0
    if cross > 0:
        return -1   # above → potential knee
    return 1        # below or on line


# ---------------------------------------------------------------------------
# calculate_angle_measure
# ---------------------------------------------------------------------------

def calculate_angle_measure(
    considering_pt: np.ndarray,
    neighbor_1:     np.ndarray,
    neighbor_2:     np.ndarray,
) -> float:
    """
    Compute the interior angle (degrees) at `considering_pt` formed by
    the vectors to neighbor_1 and neighbor_2.

    PSI_processor uses this to filter knee points: if the angle exceeds
    gamma (default 210°), the point is accepted as a knee.

    Note: angles > 180° correspond to "reflex" angles — PSI uses the
    exterior/reflex angle measure, i.e. 360° - interior_angle.
    """
    p  = np.asarray(considering_pt, dtype=float)
    p1 = np.asarray(neighbor_1,    dtype=float)
    p2 = np.asarray(neighbor_2,    dtype=float)

    v1 = p1 - p
    v2 = p2 - p

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0

    cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
    interior  = np.degrees(np.arccos(cos_angle))   # [0, 180]

    # PSI uses the reflex angle measure: 360 - interior gives values in [180, 360]
    # for concave (knee) points, which exceed the gamma=210 threshold
    return 360.0 - interior


# ---------------------------------------------------------------------------
# check_valid
# ---------------------------------------------------------------------------

def check_valid(
    hashKey:   str,
    neighbors: list | None = None,
    P:         np.ndarray | list | None = None,
    DS:        list | None = None,
) -> bool:
    """
    Return True if hashKey has not been seen in neighbors, P, or DS.

    PSI_processor calls this before evaluating a neighbour to avoid
    re-evaluating architectures already in the population or already
    visited during local search.
    """
    if neighbors and hashKey in neighbors:
        return False
    if P is not None and hasattr(P, '__iter__') and hashKey in P:
        return False
    if DS and hashKey in DS:
        return False
    return True