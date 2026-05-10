import numpy as np


# ------------------------------------------------------------------ #
# Dominance helpers
# ------------------------------------------------------------------ #

def is_dominated(a, b):
    """
    Returns True if solution a is dominated by solution b.
    Assumes minimization on both objectives.
    For accuracy: pass -accuracy so higher is better becomes lower is better.
    """
    return all(b[i] <= a[i] for i in range(len(a))) and \
           any(b[i] < a[i] for i in range(len(a)))


def compute_pareto_front(solutions: list) -> list:
    """
    solutions: list of dicts with keys 'accuracy', 'latency', 'arch_index'
    Returns the Pareto-optimal subset (maximize accuracy, minimize latency).
    """
    pareto = []
    for s in solutions:
        dominated = False
        for other in solutions:
            if other is s:
                continue
            if (other["accuracy"] >= s["accuracy"] and
                    other["latency"] <= s["latency"] and
                    (other["accuracy"] > s["accuracy"] or
                     other["latency"] < s["latency"])):
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    return pareto


# ------------------------------------------------------------------ #
# Internal conversion
# ------------------------------------------------------------------ #

def _to_array(front) -> np.ndarray:
    """
    Accept either:
      - np.ndarray of shape (N, 2)  columns: [accuracy, latency]
      - list of dicts with keys 'accuracy' and 'latency'

    Always returns an (N, 2) float64 array [accuracy, latency].
    """
    if isinstance(front, np.ndarray):
        arr = np.asarray(front, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(
                f"Expected (N, 2) array [accuracy, latency], got shape {arr.shape}"
            )
        return arr
    # list-of-dicts path (original API)
    return np.array(
        [[d["accuracy"], d["latency"]] for d in front], dtype=float
    )


# ------------------------------------------------------------------ #
# Hypervolume
# ------------------------------------------------------------------ #

def hypervolume_2d(pareto_front, reference_point: tuple) -> float:
    """
    Computes 2D hypervolume indicator.

    pareto_front:
        Either a list of dicts with keys 'accuracy' / 'latency',
        or an (N, 2) numpy array with columns [accuracy, latency].

    reference_point: (ref_latency, ref_accuracy)
        The worst-case point. ref_latency >= all front latencies,
        ref_accuracy <= all front accuracies.

    Assumes: maximize accuracy (y-axis), minimize latency (x-axis).
    """
    arr = _to_array(pareto_front)          # (N, 2) — [accuracy, latency]
    ref_lat, ref_acc = reference_point

    # Sort by latency ascending so we sweep left → right
    order = np.argsort(arr[:, 1])
    arr   = arr[order]

    hv       = 0.0
    prev_lat = ref_lat

    for point in reversed(arr):            # right → left sweep
        acc, lat = point[0], point[1]
        width  = prev_lat - lat
        height = acc - ref_acc
        if width > 0 and height > 0:
            hv      += width * height
            prev_lat = lat

    return hv


# ------------------------------------------------------------------ #
# GD / IGD
# ------------------------------------------------------------------ #

def _normalize(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    return (arr - vmin) / (vmax - vmin + 1e-8)


def generational_distance(approx_front, true_front) -> float:
    """
    Generational Distance (GD) — mean distance from each approximation
    point to its nearest true-front point (normalised space).

    Accepts (N, 2) numpy arrays or lists of dicts [accuracy, latency].
    """
    approx = _to_array(approx_front)
    true   = _to_array(true_front)

    all_   = np.vstack([approx, true])
    vmin   = all_.min(axis=0)
    vmax   = all_.max(axis=0)

    approx_n = _normalize(approx, vmin, vmax)
    true_n   = _normalize(true,   vmin, vmax)

    return float(
        np.mean([np.min(np.linalg.norm(true_n - pt, axis=1)) for pt in approx_n])
    )


def inverted_generational_distance(approx_front, true_front) -> float:
    """
    Inverted Generational Distance (IGD) — mean distance from each
    true-front point to its nearest approximation point (normalised space).

    Accepts (N, 2) numpy arrays or lists of dicts [accuracy, latency].
    """
    approx = _to_array(approx_front)
    true   = _to_array(true_front)

    all_   = np.vstack([approx, true])
    vmin   = all_.min(axis=0)
    vmax   = all_.max(axis=0)

    approx_n = _normalize(approx, vmin, vmax)
    true_n   = _normalize(true,   vmin, vmax)

    return float(
        np.mean([np.min(np.linalg.norm(approx_n - pt, axis=1)) for pt in true_n])
    )


# ------------------------------------------------------------------ #
# Unified compute_metrics
# ------------------------------------------------------------------ #

def compute_metrics(approx_front, true_front) -> dict:
    """
    Compute HV ratio, GD, and IGD for an approximation front vs the
    true Pareto front.

    Both arguments accept either:
      - (N, 2) numpy array  with columns [accuracy, latency]
      - list of dicts       with keys   'accuracy', 'latency'

    Returns
    -------
    dict with keys:
        hv_true, hv_approx, hv_ratio,
        gd, igd,
        front_size_true, front_size_approx
    """
    approx = _to_array(approx_front)
    true   = _to_array(true_front)

    all_acc = np.concatenate([approx[:, 0], true[:, 0]])
    all_lat = np.concatenate([approx[:, 1], true[:, 1]])
    margin  = 0.05

    ref_lat = float(np.max(all_lat) + margin * abs(np.max(all_lat)))
    ref_acc = float(np.min(all_acc) - margin * abs(np.min(all_acc)))
    reference_point = (ref_lat, ref_acc)

    hv_true   = hypervolume_2d(true,   reference_point)
    hv_approx = hypervolume_2d(approx, reference_point)
    hv_ratio  = (hv_approx / hv_true * 100) if hv_true > 0 else 0.0

    return {
        "hv_true":           hv_true,
        "hv_approx":         hv_approx,
        "hv_ratio":          hv_ratio,
        "gd":                generational_distance(approx, true),
        "igd":               inverted_generational_distance(approx, true),
        "front_size_true":   len(true),
        "front_size_approx": len(approx),
    }