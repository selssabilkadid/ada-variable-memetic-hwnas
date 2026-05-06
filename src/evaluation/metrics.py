"""
src/evaluation/metrics.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Quality indicators for multi-objective NAS approximation sets.

Functions
---------
is_dominated(a, b)
    Check if solution ``a`` is dominated by solution ``b`` (minimisation).

compute_pareto_front(solutions)
    Extract the Pareto-optimal subset from a list of solution dicts.

hypervolume_2d(pareto_front, reference_point)
    2-D hypervolume indicator from a list of solution dicts.

hypervolume_2d_arr(front_arr, ref_point)
    2-D hypervolume from a raw numpy array  [accuracy, latency].

generational_distance(approx_df, true_df)
    GD — convergence towards the true Pareto front (lower is better).

inverted_generational_distance(approx_df, true_df)
    IGD — convergence + spread w.r.t. the true front (lower is better).

compute_metrics(approx_df, true_df)
    Convenience wrapper: returns HV, HV-ratio, GD, IGD in one call.
    Accepts either DataFrames or numpy arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy helpers (list-of-dicts interface — kept for backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def is_dominated(a: list | np.ndarray, b: list | np.ndarray) -> bool:
    """Return True if solution ``a`` is dominated by solution ``b``.

    Assumes **minimisation** on all objectives.
    For accuracy (maximise), pass ``-accuracy`` so higher becomes lower.
    """
    return (
        all(b[i] <= a[i] for i in range(len(a)))
        and any(b[i] < a[i] for i in range(len(a)))
    )


def compute_pareto_front(solutions: list[dict]) -> list[dict]:
    """Extract the Pareto-optimal subset.

    Parameters
    ----------
    solutions : list of dicts with keys ``'accuracy'``, ``'latency'``,
                ``'arch_index'``.

    Returns
    -------
    pareto : list[dict]
        Non-dominated solutions (maximise accuracy, minimise latency).
    """
    pareto = []
    for s in solutions:
        dominated = False
        for other in solutions:
            if other is s:
                continue
            if (
                other["accuracy"] >= s["accuracy"]
                and other["latency"] <= s["latency"]
                and (
                    other["accuracy"] > s["accuracy"]
                    or other["latency"] < s["latency"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    return pareto


def hypervolume_2d(pareto_front: list[dict], reference_point: tuple) -> float:
    """Compute the 2-D hypervolume indicator from a list of solution dicts.

    Parameters
    ----------
    pareto_front    : list of dicts with keys ``'accuracy'`` and ``'latency'``.
    reference_point : (ref_latency, ref_accuracy) — worst-case corner.

    Returns
    -------
    hv : float
    """
    front = sorted(pareto_front, key=lambda x: x["latency"])
    ref_lat, ref_acc = reference_point

    hv, prev_lat = 0.0, ref_lat
    for point in reversed(front):
        width  = prev_lat - point["latency"]
        height = point["accuracy"] - ref_acc
        if width > 0 and height > 0:
            hv += width * height
        prev_lat = point["latency"]
    return hv


# ═══════════════════════════════════════════════════════════════════════════════
#  NumPy / DataFrame interface  (used by run_nsga2_fh_ca.py)
# ═══════════════════════════════════════════════════════════════════════════════

def hypervolume_2d_arr(front_arr: np.ndarray, ref_point: tuple) -> float:
    """2-D hypervolume from a raw array.

    Parameters
    ----------
    front_arr : np.ndarray, shape (n, 2)
        Columns: [accuracy, latency].
    ref_point : (ref_latency, ref_accuracy)
        Worst-case reference corner.

    Returns
    -------
    hv : float
    """
    ref_lat, ref_acc = ref_point
    # Sort by latency descending so we sweep left-to-right
    pts = front_arr[np.argsort(front_arr[:, 1])[::-1]]   # sort lat DESC

    hv, prev_lat = 0.0, ref_lat
    for acc, lat in pts:
        if lat >= ref_lat:
            continue
        w = prev_lat - lat
        h = acc - ref_acc
        if h > 0 and w > 0:
            hv += w * h
        prev_lat = lat
    return hv


def _normalise_joint(
    pts:     np.ndarray,
    ref_pts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Min-max normalise ``pts`` and ``ref_pts`` jointly per objective."""
    combined = np.vstack([pts, ref_pts])
    lo = combined.min(axis=0)
    hi = combined.max(axis=0)
    spread = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    return (pts - lo) / spread, (ref_pts - lo) / spread


def generational_distance(
    approx_df: pd.DataFrame,
    true_df:   pd.DataFrame,
) -> float:
    """Generational Distance (GD) — convergence indicator.

    GD = mean distance from each **approximated** solution to its nearest
    **true** solution.  Lower is better.  Does not penalise poor spread.

    Parameters
    ----------
    approx_df, true_df : pd.DataFrame
        Must contain columns ``'accuracy'`` and ``'latency'``.

    Returns
    -------
    gd : float  (``np.nan`` if either frame is empty)
    """
    if approx_df.empty or true_df.empty:
        return np.nan

    A = approx_df[["accuracy", "latency"]].values
    P = true_df[["accuracy",  "latency"]].values
    A_n, P_n = _normalise_joint(A, P)

    # (|A|, |P|) pairwise distances
    dists = np.sqrt(((A_n[:, None, :] - P_n[None, :, :]) ** 2).sum(axis=2))
    return float(dists.min(axis=1).mean())


def inverted_generational_distance(
    approx_df: pd.DataFrame,
    true_df:   pd.DataFrame,
) -> float:
    """Inverted Generational Distance (IGD) — convergence + spread indicator.

    IGD = mean distance from each **true** solution to its nearest
    **approximated** solution.  Lower is better.  Preferred metric in the
    NAS / EMO literature because it jointly penalises poor convergence and
    poor spread.

    Parameters
    ----------
    approx_df, true_df : pd.DataFrame
        Must contain columns ``'accuracy'`` and ``'latency'``.

    Returns
    -------
    igd : float  (``np.nan`` if either frame is empty)
    """
    if approx_df.empty or true_df.empty:
        return np.nan

    A = approx_df[["accuracy", "latency"]].values
    P = true_df[["accuracy",  "latency"]].values
    A_n, P_n = _normalise_joint(A, P)

    # (|P|, |A|) pairwise distances — swapped vs GD
    dists = np.sqrt(((P_n[:, None, :] - A_n[None, :, :]) ** 2).sum(axis=2))
    return float(dists.min(axis=1).mean())


def compute_metrics(
    approx_df: pd.DataFrame | np.ndarray,
    true_df:   pd.DataFrame | np.ndarray,
) -> dict:
    """Convenience wrapper: compute HV, HV-ratio, GD, and IGD in one call.

    Accepts either DataFrames (columns ``accuracy``, ``latency``) or raw
    numpy arrays shaped (n, 2) with columns [accuracy, latency].

    Parameters
    ----------
    approx_df : DataFrame or ndarray  — approximation set.
    true_df   : DataFrame or ndarray  — reference (true) Pareto front.

    Returns
    -------
    dict with keys:
        ``hv_true``, ``hv_approx``, ``hv_ratio``  (HV ratio as fraction 0–1),
        ``hv_ratio_pct``  (HV ratio as percentage 0–100),
        ``gd``, ``igd``,
        ``front_size_true``, ``front_size_approx``.
    """
    # Accept numpy arrays by wrapping them
    if isinstance(approx_df, np.ndarray):
        approx_df = pd.DataFrame(approx_df, columns=["accuracy", "latency"])
    if isinstance(true_df, np.ndarray):
        true_df   = pd.DataFrame(true_df,   columns=["accuracy", "latency"])

    all_acc = np.concatenate([approx_df["accuracy"].values, true_df["accuracy"].values])
    all_lat = np.concatenate([approx_df["latency"].values,  true_df["latency"].values])

    margin    = 0.05
    ref_point = (
        np.max(all_lat) * (1 + margin),
        np.min(all_acc) * (1 - margin),
    )   # (ref_lat, ref_acc)

    hv_true   = hypervolume_2d_arr(true_df[["accuracy", "latency"]].values,   ref_point)
    hv_approx = hypervolume_2d_arr(approx_df[["accuracy", "latency"]].values, ref_point)
    hv_ratio  = (hv_approx / hv_true) if hv_true > 0 else 0.0

    gd_val  = generational_distance(approx_df, true_df)
    igd_val = inverted_generational_distance(approx_df, true_df)

    return {
        "hv_true":           hv_true,
        "hv_approx":         hv_approx,
        "hv_ratio":          hv_ratio,
        "hv_ratio_pct":      hv_ratio * 100,
        "gd":                gd_val,
        "igd":               igd_val,
        "front_size_true":   len(true_df),
        "front_size_approx": len(approx_df),
    }
