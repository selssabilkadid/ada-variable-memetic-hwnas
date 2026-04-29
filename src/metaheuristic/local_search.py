# src/metaheuristic/local_search.py
"""
KL-guided local search for memetic NSGA-II on NAS-Bench-201 / HW-NAS-Bench.

Public API
----------
compute_kl_weights(nb201_loader, hw_loader, devices, dataset, pareto_fronts)
    Pre-search step: compute per-device, per-edge KL weights from a
    reference Pareto front (e.g. ground truth or a warm-start run).

apply_local_search(front_indices, arch_matrix, population_F, eval_cache,
                   nb201_loader, hw_loader, dataset, device,
                   kl_weights, K, rng)
    Post-selection step: improve every rank-1 solution by budget-K
    neighbourhood search ordered by KL weights.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_EDGES = 6      # NAS-Bench-201 cell has 6 directed edges
N_OPS   = 5      # {none, skip, conv1x1, conv3x3, avgpool3x3}
N_ARCHS = 15625  # HWNASLoader.SEARCH_SPACE_SIZE  (5^6)

_SMOOTH = 1e-6   # Laplace-style smoothing to avoid log(0) in KL


# ---------------------------------------------------------------------------
# Step 1 — Decode / encode arch_index ↔ op-vector
# ---------------------------------------------------------------------------

def index_to_ops(arch_index: int) -> Tuple[int, ...]:
    """
    Convert a flat arch_index in [0, 15624] to a 6-tuple of op codes.

    NAS-Bench-201 encodes each cell as a base-5 number over 6 edges:
        arch_index = op[0]*5^0 + op[1]*5^1 + ... + op[5]*5^5
    """
    ops = []
    idx = arch_index
    for _ in range(N_EDGES):
        ops.append(idx % N_OPS)
        idx //= N_OPS
    return tuple(ops)


def ops_to_index(ops: Tuple[int, ...]) -> int:
    """Inverse of index_to_ops."""
    return sum(op * (N_OPS ** e) for e, op in enumerate(ops))


# ---------------------------------------------------------------------------
# Step 2 — Marginal operation frequencies across all 15,625 architectures
# ---------------------------------------------------------------------------

def _compute_marginal_freq() -> np.ndarray:
    """
    Returns freq_all: shape (N_EDGES, N_OPS).

    Because NAS-Bench-201 uses a pure Cartesian product of 5 ops × 6 edges,
    every operation appears exactly 15625/5 = 3125 times on every edge.
    The marginal is therefore uniform: freq_all[e, op] = 1/5 for all e, op.

    This function computes it explicitly rather than hard-coding 0.2 so that
    the logic is clear and survives any future change to the search space.
    """
    counts = np.zeros((N_EDGES, N_OPS), dtype=np.float64)
    for arch_index in range(N_ARCHS):
        for e, op in enumerate(index_to_ops(arch_index)):
            counts[e, op] += 1.0
    # Normalize rows → probability distribution per edge
    return counts / counts.sum(axis=1, keepdims=True)


# Cache: computed once on first call
_MARGINAL_FREQ: np.ndarray | None = None


def get_marginal_freq() -> np.ndarray:
    global _MARGINAL_FREQ
    if _MARGINAL_FREQ is None:
        _MARGINAL_FREQ = _compute_marginal_freq()
    return _MARGINAL_FREQ


# ---------------------------------------------------------------------------
# Step 3 — KL weight computation  (pre-search step)
# ---------------------------------------------------------------------------

def compute_kl_weights(
    pareto_indices_by_device: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Compute per-device, per-edge KL divergence weights.

    Parameters
    ----------
    pareto_indices_by_device : dict
        { device_name : array of arch_index values on the Pareto front }
        Typically loaded from ground-truth CSVs or from a warm-start run.

    Returns
    -------
    kl_weights : dict
        { device_name : np.ndarray of shape (N_EDGES,) }
        Normalized weights (sum to 1) over edges for each device.

    Implementation notes
    --------------------
    Order of operations is critical:
        1. Count raw op frequencies on the Pareto front.
        2. Add epsilon BEFORE normalizing (not after).
        3. Normalize to get freq_pareto[e] as a probability vector.
        4. Compute KL( freq_pareto[e] || freq_all[e] ) per edge.
        5. Normalize KL values across edges → sampling distribution.
    Doing step 2 after step 3 gives a different (wrong) distribution.
    """
    freq_all = get_marginal_freq()   # shape (N_EDGES, N_OPS) — uniform 0.2
    kl_weights: Dict[str, np.ndarray] = {}

    for device, pareto_indices in pareto_indices_by_device.items():
        if len(pareto_indices) == 0:
            # Degenerate case: no Pareto front → fall back to uniform weights
            kl_weights[device] = np.ones(N_EDGES) / N_EDGES
            continue

        # --- 3a. Count op occurrences on the Pareto front -----------------
        counts = np.zeros((N_EDGES, N_OPS), dtype=np.float64)
        for arch_index in pareto_indices:
            for e, op in enumerate(index_to_ops(int(arch_index))):
                counts[e, op] += 1.0

        # --- 3b. Smooth THEN normalize ------------------------------------
        #   Adding epsilon before normalization ensures no zero probability,
        #   which would make KL divergence undefined.
        counts += _SMOOTH                              # shape (N_EDGES, N_OPS)
        freq_pareto = counts / counts.sum(axis=1, keepdims=True)

        # --- 3c. KL( freq_pareto[e] || freq_all[e] ) per edge -------------
        #   KL(P || Q) = sum_x P(x) * log( P(x) / Q(x) )
        #   Both distributions are already smoothed/normalized so log is safe.
        kl_per_edge = np.sum(
            freq_pareto * np.log(freq_pareto / freq_all),
            axis=1,
        )   # shape (N_EDGES,)

        # KL is non-negative; numerical noise can push near-zero values
        # slightly below 0 — clip defensively.
        kl_per_edge = np.clip(kl_per_edge, 0.0, None)

        # --- 3d. Normalize across edges → probability distribution --------
        total = kl_per_edge.sum()
        if total < 1e-12:
            # All edges equally uninformative → uniform fallback
            kl_weights[device] = np.ones(N_EDGES) / N_EDGES
        else:
            kl_weights[device] = kl_per_edge / total

    return kl_weights


# ---------------------------------------------------------------------------
# Step 4 — Neighbourhood generation with KL-ordered edge priority
# ---------------------------------------------------------------------------

def get_ordered_neighbours(
    arch: Tuple[int, ...],
    device: str,
    kl_weights: Dict[str, np.ndarray],
    rng: np.random.Generator,
) -> List[Tuple[float, int, int, Tuple[int, ...]]]:
    """
    Generate all single-edge-flip neighbours of `arch`, sorted by the KL
    weight of the flipped edge (descending).  Ties within the same weight
    are broken by a random shuffle to avoid op-index bias.

    Parameters
    ----------
    arch        : 6-tuple of op codes (decoded arch)
    device      : hardware platform key into kl_weights
    kl_weights  : output of compute_kl_weights()
    rng         : numpy Generator for reproducible tie-breaking

    Returns
    -------
    List of (weight, edge, op, neighbour_ops_tuple) sorted by -weight.
    Ties in weight are shuffled so no op index is systematically preferred.
    """
    weights = kl_weights[device]   # shape (N_EDGES,)

    neighbours: List[Tuple[float, int, int, Tuple[int, ...]]] = []
    for edge in range(N_EDGES):
        w = float(weights[edge])
        for op in range(N_OPS):
            if op == arch[edge]:
                continue                       # skip identity
            neighbour = list(arch)
            neighbour[edge] = op
            neighbours.append((w, edge, op, tuple(neighbour)))

    # --- Tiebreak fix: shuffle within same-weight groups ------------------
    # Python's sort is stable, so without this, same-weight entries stay in
    # insertion order (edge 0 op 0, 1, 2 … edge 1 op 0 …), biasing toward
    # low edge and low op indices.  We shuffle first, then sort to break ties
    # randomly while preserving the descending weight order.
    rng.shuffle(neighbours)                    # random permutation in-place
    neighbours.sort(key=lambda x: -x[0])      # stable sort keeps shuffle within ties

    return neighbours


# ---------------------------------------------------------------------------
# Step 5 — Dominance utilities
# ---------------------------------------------------------------------------

def _dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    """
    Return True if solution a dominates solution b (minimisation on both axes).
    a dominates b iff a is no worse on all objectives AND strictly better on
    at least one.
    """
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


def _is_dominated_by_front(f_new: np.ndarray, front_F: np.ndarray) -> bool:
    """Return True if f_new is dominated by any member of front_F."""
    for f in front_F:
        if _dominates(f, f_new):
            return True
    return False


# ---------------------------------------------------------------------------
# Step 6 — Local search (applied once per generation, after survivor selection)
# ---------------------------------------------------------------------------

def apply_local_search(
    front_indices: np.ndarray,
    arch_matrix: np.ndarray,
    population_F: np.ndarray,
    eval_cache: Dict[int, np.ndarray],
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    kl_weights: Dict[str, np.ndarray],
    K: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    KL-guided local search over all rank-1 (Pareto-front) solutions.

    Design rules implemented here
    ──────────────────────────────
    1. Full pass first, then update.
       All front members are improved in the snapshot of F1 taken at the
       START of this call.  Accepted neighbours are collected in `pending`;
       the population and front are updated only after all members have been
       processed.  This eliminates order-dependency artifacts.

    2. Budget accounting.
       K counts only NEW evaluations (cache misses).  A cache hit that leads
       to acceptance does NOT consume budget — you already paid for it.

    3. Duplicate / cache handling.
       Before calling the loaders, check eval_cache[arch_index].  If present,
       use the cached objectives.  eval_cache is mutated in-place so new
       evaluations are available to subsequent generations.

    Parameters
    ----------
    front_indices  : indices into population arrays for rank-1 solutions
    arch_matrix    : (pop_size, 1) integer array of arch_index values
    population_F   : (pop_size, 2) objective matrix [-acc, latency]
    eval_cache     : { arch_index: np.array([-acc, latency]) } — shared cache
    nb201_loader   : NASBench201CSVLoader instance
    hw_loader      : HWNASLoader instance
    dataset        : e.g. "cifar10"
    device         : e.g. "raspi4"
    kl_weights     : output of compute_kl_weights()
    K              : neighbourhood budget (new evals per front member)
    rng            : numpy Generator

    Returns
    -------
    new_arch_matrix   : updated (pop_size, 1) integer array
    new_population_F  : updated (pop_size, 2) objective matrix
    n_new_evals       : number of new (non-cached) evaluations performed
    """
    # Snapshot of the current Pareto front objectives for dominance checks.
    # This is the fixed reference used throughout the entire local search pass.
    front_F_snapshot = population_F[front_indices].copy()   # (|F1|, 2)

    # Collect accepted replacements: { population_index: (arch_index, F) }
    pending: Dict[int, Tuple[int, np.ndarray]] = {}

    n_new_evals = 0

    for pop_idx in front_indices:
        arch_index = int(arch_matrix[pop_idx, 0])
        arch_ops   = index_to_ops(arch_index)

        neighbours = get_ordered_neighbours(arch_ops, device, kl_weights, rng)

        budget_used = 0   # new evaluations consumed for this front member

        for weight, edge, op, neighbour_ops in neighbours:
            if budget_used >= K:
                break

            n_arch_index = ops_to_index(neighbour_ops)

            # --- Cache lookup -------------------------------------------
            if n_arch_index in eval_cache:
                n_F = eval_cache[n_arch_index]
                # Cache hit: does NOT consume budget
            else:
                # --- New evaluation -------------------------------------
                acc     = nb201_loader.get_accuracy(n_arch_index, dataset)
                latency = hw_loader.get_latency(n_arch_index,
                                                hw_platform=device,
                                                dataset=dataset)
                n_F = np.array([-acc, latency])
                eval_cache[n_arch_index] = n_F
                budget_used  += 1
                n_new_evals  += 1

            # --- Acceptance criterion -----------------------------------
            # Accept if neighbour is NOT dominated by any member of the
            # SNAPSHOT front (not the pending-updated front).
            if not _is_dominated_by_front(n_F, front_F_snapshot):
                pending[pop_idx] = (n_arch_index, n_F)
                break   # first accepted neighbour replaces current solution

    # --- Apply all pending replacements at once -------------------------
    new_arch_matrix  = arch_matrix.copy()
    new_population_F = population_F.copy()

    for pop_idx, (n_arch_index, n_F) in pending.items():
        new_arch_matrix[pop_idx, 0] = n_arch_index
        new_population_F[pop_idx]   = n_F

    return new_arch_matrix, new_population_F, n_new_evals