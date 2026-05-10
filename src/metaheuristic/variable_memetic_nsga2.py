# src/metaheuristic/variable_memetic_nsga2.py
"""
Memetic NSGA-II: custom NSGA-II (src/metaheuristic/nsga2/core/nsga2.py)
+ variance-guided local search applied to the rank-0 front after each generation.

Identical in structure to memetic_nsga2.py (same custom NSGA-II baseline,
same callable callback pattern, same adaptive_mutation hook), but replaces
the uniform-random gene selection in the local search with variance-weighted
edge-priority selection:

  - Edge weights (precomputed offline via src/utils/variance_weights.py)
    are used as a probability distribution over which gene to perturb:
    high-variance edges are sampled more often.
  - The new value for the chosen gene is still sampled uniformly from
    the remaining choices (all values except the current one).
  - Neighbour evaluation and greedy dominance acceptance are otherwise
    identical to the uniform variant in memetic_nsga2.py.

Edge weights are precomputed offline via:
    python src/utils/variance_weights.py --dataset cifar10
and loaded from data/variance_weights/weights_<device>_<dataset>.npy.

Two runners call this module:

  run_variable_memetic_nsga2.py (plain variance-guided memetic):
      run_memetic(...) with default adaptive_mutation=None

  run_adaptive_variable_memetic_nsga2.py (variance-guided + UCB1):
      run_memetic(..., adaptive_mutation={"initial_probs": ..., "update_period": ...})

Usage
-----
from src.metaheuristic.variable_memetic_nsga2 import run_memetic

# Plain variance-guided memetic
results = run_memetic(
    nb201_loader=nb201_loader,
    hw_loader=hw_loader,
    dataset="cifar10",
    hw_platform="raspi4",
    pop_size=50,
    n_gen=20,
    K=4,
    seed=42,
    verbose=True,
)

# Variance-guided memetic + adaptive mutation (UCB1)
results = run_memetic(
    nb201_loader=nb201_loader,
    hw_loader=hw_loader,
    dataset="cifar10",
    hw_platform="raspi4",
    pop_size=50,
    n_gen=20,
    K=4,
    seed=42,
    verbose=True,
    adaptive_mutation={
        "initial_probs": np.array([0.33, 0.33, 0.34]),
        "update_period": 5,
    },
)
"""

from __future__ import annotations
import os
import sys
import numpy as np
from typing import List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.metaheuristic.nsga2.core.nsga2 import NSGA2
from src.metaheuristic.nsga2.core.individual import Individual
from src.benchmark.hwnas_problem_def import HWNASProblem
from src.utils.variance_weights import load_weights

# Default location of precomputed weight files
DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "variance_weights")


# ---------------------------------------------------------------------------
# Variance-guided local search callback
# ---------------------------------------------------------------------------

class VarianceLocalSearchCallback:
    """
    Callable passed to NSGA2.run(callback=...).

    Called every generation as ``callback(gen, population)`` after survivor
    selection, so every Individual already has a valid ``.rank``.

    Mirrors UniformLocalSearchCallback in memetic_nsga2.py exactly in
    structure and data model, but selects which gene to perturb using
    variance weights rather than uniformly at random:

      - gene_probs (normalised edge_weights) bias perturbation toward
        high-variance edges.
      - The new value for the chosen gene is still sampled uniformly from
        the remaining choices (all values except the current one).
      - Greedy acceptance: accept neighbour only if it strictly dominates
        the current best (≤ all objectives, < at least one).
      - Improvements compound within the K-budget (move-then-continue).

    Operates entirely on Individual.decision_vars (6-gene int vectors).
    Never constructs or passes flat arch indices — those are produced
    internally by HWNASProblem.evaluate() via vector_to_index().
    """

    def __init__(
        self,
        problem,
        edge_weights: np.ndarray,   # shape (n_var,) precomputed variance weights
        K: int,
        seed: int,
    ) -> None:
        self.problem     = problem
        self.K           = K
        self.rng         = np.random.RandomState(seed)
        self.total_evals = 0

        # Normalise edge weights → valid probability distribution over genes.
        # If all weights are zero (degenerate), fall back to uniform.
        w = np.asarray(edge_weights, dtype=float)
        total = w.sum()
        self.gene_probs = w / total if total > 0 else np.ones(len(w)) / len(w)

        # num_choices per gene taken directly from the problem (shape: n_var,)
        self.num_choices = np.asarray(problem.num_choices, dtype=int)

    def __call__(self, gen: int, population: List[Individual]) -> None:
        front0 = [ind for ind in population if ind.rank == 0]

        for ind in front0:
            best_vars = ind.decision_vars.copy()
            best_obj  = ind.objectives.copy()

            for _ in range(self.K):
                # --- Variance-weighted gene selection ---
                gene = self.rng.choice(len(best_vars), p=self.gene_probs)
                nc   = int(self.num_choices[gene])
                if nc <= 1:
                    continue

                # Sample any value other than the current one (uniform over rest)
                old     = int(best_vars[gene])
                others  = [v for v in range(nc) if v != old]
                new_val = int(self.rng.choice(others))

                neighbour_vars       = best_vars.copy()
                neighbour_vars[gene] = new_val

                # problem.evaluate() expects a 6-gene vector and calls
                # vector_to_index() internally — never pass a flat arch index
                neighbour_obj = np.asarray(
                    self.problem.evaluate(neighbour_vars), dtype=float
                )
                self.total_evals += 1

                # Accept if neighbour strictly dominates current best
                if (np.all(neighbour_obj <= best_obj)
                        and np.any(neighbour_obj < best_obj)):
                    best_vars = neighbour_vars
                    best_obj  = neighbour_obj

            # Write improvements back into the Individual in-place
            ind.decision_vars = best_vars
            ind.objectives    = best_obj


# ---------------------------------------------------------------------------
# Population snapshot callback (chains local search + captures last population)
# ---------------------------------------------------------------------------

class _PopulationSnapshot:
    """Thin wrapper so two callbacks can be chained from a single callable.

    Mirrors the same helper in memetic_nsga2.py exactly.
    """

    def __init__(self, local_search: VarianceLocalSearchCallback) -> None:
        self._ls  = local_search
        self.last_population: List[Individual] = []

    def __call__(self, gen: int, population: List[Individual]) -> None:
        self._ls(gen, population)
        self.last_population = population   # reference, updated each gen


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_memetic(
    nb201_loader,
    hw_loader,
    dataset: str                        = "cifar10",
    hw_platform: str                    = "raspi4",
    pop_size: int                       = 50,
    n_gen: int                          = 20,
    K: int                              = 4,
    seed: int                           = 42,
    verbose: bool                       = True,
    weights_dir: str                    = DEFAULT_WEIGHTS_DIR,
    adaptive_mutation: Optional[dict]   = None,   # None → plain; dict → UCB1
) -> dict:
    """
    Run Memetic NSGA-II (custom NSGA-II + variance-guided local search).

    Structurally identical to memetic_nsga2.run_memetic(); the only difference
    is that local search uses variance-weighted gene selection rather than
    uniform random selection.

    Parameters
    ----------
    nb201_loader      : NASBench201CSVLoader
    hw_loader         : HWNASLoader
    dataset           : one of ["cifar10", "cifar100", "ImageNet16-120"]
    hw_platform       : hardware target, e.g. "raspi4"
    pop_size          : NSGA-II population size
    n_gen             : number of generations
    K                 : local search budget — max neighbours evaluated per
                        rank-0 individual per generation
    seed              : random seed
    verbose           : print per-generation progress
    weights_dir       : path to precomputed variance weight .npy files
                        (default: data/variance_weights/)
    adaptive_mutation : if None  → plain variance-guided memetic (Contribution 2)
                        if dict  → passed to NSGA2 to enable UCB1 operator
                                   selection (Contribution 3).
                        Expected keys: "initial_probs", "update_period".

    Returns
    -------
    dict with keys:
        pareto_X        : 1-D array of row indices for Pareto-front solutions
        pareto_acc      : 1-D array of accuracy values  (%)
        pareto_lat      : 1-D array of latency values   (ms)
        F               : (N, 2) objectives array for the full final population
        edge_weights    : variance weight vector used  (shape n_var,)
        total_ls_evals  : total new evaluations spent in local search
    """

    # ------------------------------------------------------------------
    # Load precomputed variance weights
    # ------------------------------------------------------------------
    print(
        f"[Variance Memetic] Loading variance weights for "
        f"device={hw_platform}, dataset={dataset} ..."
    )
    edge_weights = load_weights(hw_platform, dataset, weights_dir)

    if verbose:
        print(
            f"[Variance Memetic] Edge weights (edges 0-5): "
            f"{', '.join(f'e{i}={v:.4f}' for i, v in enumerate(edge_weights))}"
        )

    # ------------------------------------------------------------------
    # Problem
    # ------------------------------------------------------------------
    problem = HWNASProblem(
        acc_loader = nb201_loader,
        hw_loader  = hw_loader,
        dataset    = dataset,
        platform   = hw_platform,
    )

    # ------------------------------------------------------------------
    # Local search callback
    # ------------------------------------------------------------------
    ls_callback = VarianceLocalSearchCallback(
        problem      = problem,
        edge_weights = edge_weights,
        K            = K,
        seed         = seed + 1,   # offset from main seed, same convention as memetic_nsga2
    )
    snapshot = _PopulationSnapshot(ls_callback)

    # ------------------------------------------------------------------
    # Algorithm — same custom NSGA2 baseline as memetic_nsga2.
    # adaptive_mutation is passed through: None → standard mutation,
    # dict → UCB1 operator selection activated inside NSGA2.
    # ------------------------------------------------------------------
    mode_label = "Adaptive" if adaptive_mutation is not None else "Plain"
    algo = NSGA2(
        problem           = problem,
        pop_size          = pop_size,
        n_var             = problem.n_var,
        n_obj             = problem.n_obj,
        num_choices       = problem.num_choices,
        seed              = seed,
        prob_crossover    = 0.9,
        adaptive_mutation = adaptive_mutation,   # None or UCB1 config dict
    )

    print(
        f"[Variance Memetic NSGA-II | {mode_label}] Starting | dataset={dataset}  "
        f"hw={hw_platform}  pop={pop_size}  gen={n_gen}  K={K}  seed={seed}"
    )

    pareto_set, pareto_front = algo.run(
        generations = n_gen,
        verbose     = verbose,
        callback    = snapshot,
    )

    # ------------------------------------------------------------------
    # Unpack results
    # ------------------------------------------------------------------
    pareto_acc = -pareto_front[:, 0]
    pareto_lat =  pareto_front[:, 1]

    # Full population objectives for scatter plot
    F = np.array([ind.objectives for ind in snapshot.last_population])

    # Row-number index (no benchmark index available from custom NSGA-II)
    pareto_X = np.arange(len(pareto_acc))

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(
            f"\n[Variance Memetic NSGA-II | {mode_label}] Done -- "
            f"Pareto front size: {len(pareto_X)}"
        )
        print(
            f"  Best accuracy : {pareto_acc[best_acc_idx]:.2f}%  "
            f"lat={pareto_lat[best_acc_idx]:.3f} ms"
        )
        print(
            f"  Best latency  : {pareto_lat[best_lat_idx]:.3f} ms  "
            f"acc={pareto_acc[best_lat_idx]:.2f}%"
        )
        print(
            f"  Local search new evals: {ls_callback.total_evals}  "
            f"(cached hits not counted)"
        )

    return {
        "pareto_X":       pareto_X,
        "pareto_acc":     pareto_acc,
        "pareto_lat":     pareto_lat,
        "F":              F,
        "edge_weights":   edge_weights,
        "total_ls_evals": ls_callback.total_evals,
    }