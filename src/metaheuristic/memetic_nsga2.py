# src/metaheuristic/memetic_nsga2.py
"""
Memetic NSGA-II: custom NSGA-II (src/metaheuristic/nsga2/core/nsga2.py)
+ uniform random local search applied to the rank-0 front after each generation.

Local search is neighbourhood-based and operator-agnostic:
  for each rank-0 individual, generate K neighbours by changing one randomly
  chosen gene to a uniformly-sampled different value, then accept the
  neighbour if it strictly dominates the current individual.

Two runners call this module:

  run_memetic_nsga2.py (Contribution 2 — plain memetic):
      run_memetic(...) with default adaptive_mutation=None

  run_adaptive_memetic_nsga2.py (Contribution 3 — memetic + UCB1):
      run_memetic(..., adaptive_mutation={"initial_probs": ..., "update_period": ...})


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


# --------------------------------------------------------------------------- #
# Uniform random local search callback
# --------------------------------------------------------------------------- #

class UniformLocalSearchCallback:
    """
    Callable passed to NSGA2.run(callback=...).

    Called every generation as ``callback(gen, population)`` after
    survivor selection, so every Individual already has a valid ``.rank``.

    For each rank-0 individual:
      - generate K neighbours, each differing in exactly one randomly
        chosen gene (new value sampled uniformly from the remaining choices)
      - greedily accept a neighbour if it strictly dominates the incumbent
        (Pareto dominance: ≤ on all objectives, < on at least one)
      - move to the accepted neighbour before generating the next one,
        so improvements compound within the K-budget
    """

    def __init__(
        self,
        problem,
        num_choices: np.ndarray,
        K: int,
        seed: int,
    ) -> None:
        self.problem     = problem
        self.num_choices = np.asarray(num_choices, dtype=int)
        self.K           = K
        self.rng         = np.random.RandomState(seed)
        self.total_evals = 0

    def __call__(self, gen: int, population: List[Individual]) -> None:
        front0 = [ind for ind in population if ind.rank == 0]

        for ind in front0:
            best_vars = ind.decision_vars.copy()
            best_obj  = ind.objectives.copy()

            for _ in range(self.K):
                # Pick a gene to perturb
                gene = self.rng.randint(0, len(best_vars))
                nc   = int(self.num_choices[gene])
                if nc <= 1:
                    continue

                # Sample any value other than the current one (uniform)
                old     = int(best_vars[gene])
                others  = [v for v in range(nc) if v != old]
                new_val = int(self.rng.choice(others))

                neighbour_vars       = best_vars.copy()
                neighbour_vars[gene] = new_val

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


# --------------------------------------------------------------------------- #
# Population snapshot callback (captures last-generation population for F)
# --------------------------------------------------------------------------- #

class _PopulationSnapshot:
    """Thin wrapper so two callbacks can be chained from a single callable."""

    def __init__(self, local_search: UniformLocalSearchCallback) -> None:
        self._ls  = local_search
        self.last_population: List[Individual] = []

    def __call__(self, gen: int, population: List[Individual]) -> None:
        self._ls(gen, population)
        self.last_population = population   # reference, updated each gen


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #

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
    weights_dir: Optional[str]          = None,        # accepted but unused
    adaptive_mutation: Optional[dict]   = None,        # None → plain memetic
) -> dict:
    """
    Run Memetic NSGA-II (custom NSGA-II + uniform random local search).

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
    weights_dir       : ignored (kept for runner back-compatibility)
    adaptive_mutation : if None  → plain memetic (Contribution 2)
                        if dict  → passed to NSGA2 to enable UCB1 operator
                                   selection (Contribution 3).
                        Expected keys: "initial_probs", "update_period".

    Returns
    -------
    dict with keys:
        pareto_X        : 1-D array of row indices for Pareto-front solutions
        pareto_acc      : 1-D array of accuracy values
        pareto_lat      : 1-D array of latency values
        F               : (N, 2) objectives array for the full final population
        edge_weights    : uniform weight vector (1/n_edges each); placeholder
                          returned so the experiment runner can save it without
                          changes
        total_ls_evals  : total new evaluations spent in local search
    """

    # ------------------------------------------------------------------ #
    # Problem
    # ------------------------------------------------------------------ #
    problem = HWNASProblem(
        acc_loader = nb201_loader,
        hw_loader  = hw_loader,
        dataset    = dataset,
        platform   = hw_platform,
    )

    # ------------------------------------------------------------------ #
    # Local search callback
    # ------------------------------------------------------------------ #
    ls_callback = UniformLocalSearchCallback(
        problem     = problem,
        num_choices = problem.num_choices,
        K           = K,
        seed        = seed + 1,   # offset from main seed
    )
    snapshot = _PopulationSnapshot(ls_callback)

    # ------------------------------------------------------------------ #
    # Algorithm
    # ------------------------------------------------------------------ #
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
        f"[Memetic NSGA-II | {mode_label}] Starting | dataset={dataset}  "
        f"hw={hw_platform}  pop={pop_size}  gen={n_gen}  K={K}  seed={seed}"
    )

    pareto_set, pareto_front = algo.run(
        generations = n_gen,
        verbose     = verbose,
        callback    = snapshot,
    )

    # ------------------------------------------------------------------ #
    # Unpack results
    # ------------------------------------------------------------------ #
    pareto_acc = -pareto_front[:, 0]
    pareto_lat =  pareto_front[:, 1]

    # Full population objectives for scatter plot
    F = np.array([ind.objectives for ind in snapshot.last_population])

    # Row-number index (no benchmark index available from custom NSGA-II)
    pareto_X = np.arange(len(pareto_acc))

    # Uniform placeholder weights (6 NASBench-201 edges)
    n_edges      = 6
    edge_weights = np.full(n_edges, 1.0 / n_edges)

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(f"\n[Memetic NSGA-II | {mode_label}] Done -- Pareto front size: {len(pareto_X)}")
        print(
            f"  Best accuracy : {pareto_acc[best_acc_idx]:.2f}%  "
            f"lat={pareto_lat[best_acc_idx]:.3f} ms"
        )
        print(
            f"  Best latency  : {pareto_lat[best_lat_idx]:.3f} ms  "
            f"acc={pareto_acc[best_lat_idx]:.2f}%"
        )
        print(f"  Local search new evals: {ls_callback.total_evals}")

    return {
        "pareto_X":       pareto_X,
        "pareto_acc":     pareto_acc,
        "pareto_lat":     pareto_lat,
        "F":              F,
        "edge_weights":   edge_weights,
        "total_ls_evals": ls_callback.total_evals,
    }