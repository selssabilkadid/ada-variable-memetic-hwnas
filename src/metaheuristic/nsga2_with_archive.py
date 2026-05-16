"""
src/metaheuristic/nsga2_with_archive.py

NSGA-II with elitist archive injection.

Uses the custom NSGA-II from src/metaheuristic/nsga2/core/nsga2.py — the
same algorithm used by every other runner in the project.

The archive logic (FHArchive / CrowdingArchive from archives.py) is plugged
in via the callback(gen, population) hook that the custom NSGA2.run() already
exposes.  No changes to the core algorithm are needed.

Archive callback steps (called every generation after survivor selection):
  1. Feed every individual in the current population into the archive.
  2. Inject up to pop_size // 10 archive members back into the population
     by replacing the worst-ranked / lowest-crowding-distance individuals.
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
from src.metaheuristic.nsga2.core.nondominated_sort import fast_non_dominated_sort
from src.benchmark.hwnas_problem_def import HWNASProblem
from src.metaheuristic.archives import FHArchive, CrowdingArchive


# ---------------------------------------------------------------------------
# Archive injection callback
# ---------------------------------------------------------------------------

class ArchiveCallback:
    """
    Callable passed to NSGA2.run(callback=...).

    Called every generation as callback(gen, population) after survivor
    selection, so every Individual already has a valid .rank assigned by
    fast_non_dominated_sort inside the NSGA2 loop.

    Steps:
      1. Update archive with every individual in the current population.
      2. Inject up to pop_size // 10 archive members into the population
         by replacing the worst individuals (highest rank, then lowest
         crowding distance).
    """

    def __init__(self, archive, pop_size: int):
        self.archive          = archive
        self.pop_size         = pop_size
        self.history: list    = []
        self._gen             = 0

    def __call__(self, gen: int, population: List[Individual]) -> None:
        self._gen += 1

        # ── 1. Feed current population into archive ───────────────────────
        for ind in population:
            arch_idx = id(ind)          # unique object id as surrogate key
            acc      = float(-ind.objectives[0])   # undo negation
            lat      = float(ind.objectives[1])
            self.archive.update([(arch_idx, acc, lat)])

        # ── 2. Inject archive members into population ─────────────────────
        n_inject   = max(1, self.pop_size // 10)
        pop_ids    = {id(ind) for ind in population}

        archive_df = self.archive.to_dataframe()
        if archive_df.empty:
            self._record(population)
            return

        # Collect archive candidates not already in population
        candidates_rows = archive_df.head(n_inject)   # archive is already non-dom

        # Identify worst individuals: highest rank, then lowest crowding dist
        sorted_pop = sorted(
            range(len(population)),
            key=lambda i: (
                getattr(population[i], "rank", 999),
                -(getattr(population[i], "crowding_distance", 0.0)
                  if np.isfinite(getattr(population[i], "crowding_distance", 0.0))
                  else 0.0),
            ),
            reverse=True,   # worst first
        )

        slots_to_replace = sorted_pop[:len(candidates_rows)]

        for slot, (_, row) in zip(slots_to_replace, candidates_rows.iterrows()):
            
            acc_stored = float(row["accuracy"])
            lat_stored = float(row["latency"])

            new_obj = np.array([-acc_stored, lat_stored])   # back to min form

            # Replace worst individual's objectives in-place so the Individual
            population[slot].objectives = new_obj

        # ── 3. Record history ─────────────────────────────────────────────
        self._record(population)

    def _record(self, population: List[Individual]) -> None:
        fronts   = fast_non_dominated_sort(population)
        pf_size  = len(fronts[0]) if fronts else 0
        self.history.append({
            "generation":        self._gen,
            "pareto_front_size": pf_size,
            "archive_size":      self.archive.size,
        })


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_nsga2_with_archive(
    nb201_loader,
    hw_loader,
    dataset: str                                      = "cifar10",
    hw_platform: str                                  = "raspi4",
    archive: Optional[FHArchive | CrowdingArchive]    = None,
    pop_size: int                                     = 100,
    n_gen: int                                        = 50,
    seed: int                                         = 42,
    verbose: bool                                     = True,
    adaptive_mutation: Optional[dict]                 = None,
) -> dict:
    """
    Run NSGA-II with elitist archive injection.

    Parameters
    ----------
    nb201_loader      : NASBench201CSVLoader
    hw_loader         : HWNASLoader
    dataset           : one of ["cifar10", "cifar100", "ImageNet16-120"]
    hw_platform       : hardware target, e.g. "raspi4"
    archive           : FHArchive or CrowdingArchive instance.
                        If None, a default CrowdingArchive(max_size=500) is used.
    pop_size          : NSGA-II population size
    n_gen             : number of generations
    seed              : random seed
    verbose           : print per-generation progress
    adaptive_mutation : if None  → standard mutation
                        if dict  → UCB1 operator selection (same as other runners)
                        Expected keys: "initial_probs", "update_period"

    Returns
    -------
    dict with keys:
        "pareto_set"    : (P, n_var) decision variables of Pareto front
        "pareto_X"      : 1-D row indices (arange)
        "pareto_acc"    : Pareto-front accuracies  (%)
        "pareto_lat"    : Pareto-front latencies   (ms)
        "F"             : (N, 2) full final population objectives
        "archive"       : the archive object after the run
        "archive_df"    : archive.to_dataframe()
        "history"       : list[dict] — gen, pareto_front_size, archive_size
    """

    if archive is None:
        archive = CrowdingArchive(max_size=500)

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
    # Archive callback
    # ------------------------------------------------------------------
    archive_cb = ArchiveCallback(archive=archive, pop_size=pop_size)

    # ------------------------------------------------------------------
    # Algorithm 
    # ------------------------------------------------------------------
    algo = NSGA2(
        problem           = problem,
        pop_size          = pop_size,
        n_var             = problem.n_var,
        n_obj             = problem.n_obj,
        num_choices       = problem.num_choices,
        seed              = seed,
        prob_crossover    = 0.9,
        adaptive_mutation = adaptive_mutation,
    )

    mode_label = "Adaptive" if adaptive_mutation is not None else "Plain"
    print(
        f"[NSGA2+Archive | {type(archive).__name__} | {mode_label}] Starting | "
        f"dataset={dataset}  hw={hw_platform}  "
        f"pop={pop_size}  gen={n_gen}  seed={seed}"
    )

    pareto_set, pareto_front = algo.run(
        generations = n_gen,
        verbose     = verbose,
        callback    = archive_cb,
    )

    # ------------------------------------------------------------------
    # Unpack results 
    # ------------------------------------------------------------------
    pareto_acc = -pareto_front[:, 0]
    pareto_lat =  pareto_front[:, 1]
    pareto_X   = np.arange(len(pareto_acc))


    F = algo.history[-1]["objectives"] if algo.history else pareto_front

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(f"\n[NSGA2+Archive] Done -- Pareto front size: {len(pareto_X)}")
        print(
            f"  Best accuracy : {pareto_acc[best_acc_idx]:.2f}%  "
            f"lat={pareto_lat[best_acc_idx]:.3f} ms"
        )
        print(
            f"  Best latency  : {pareto_lat[best_lat_idx]:.3f} ms  "
            f"acc={pareto_acc[best_lat_idx]:.2f}%"
        )
        print(f"  Final archive size: {archive.size}")

    return {
        "pareto_set": pareto_set,
        "pareto_X":   pareto_X,
        "pareto_acc": pareto_acc,
        "pareto_lat": pareto_lat,
        "F":          F,
        "archive":    archive,
        "archive_df": archive.to_dataframe(),
        "history":    archive_cb.history,
    }