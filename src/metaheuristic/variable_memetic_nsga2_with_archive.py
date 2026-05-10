# src/metaheuristic/variable_memetic_nsga2_with_archive.py
"""
Memetic NSGA-II with online variance-guided local search + elitist archive.

Extends variable_memetic_nsga2.py in one key way:

  OFFLINE variant (variable_memetic_nsga2.py):
      Edge weights are precomputed once from the full benchmark population
      before the run starts and stay fixed throughout.

  THIS (ONLINE) variant:
      Edge weights are recomputed from scratch every generation using the
      current archive members' decision variables:
          weight[gene] = Var(archive[:, gene]) / sum(Var(archive[:, gene]))
      When the archive is empty (generation 0) or has only one member,
      the callback falls back to uniform weights so the run never stalls.

      This lets the local search focus on genes that are still diverse in
      the non-dominated archive, naturally shifting attention as the search
      converges.

Archive:
    FHArchive or CrowdingArchive (from src/metaheuristic/archives.py).
    Updated every generation with the current population before weights
    are recomputed, so weights reflect the latest non-dominated solutions.

Two runners call this module:

  run_variable_memetic_nsga2_archive.py  (plain, no UCB1):
      run_memetic_archive(...) with default adaptive_mutation=None

  run_adaptive_variable_memetic_nsga2_archive.py (UCB1 on top):
      run_memetic_archive(..., adaptive_mutation={...})

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
from src.metaheuristic.archives import FHArchive, CrowdingArchive


# ---------------------------------------------------------------------------
# Online variance weight computation
# ---------------------------------------------------------------------------

def _compute_online_weights(
    archive,
    n_var: int,
    fallback: str = "uniform",
) -> np.ndarray:
    """
    Compute per-gene variance weights from the current archive members.

    Parameters
    ----------
    archive  : FHArchive or CrowdingArchive — must expose .to_dataframe()
               with an "arch_index" column AND ._archive list of
               (arch_idx, acc, lat) tuples whose arch_idx maps to a stored
               decision_vars vector.
    n_var    : number of genes (6 for NASBench-201).
    fallback : "uniform" — what to return when archive is too small to
               compute meaningful variance (< 2 members).

    Returns
    -------
    weights : np.ndarray shape (n_var,), sum = 1.0
    """
    # Collect decision_vars stored in archive members
    # Both FHArchive and CrowdingArchive store (arch_idx, acc, lat) tuples
    # in their internal structures. arch_idx here is the Individual object
    # itself (we store id→vars mapping in the callback).
    # The callback maintains a separate id→decision_vars dict for this purpose.
    # This function is called with that dict passed as `var_store`.
    # ── see ArchiveVarCallback._compute_weights() which calls this ──────────
    raise NotImplementedError(
        "_compute_online_weights must be called via "
        "ArchiveVarCallback._compute_weights()"
    )


# ---------------------------------------------------------------------------
# Combined archive + variance-guided local search callback
# ---------------------------------------------------------------------------

class ArchiveVarCallback:
    """
    Callable passed to NSGA2.run(callback=...).

    Every generation (after survivor selection):

      Step 1 — Archive update:
          Feed every individual in the current population into the archive.
          Store their decision_vars in a local id→vars dict so we can
          reconstruct the gene matrix for variance computation.

      Step 2 — Online weight recomputation:
          Build the (archive_size × n_var) matrix of decision_vars from
          archive members. Compute per-gene variance. Normalise → gene_probs.
          Falls back to uniform if archive has < 2 members.

      Step 3 — Variance-guided local search on rank-0 front:
          Same greedy K-neighbour dominance search as
          VarianceLocalSearchCallback in variable_memetic_nsga2.py, but
          using the freshly recomputed gene_probs.
    """

    def __init__(
        self,
        archive,
        problem,
        n_var: int,
        K: int,
        seed: int,
    ) -> None:
        self.archive     = archive
        self.problem     = problem
        self.n_var       = n_var
        self.K           = K
        self.rng         = np.random.RandomState(seed)
        self.total_evals = 0
        self.history: list = []
        self._gen        = 0

        # num_choices per gene
        self.num_choices = np.asarray(problem.num_choices, dtype=int)

        # id(Individual) → decision_vars — lets us rebuild gene matrix from
        # archive members without storing decision_vars inside the archive itself
        self._var_store: dict[int, np.ndarray] = {}

        # Start with uniform weights; overwritten after first archive update
        self._gene_probs = np.ones(n_var) / n_var

    # ------------------------------------------------------------------
    # Weight recomputation
    # ------------------------------------------------------------------

    def _compute_weights(self) -> np.ndarray:
        """
        Build per-gene variance weights from current archive members.

        Returns normalised weight vector (sums to 1). Falls back to
        uniform when archive has < 2 members with known decision_vars.
        """
        known_ids = set(self._var_store.keys())

        # Collect decision_vars for every archive member we have vars for
        if isinstance(self.archive, CrowdingArchive):
            members = self.archive._archive          # list of (idx, acc, lat)
        else:
            # FHArchive: flatten all cells
            members = [
                s for cell in self.archive._cells.values() for s in cell
            ]

        var_matrix_rows = []
        for (arch_id, acc, lat) in members:
            if arch_id in self._var_store:
                var_matrix_rows.append(self._var_store[arch_id])

        if len(var_matrix_rows) < 2:
            # Not enough data yet → uniform
            return np.ones(self.n_var) / self.n_var

        mat      = np.vstack(var_matrix_rows)          # (m, n_var)
        variances = mat.var(axis=0)                    # (n_var,)
        total     = variances.sum()

        if total == 0:
            # All archive members are identical → uniform
            return np.ones(self.n_var) / self.n_var

        return variances / total

    # ------------------------------------------------------------------
    # Local search (mirrors VarianceLocalSearchCallback exactly)
    # ------------------------------------------------------------------

    def _local_search(self, population: List[Individual]) -> None:
        front0 = [ind for ind in population if ind.rank == 0]

        for ind in front0:
            best_vars = ind.decision_vars.copy()
            best_obj  = ind.objectives.copy()

            for _ in range(self.K):
                # Variance-weighted gene selection (online weights)
                gene = self.rng.choice(self.n_var, p=self._gene_probs)
                nc   = int(self.num_choices[gene])
                if nc <= 1:
                    continue

                old     = int(best_vars[gene])
                others  = [v for v in range(nc) if v != old]
                new_val = int(self.rng.choice(others))

                neighbour_vars       = best_vars.copy()
                neighbour_vars[gene] = new_val

                neighbour_obj = np.asarray(
                    self.problem.evaluate(neighbour_vars), dtype=float
                )
                self.total_evals += 1

                # Greedy dominance acceptance
                if (np.all(neighbour_obj <= best_obj)
                        and np.any(neighbour_obj < best_obj)):
                    best_vars = neighbour_vars
                    best_obj  = neighbour_obj

            ind.decision_vars = best_vars
            ind.objectives    = best_obj

    # ------------------------------------------------------------------
    # Main callback entry point
    # ------------------------------------------------------------------

    def __call__(self, gen: int, population: List[Individual]) -> None:
        self._gen += 1

        # ── Step 1: update archive + var_store ────────────────────────
        for ind in population:
            ind_id = id(ind)
            acc    = float(-ind.objectives[0])   # undo negation
            lat    = float(ind.objectives[1])
            self.archive.update([(ind_id, acc, lat)])
            # Always refresh var_store (Individual may have been updated
            # in-place by local search in a previous generation)
            self._var_store[ind_id] = ind.decision_vars.copy()

        # ── Step 2: recompute online variance weights ─────────────────
        self._gene_probs = self._compute_weights()

        # ── Step 3: variance-guided local search on rank-0 front ──────
        self._local_search(population)

        # ── Step 4: record history ────────────────────────────────────
        pf_size = sum(1 for ind in population if ind.rank == 0)
        self.history.append({
            "generation":        self._gen,
            "pareto_front_size": pf_size,
            "archive_size":      self.archive.size,
            "gene_probs":        self._gene_probs.tolist(),
        })


# ---------------------------------------------------------------------------
# Population snapshot (chains archive callback + captures last population)
# ---------------------------------------------------------------------------

class _PopulationSnapshot:
    """Mirrors the same helper in memetic_nsga2.py and variable_memetic_nsga2.py."""

    def __init__(self, archive_cb: ArchiveVarCallback) -> None:
        self._cb  = archive_cb
        self.last_population: List[Individual] = []

    def __call__(self, gen: int, population: List[Individual]) -> None:
        self._cb(gen, population)
        self.last_population = population


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_memetic_archive(
    nb201_loader,
    hw_loader,
    dataset: str                                    = "cifar10",
    hw_platform: str                                = "raspi4",
    archive: Optional[FHArchive | CrowdingArchive]  = None,
    pop_size: int                                   = 50,
    n_gen: int                                      = 20,
    K: int                                          = 4,
    seed: int                                       = 42,
    verbose: bool                                   = True,
    adaptive_mutation: Optional[dict]               = None,
) -> dict:
    """
    Run Memetic NSGA-II with online variance-guided local search and
    elitist archive.

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
    K                 : local search budget per rank-0 individual per generation
    seed              : random seed
    verbose           : print per-generation progress
    adaptive_mutation : if None  → plain variance-guided memetic
                        if dict  → UCB1 operator selection on top
                        Expected keys: "initial_probs", "update_period"

    Returns
    -------
    dict with keys:
        pareto_X        : 1-D array of row indices for Pareto-front solutions
        pareto_acc      : 1-D array of accuracy values  (%)
        pareto_lat      : 1-D array of latency values   (ms)
        F               : (N, 2) objectives for the full final population
        edge_weights    : final gene_probs used in the last generation
        total_ls_evals  : total new evaluations spent in local search
        archive         : the archive object after the run
        archive_df      : archive.to_dataframe()
        history         : list[dict] per generation:
                          gen, pareto_front_size, archive_size, gene_probs
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
    # Combined archive + local search callback
    # ------------------------------------------------------------------
    archive_cb = ArchiveVarCallback(
        archive  = archive,
        problem  = problem,
        n_var    = problem.n_var,
        K        = K,
        seed     = seed + 1,
    )
    snapshot = _PopulationSnapshot(archive_cb)

    # ------------------------------------------------------------------
    # Algorithm — same custom NSGA2 baseline as all other runners
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
        adaptive_mutation = adaptive_mutation,
    )

    print(
        f"[Online Variance Memetic | {type(archive).__name__} | {mode_label}] "
        f"Starting | dataset={dataset}  hw={hw_platform}  "
        f"pop={pop_size}  gen={n_gen}  K={K}  seed={seed}"
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
    pareto_X   = np.arange(len(pareto_acc))

    F = np.array([ind.objectives for ind in snapshot.last_population])

    # Final gene_probs = weights from last generation
    final_weights = archive_cb._gene_probs.copy()

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(
            f"\n[Online Variance Memetic | {mode_label}] Done -- "
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
        print(f"  Local search new evals : {archive_cb.total_evals}")
        print(f"  Final archive size     : {archive.size}")
        print(
            f"  Final gene weights     : "
            f"{', '.join(f'e{i}={v:.4f}' for i, v in enumerate(final_weights))}"
        )

    return {
        "pareto_X":       pareto_X,
        "pareto_acc":     pareto_acc,
        "pareto_lat":     pareto_lat,
        "F":              F,
        "edge_weights":   final_weights,
        "total_ls_evals": archive_cb.total_evals,
        "archive":        archive,
        "archive_df":     archive.to_dataframe(),
        "history":        archive_cb.history,
    }