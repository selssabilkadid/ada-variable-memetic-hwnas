# src/metaheuristic/memetic_nsga2.py
"""
Memetic NSGA-II: vanilla NSGA-II + variance-guided local search.

The only structural change vs. the baseline is that after each generation's
survivor selection we call apply_local_search() on the rank-1 front before
advancing to the next generation.

Edge weights are precomputed offline via:
    python src/utils/variance_weights.py --dataset cifar10
and loaded from data/variance_weights/weights_<device>_<dataset>.npy.

Usage
-----
from src.metaheuristic.memetic_nsga2 import run_memetic

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
"""

from __future__ import annotations
import sys
import os
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.metaheuristic.local_search import apply_local_search
from src.metaheuristic.nsga2 import HWNASProblem
from src.utils.variance_weights import load_weights

# Default location of precomputed weight files
DEFAULT_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "variance_weights")


# ---------------------------------------------------------------------------
# Callback: inject local search after each generation
# ---------------------------------------------------------------------------

class VarianceLocalSearchCallback(Callback):
    """
    pymoo Callback that runs variance-guided local search on the rank-1 front
    after every generation's survivor selection step.
    """

    def __init__(
        self,
        nb201_loader,
        hw_loader,
        dataset: str,
        device: str,
        edge_weights: np.ndarray,   # shape (6,) precomputed variance weights
        K: int,
        seed: int,
    ):
        super().__init__()
        self.nb201_loader = nb201_loader
        self.hw_loader    = hw_loader
        self.dataset      = dataset
        self.device       = device
        self.K            = K
        self.rng          = np.random.default_rng(seed)

        # Wrap in the same dict format apply_local_search expects
        self.kl_weights = {device: edge_weights}

        # Shared evaluation cache — persists across all generations
        self.eval_cache: dict    = {}
        self.total_new_evals: int = 0

    def notify(self, algorithm, **kwargs):
        """Called by pymoo after each generation."""
        pop = algorithm.pop

        arch_matrix  = pop.get("X").astype(int)
        population_F = pop.get("F").astype(float)

        ranks = pop.get("rank")
        if ranks is None:
            return

        front_indices = np.where(ranks == 0)[0]
        if len(front_indices) == 0:
            return

        new_arch_matrix, new_population_F, n_new = apply_local_search(
            front_indices = front_indices,
            arch_matrix   = arch_matrix,
            population_F  = population_F,
            eval_cache    = self.eval_cache,
            nb201_loader  = self.nb201_loader,
            hw_loader     = self.hw_loader,
            dataset       = self.dataset,
            device        = self.device,
            kl_weights    = self.kl_weights,
            K             = self.K,
            rng           = self.rng,
        )

        self.total_new_evals += n_new

        for i, ind in enumerate(pop):
            ind.set("X", new_arch_matrix[i])
            ind.set("F", new_population_F[i])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_memetic(
    nb201_loader,
    hw_loader,
    dataset: str     = "cifar10",
    hw_platform: str = "raspi4",
    pop_size: int    = 50,
    n_gen: int       = 20,
    K: int           = 4,
    seed: int        = 42,
    verbose: bool    = True,
    weights_dir: str = DEFAULT_WEIGHTS_DIR,
) -> dict:
    """
    Run Memetic NSGA-II (NSGA-II + variance-guided local search).

    Parameters
    ----------
    nb201_loader : NASBench201CSVLoader
    hw_loader    : HWNASLoader
    dataset      : one of ["cifar10", "cifar100", "ImageNet16-120"]
    hw_platform  : hardware target, e.g. "raspi4"
    pop_size     : NSGA-II population size
    n_gen        : number of generations
    K            : local search neighbourhood budget (new evals per front solution)
    seed         : random seed
    verbose      : print per-generation progress
    weights_dir  : path to precomputed variance weight .npy files
                   (default: data/variance_weights/)

    Returns
    -------
    dict with keys matching nsga2.run() plus:
        "total_ls_evals"  : total new evaluations spent in local search
        "edge_weights"    : the variance weight vector used (shape 6,)
    """

    # ------------------------------------------------------------------
    # Load precomputed variance weights
    # ------------------------------------------------------------------
    print(f"[Memetic] Loading variance weights for "
          f"device={hw_platform}, dataset={dataset} ...")

    edge_weights = load_weights(hw_platform, dataset, weights_dir)

    if verbose:
        print(f"[Memetic] Edge weights (edges 0-5): "
              f"{', '.join(f'e{i}={v:.4f}' for i, v in enumerate(edge_weights))}")

    # ------------------------------------------------------------------
    # Problem
    # ------------------------------------------------------------------
    problem = HWNASProblem(
        nb201_loader=nb201_loader,
        hw_loader=hw_loader,
        dataset=dataset,
        hw_platform=hw_platform,
    )

    # ------------------------------------------------------------------
    # Callback
    # ------------------------------------------------------------------
    callback = VarianceLocalSearchCallback(
        nb201_loader = nb201_loader,
        hw_loader    = hw_loader,
        dataset      = dataset,
        device       = hw_platform,
        edge_weights = edge_weights,
        K            = K,
        seed         = seed + 1,
    )

    # ------------------------------------------------------------------
    # Algorithm (identical operator set as baseline)
    # ------------------------------------------------------------------
    algorithm = NSGA2(
        pop_size  = pop_size,
        sampling  = IntegerRandomSampling(),
        crossover = SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
        mutation  = PM(prob=1.0 / problem.n_var, eta=20,
                       vtype=float, repair=RoundingRepair()),
        eliminate_duplicates = True,
        callback  = callback,
    )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    print(
        f"[Memetic NSGA2] Starting | dataset={dataset}  hw={hw_platform}  "
        f"pop={pop_size}  gen={n_gen}  K={K}  seed={seed}"
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", n_gen),
        seed    = seed,
        verbose = verbose,
    )

    # ------------------------------------------------------------------
    # Extract results (same shape contract as nsga2.run)
    # ------------------------------------------------------------------
    X = result.pop.get("X").flatten().astype(int)
    F = result.pop.get("F")

    pareto_X   = result.opt.get("X").flatten().astype(int)
    pareto_F   = result.opt.get("F")
    pareto_acc = -pareto_F[:, 0]
    pareto_lat =  pareto_F[:, 1]

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(f"\n[Memetic NSGA2] Done -- Pareto front size: {len(pareto_X)}")
        print(
            f"  Best accuracy : {pareto_acc[best_acc_idx]:.2f}%  "
            f"(arch={pareto_X[best_acc_idx]}, "
            f"lat={pareto_lat[best_acc_idx]:.3f} ms)"
        )
        print(
            f"  Best latency  : {pareto_lat[best_lat_idx]:.3f} ms  "
            f"(arch={pareto_X[best_lat_idx]}, "
            f"acc={pareto_acc[best_lat_idx]:.2f}%)"
        )
        print(
            f"  Local search new evals: {callback.total_new_evals}  "
            f"(cached hits not counted)"
        )

    return {
        "result":         result,
        "X":              X,
        "F":              F,
        "pareto_X":       pareto_X,
        "pareto_F":       pareto_F,
        "pareto_acc":     pareto_acc,
        "pareto_lat":     pareto_lat,
        "total_ls_evals": callback.total_new_evals,
        "edge_weights":   edge_weights,
    }