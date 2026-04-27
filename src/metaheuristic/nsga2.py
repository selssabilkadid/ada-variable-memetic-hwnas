import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.termination import get_termination


# ---------------------------------------------------------------------------
# Custom pymoo Problem
# ---------------------------------------------------------------------------

class HWNASProblem(ElementwiseProblem):
    """
    Single-index NASBench-201 bi-objective problem for pymoo NSGA-II.

    Decision variable : x[0] = arch_index  (integer, scalar)
    Objectives        : f[0] = -accuracy   (minimise → maximise accuracy)
                        f[1] =  latency    (minimise, raspi4 by default)

    Both objectives are pure benchmark lookups — no training involved.
    """

    def __init__(
        self,
        nb201_loader,           # NASBench201Loader instance
        hw_loader,              # HWNASLoader instance
        dataset: str = "cifar10",
        hw_platform: str = "raspi4",
    ):
        # Pull the search space upper bound directly from the loader constant
        # so this never goes out of sync with HWNASLoader.SEARCH_SPACE_SIZE
        n_archs = hw_loader.SEARCH_SPACE_SIZE   # 15625 for NASBench-201

        super().__init__(
            n_var=1,                        # single integer decision variable
            n_obj=2,                        # f1 = -acc,  f2 = latency
            n_ieq_constr=0,
            xl=np.array([0]),
            xu=np.array([n_archs - 1]),
            vtype=int,                      # pymoo >= 0.6: marks variable as integer
        )

        self.nb201_loader = nb201_loader
        self.hw_loader    = hw_loader
        self.dataset      = dataset
        self.hw_platform  = hw_platform

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Called once per individual by ElementwiseProblem.

        x[0] arrives as an integer thanks to vtype=int + RoundingRepair.
        We defensively round anyway to guard against any floating-point drift.
        """
        arch_index = int(round(x[0]))

        # --- Objective 1: accuracy (NAS-Bench-201) ----------------------------
        # get_accuracy returns a float in [0, 100]
        # We negate so NSGA-II minimises → we effectively maximise accuracy
        accuracy = self.nb201_loader.get_accuracy(arch_index, self.dataset)
        f1 = -accuracy

        # --- Objective 2: hardware latency (HW-NAS-Bench) ---------------------
        # get_latency returns latency in ms for the chosen hw_platform
        latency = self.hw_loader.get_latency(
            arch_index,
            hw_platform=self.hw_platform,
            dataset=self.dataset,
        )
        f2 = latency   # already in minimisation form

        out["F"] = [f1, f2]


# ---------------------------------------------------------------------------
# NSGA-II setup + run
# ---------------------------------------------------------------------------

def run(
    nb201_loader,
    hw_loader,
    dataset: str       = "cifar10",
    hw_platform: str   = "raspi4",
    pop_size: int      = 100,
    n_gen: int         = 50,
    seed: int          = 42,
    verbose: bool      = True,
) -> dict:
    """
    Run NSGA-II on the NASBench-201 / HW-NAS-Bench search space.

    Parameters
    ----------
    nb201_loader  : NASBench201Loader
        Provides accuracy via get_accuracy(arch_index, dataset).
    hw_loader     : HWNASLoader
        Provides latency via get_latency(arch_index, hw_platform, dataset).
    dataset       : str
        One of ["cifar10", "cifar100", "ImageNet16-120"].
    hw_platform   : str
        Hardware target for latency. Default "raspi4".
    pop_size      : int
        NSGA-II population size.
    n_gen         : int
        Number of generations (termination criterion).
    seed          : int
        Random seed for full reproducibility.
    verbose       : bool
        Print pymoo generation-by-generation progress.

    Returns
    -------
    dict with keys
        "result"        : raw pymoo Result object
        "X"             : arch indices of final population          shape (pop_size,)
        "F"             : objective matrix [-acc, latency]          shape (pop_size, 2)
        "pareto_X"      : arch indices on the Pareto front          shape (n_pareto,)
        "pareto_F"      : objective matrix on the Pareto front      shape (n_pareto, 2)
        "pareto_acc"    : accuracy (%)  on the Pareto front         shape (n_pareto,)
        "pareto_lat"    : latency  (ms) on the Pareto front         shape (n_pareto,)
    """

    # -- Problem -----------------------------------------------------------------
    problem = HWNASProblem(
        nb201_loader=nb201_loader,
        hw_loader=hw_loader,
        dataset=dataset,
        hw_platform=hw_platform,
    )

    # -- Integer-safe operator set -----------------------------------------------
    #
    # Standard pymoo recipe for single integer variables:
    #
    #   IntegerRandomSampling   -- draws uniform integers in [xl, xu]
    #   SBX + RoundingRepair    -- applies continuous crossover then snaps to int
    #   PM  + RoundingRepair    -- applies polynomial mutation  then snaps to int
    #
    # eta=15 / eta=20 are standard NSGA-II defaults for SBX / PM respectively.
    # prob=1/n_var for PM means on average one variable is mutated per individual,
    # consistent with the single-edge mutation used in RegularizedEvolution.
    #
    # eliminate_duplicates=True is critical: with a 15 625-point discrete space
    # and pop_size=100, collisions after mutation are frequent, especially early on.
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=SBX(
            prob=0.9,
            eta=15,
            vtype=float,
            repair=RoundingRepair(),
        ),
        mutation=PM(
            prob=1.0 / problem.n_var,   # = 1.0 for a single decision variable
            eta=20,
            vtype=float,
            repair=RoundingRepair(),
        ),
        eliminate_duplicates=True,
    )

    # -- Termination -------------------------------------------------------------
    termination = get_termination("n_gen", n_gen)

    # -- Run ---------------------------------------------------------------------
    print(
        f"[NSGA2] Starting | dataset={dataset}  hw={hw_platform}  "
        f"pop={pop_size}  gen={n_gen}  seed={seed}"
    )

    result = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
    )

    # -- Extract final population ------------------------------------------------
    X = result.pop.get("X").flatten().astype(int)   # (pop_size,)
    F = result.pop.get("F")                          # (pop_size, 2)

    # -- Extract Pareto front ----------------------------------------------------
    # pymoo stores the non-dominated set of the final population in result.opt
    pareto_X = result.opt.get("X").flatten().astype(int)   # (n_pareto,)
    pareto_F = result.opt.get("F")                         # (n_pareto, 2)

    # Convert back to human-readable form
    pareto_acc = -pareto_F[:, 0]   # accuracy  (%)  -- undo the negation
    pareto_lat =  pareto_F[:, 1]   # latency   (ms)

    if verbose:
        best_acc_idx = pareto_acc.argmax()
        best_lat_idx = pareto_lat.argmin()
        print(f"\n[NSGA2] Done -- Pareto front size: {len(pareto_X)}")
        print(
            f"  Best accuracy : {pareto_acc[best_acc_idx]:.2f}%  "
            f"(arch_index={pareto_X[best_acc_idx]}, "
            f"latency={pareto_lat[best_acc_idx]:.3f} ms)"
        )
        print(
            f"  Best latency  : {pareto_lat[best_lat_idx]:.3f} ms  "
            f"(arch_index={pareto_X[best_lat_idx]}, "
            f"accuracy={pareto_acc[best_lat_idx]:.2f}%)"
        )

    return {
        "result":     result,       # full pymoo Result (history, algorithm state, ...)
        "X":          X,            # all final-population arch indices
        "F":          F,            # all final-population objectives [-acc, lat]
        "pareto_X":   pareto_X,     # Pareto-front arch indices
        "pareto_F":   pareto_F,     # Pareto-front objectives [-acc, lat]
        "pareto_acc": pareto_acc,   # Pareto-front accuracies  (%)
        "pareto_lat": pareto_lat,   # Pareto-front latencies   (ms)
    }