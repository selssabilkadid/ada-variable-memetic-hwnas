from __future__ import annotations

import numpy as np
from pymoo.core.problem import ElementwiseProblem


N_EDGES = 6
N_OPS   = 5


def index_to_ops(arch_index: int) -> list:
    ops, idx = [], arch_index
    for _ in range(N_EDGES):
        ops.append(idx % N_OPS)
        idx //= N_OPS
    return ops          # list of 6 ints, length N_EDGES


def ops_to_index(ops) -> int:
    return int(sum(op * (N_OPS ** e) for e, op in enumerate(ops)))


# ---------------------------------------------------------------------------
# Elitist archive  (satisfies E_Archive_search interface)
# ---------------------------------------------------------------------------

class ElitistArchive:
    """
    Minimal non-dominated archive.

    PSI_processor calls:
        kwargs['algorithm'].E_Archive_search.update(ind, algorithm=algo)

    where `ind` is a pymoo Individual with X, hashKey, F already set.
    We store a copy of every non-dominated individual seen so far.
    The runner extracts the final Pareto front from here after the search.
    """

    def __init__(self):
        self._members: list = []

    def update(self, ind, algorithm=None):
        f_new = ind.get("F")
        # Drop archive members dominated by the incoming individual
        self._members = [
            m for m in self._members
            if not _dominates(f_new, m.get("F"))
        ]
        # Add only if not dominated by any remaining member
        if not any(_dominates(m.get("F"), f_new) for m in self._members):
            self._members.append(ind.copy())

    def pareto_F(self) -> np.ndarray:
        if not self._members:
            return np.empty((0, 2))
        return np.array([m.get("F") for m in self._members])

    def pareto_X(self) -> list:
        return [m.get("X") for m in self._members]


def _dominates(f_a: np.ndarray, f_b: np.ndarray) -> bool:
    return bool(np.all(f_a <= f_b) and np.any(f_a < f_b))


# ---------------------------------------------------------------------------
# Problem class
# ---------------------------------------------------------------------------

class HWNASProblemMOENAS(ElementwiseProblem):
    """
    Bi-objective NAS-Bench-201 / HW-NAS-Bench problem for MOENAS-TF-PSI.

    Decision variables : op-vector X of length 6, each in {0,1,2,3,4}
    Objective 0        : test error rate  = 1 - accuracy/100   (minimise)
    Objective 1        : hardware latency (ms)                  (minimise)

    The op-vector encoding is mandatory here: PSI_processor mutates X[idx]
    to flip individual edges, so a flat integer encoding does not work.
    """

    # Attributes read directly by PSI_processor -------------------------
    name          = "NASBench201"       # used by their get_hashKey() utility
    maxLength     = N_EDGES             # number of mutable positions
    available_ops = list(range(N_OPS))  # [0, 1, 2, 3, 4]

    def __init__(
        self,
        nb201_loader,
        hw_loader,
        dataset:     str = "cifar10",
        hw_platform: str = "raspi4",
    ):
        super().__init__(
            n_var        = N_EDGES,
            n_obj        = 2,
            n_ieq_constr = 0,
            xl           = np.zeros(N_EDGES, dtype=int),
            xu           = np.full(N_EDGES, N_OPS - 1, dtype=int),
            vtype        = int,
        )

        self.nb201_loader = nb201_loader
        self.hw_loader    = hw_loader
        self.dataset      = dataset
        self.hw_platform  = hw_platform

        # Shared evaluation cache, keyed by arch_index int
        self._cache:  dict[int, np.ndarray] = {}
        self.n_evals: int = 0   # counts only cache misses

    # ------------------------------------------------------------------
    # pymoo entry point (called for every individual by NSGA-II)
    # ------------------------------------------------------------------

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self._eval_ops(x)

    # ------------------------------------------------------------------
    # Interface required by PSI_processor
    # ------------------------------------------------------------------

    def isValid(self, X) -> bool:
        """
        NAS-Bench-201 has no structural constraints — every op-vector maps
        to a valid architecture.
        """
        return True

    def get_computational_metric(self, X) -> float:
        """
        Returns hardware latency for op-vector X.

        PSI_processor uses this as the 'complexity' proxy during the
        training-free phase to rank neighbours cheaply before committing
        to a full evaluation.  In our benchmark setting, lookups are free,
        so this is exact — no approximation involved.

        Note: PSI_processor stores this as potentialSol_F_proxy[0], so it
        must be in minimisation form (lower = better hardware performance).
        Latency in ms is already in minimisation form.
        """
        arch_index = ops_to_index(X)
        return float(self.hw_loader.get_latency(
            arch_index,
            hw_platform=self.hw_platform,
            dataset=self.dataset,
        ))

    def get_performance_metric(self, X, using_zc: bool = True):
        """
        Returns (performance_score, None, None).

        PSI_processor unpacks as:
            tf_ind, _, _ = problem.get_performance_metric(X, using_zc=True)

        In the original code, using_zc=True returns a Synaptic Flow score
        where HIGHER = better, so PSI_processor negates it internally via
        potentialSol_F_proxy = [computational_metric, tf_ind].
        It then calls find_the_better() which expects minimisation on both.

        IMPORTANT sign convention:
        Their proxy vector is [latency, tf_score] and find_the_better()
        minimises both.  Synaptic Flow is a "higher is better" metric, so
        their tf_ind is already negated (negative synflow) before being
        stored.  We must match this: return NEGATIVE accuracy so that
        minimising it = maximising accuracy.

            tf_ind = -(accuracy / 100)   →  minimising this maximises accuracy
        """
        arch_index = ops_to_index(X)
        acc = self.nb201_loader.get_accuracy(arch_index, self.dataset)
        tf_ind = -(acc / 100.0)   # negated: minimising = maximising accuracy
        return tf_ind, None, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _eval_ops(self, X) -> np.ndarray:
        return self._eval_index(ops_to_index(list(X)))

    def _eval_index(self, arch_index: int) -> np.ndarray:
        if arch_index in self._cache:
            return self._cache[arch_index]
        acc     = self.nb201_loader.get_accuracy(arch_index, self.dataset)
        latency = self.hw_loader.get_latency(
            arch_index,
            hw_platform=self.hw_platform,
            dataset=self.dataset,
        )
        f = np.array([1.0 - acc / 100.0, latency])
        self._cache[arch_index] = f
        self.n_evals += 1
        return f