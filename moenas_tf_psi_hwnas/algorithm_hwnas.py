# moenas_tf_psi_hwnas/algorithm_hwnas.py
"""
NSGA-II subclass that injects PSI_processor into the generation loop.

Import strategy
---------------
The root cause of the import error is:

    PSI/PSI.py
        → from utils import get_hashKey
        → utils/__init__.py imports utils/utils.py
        → utils/utils.py: from nasbench import wrap_api as api
        → nasbench/api.py → nasbench/lib/config.py
        → from absl import flags   ← ModuleNotFoundError

We break this chain at the PSI.py level by monkey-patching the 'utils'
module in sys.modules BEFORE importing PSI_processor.  Python's import
system then hands our shim to PSI.py instead of their utils package.

This requires zero changes to any file in the MOENAS-TF-PSI clone.

Fixes applied
-------------
1. _SafeEvaluator.__getattr__ guards 'name == _inner' to prevent
   infinite recursion during copy.deepcopy (deepcopy reconstructs
   without calling __init__, so self._inner may not exist yet).

2. _sanitize_pop() extracted as a shared helper and called BOTH in
   _advance (after PSI) AND at the top of _post_advance (before
   super() triggers pymoo's XTolerance termination criterion which
   does np.row_stack on all X arrays and crashes on wrong shape/dtype).

3. X is cast to np.ndarray(dtype=int) inside _sanitize_pop so that
   row_stack never produces a (n, 0) shaped array from a mix of lists
   and ndarrays.
"""

from __future__ import annotations

import sys
import os
import copy
import math
import types
import importlib.util
import numpy as np

# ---------------------------------------------------------------------------
# Step 0 — Make sure the project root is on sys.path so that imports like
#           'from moenas_tf_psi_hwnas.problem_hwnas import ...' work whether
#           this file is run as a script or imported as part of a package.
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../moenas_tf_psi_hwnas/
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)                   # .../hwnas-with-metaheuristics-project/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Step 1 — Load utils_shim directly from its file path, bypassing any
#           package resolution.  This works whether the project root is on
#           sys.path or not, and whether running as a script or as a module.
# ---------------------------------------------------------------------------
_SHIM_PATH = os.path.join(_THIS_DIR, "utils_shim.py")
_spec      = importlib.util.spec_from_file_location("utils_shim", _SHIM_PATH)
_shim      = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shim)

# Build a fake module object that exposes every name PSI.py imports
_utils_mod = types.ModuleType("utils")
_utils_mod.get_hashKey                 = _shim.get_hashKey
_utils_mod.find_all_neighbors          = _shim.find_all_neighbors
_utils_mod.find_the_better             = _shim.find_the_better
_utils_mod.get_idx_non_dominated_front = _shim.get_idx_non_dominated_front
_utils_mod.check_above_or_below        = _shim.check_above_or_below
_utils_mod.calculate_angle_measure     = _shim.calculate_angle_measure
_utils_mod.check_valid                 = _shim.check_valid

sys.modules["utils"]     = _utils_mod   # intercepts bare 'from utils import ...'

# PSI/utils.py does 'from utils import ...' via a relative-style bare import;
# also register under the PSI sub-path in case Python resolves it that way.
sys.modules["PSI.utils"] = _utils_mod

# ---------------------------------------------------------------------------
# Step 2 — Add MOENAS-TF-PSI root to sys.path so 'from PSI.PSI import ...'
#           resolves correctly, now that utils is patched.
# ---------------------------------------------------------------------------
_MOENAS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../MOENAS-TF-PSI")
)
if _MOENAS_DIR not in sys.path:
    sys.path.insert(0, _MOENAS_DIR)

# ---------------------------------------------------------------------------
# Step 3 — Now it is safe to import PSI_processor
# ---------------------------------------------------------------------------
from PSI.PSI import PSI_processor   # noqa: E402  (must come after monkey-patch)

# ---------------------------------------------------------------------------
# Normal imports
# ---------------------------------------------------------------------------
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling

# Load problem_hwnas the same way — path-safe, works as script or module
_PROB_PATH = os.path.join(_THIS_DIR, "problem_hwnas.py")
_spec2     = importlib.util.spec_from_file_location("problem_hwnas", _PROB_PATH)
_prob_mod  = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_prob_mod)
ElitistArchive = _prob_mod.ElitistArchive


# ---------------------------------------------------------------------------
# hashKey injection helper
# ---------------------------------------------------------------------------

def _inject_hash_keys(pop, problem_name: str) -> None:
    """
    Stamp every individual in `pop` with a hashKey string.

    PSI_processor accesses hashKey in TWO different ways:
        pop.get('hashKey')           → reads from pymoo's internal data dict
        non_dominated_set[i].hashKey → reads as a direct Python attribute

    We must set BOTH so both access patterns work.
    """
    for ind in pop:
        X = ind.get("X")
        if X is not None:
            hk = _shim.get_hashKey(list(X), problem_name)
            ind.set("hashKey", hk)   # satisfies pop.get('hashKey')
            ind.hashKey = hk         # satisfies ind.hashKey attribute access


# ---------------------------------------------------------------------------
# Population.new() / Population.merge() compatibility shim
# ---------------------------------------------------------------------------

def _patch_pymoo_population_api():
    """
    Globally patch Population.new() and Population.merge() so PSI_processor
    old-API calls work with new pymoo.

    Old API (PSI uses):
        P.new(1)             -> 1 empty individual
        pop_a.merge(pop_b)   -> instance method

    New API (pymoo provides):
        Population.new("X", arr)   -> keyword pairs
        Population.merge(a, b)     -> static/class, both args required

    Patching at CLASS level means every Population instance is covered,
    including those created inside PSI — no per-instance patching needed.
    """
    from pymoo.core.population import Population
    from pymoo.core.individual  import Individual

    _real_new = staticmethod(Population.__dict__.get("new", Population.new))

    @classmethod
    def _new_compat(cls, *args, **kwargs):
        # Old API: P.new(1) -> n empty individuals
        # PSI calls this to get a blank Individual, then immediately sets
        # X, hashKey, and F on it before using it.
        if len(args) == 1 and isinstance(args[0], int):
            n = args[0]
            return cls(individuals=[Individual() for _ in range(n)])
        # New API: Population.new("X", array, ...) keyword pairs
        try:
            return _real_new.__func__(cls, *args, **kwargs)
        except Exception:
            return cls(individuals=[Individual()])

    @classmethod
    def _merge_compat(cls, *args):
        # Handles both:
        #   instance call: pop_a.merge(pop_b)      -> cls=pop_a, args=(pop_b,)
        #   class call:    Population.merge(a, b)  -> cls=Population, args=(a,b)
        if isinstance(cls, Population):
            a, b = cls, args[0]
        elif len(args) == 2:
            a, b = args[0], args[1]
        else:
            raise TypeError("merge() expected instance call or 2 args")
        return Population(individuals=list(a) + list(b))

    Population.new   = _new_compat
    Population.merge = _merge_compat


_patch_pymoo_population_api()


# ---------------------------------------------------------------------------
# Safe evaluator — skips individuals with malformed X
# ---------------------------------------------------------------------------

class _SafeEvaluator:
    """
    Wraps pymoo Evaluator to discard blank PSI placeholder individuals
    (X=None or wrong shape) before they reach problem.evaluate().
    PSI creates these via P.new(1) and sometimes leaks them into self.pop.
    """

    def __init__(self, n_var: int):
        from pymoo.core.evaluator import Evaluator
        self._inner = Evaluator()
        self.n_var  = n_var
        self.n_eval = 0

    def eval(self, problem, pop, **kwargs):
        from pymoo.core.population import Population
        valid = [
            ind for ind in pop
            if ind.get("X") is not None
            and np.asarray(ind.get("X")).shape == (self.n_var,)
        ]
        if not valid:
            return
        self._inner.eval(problem, Population(individuals=valid), **kwargs)
        self.n_eval = self._inner.n_eval

    def __getattr__(self, name):
        # FIX 1: Guard against bootstrap recursion during copy.deepcopy.
        #
        # deepcopy reconstructs the object WITHOUT calling __init__, so
        # self._inner does not exist yet when deepcopy probes for optional
        # dunder methods (e.g. __setstate__) via hasattr().
        #
        # Without this guard the call chain is:
        #   hasattr(obj, '__setstate__')
        #     → __getattr__('__setstate__')
        #       → getattr(self._inner, '__setstate__')
        #         → __getattr__('_inner')          ← _inner missing
        #           → getattr(self._inner, '_inner')
        #             → __getattr__('_inner')       ← infinite recursion
        #
        # Raising AttributeError for '_inner' is the correct contract:
        # it tells Python "this attribute does not exist", which is exactly
        # what hasattr() and deepcopy need to hear.
        if name == "_inner":
            raise AttributeError(name)
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# No-op display — satisfies every method pymoo calls on self.display
# ---------------------------------------------------------------------------

class _NoOpDisplay:
    """
    Replaces pymoo's display object when display is None or uncallable.
    pymoo calls the following on self.display across different versions:
        self.display(self)       — _post_advance()
        self.display.finalize()  — finalize()
        self.display.reset()     — setup()  (some versions)
        self.display.update(...) — some versions
    All are silently ignored here.
    """
    def __call__(self, *a, **kw): pass
    def finalize(self):           pass
    def reset(self, *a, **kw):    pass
    def update(self, *a, **kw):   pass


# ---------------------------------------------------------------------------
# Algorithm subclass
# ---------------------------------------------------------------------------

class NSGA2_PSI(NSGA2):
    """
    NSGA-II with optional PSI local search injected after each generation.

    Parameters
    ----------
    problem      : HWNASProblemMOENAS instance
    use_psi      : False → vanilla NSGA-II baseline (no local search)
    psi_npoints  : k, number of simultaneous edge flips per neighbour
    psi_using_zc : True  → TF-PSI (uses get_performance_metric proxy)
                   False → PSI with true benchmark objectives
    gamma        : knee-point angle threshold (PSI default 210)
    pop_size     : population size (paper default 20)
    """

    def __init__(
        self,
        problem,
        use_psi:      bool = True,
        psi_npoints:  int  = 1,
        psi_using_zc: bool = True,
        gamma:        int  = 210,
        pop_size:     int  = 20,
        **kwargs,
    ):
        super().__init__(
            pop_size  = pop_size,
            sampling  = IntegerRandomSampling(),
            crossover = SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
            mutation  = PM(
                prob   = 1.0 / problem.n_var,
                eta    = 20,
                vtype  = float,
                repair = RoundingRepair(),
            ),
            eliminate_duplicates = True,
            **kwargs,
        )

        self.problem          = problem
        self.use_psi          = use_psi
        self.E_Archive_search = ElitistArchive()

        # Wrap the evaluator to silently skip individuals with malformed X.
        # PSI leaks blank placeholder individuals (X=None or wrong shape)
        # into self.pop via survival.do() merge.  pymoo's next() then tries
        # to evaluate them as infills and crashes on X.shape[1].
        # We intercept at the evaluator level — the safest choke point.
        self.evaluator = _SafeEvaluator(n_var=self.problem.n_var)

        self.psi = PSI_processor(
            gamma    = gamma,
            nPoints  = psi_npoints,
            using_zc = psi_using_zc,
        ) if use_psi else None

    # ------------------------------------------------------------------
    # Interface required by PSI_processor
    # ------------------------------------------------------------------

    def evaluate(self, X) -> np.ndarray:
        """
        Called by PSI_processor as kwargs['algorithm'].evaluate(neighbor_X).
        X is an op-vector (list of 6 ints).
        Returns np.array([error_rate, latency]).
        """
        return self.problem._eval_ops(X)

    # ------------------------------------------------------------------
    # Shared population sanitization
    # ------------------------------------------------------------------

    def _sanitize_pop(self) -> None:
        """
        Ensure every individual in self.pop has:
          • X as np.ndarray of shape (n_var,) and dtype int
          • F as np.ndarray of shape (n_obj,)

        Any individual failing these checks is replaced with a copy of a
        valid one (cycling through valid individuals to preserve pop_size).

        FIX 2: This helper is called in TWO places:
          (a) _advance()      — immediately after PSI modifies self.pop
          (b) _post_advance() — BEFORE super() is called, because pymoo's
              XTolerance termination criterion runs inside super()._post_advance()
              and does np.row_stack on every X in the population.  If any X has
              wrong shape or is a plain Python list, row_stack produces a (n, 0)
              array and the subsequent boolean-mask indexing crashes:

                IndexError: boolean index did not match indexed array along axis 1;
                size of axis is 0 but size of corresponding boolean axis is 6

        FIX 3: X is cast to np.ndarray(dtype=int) to prevent row_stack from
        producing a (n, 0) shaped array when X values are stored as Python lists.
        """
        from pymoo.core.population import Population as _Pop

        n_var, n_obj      = self.problem.n_var, self.problem.n_obj
        valid, invalid_idx = [], []

        for k, ind in enumerate(self.pop):
            X = ind.get("X")
            F = ind.get("F")
            x_ok = (X is not None and np.asarray(X).shape == (n_var,))
            f_ok = (F is not None and np.asarray(F).shape == (n_obj,))
            if x_ok and f_ok:
                valid.append(ind)
            else:
                invalid_idx.append(k)

        if not valid:
            # Nothing usable — leave pop alone, let pymoo surface the error
            return

        if invalid_idx:
            all_inds = list(self.pop)
            for k in invalid_idx:
                # Replace with a cycled copy of a valid individual
                all_inds[k] = copy.copy(valid[k % len(valid)])
            self.pop = _Pop(individuals=all_inds)

        # Enforce int dtype on every individual's X so np.row_stack always
        # produces a well-shaped (n, n_var) matrix inside pymoo's termination.
        for ind in self.pop:
            X = ind.get("X")
            if X is not None:
                arr = np.asarray(X, dtype=int)
                # Only set if dtype or type has changed (avoid redundant work)
                if not isinstance(X, np.ndarray) or X.dtype != arr.dtype:
                    ind.set("X", arr)

    # ------------------------------------------------------------------
    # pymoo generation hooks
    # ------------------------------------------------------------------

    def _post_advance(self):
        """
        1. Ensure self.display is always a safe callable object.
        2. Sanitize self.pop BEFORE calling super()._post_advance().

        The order matters: super()._post_advance() calls
        self.termination.update(self), which internally runs XTolerance.
        XTolerance calls normalize(X, xl=..., xu=...) which does
        np.row_stack on all X arrays.  If any X is a plain list or has
        wrong shape the boolean-mask indexing inside normalize() crashes.
        _sanitize_pop() must run first to guarantee all X arrays are
        proper np.ndarray(dtype=int) of shape (n_var,).
        """
        if not isinstance(getattr(self, "display", None), _NoOpDisplay):
            if not hasattr(getattr(self, "display", None), "finalize"):
                self.display = _NoOpDisplay()

        # KEY FIX: sanitize before termination criterion accesses X arrays
        self._sanitize_pop()

        super()._post_advance()

    def _initialize_advance(self, infills=None, **kwargs):
        """
        Called by pymoo after the initial population is evaluated.
        Inject hashKeys here so PSI has them from generation 0.
        """
        super()._initialize_advance(infills=infills, **kwargs)
        _inject_hash_keys(self.pop, self.problem.name)

    def _advance(self, infills=None, **kwargs):
        """
        Standard NSGA-II advance first, then PSI local search.
        Called by pymoo once per generation after offspring evaluation
        and survivor selection.
        """
        super()._advance(infills=infills, **kwargs)

        if not self.use_psi or self.psi is None:
            return

        # Re-stamp hashKeys after crossover/mutation produced new individuals
        _inject_hash_keys(self.pop, self.problem.name)

        # Reset per-generation tracking inside PSI_processor
        self.psi.reset()

        # Run PSI — may mutate or replace individuals in self.pop
        self.pop = self.psi.do(
            self.pop,
            problem   = self.problem,
            algorithm = self,
        )

        # First sanitization pass — cleans up PSI's blank placeholders.
        # A second pass runs in _post_advance() before termination checks.
        self._sanitize_pop()