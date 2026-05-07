"""
Self-adaptive mutation strategy pool for NSGA-II (SaMuNet-style).

Implements the frequency-based Adaptive Operator Selection (AOS) from:
  Xue et al., "A Self-Adaptive Mutation Neural Architecture Search
  Algorithm Based on Blocks", IEEE CIM, 2021.

Three mutation strategies with adaptive selection probabilities:
  - conservative: change 1 gene to a neighbour value (±1 mod n_choices)
  - uniform:      change 1 gene to any other value
  - aggressive:   change 2-3 genes simultaneously

Feedback uses Pareto dominance (not single-objective accuracy):
  - offspring dominates pre-mutation  →  SUCCESS
  - pre-mutation dominates offspring  →  FAILURE
  - otherwise                         →  NEUTRAL (not recorded)

Probabilities are recomputed every update_period generations using
the success-rate normalisation from Equations 1-4 of the paper.
"""

from __future__ import annotations

import numpy as np
from typing import Callable, Dict, List, Optional


# ── Pareto dominance (minimisation) ──────────────────────────────────────────

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff a <= b component-wise and a < b in at least one component."""
    return bool(np.all(a <= b) and np.any(a < b))


# ── Strategy functions ────────────────────────────────────────────────────────
# Each takes (vars_, num_choices, rng) and modifies vars_ IN-PLACE.
# Returns the index of the first mutated gene.

def conservative_mutation(
    vars_: np.ndarray,
    num_choices: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    """Change 1 gene to a neighbour (±1 mod n_choices).

    Exploitation — small perturbation, stays close to current architecture.
    E.g. for 5 ops: conv_3x3(0) → conv_1x1(1), or avg_pool(2) → skip(3).
    """
    n = len(vars_)
    gene = int(rng.randint(0, n))
    nc = int(num_choices[gene])
    old = int(vars_[gene])
    # Pick +1 or -1, wrapping around
    direction = rng.choice([-1, 1])
    new_val = (old + direction) % nc
    vars_[gene] = new_val
    return gene


def uniform_mutation(
    vars_: np.ndarray,
    num_choices: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    """Change 1 gene to any other value (uniform random).

    Balanced — moderate exploration. Equivalent to the standard
    discrete_mutation applied to a single gene.
    """
    n = len(vars_)
    gene = int(rng.randint(0, n))
    nc = int(num_choices[gene])
    old = int(vars_[gene])
    others = [v for v in range(nc) if v != old]
    vars_[gene] = int(rng.choice(others))
    return gene


def aggressive_mutation(
    vars_: np.ndarray,
    num_choices: np.ndarray,
    rng: np.random.RandomState,
) -> int:
    """Change 2-3 genes simultaneously.

    Exploration — large structural change, can jump far across
    the search space in one step.
    """
    n = len(vars_)
    n_mutate = int(rng.randint(2, min(4, n + 1)))  # 2 or 3
    genes = rng.choice(n, size=n_mutate, replace=False)
    for g in genes:
        nc = int(num_choices[g])
        old = int(vars_[g])
        others = [v for v in range(nc) if v != old]
        vars_[g] = int(rng.choice(others))
    return int(genes[0])


# ── Registry ──────────────────────────────────────────────────────────────────

STRATEGY_NAMES: List[str] = ["conservative", "uniform", "aggressive"]
STRATEGY_FUNCS: List[Callable] = [conservative_mutation, uniform_mutation, aggressive_mutation]


# ── Adaptive Mutation Pool ────────────────────────────────────────────────────

class AdaptiveMutationPool:
    """Self-adaptive mutation strategy pool.

    Maintains N mutation strategies with selection probabilities that
    are recomputed every *update_period* generations based on the
    historical success rate of each strategy.

    Parameters
    ----------
    num_choices : array-like
        Per-gene number of valid choices.
    strategies : list of callable, optional
        Strategy functions.  Default: [conservative, uniform, aggressive].
    initial_probs : list of float, optional
        Starting selection probabilities.  Default: [0.3, 0.4, 0.3].
    update_period : int
        Recompute probabilities every N generations.  Default: 5.
    """

    def __init__(
        self,
        num_choices: np.ndarray,
        strategies: Optional[List[Callable]] = None,
        initial_probs: Optional[List[float]] = None,
        update_period: int = 5,
    ) -> None:
        self.strategies = strategies or STRATEGY_FUNCS
        self.n_strategies = len(self.strategies)
        self.num_choices = np.atleast_1d(np.asarray(num_choices, dtype=int))
        self.update_period = update_period

        if initial_probs is not None:
            self.probs = np.array(initial_probs, dtype=np.float64)
        else:
            self.probs = np.ones(self.n_strategies, dtype=np.float64) / self.n_strategies

        # Per-window counters (reset every update_period generations)
        self._success = np.zeros(self.n_strategies, dtype=np.float64)
        self._fail = np.zeros(self.n_strategies, dtype=np.float64)
        self._gen_counter = 0

        # Full history (never reset) — useful for plotting
        self.prob_history: List[Dict] = []

    # ── Selection ─────────────────────────────────────────────────────────

    def select_strategy(self, rng: np.random.RandomState) -> int:
        """UCB1 instead of roulette wheel."""
        total = self._success.sum() + self._fail.sum()
        if total == 0:
            return int(rng.randint(0, self.n_strategies))
        
        pulls = self._success + self._fail
        pulls[pulls == 0] = 1  # avoid division by zero
        mean_reward = self._success / pulls
        exploration = np.sqrt(np.log(total + 1) / pulls)
        
        ucb = mean_reward + 1.41 * exploration  # C = sqrt(2)
        return int(np.argmax(ucb))

    def apply_strategy(
        self,
        vars_: np.ndarray,
        strategy_idx: int,
        rng: np.random.RandomState,
    ) -> int:
        """Apply the chosen strategy *in-place* to vars_.

        Returns the index of the first mutated gene.
        """
        return self.strategies[strategy_idx](vars_, self.num_choices, rng)

    # ── Feedback ──────────────────────────────────────────────────────────

    def record_outcome(self, strategy_idx: int, success: bool) -> None:
        """Manually record success (True) or failure (False)."""
        if success:
            self._success[strategy_idx] += 1.0
        else:
            self._fail[strategy_idx] += 1.0

    def record_dominance(self, strategy_idx, post_obj, pre_obj):
        if np.any(np.isinf(post_obj)) or np.any(np.isinf(pre_obj)):
            post_inf = np.sum(np.isinf(post_obj))
            pre_inf  = np.sum(np.isinf(pre_obj))
            if post_inf < pre_inf:
                self._success[strategy_idx] += 1.0
            elif pre_inf < post_inf:
                self._fail[strategy_idx] += 1.0
            else:
                post_finite = post_obj[np.isfinite(post_obj)]
                pre_finite  = pre_obj[np.isfinite(pre_obj)]
                if len(post_finite) > 0 and _dominates(post_finite, pre_finite):
                    self._success[strategy_idx] += 1.0
                elif len(pre_finite) > 0 and _dominates(pre_finite, post_finite):
                    self._fail[strategy_idx] += 1.0
            return

        if _dominates(post_obj, pre_obj):
            self._success[strategy_idx] += 1.0
        elif _dominates(pre_obj, post_obj):
            self._fail[strategy_idx] += 1.0
    # ── Periodic update ───────────────────────────────────────────────────

    def step(self) -> bool:
        """Call once per generation.

        Returns True on the generation when probabilities are recomputed.
        """
        self._gen_counter += 1
        if self._gen_counter % self.update_period == 0:
            self._update()
            return True
        return False

    def _update(self) -> None:
        """Recompute probabilities from accumulated success/failure counts.

        Implements Equations 1-4 from the SaMuNet paper.
        """
        # Snapshot before update
        self.prob_history.append({
            "generation": self._gen_counter,
            "probs": self.probs.copy(),
            "success": self._success.copy(),
            "fail": self._fail.copy(),
        })

        rates = np.zeros(self.n_strategies, dtype=np.float64)
        for i in range(self.n_strategies):
            s = self._success[i]
            f = self._fail[i]
            if s == 0.0 and f == 0.0:
                s = 1.0                       # floor to prevent extinction
            rates[i] = s / (s + f)           # Eq. 3: success rate

        total = rates.sum()
        if total > 0:
            self.probs = rates / total       # Eq. 4: normalise
        else:
            self.probs[:] = 1.0 / self.n_strategies

        # Reset counters for next window
        self._success[:] = 0.0
        self._fail[:] = 0.0

    # ── Diagnostics ───────────────────────────────────────────────────────

    @property
    def summary(self) -> str:
        """One-line summary of current probabilities."""
        parts = [f"{STRATEGY_NAMES[i]}={self.probs[i]:.3f}"
                 for i in range(self.n_strategies)]
        return "  ".join(parts)

    @property
    def generation(self) -> int:
        return self._gen_counter