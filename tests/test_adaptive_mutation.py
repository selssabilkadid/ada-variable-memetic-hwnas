# import numpy as np
# from src.metaheuristic.nsga2.core.nsga2 import NSGA2

# class DummyProblem:
#     def evaluate(self, x):
#         return [float(np.sum(x)), float(np.sum(x))]

# def test_adaptive_pool_initialization():
#     algo = NSGA2(
#         problem=DummyProblem(),
#         pop_size=10,
#         n_var=5,
#         n_obj=2,
#         num_choices=np.full(5, 3),
#         seed=0,
#         adaptive_mutation={
#             "initial_probs": [0.3, 0.4, 0.3],
#             "update_period": 2,
#         }
#     )

#     assert algo._adaptive_pool is not None
#     assert len(algo._adaptive_pool.probs) == 3
    
# def test_adaptive_runs_without_crash():
#     algo = NSGA2(
#         problem=DummyProblem(),
#         pop_size=10,
#         n_var=5,
#         n_obj=2,
#         num_choices=np.full(5, 3),
#         seed=1,
#         adaptive_mutation={
#             "initial_probs": [0.3, 0.4, 0.3],
#             "update_period": 2,
#         }
#     )

#     pareto_set, pareto_front = algo.run(generations=3, verbose=False)

#     assert pareto_front.shape[1] == 2
#     assert len(pareto_front) > 0
    

# def test_adaptive_mutation_updates_probs():
#     algo = NSGA2(
#         problem=DummyProblem(),
#         pop_size=20,
#         n_var=5,
#         n_obj=2,
#         num_choices=np.full(5, 3),
#         seed=0,
#         adaptive_mutation={
#             "initial_probs": [0.3, 0.4, 0.3],
#             "update_period": 1,  # force frequent updates
#         }
#     )

#     initial = algo._adaptive_pool.probs.copy()

#     algo.run(generations=5, verbose=False)

#     final = algo._adaptive_pool.probs

#     # probabilities should change OR at least evolve
#     assert not np.allclose(initial, final)
    
# def test_mutation_strategies_are_used():
#     algo = NSGA2(
#         problem=DummyProblem(),
#         pop_size=20,
#         n_var=5,
#         n_obj=2,
#         num_choices=np.full(5, 3),
#         seed=0,
#         adaptive_mutation={
#             "initial_probs": [0.3, 0.4, 0.3],
#             "update_period": 2,
#         }
#     )

#     algo.run(generations=3, verbose=False)

#     # check mutation_type was assigned
#     used = False
#     for ind in algo._feedback_queue:
#         if hasattr(ind, "strategy_idx"):
#             used = True

#     assert True  # (or replace with better internal logging if needed)            

import numpy as np
import pytest

from src.metaheuristic.nsga2.core.adaptive_mutation import (
    AdaptiveMutationPool,
    conservative_mutation,
    uniform_mutation,
    aggressive_mutation,
    _dominates,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pool(update_period=5, initial_probs=None, n_choices=3, n_genes=5):
    return AdaptiveMutationPool(
        num_choices=np.full(n_genes, n_choices),
        initial_probs=initial_probs,
        update_period=update_period,
    )

def _rng(seed=42):
    return np.random.RandomState(seed)


# ── _dominates ────────────────────────────────────────────────────────────────

class TestDominates:

    def test_strict_dominance(self):
        assert _dominates(np.array([1.0, 2.0]), np.array([3.0, 4.0]))

    def test_equal_in_one_dim(self):
        # a <= b everywhere, a < b in at least one → True
        assert _dominates(np.array([1.0, 2.0]), np.array([2.0, 2.0]))

    def test_identical_arrays_not_dominant(self):
        assert not _dominates(np.array([2.0, 2.0]), np.array([2.0, 2.0]))

    def test_reverse_dominance(self):
        assert not _dominates(np.array([3.0, 4.0]), np.array([1.0, 2.0]))

    def test_incomparable_both_directions(self):
        # [1, 3] vs [2, 2] — neither dominates
        assert not _dominates(np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        assert not _dominates(np.array([2.0, 2.0]), np.array([1.0, 3.0]))

    def test_single_element_true(self):
        assert _dominates(np.array([1.0]), np.array([2.0]))

    def test_single_element_equal(self):
        assert not _dominates(np.array([1.0]), np.array([1.0]))

    def test_many_objectives(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([2.0, 2.0, 4.0, 5.0])
        assert _dominates(a, b)


# ── conservative_mutation ─────────────────────────────────────────────────────

class TestConservativeMutation:

    def test_exactly_one_gene_changes(self):
        rng = _rng()
        nc  = np.full(5, 5)
        for _ in range(50):
            v = np.array([2, 2, 2, 2, 2])
            old = v.copy()
            conservative_mutation(v, nc, rng)
            assert np.sum(v != old) == 1

    def test_mutation_is_adjacent(self):
        # Mutated value must be exactly ±1 mod n_choices from original.
        rng = _rng()
        nc  = np.full(5, 5)
        for _ in range(100):
            v = np.array([2, 2, 2, 2, 2])
            g = conservative_mutation(v, nc, rng)
            diff = (int(v[g]) - 2) % 5
            assert diff in (1, 4), f"Expected ±1 mod 5, got diff={diff}"

    def test_wraps_at_lower_boundary(self):
        # gene=0, n=4 → neighbours are 1 and 3 (wrapping)
        rng  = _rng(0)
        nc   = np.full(1, 4)
        seen = set()
        for _ in range(200):
            v = np.array([0])
            conservative_mutation(v, nc, rng)
            seen.add(int(v[0]))
        assert seen == {1, 3}

    def test_wraps_at_upper_boundary(self):
        rng  = _rng(0)
        nc   = np.full(1, 4)
        seen = set()
        for _ in range(200):
            v = np.array([3])
            conservative_mutation(v, nc, rng)
            seen.add(int(v[0]))
        assert seen == {0, 2}

    def test_returns_correct_gene_index(self):
        rng = _rng()
        nc  = np.full(5, 5)
        v   = np.array([2, 2, 2, 2, 2])
        old = v.copy()
        g   = conservative_mutation(v, nc, rng)
        assert v[g] != old[g]
        assert np.sum(v != old) == 1


# ── uniform_mutation ──────────────────────────────────────────────────────────

class TestUniformMutation:

    def test_exactly_one_gene_changes(self):
        rng = _rng()
        nc  = np.full(5, 4)
        for _ in range(50):
            v   = np.array([1, 1, 1, 1, 1])
            old = v.copy()
            uniform_mutation(v, nc, rng)
            assert np.sum(v != old) == 1

    def test_never_same_value(self):
        rng = _rng()
        nc  = np.full(5, 4)
        for _ in range(100):
            v   = np.array([1, 1, 1, 1, 1])
            old = v.copy()
            g   = uniform_mutation(v, nc, rng)
            assert v[g] != old[g], "Uniform must always change the value"

    def test_all_non_current_values_reachable(self):
        # Over enough trials, every value except the current must appear.
        rng  = _rng(0)
        nc   = np.full(1, 5)
        seen = set()
        for _ in range(500):
            v = np.array([2])
            uniform_mutation(v, nc, rng)
            seen.add(int(v[0]))
        assert seen == {0, 1, 3, 4}

    def test_returns_correct_gene_index(self):
        rng = _rng()
        nc  = np.full(5, 4)
        v   = np.array([1, 1, 1, 1, 1])
        old = v.copy()
        g   = uniform_mutation(v, nc, rng)
        assert v[g] != old[g]


# ── aggressive_mutation ───────────────────────────────────────────────────────

class TestAggressiveMutation:

    def test_changes_two_or_three_genes(self):
        rng = _rng()
        nc  = np.full(6, 4)
        for _ in range(100):
            v   = np.array([1, 1, 1, 1, 1, 1])
            old = v.copy()
            aggressive_mutation(v, nc, rng)
            n_changed = int(np.sum(v != old))
            assert 2 <= n_changed <= 3, f"Expected 2-3 changes, got {n_changed}"

    def test_each_changed_gene_has_new_value(self):
        rng = _rng()
        nc  = np.full(6, 4)
        for _ in range(100):
            v   = np.array([1, 1, 1, 1, 1, 1])
            old = v.copy()
            aggressive_mutation(v, nc, rng)
            for g in np.where(v != old)[0]:
                assert v[g] != old[g]

    def test_two_gene_array_changes_both(self):
        # Edge case: n=2 → always changes exactly 2 genes.
        rng = _rng()
        nc  = np.full(2, 4)
        for _ in range(50):
            v   = np.array([1, 1])
            old = v.copy()
            aggressive_mutation(v, nc, rng)
            assert np.sum(v != old) == 2

    def test_returns_int_gene_index(self):
        rng = _rng()
        nc  = np.full(5, 4)
        v   = np.array([1, 1, 1, 1, 1])
        g   = aggressive_mutation(v, nc, rng)
        assert isinstance(g, int)
        assert 0 <= g < len(v)


# ── record_outcome & record_dominance ─────────────────────────────────────────

class TestFeedback:

    def test_record_outcome_success(self):
        pool = _pool()
        pool.record_outcome(0, success=True)
        assert pool._success[0] == 1.0
        assert pool._fail[0]    == 0.0

    def test_record_outcome_failure(self):
        pool = _pool()
        pool.record_outcome(1, success=False)
        assert pool._fail[1]    == 1.0
        assert pool._success[1] == 0.0

    def test_dominance_success(self):
        pool = _pool()
        pool.record_dominance(0, np.array([1.0, 2.0]), np.array([2.0, 3.0]))
        assert pool._success[0] == 1.0
        assert pool._fail[0]    == 0.0

    def test_dominance_failure(self):
        pool = _pool()
        pool.record_dominance(0, np.array([2.0, 3.0]), np.array([1.0, 2.0]))
        assert pool._fail[0]    == 1.0
        assert pool._success[0] == 0.0

    def test_dominance_neutral_records_nothing(self):
        pool = _pool()
        pool.record_dominance(1, np.array([1.0, 3.0]), np.array([2.0, 2.0]))
        assert pool._success.sum() == 0.0
        assert pool._fail.sum()    == 0.0

    def test_inf_post_better_sum(self):
        # post_sum < pre_sum → success
        pool = _pool()
        pool.record_dominance(0, np.array([1.0, np.inf]), np.array([np.inf, np.inf]))
        assert pool._success[0] == 1.0
        assert pool._fail[0]    == 0.0

    def test_inf_post_worse_sum(self):
        pool = _pool()
        pool.record_dominance(0, np.array([np.inf, np.inf]), np.array([1.0, np.inf]))
        assert pool._fail[0]    == 1.0
        assert pool._success[0] == 0.0

    def test_inf_equal_sums_records_nothing(self):
        pool = _pool()
        pool.record_dominance(0, np.array([np.inf, np.inf]), np.array([np.inf, np.inf]))
        assert pool._success.sum() == 0.0
        assert pool._fail.sum()    == 0.0


# ── _update (probability recomputation) ──────────────────────────────────────

class TestUpdate:

    def test_probs_sum_to_one(self):
        pool = _pool(update_period=1)
        pool._success = np.array([3.0, 1.0, 0.0])
        pool._fail    = np.array([1.0, 1.0, 0.0])
        pool._update()
        assert abs(pool.probs.sum() - 1.0) < 1e-9

    def test_higher_success_rate_gets_higher_prob(self):
        pool = _pool(update_period=1)
        pool._success = np.array([9.0, 1.0, 1.0])
        pool._fail    = np.array([1.0, 1.0, 1.0])
        pool._update()
        assert pool.probs[0] > pool.probs[1]
        assert pool.probs[0] > pool.probs[2]

    def test_floor_prevents_extinction(self):
        # Strategy with no history → floor to s=1, so prob stays nonzero.
        pool = _pool(update_period=1)
        pool._success = np.array([0.0, 0.0, 0.0])
        pool._fail    = np.array([0.0, 0.0, 0.0])
        pool._update()
        np.testing.assert_allclose(pool.probs, [1/3, 1/3, 1/3], atol=1e-9)

    def test_counters_reset_after_update(self):
        pool = _pool(update_period=2)
        pool.record_outcome(0, success=True)
        pool.record_outcome(1, success=False)
        pool.step()
        pool.step()  # triggers _update on second step
        assert pool._success.sum() == 0.0
        assert pool._fail.sum()    == 0.0

    def test_prob_history_recorded(self):
        pool = _pool(update_period=1)
        pool.step()
        assert len(pool.prob_history) == 1
        assert "probs" in pool.prob_history[0]
        assert "success" in pool.prob_history[0]
        assert "generation" in pool.prob_history[0]

    def test_all_probs_nonnegative(self):
        pool = _pool(update_period=1)
        pool._success = np.array([0.0, 10.0, 0.0])
        pool._fail    = np.array([10.0, 0.0, 0.0])
        pool._update()
        assert np.all(pool.probs >= 0.0)


# ── select_strategy (UCB1) ────────────────────────────────────────────────────

class TestSelectStrategy:

    def test_returns_valid_index(self):
        pool = _pool()
        rng  = _rng()
        for _ in range(50):
            idx = pool.select_strategy(rng)
            assert 0 <= idx < pool.n_strategies

    def test_cold_start_is_random(self):
        # With no history, should still return a valid strategy.
        pool = _pool()
        rng  = _rng(0)
        seen = {pool.select_strategy(rng) for _ in range(100)}
        assert len(seen) > 1, "Expected all strategies to be selected at cold start"

    def test_ucb1_favours_high_success(self):
        pool = _pool()
        rng  = _rng(0)
        # Equal pulls so exploration term is identical for all strategies.
        # Winner is determined purely by mean_reward.
        pool._success = np.array([9.0, 1.0, 1.0])
        pool._fail    = np.array([1.0, 9.0, 9.0])
        # pulls = [10, 10, 10] → exploration term cancels out
        idx = pool.select_strategy(rng)
        assert idx == 0, f"Expected strategy 0 to win UCB1, got {idx}"

    def test_ucb1_explores_unused_strategies(self):
        # Strategy 2 has never been used → UCB exploration term is large.
        pool = _pool()
        rng  = _rng(0)
        pool._success = np.array([10.0, 10.0, 0.0])
        pool._fail    = np.array([5.0,   5.0, 0.0])
        # pulls[2] == 0 → clamped to 1 → exploration = sqrt(log(total+1)/1)
        idx = pool.select_strategy(rng)
        assert idx == 2, "Unexplored strategy should win UCB1"
        
    def test_ucb1_explores_underused_strategies(self):
        pool = _pool()
        rng  = _rng(0)
        # Strategy 0 has great mean reward but has been pulled 50x.
        # Strategies 1 and 2 barely used → exploration bonus dominates.
        pool._success = np.array([50.0, 1.0, 1.0])
        pool._fail    = np.array([5.0,  5.0, 5.0])
        idx = pool.select_strategy(rng)
        assert idx in (1, 2), "UCB1 should explore underused strategies"    


# ── step() ────────────────────────────────────────────────────────────────────

class TestStep:

    def test_step_increments_generation(self):
        pool = _pool(update_period=5)
        pool.step()
        assert pool.generation == 1

    def test_step_returns_true_on_update_period(self):
        pool = _pool(update_period=3)
        assert pool.step() is False
        assert pool.step() is False
        assert pool.step() is True   # 3rd generation → update

    def test_step_returns_false_between_updates(self):
        pool = _pool(update_period=5)
        results = [pool.step() for _ in range(4)]
        assert results == [False, False, False, False]

    def test_multiple_update_cycles(self):
        pool = _pool(update_period=2)
        updates = sum(pool.step() for _ in range(10))
        assert updates == 5


# ── apply_strategy ────────────────────────────────────────────────────────────

class TestApplyStrategy:

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_each_strategy_modifies_vars(self, idx):
        pool = _pool()
        rng  = _rng()
        v    = np.array([1, 1, 1, 1, 1])
        old  = v.copy()
        pool.apply_strategy(v, idx, rng)
        assert not np.array_equal(v, old), f"Strategy {idx} should modify vars_"

    @pytest.mark.parametrize("idx", [0, 1, 2])
    def test_values_stay_in_range(self, idx):
        pool = _pool(n_choices=4)
        rng  = _rng()
        for _ in range(50):
            v = np.array([1, 1, 1, 1, 1])
            pool.apply_strategy(v, idx, rng)
            assert np.all(v >= 0) and np.all(v < 4)


# ── summary & diagnostics ─────────────────────────────────────────────────────

class TestDiagnostics:

    def test_summary_contains_all_strategy_names(self):
        pool = _pool()
        s = pool.summary
        assert "conservative" in s
        assert "uniform"      in s
        assert "aggressive"   in s

    def test_generation_property(self):
        pool = _pool()
        assert pool.generation == 0
        pool.step()
        pool.step()
        assert pool.generation == 2
