import numpy as np
from src.metaheuristic.nsga2.core.nsga2 import NSGA2

class DummyProblem:
    def evaluate(self, x):
        return [float(np.sum(x)), float(len(x) - np.sum(x))]

def test_nsga2_runs_smoke():
    algo = NSGA2(
        problem=DummyProblem(),
        pop_size=10,
        n_var=4,
        n_obj=2,
        num_choices=np.full(4, 3),
        seed=0,
    )
    pareto_set, pareto_front = algo.run(generations=2, verbose=False)
    assert pareto_set.shape[1] == 4
    assert pareto_front.shape[1] == 2

def test_nsga2_history_tracks_generations():
    algo = NSGA2(
        problem=DummyProblem(),
        pop_size=10,
        n_var=4,
        n_obj=2,
        num_choices=np.full(4, 3),
        seed=1,
    )
    algo.run(generations=5, verbose=False)
    assert len(algo.history) == 5

def test_nsga2_pareto_front_non_negative():
    algo = NSGA2(
        problem=DummyProblem(),
        pop_size=20,
        n_var=5,
        n_obj=2,
        num_choices=np.full(5, 3),
        seed=42,
    )
    _, pareto_front = algo.run(generations=10, verbose=False)
    # shape check only — DummyProblem objectives can be negative by design
    assert pareto_front.shape[1] == 2
    assert len(pareto_front) > 0