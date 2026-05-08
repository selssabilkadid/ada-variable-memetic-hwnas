import numpy as np
from src.metaheuristic.nsga2.core.individual import Individual

def test_individual_creation():
    ind = Individual(decision_vars=[1, 2, 3], n_objectives=2)
    assert list(ind.decision_vars) == [1, 2, 3]
    assert np.all(ind.objectives == np.inf)  # pre-allocated to inf, not None

def test_individual_objectives_assignment():
    ind = Individual(decision_vars=[0, 0, 0], n_objectives=2)
    ind.objectives = np.array([1.0, 2.0])
    assert len(ind.objectives) == 2
    assert ind.objectives[0] == 1.0

def test_individual_copy_is_independent():
    ind = Individual(decision_vars=[1, 2], n_objectives=2)
    ind.objectives = np.array([3.0, 4.0])
    clone = ind.copy()
    clone.objectives[0] = 99.0
    assert ind.objectives[0] == 3.0

def test_individual_evaluate():
    class DummyProblem:
        def evaluate(self, x):
            return [float(x[0]), float(x[1])]
    ind = Individual(decision_vars=[3, 7], n_objectives=2)
    ind.evaluate(DummyProblem())
    assert ind.objectives[0] == 3.0
    assert ind.objectives[1] == 7.0