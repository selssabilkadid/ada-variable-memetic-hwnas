import numpy as np
from src.metaheuristic.nsga2.core.nondominated_sort import fast_non_dominated_sort
from src.metaheuristic.nsga2.core.crowding_distance import crowding_distance_assignment
from src.metaheuristic.nsga2.core.individual import Individual

def _make(obj):
    ind = Individual(decision_vars=[0], n_objectives=len(obj))
    ind.objectives = np.array(obj, dtype=float)
    return ind

def test_dominated_individual_is_in_later_front():
    a = _make([1.0, 1.0])  # dominates b on both
    b = _make([2.0, 2.0])
    fronts = fast_non_dominated_sort([a, b])
    assert a in fronts[0]
    assert b in fronts[1]

def test_nondominated_set_all_in_front_zero():
    inds = [_make([1.0, 4.0]), _make([2.0, 3.0]), _make([3.0, 2.0]), _make([4.0, 1.0])]
    fronts = fast_non_dominated_sort(inds)
    assert len(fronts) == 1
    assert len(fronts[0]) == 4

def test_crowding_extremes_are_inf():
    inds = [_make([1.0, 4.0]), _make([2.0, 3.0]), _make([3.0, 2.0]), _make([4.0, 1.0])]
    crowding_distance_assignment(inds)
    assert inds[0].crowding_distance == np.inf
    assert inds[-1].crowding_distance == np.inf