import numpy as np
from src.metaheuristic.nsga2.core.selection import tournament_selection
from src.metaheuristic.nsga2.core.individual import Individual

def _make_ind(decision_vars, objectives, rank, crowding_distance):
    ind = Individual(decision_vars=decision_vars, n_objectives=len(objectives))
    ind.objectives = np.array(objectives, dtype=float)
    ind.rank = rank
    ind.crowding_distance = crowding_distance
    return ind

def test_tournament_prefers_lower_rank():
    rng = np.random.RandomState(0)
    population = [
        _make_ind([0,0,0], [1.0, 5.0], rank=2, crowding_distance=0.1),
        _make_ind([1,1,1], [1.0, 5.0], rank=1, crowding_distance=0.1),
    ]
    winners = [tournament_selection(population, rng=rng) for _ in range(20)]
    assert all(w.rank == 1 for w in winners)

def test_tournament_returns_individual():
    rng = np.random.RandomState(1)
    population = [
        _make_ind([0,0,0], [1.0, 5.0], rank=1, crowding_distance=0.5),
        _make_ind([1,1,1], [2.0, 4.0], rank=2, crowding_distance=0.2),
    ]
    winner = tournament_selection(population, rng=rng)
    assert isinstance(winner, Individual)