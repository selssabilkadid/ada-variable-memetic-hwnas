import numpy as np
from src.metaheuristic.nsga2.core.mutation import discrete_mutation
from src.metaheuristic.nsga2.core.crossover import uniform_crossover

def test_discrete_mutation_with_prob_1():
    rng = np.random.RandomState(0)
    x = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    num_choices = np.array([5, 5, 5, 5, 5, 5])
    discrete_mutation(x, prob_mutation=1.0, num_choices=num_choices, rng=rng)
    # with prob=1.0 every gene is replaced — result is still in valid range
    assert all(0 <= v < 5 for v in x)

def test_uniform_crossover_output_shape():
    rng = np.random.RandomState(1)
    p1 = np.array([0, 0, 0, 0, 0, 0], dtype=np.int32)
    p2 = np.array([4, 4, 4, 4, 4, 4], dtype=np.int32)
    c1, c2 = uniform_crossover(p1, p2, prob_crossover=0.9, rng=rng)
    assert len(c1) == len(p1)
    assert len(c2) == len(p2)
    assert set(c1.tolist()).issubset({0, 4})
    assert set(c2.tolist()).issubset({0, 4})