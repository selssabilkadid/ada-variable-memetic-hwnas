from __future__ import annotations
import numpy as np


def uniform_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    prob_crossover: float = 0.9,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Uniform Crossover on two parent vectors.

    Args:
        parent1: Decision variables of parent 1.
        parent2: Decision variables of parent 2.
        prob_crossover: Probability that crossover occurs.
        rng: Optional RandomState for reproducibility.

    Returns:
        Tuple (child1, child2) with the same shape as the parents.
    """
    if rng is None:
        rng = np.random

    # Start as copies of parents
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Decide if crossover happens
    if rng.rand() >= prob_crossover:
        return child1, child2

    # Iterate through every gene 
    for i in range(len(parent1)):
        # 50% chance to swap this specific gene
        if rng.rand() < 0.5:
            child1[i] = parent2[i]
            child2[i] = parent1[i]

    return child1, child2