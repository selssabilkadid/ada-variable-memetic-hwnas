from __future__ import annotations
import numpy as np
from typing import List
from .individual import Individual


def tournament_selection(
    population: List[Individual], 
    rng: np.random.RandomState | None = None
) -> Individual:
    """Select one individual via binary tournament using crowded-comparison.

    Args:
        population: The current population.
        rng: Optional RandomState for reproducibility.

    Returns:
        The winning Individual.
    """
    if rng is None:
        rng = np.random

    # Pick 2 unique random indices
    indices = rng.choice(len(population), 2, replace=False)
    a, b = population[indices[0]], population[indices[1]]

    # 1. Prefer lower (better) rank
    if a.rank < b.rank:
        return a
    if b.rank < a.rank:
        return b

    # 2. If ranks equal, prefer larger crowding distance
    if a.crowding_distance > b.crowding_distance:
        return a
    
    return b
