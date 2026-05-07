from __future__ import annotations
from typing import List
import numpy as np
from .individual import Individual


def _dominates(obj_a: np.ndarray, obj_b: np.ndarray) -> bool:
    """Standard Pareto dominance"""
    return bool(np.all(obj_a <= obj_b) and np.any(obj_a < obj_b))


def fast_non_dominated_sort(population: List[Individual]) -> List[List[Individual]]:
    """
    Partition population into nondomination fronts.

    Args:
        population: List of evaluated Individual objects.

    Returns:
        A list of fronts, where each front is a list of Individuals, 
        and Fronts[0] is the first Pareto front.
    """
    n = len(population)
    if n == 0:
        return []

    # Domination count for each individual
    domination_count = np.zeros(n, dtype=int)
    # Set of indices each individual dominates
    dominated_sets: List[List[int]] = [[] for _ in range(n)]

    # Count pairwise dominations 
    for i in range(n):
        for j in range(i + 1, n):
            # Compare objectives directly
            if _dominates(population[i].objectives, population[j].objectives):
                domination_count[j] += 1
                dominated_sets[i].append(j)
            elif _dominates(population[j].objectives, population[i].objectives):
                domination_count[i] += 1
                dominated_sets[j].append(i)

    # Initialize fronts list
    fronts: List[List[Individual]] = []

    # First front: all individuals with domination_count == 0
    front_indices = [i for i in range(n) if domination_count[i] == 0]
    front = [population[i] for i in front_indices]
    for ind in front:
        ind.rank = 0
    fronts.append(front)

    # Build subsequent fronts
    while front_indices:
        next_front_indices: List[int] = []
        for i in front_indices:
            for j in dominated_sets[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front_indices.append(j)
        if not next_front_indices:
            break
        front_indices = next_front_indices
        front = [population[i] for i in front_indices]
        for ind in front:
            ind.rank = len(fronts)
        fronts.append(front)

    return fronts
