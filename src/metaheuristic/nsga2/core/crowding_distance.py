from __future__ import annotations
from typing import List
import numpy as np
from .individual import Individual


def crowding_distance_assignment(front: List[Individual]) -> None:
    """
    Assign crowding distance to every individual in front.

    Args:
        front: List of evaluated Individuals belonging to the same nondominated front.
    """
    n = len(front)
    if n <= 2:
        # In a front with 0-2 solutions, all are boundary solutions with infinite distance
        for ind in front:
            ind.crowding_distance = float("inf")
        return

    n_obj = len(front[0].objectives)

    # Reset distances
    for ind in front:
        ind.crowding_distance = 0.0

    # Accumulate crowding distance per objective
    for m in range(n_obj):
        # Sort by objective m
        front.sort(key=lambda ind: ind.objectives[m])

        # Boundary solutions get infinite distance
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        # Objective range for normalisation
        obj_min = front[0].objectives[m]
        obj_max = front[-1].objectives[m]
        obj_range = obj_max - obj_min

        # If all values are identical, skip this objective 
        if obj_range < 1e-12:
            continue

        # Accumulate normalised gap for intermediate solutions
        for i in range(1, n - 1):
            if front[i].crowding_distance == float("inf"):
                # Already a boundary in another objective, keep inf
                continue
            gap = front[i + 1].objectives[m] - front[i - 1].objectives[m]
            front[i].crowding_distance += gap / obj_range
