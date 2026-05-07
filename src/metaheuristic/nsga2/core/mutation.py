from __future__ import annotations
import numpy as np

def discrete_mutation(
    individual: np.ndarray,
    prob_mutation: float,
    num_choices: np.ndarray | int,
    rng: np.random.RandomState | None = None,
) -> np.ndarray:
    """
    Discrete mutation operation for NSGA-II.

    Args:
        individual: Decision variables.
        prob_mutation: Per-gene mutation probability.
        num_choices: number of valid options for each gene.
        rng: Optional RandomState for reproducibility.

    Returns:
        The mutated individual.
    """
    if rng is None:
        rng = np.random

    n = len(individual)

    # Normalise num_choices to an array if it's a scalar
    if np.isscalar(num_choices):
        choices_per_gene = np.full(n, int(num_choices), dtype=int)
    else:
        choices_per_gene = np.asarray(num_choices, dtype=int)

    for i in range(n):
        if rng.rand() >= prob_mutation:
            continue

        k = choices_per_gene[i]
        if k <= 1:
            continue 

        # Sample a new value that is different from the current one
        current = individual[i]
        candidates = [v for v in range(k) if v != current]
        individual[i] = rng.choice(candidates)

    return individual