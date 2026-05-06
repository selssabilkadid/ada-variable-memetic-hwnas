"""
src/metaheuristic/nsga2_with_archive.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

_DEFAULT_POP_SIZE       = 100
_DEFAULT_N_GEN          = 50
_DEFAULT_CROSSOVER_PROB = 0.9
_DEFAULT_MUTATION_PROB  = 0.1
_DEFAULT_TOURNAMENT_K   = 2
_SEARCH_SPACE_SIZE      = 15_625


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    n   = len(objectives)
    acc = objectives[:, 0]
    lat = objectives[:, 1]
    dom_count = np.zeros(n, dtype=int)
    dom_set   = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if acc[j] >= acc[i] and lat[j] <= lat[i] and (
                acc[j] > acc[i] or lat[j] < lat[i]
            ):
                dom_count[i] += 1
            elif acc[i] >= acc[j] and lat[i] <= lat[j] and (
                acc[i] > acc[j] or lat[i] < lat[j]
            ):
                dom_set[i].append(j)
    fronts: list[list[int]] = []
    current_front = [i for i in range(n) if dom_count[i] == 0]
    while current_front:
        fronts.append(current_front)
        next_front: list[int] = []
        for i in current_front:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        current_front = next_front
    return fronts


def crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    size      = len(front)
    distances = np.zeros(size)
    if size <= 2:
        distances[:] = np.inf
        return distances
    for m in range(objectives.shape[1]):
        vals   = objectives[front, m]
        order  = np.argsort(vals)
        f_min, f_max = vals[order[0]], vals[order[-1]]
        spread = (f_max - f_min) if f_max != f_min else 1e-8
        distances[order[0]]  = np.inf
        distances[order[-1]] = np.inf
        for k in range(1, size - 1):
            distances[order[k]] += (
                vals[order[k + 1]] - vals[order[k - 1]]
            ) / spread
    return distances


def tournament_select(
    pop:   list[int],
    ranks: np.ndarray,
    cd:    np.ndarray,
    rng:   np.random.RandomState,
    k:     int = 2,
) -> int:
    candidates = rng.choice(len(pop), k, replace=False)
    best = candidates[0]
    for c in candidates[1:]:
        if ranks[c] < ranks[best] or (
            ranks[c] == ranks[best] and cd[c] > cd[best]
        ):
            best = c
    return best


def crossover(
    p1:                int,
    p2:                int,
    rng:               np.random.RandomState,
    prob:              float = _DEFAULT_CROSSOVER_PROB,
    search_space_size: int   = _SEARCH_SPACE_SIZE,
) -> tuple[int, int]:
    if rng.rand() > prob:
        return p1, p2
    point = rng.randint(1, 14)
    mask  = (1 << point) - 1
    c1 = int(((p1 & ~mask) | (p2 & mask)) % search_space_size)
    c2 = int(((p2 & ~mask) | (p1 & mask)) % search_space_size)
    return c1, c2


def mutate(
    ind:               int,
    rng:               np.random.RandomState,
    prob:              float = _DEFAULT_MUTATION_PROB,
    search_space_size: int   = _SEARCH_SPACE_SIZE,
) -> int:
    if rng.rand() > prob:
        return ind
    return int(ind ^ (1 << rng.randint(0, 14))) % search_space_size


def _is_pareto_mask(acc_arr: np.ndarray, lat_arr: np.ndarray) -> np.ndarray:
    n    = len(acc_arr)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        dominated = (
            (acc_arr >= acc_arr[i]) & (lat_arr <= lat_arr[i])
            & ((acc_arr > acc_arr[i]) | (lat_arr < lat_arr[i]))
        )
        dominated[i] = False
        if dominated.any():
            mask[i] = False
    return mask


def compute_pareto_df(df: pd.DataFrame) -> pd.DataFrame:
    mask = _is_pareto_mask(df["accuracy"].values, df["latency"].values)
    return (
        df[mask]
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )


def run_nsga2_with_archive(
    eval_fn:            Callable[[int], tuple[float, float]],
    archive=None,
    pop_size:           int   = _DEFAULT_POP_SIZE,
    n_generations:      int   = _DEFAULT_N_GEN,
    crossover_prob:     float = _DEFAULT_CROSSOVER_PROB,
    mutation_prob:      float = _DEFAULT_MUTATION_PROB,
    tournament_k:       int   = _DEFAULT_TOURNAMENT_K,
    search_space_size:  int   = _SEARCH_SPACE_SIZE,
    seed:               int   = 42,
) -> tuple[list[int], list[dict]]:

    # ── Isolated local RNG — does NOT touch np.random global state ────────
    rng = np.random.RandomState(seed)

    population: list[int] = list(
        rng.choice(search_space_size, pop_size, replace=False)
    )
    history: list[dict] = []

    for gen in range(n_generations):

        obj = np.array([eval_fn(ind) for ind in population])

        fronts = fast_non_dominated_sort(obj)
        ranks  = np.empty(pop_size, dtype=int)
        cd     = np.zeros(pop_size)
        for rank_idx, front in enumerate(fronts):
            for i in front:
                ranks[i] = rank_idx
            cd_vals = crowding_distance(obj, front)
            for k, i in enumerate(front):
                cd[i] = cd_vals[k]

        offspring: list[int] = []
        while len(offspring) < pop_size:
            p1 = population[tournament_select(population, ranks, cd, rng, tournament_k)]
            p2 = population[tournament_select(population, ranks, cd, rng, tournament_k)]
            c1, c2 = crossover(p1, p2, rng, crossover_prob, search_space_size)
            offspring.append(mutate(c1, rng, mutation_prob, search_space_size))
            if len(offspring) < pop_size:
                offspring.append(mutate(c2, rng, mutation_prob, search_space_size))

        combined     = population + offspring
        obj_combined = np.array([eval_fn(ind) for ind in combined])
        fronts_c     = fast_non_dominated_sort(obj_combined)

        new_pop: list[int] = []
        for front in fronts_c:
            if len(new_pop) + len(front) <= pop_size:
                new_pop += [combined[i] for i in front]
            else:
                needed   = pop_size - len(new_pop)
                cd_front = crowding_distance(obj_combined, front)
                sorted_f = sorted(range(len(front)), key=lambda k: -cd_front[k])
                for k in sorted_f[:needed]:
                    new_pop.append(combined[front[k]])
                break

        population = new_pop

        # ── Update archive + inject back into population ───────────────────
        if archive is not None:
            archive.update(
                (combined[i], obj_combined[i, 0], obj_combined[i, 1])
                for i in range(len(combined))
            )

            pop_set    = set(population)
            archive_df = archive.to_dataframe()
            candidates = [
                int(row["arch_index"])
                for _, row in archive_df.iterrows()
                if int(row["arch_index"]) not in pop_set
            ]
            if candidates:
                n_inject = min(len(candidates), max(1, pop_size // 10))
                pop_obj_cur = np.array([eval_fn(ind) for ind in population])
                fronts_cur  = fast_non_dominated_sort(pop_obj_cur)
                ranks_cur   = np.empty(pop_size, dtype=int)
                cd_cur      = np.zeros(pop_size)
                for rank_idx, front in enumerate(fronts_cur):
                    for i in front:
                        ranks_cur[i] = rank_idx
                    cd_vals = crowding_distance(pop_obj_cur, front)
                    for k, i in enumerate(front):
                        cd_cur[i] = cd_vals[k]
                worst_order = sorted(
                    range(pop_size),
                    key=lambda i: (
                        -ranks_cur[i],
                        cd_cur[i] if np.isfinite(cd_cur[i]) else 0
                    )
                )
                for slot, arch_idx in zip(worst_order[:n_inject],
                                          candidates[:n_inject]):
                    population[slot] = arch_idx

        pop_obj = np.array([eval_fn(i) for i in population])
        pf_mask = _is_pareto_mask(pop_obj[:, 0], pop_obj[:, 1])
        history.append({
            "generation":        gen + 1,
            "pareto_front_size": int(pf_mask.sum()),
            "archive_size":      archive.size if archive is not None else 0,
        })

    return population, history