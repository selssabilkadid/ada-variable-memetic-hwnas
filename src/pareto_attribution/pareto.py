import numpy as np

def is_dominated(a, b):
    return np.all(b <= a) and np.any(b < a)

def get_pareto_mask(Y):
    n = len(Y)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and is_dominated(Y[i], Y[j]):
                mask[i] = False
                break
    return mask

def compute_distance_to_front(Y, pareto_Y):
    dists = []
    for y in Y:
        d = np.min(np.linalg.norm(pareto_Y - y, axis=1))
        dists.append(d)
    return np.array(dists)