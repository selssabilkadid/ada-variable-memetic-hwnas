import numpy as np


def is_dominated(a, b):
    """
    Returns True if solution a is dominated by solution b.
    Assumes minimization on both objectives.
    For accuracy: pass -accuracy so higher is better becomes lower is better.
    """
    return all(b[i] <= a[i] for i in range(len(a))) and \
           any(b[i] < a[i] for i in range(len(a)))


def compute_pareto_front(solutions: list) -> list:
    """
    solutions: list of dicts with keys 'accuracy', 'latency', 'arch_index'
    Returns the Pareto-optimal subset (maximize accuracy, minimize latency).
    """
    pareto = []
    for s in solutions:
        dominated = False
        for other in solutions:
            if other is s:
                continue
            # other dominates s if: other.acc >= s.acc AND other.lat <= s.lat
            # with at least one strict
            if (other["accuracy"] >= s["accuracy"] and
                    other["latency"] <= s["latency"] and
                    (other["accuracy"] > s["accuracy"] or
                     other["latency"] < s["latency"])):
                dominated = True
                break
        if not dominated:
            pareto.append(s)
    return pareto


def hypervolume_2d(pareto_front: list, reference_point: tuple) -> float:
    """
    Computes 2D hypervolume indicator.
    reference_point: (worst_accuracy, worst_latency) — e.g., (0.0, max_latency)
    Pareto front should be sorted by latency ascending.
    
    Assumes: maximize accuracy (y-axis), minimize latency (x-axis).
    reference_point = (ref_latency, ref_accuracy) where ref is the worst case.
    """
    front = sorted(pareto_front, key=lambda x: x["latency"])
    ref_lat, ref_acc = reference_point

    hv = 0.0
    prev_lat = ref_lat
    for point in reversed(front):
        width = prev_lat - point["latency"]
        height = point["accuracy"] - ref_acc
        if width > 0 and height > 0:
            hv += width * height
        prev_lat = point["latency"]
    return hv