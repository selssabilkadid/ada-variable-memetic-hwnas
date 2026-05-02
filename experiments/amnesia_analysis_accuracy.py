"""
experiments/amnesia_analysis_accuracy.py
========================================

Accuracy-only Structural Amnesia Experiment compatible with E/O framework.

Key idea:
- Use ONLY accuracy (no latency, no HV)
- Initialize population with optimal architectures
- Track when they are evicted
- Show accuracy drop when they die
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.benchmark.hwnas_loader import HWNASLoader

# ── Config ─────────────────────────────────────────

DATASET   = "cifar10"
DEVICES   = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]

POPULATION  = 50
SAMPLE_SIZE = 10
CYCLES      = 3000
K_OPTIMAL   = 10
SEEDS       = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../ground_truth")
OUTPUT_DIR       = os.path.join(BASE_DIR, "../outputs", "amnesia_accuracy", DATASET)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────

def load_df(dataset, device):
    folder = os.path.join(GROUND_TRUTH_DIR, dataset, device)
    df = pd.read_csv(os.path.join(folder, "search_results.csv"))

    if "accuracy" not in df.columns:
        for candidate in ["acc_cifar10", "acc_cifar100"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "accuracy"})
                break

    return df

def get_optimal_archs(df, k):
    sorted_df = df.sort_values("accuracy", ascending=False)
    return sorted_df.head(k)["arch_index"].tolist()

def plot_accuracy_with_evictions(acc_history, eviction_events, optimal_archs, device, seed=None):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(acc_history, linewidth=1.5, label="Best Accuracy")

    colors = plt.cm.tab10(np.linspace(0, 1, len(optimal_archs)))
    color_map = {g: colors[i] for i, g in enumerate(optimal_archs)}

    for ev in eviction_events:
        g   = ev["arch_index"]
        gen = ev["gen"]

        ax.axvline(x=gen, color=color_map[g], linestyle="--", alpha=0.7)
        ax.annotate(f"-{ev['acc_drop']:.2f}", xy=(gen, acc_history[gen]), fontsize=7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Population Accuracy")
    title = f"Structural Amnesia (Accuracy-only RE)\n{DATASET} / {device}"
    if seed is not None:
        title += f" / seed {seed}"
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()

    device_dir = os.path.join(OUTPUT_DIR, device)
    os.makedirs(device_dir, exist_ok=True)

    filename = f"accuracy_amnesia_seed_{seed}.png" if seed is not None else f"accuracy_amnesia_{device}.png"
    path = os.path.join(device_dir, filename)

    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved plot] {path}")

def run_accuracy_amnesia(df, device, optimal_archs, seed=None):
    """
    Run accuracy-only RE experiment and return results
    """
    random.seed(seed)
    np.random.seed(seed)

    lookup = {int(row["arch_index"]): row for _, row in df.iterrows()}

    all_indices = list(lookup.keys())
    population = list(optimal_archs)  # biased init
    while len(population) < POPULATION:
        population.append(random.choice(all_indices))

    alive = {g: True for g in optimal_archs}
    eviction_events = []
    acc_history = []

    def accuracy(idx):
        return lookup[idx]["accuracy"]
     
    best_ever = [] 
    for gen in range(CYCLES):
        best_acc = max(accuracy(i) for i in population)
        acc_history.append(best_acc)
        
        if gen == 0:
            best_ever.append(best_acc)
        else:
            best_ever.append(max(best_ever[-1], best_acc))    

        # Parent selection
        sample = random.sample(population, SAMPLE_SIZE)
        parent = max(sample, key=accuracy)

        # Mutation: random child
        child = random.choice(all_indices)
        population.append(child)

        # Aging
        acc_before = max(accuracy(i) for i in population)
        removed = population.pop(0)
        acc_after = max(accuracy(i) for i in population)
        acc_drop = acc_before - acc_after

        if removed in optimal_archs:
            alive[removed] = False
            eviction_events.append({"gen": gen, "arch_index": removed, "acc_drop": acc_drop})

    return acc_history, eviction_events, best_ever

# ──────────────────────────────────────────────────
# E/O-compatible API
# ──────────────────────────────────────────────────
# Add this function somewhere after imports in amnesia_analysis_accuracy.py

def plot_amnesia_dual_curve(
    best_alive,
    best_ever,
    eviction_events,
    optimal_archs,
    save_path
):
    """
    Plot structural amnesia with two curves:
    - Best Alive in population
    - Best Ever Seen so far
    Shows eviction events and accuracy drops.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Best Alive
    ax.plot(best_alive, label="Best Alive", linewidth=2)

    # Best Ever
    ax.plot(best_ever, label="Best Ever Seen", linestyle="--", linewidth=2)

    colors = plt.cm.tab10(np.linspace(0, 1, len(optimal_archs)))
    color_map = {g: colors[i] for i, g in enumerate(optimal_archs)}

    # Eviction lines
    for ev in eviction_events:
        g = ev["arch_index"]
        gen = ev["gen"]

        ax.axvline(x=gen, color=color_map[g], linestyle="--", alpha=0.6)
        # Adjusted to prevent negative index at gen=0
        y_val = best_alive[gen-1] if gen > 0 else best_alive[0]
        ax.annotate(f"-{ev['acc_drop']:.2f}", xy=(gen, y_val), fontsize=7)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title("Structural Amnesia: Best Alive vs Best Ever")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print("[Saved]", save_path)
    
def run_experiment(dataset=DATASET, devices=DEVICES, seeds=SEEDS, k_optimal=K_OPTIMAL, plot=True):
    results = {}

    loader = HWNASLoader()  # placeholder if needed in other E/O experiments

    for device in devices:
        df = load_df(dataset, device)
        optimal_archs = get_optimal_archs(df, k_optimal)
        print(f"[{device}] {len(df)} architectures loaded, optimal: {optimal_archs}")

        device_results = []

        for seed in range(seeds):
            acc_history, eviction_events, best_ever = run_accuracy_amnesia(df, device, optimal_archs, seed=seed)
            if plot:
                # Single-curve plot
                plot_accuracy_with_evictions(acc_history, eviction_events, optimal_archs, device, seed=seed)

                # Dual-curve plot
                save_path = os.path.join(OUTPUT_DIR, device, f"dual_curve_seed_{seed}.png")
                plot_amnesia_dual_curve(
                    best_alive=acc_history,
                    best_ever=best_ever,
                    eviction_events=eviction_events,
                    optimal_archs=optimal_archs,
                    save_path=save_path
                )

        results[device] = device_results

    return results

# ──────────────────────────────────────────────────
# Legacy standalone run
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    run_experiment()