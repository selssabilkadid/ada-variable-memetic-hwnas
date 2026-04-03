# import os
# import pandas as pd
# import numpy as np

# from src.metaheuristic.regularised_evolution import RegularizedEvolutionTracker
# from experiments.pareto_analysis import compute_pareto_front, hv_contribution

# # Config
# DATASET = "cifar10"
# DEVICE = "edgegpu"
# K = 10

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../ground_truth")

# def main():
#     folder = os.path.join(GROUND_TRUTH_DIR, DATASET, DEVICE)
#     df = pd.read_csv(os.path.join(folder, "search_results.csv"))

#     # Compute Pareto front
#     df["error_rate"] = 1 - df["accuracy"]/100.0
#     pareto_indices = compute_pareto_front(df)

#     # Get "God architectures"
#     contributions = hv_contribution(df, pareto_indices)
#     top_k = np.argsort(-contributions)[:K]
#     god_archs = [pareto_indices[i] for i in top_k]

#     print("God architectures:", god_archs)

#     # Run RE
#     re = RegularizedEvolutionTracker(df, DEVICE)
#     tracker, hv_history = re.run(cycles=500, god_archs=god_archs)

#     print("\n=== Results ===")
#     for g, stats in tracker.items():
#         print(f"Arch {g}: rediscovery times = {stats['rediscovery_time']}")

# if __name__ == "__main__":
#     main()

"""
experiments/amnesia_analysis.py
================================
Amnesia Diagnostic Experiment for Regularized Evolution on HW-NAS-Bench.

Proves structural amnesia by showing:
1. God architectures (top-K HV contributors on ground-truth Pareto front)
2. RE discovers and evicts them blindly (HV drop at eviction)
3. Rediscovery is rare and costly (rediscovery gap in generations)
4. HV-over-time plot with eviction events marked visually

Usage:
    python -m experiments.amnesia_analysis
"""

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.benchmark.hwnas_loader import HWNASLoader
from src.metaheuristic.regularised_evolution import RegularizedEvolutionTracker
from experiments.pareto_analysis import compute_pareto_front, hv_contribution

# ── Config ────────────────────────────────────────────────────────────────────
DATASET        = "cifar10"
DEVICE         = "edgegpu"
K              = 10          # number of god architectures
CYCLES         = 3000        # generations (enough for amnesia to manifest)
POPULATION     = 50
SAMPLE_SIZE    = 10
LATENCY_COL    = "latency"   # column name in CSV

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../ground_truth")
OUTPUT_DIR      = os.path.join(BASE_DIR, "../outputs", "amnesia", DATASET, DEVICE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_df(dataset, device):
    """Load CSV and normalise column names."""
    folder = os.path.join(GROUND_TRUTH_DIR, dataset, device)
    df = pd.read_csv(os.path.join(folder, "search_results.csv"))

    # Normalise accuracy column
    if "accuracy" not in df.columns:
        for candidate in ["acc_cifar10", "acc_cifar100"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "accuracy"})
                break

    # Normalise latency column — try common names
    if LATENCY_COL not in df.columns:
        for candidate in [f"{device}_latency", "latency_ms", "lat"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "latency"})
                break

    if "latency" not in df.columns:
        raise ValueError(
            f"Cannot find latency column in CSV. Columns present: {df.columns.tolist()}"
        )

    df["error_rate"] = 1.0 - df["accuracy"] / 100.0
    return df


def get_god_archs(df, k):
    """Return arch_indices of top-K HV contributors on ground-truth Pareto front."""
    pareto_indices = compute_pareto_front(df)
    contributions  = hv_contribution(df, pareto_indices)
    top_k          = np.argsort(-contributions)[:k]
    god_archs      = [pareto_indices[i] for i in top_k]
    god_contrib    = contributions[top_k]
    return god_archs, god_contrib, pareto_indices


def print_summary(tracker, god_archs, god_contrib):
    print("\n" + "="*60)
    print("AMNESIA TRACKER SUMMARY")
    print("="*60)
    total_evictions  = 0
    total_hv_lost    = 0.0
    never_rediscovered = 0

    for g, contrib in zip(god_archs, god_contrib):
        t = tracker[g]
        n_evictions = len(t["eviction_gen"])
        total_evictions += n_evictions
        hv_lost = sum(t["hv_drop"])
        total_hv_lost += hv_lost
        gaps = t["rediscovery_gap"]

        if n_evictions > 0 and len(gaps) < n_evictions:
            never_rediscovered += 1

        print(f"\nArch {g:6d}  (HV contrib={contrib:.3f})")
        print(f"  First discovered : gen {t['first_discovery_gen']}")
        print(f"  Evictions        : {n_evictions}  at gens {t['eviction_gen']}")
        print(f"  HV drops         : {[f'{h:.4f}' for h in t['hv_drop']]}")
        print(f"  Rediscovery gaps : {gaps}"
              + (" ← NEVER REDISCOVERED" if n_evictions > 0 and not gaps else ""))

    print("\n" + "-"*60)
    print(f"Total god-arch evictions : {total_evictions}")
    print(f"Total HV lost to amnesia : {total_hv_lost:.4f}")
    print(f"Never rediscovered       : {never_rediscovered}/{len(god_archs)}")
    print("="*60 + "\n")


def save_results(tracker, hv_history, eviction_events, god_archs, god_contrib):
    """Persist results as JSON for reproducibility."""
    results = {
        "config": {
            "dataset": DATASET,
            "device": DEVICE,
            "cycles": CYCLES,
            "population": POPULATION,
            "k": K,
        },
        "god_archs": [
            {"arch_index": int(g), "hv_contribution": float(c)}
            for g, c in zip(god_archs, god_contrib)
        ],
        "tracker": {
            str(g): {
                "first_discovery_gen": t["first_discovery_gen"],
                "eviction_gen": t["eviction_gen"],
                "hv_drop": t["hv_drop"],
                "rediscovery_gap": t["rediscovery_gap"],
            }
            for g, t in tracker.items()
        },
        "eviction_events": eviction_events,
        "hv_history": hv_history,
    }
    path = os.path.join(OUTPUT_DIR, "amnesia_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[Saved] {path}")


def plot_hv_with_evictions(hv_history, eviction_events, tracker, god_archs):
    """
    HV-over-generations curve with vertical lines at eviction events.
    Each god arch gets a distinct colour.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(hv_history, color="steelblue", linewidth=1.2, label="Population HV")

    colours = plt.cm.tab10(np.linspace(0, 1, len(god_archs)))
    colour_map = {g: colours[i] for i, g in enumerate(god_archs)}

    for ev in eviction_events:
        g   = ev["arch_index"]
        gen = ev["gen"]
        ax.axvline(x=gen, color=colour_map[g], alpha=0.7, linewidth=1.2, linestyle="--")
        ax.annotate(
            f"−{ev['hv_drop']:.3f}",
            xy=(gen, hv_history[gen]),
            xytext=(gen + 10, hv_history[gen] + 0.01),
            fontsize=6,
            color=colour_map[g],
        )

    # Legend patches
    patches = [
        mpatches.Patch(color=colour_map[g], label=f"Arch {g} evicted")
        for g in god_archs
        if tracker[g]["eviction_gen"]
    ]
    patches.insert(0, mpatches.Patch(color="steelblue", label="Population HV"))
    ax.legend(handles=patches, fontsize=7, loc="lower right")

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title(
        f"Structural Amnesia: HV over generations with god-arch evictions\n"
        f"({DATASET} / {DEVICE}, pop={POPULATION}, cycles={CYCLES})"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "hv_amnesia_plot.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_eviction_hv_drops(eviction_events, god_archs, god_contrib):
    """Bar chart: HV drop per eviction event, coloured by arch."""
    if not eviction_events:
        print("[Plot] No eviction events to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    contrib_map = dict(zip(god_archs, god_contrib))
    colours     = plt.cm.tab10(np.linspace(0, 1, len(god_archs)))
    colour_map  = {g: colours[i] for i, g in enumerate(god_archs)}

    xs     = range(len(eviction_events))
    drops  = [ev["hv_drop"] for ev in eviction_events]
    colors = [colour_map[ev["arch_index"]] for ev in eviction_events]
    labels = [f"Gen {ev['gen']}\nArch {ev['arch_index']}" for ev in eviction_events]

    bars = ax.bar(xs, drops, color=colors)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("HV drop at eviction")
    ax.set_title("HV lost per god-arch eviction event")

    patches = [
        mpatches.Patch(
            color=colour_map[g],
            label=f"Arch {g} (contrib={contrib_map[g]:.3f})"
        )
        for g in god_archs if tracker_global.get(g, {}).get("eviction_gen")
    ]
    ax.legend(handles=patches, fontsize=7)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "hv_drops_per_eviction.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


def plot_rediscovery_gaps(tracker, god_archs):
    """Histogram of rediscovery gaps across all god archs."""
    all_gaps = []
    for g in god_archs:
        all_gaps.extend(tracker[g]["rediscovery_gap"])

    if not all_gaps:
        print("[Plot] No rediscoveries occurred — all evictions were permanent (worst case amnesia).")
        # Still save a text note
        path = os.path.join(OUTPUT_DIR, "rediscovery_note.txt")
        with open(path, "w") as f:
            f.write("No god architectures were rediscovered during the run.\n"
                    "All evictions were permanent — maximum amnesia severity.\n")
        print(f"[Saved] {path}")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_gaps, bins=20, color="tomato", edgecolor="white")
    ax.set_xlabel("Generations to rediscovery")
    ax.set_ylabel("Count")
    ax.set_title("Rediscovery gap distribution for god architectures")
    ax.axvline(np.mean(all_gaps), color="black", linestyle="--",
               label=f"Mean = {np.mean(all_gaps):.0f} gens")
    ax.legend()
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, "rediscovery_gaps.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[Saved] {path}")


# ── Global tracker reference for plotting helper ──────────────────────────────
tracker_global = {}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global tracker_global

    print(f"[Config] dataset={DATASET}, device={DEVICE}, "
          f"K={K}, cycles={CYCLES}, pop={POPULATION}")

    # 1. Load data
    print("\n[1] Loading data...")
    df = load_df(DATASET, DEVICE)
    print(f"    {len(df)} architectures loaded.")

    # 2. Identify god architectures from ground-truth Pareto front
    print("\n[2] Computing god architectures...")
    god_archs, god_contrib, pareto_indices = get_god_archs(df, K)
    print(f"    Pareto front size : {len(pareto_indices)}")
    print(f"    God archs         : {god_archs}")
    for g, c in zip(god_archs, god_contrib):
        print(f"      Arch {g:6d} — HV contribution = {c:.4f}")

    # 3. Run RE with amnesia tracker
    print(f"\n[3] Running RE ({CYCLES} cycles)...")
    loader = HWNASLoader()
    re = RegularizedEvolutionTracker(
        df, loader, DEVICE,
        population_size=POPULATION,
        sample_size=SAMPLE_SIZE
    )
    # tracker, hv_history, eviction_events = re.run(
    #     cycles=CYCLES,
    #     god_archs=god_archs
    # )
    # tracker_global = tracker

    # 4. Print summary
    # print_summary(tracker, god_archs, god_contrib)
    
    # Wrap main RE run in a loop
    SEEDS = 5
    discoveries = {g: 0 for g in god_archs}
    for seed in range(SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        tracker, hv_history, eviction_events = re.run(cycles=CYCLES, god_archs=god_archs)
        for g in god_archs:
            if tracker[g]["first_discovery_gen"] is not None:
                discoveries[g] += 1
        print_summary(tracker, god_archs, god_contrib)
    print({g: f"{discoveries[g]}/{SEEDS} runs" for g in god_archs})

    # # 5. Save results
    # print("[5] Saving results...")
    # save_results(tracker, hv_history, eviction_events, god_archs, god_contrib)

    # # 6. Plots
    # print("[6] Generating plots...")
    # plot_hv_with_evictions(hv_history, eviction_events, tracker, god_archs)
    # plot_eviction_hv_drops(eviction_events, god_archs, god_contrib)
    # plot_rediscovery_gaps(tracker, god_archs)

    # print(f"\n[Done] All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()