# runners/run_ablation_pop_gen.py
"""
2-D ablation: pop_size × n_gen for Adaptive Variance-Guided Memetic NSGA-II.

For every (dataset, device) pair it sweeps a configurable grid of pop_size
and n_gen values, records HV ratio, GD, and IGD, then produces:

  - Per-cell CSV + scatter plot  (same layout as run_single)
  - Per-(dataset, device) summary table  → ablation_pop_gen.csv
  - Per-(dataset, device) heatmaps       → ablation_pop_gen_hv.png
                                            ablation_pop_gen_gd.png
                                            ablation_pop_gen_igd.png

Output root
-----------
results/results_ablation_pop_gen/
  <dataset>/
    <device>/
      pop<P>_gen<G>/
        adaptive_variable_memetic_nsga2_front.csv
        comparison.png
        edge_weights.csv
      ablation_pop_gen.csv
      ablation_pop_gen_hv.png
      ablation_pop_gen_gd.png
      ablation_pop_gen_igd.png
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader
from src.metaheuristic.variable_memetic_nsga2 import run_memetic
from src.evaluation.metrics import compute_metrics

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "variance_weights")

# ------------------------------------------------------------------ #
# Grid to sweep — edit these lists to change the search space
# ------------------------------------------------------------------ #

POP_SIZES = [20, 50, 75, 100]        # population sizes to test
N_GENS    = [20, 30, 40, 50]          # number of generations to test
K_DEFAULT = 4                          # kept fixed during this ablation

DATASETS  = ["cifar10", "cifar100", "ImageNet16-120"]
DEVICES   = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]

ADAPTIVE_MUTATION_CONFIG = {
    "initial_probs": np.array([0.33, 0.33, 0.34]),
    "update_period": 5,
}


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def load_gt(dataset: str, device: str) -> pd.DataFrame | None:
    gt_path = os.path.join(
        BASE_DIR, "../ground_truth", dataset, device, "pareto_front.csv"
    )
    if os.path.isfile(gt_path):
        df = pd.read_csv(gt_path).sort_values("latency")
        print(f"    [GT] {len(df)} pts loaded")
        return df
    return None


def _bar_with_values(ax, x, values, width, label, color=None, offset=0.0):
    bars = ax.bar(x + offset, values, width, label=label, color=color, alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center", va="bottom", fontsize=7,
        )


# ------------------------------------------------------------------ #
# Single (pop_size, n_gen) run
# ------------------------------------------------------------------ #

def run_single(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    pop_size: int,
    n_gen: int,
    output_dir: str,
) -> dict:
    """Run one cell of the grid and return its metrics dict."""

    results = run_memetic(
        nb201_loader      = nb201_loader,
        hw_loader         = hw_loader,
        dataset           = dataset,
        hw_platform       = device,
        pop_size          = pop_size,
        n_gen             = n_gen,
        K                 = K_DEFAULT,
        seed              = 42,
        verbose           = False,          # keep console tidy during sweeps
        weights_dir       = WEIGHTS_DIR,
        adaptive_mutation = ADAPTIVE_MUTATION_CONFIG,
    )

    os.makedirs(output_dir, exist_ok=True)

    # --- Pareto front CSV ---
    df_pareto = pd.DataFrame({
        "arch_index": results["pareto_X"],
        "accuracy":   results["pareto_acc"],
        "latency":    results["pareto_lat"],
    })
    df_pareto.to_csv(
        os.path.join(output_dir, "adaptive_variable_memetic_nsga2_front.csv"),
        index=False,
    )

    # --- Edge weights CSV ---
    w = results["edge_weights"]
    pd.DataFrame({"edge": range(len(w)), "variance_weight": w}).to_csv(
        os.path.join(output_dir, "edge_weights.csv"), index=False
    )

    # --- Metrics ---
    df_gt   = load_gt(dataset, device)
    metrics = {"hv_ratio": 0.0, "gd": 0.0, "igd": 0.0}

    if df_gt is not None:
        approx  = df_pareto[["accuracy", "latency"]].values
        true    = df_gt[["accuracy", "latency"]].values
        metrics = compute_metrics(approx, true)

    # --- Scatter + bar plot ---
    df_all = pd.DataFrame({
        "latency":  results["F"][:, 1],
        "accuracy": -results["F"][:, 0],
    })
    df_pareto_sorted = df_pareto.sort_values("latency")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Adaptive Memetic NSGA-II | {dataset} | {device} "
        f"| pop={pop_size} gen={n_gen}"
    )

    ax1.scatter(df_all["latency"], df_all["accuracy"], alpha=0.3, label="Population")
    ax1.plot(
        df_pareto_sorted["latency"], df_pareto_sorted["accuracy"],
        "r-s", label="Adaptive memetic front",
    )
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"], "k--o", label="True Pareto")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    if df_gt is not None:
        names  = ["HV (%)", "GD×100", "IGD×100"]
        values = [metrics["hv_ratio"], metrics["gd"] * 100, metrics["igd"] * 100]
        x      = np.arange(len(names))
        _bar_with_values(ax2, x, values, 0.5,
                         label="Adaptive Variance-Guided Memetic",
                         color="tab:orange")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.axhline(100, linestyle="--", color="black", linewidth=0.8)
        ax2.set_title("Metrics vs Ground Truth")
        ax2.legend(fontsize=8)
    else:
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

    return metrics


# ------------------------------------------------------------------ #
# Heatmap helper
# ------------------------------------------------------------------ #

def _save_heatmap(
    matrix: np.ndarray,
    row_labels,
    col_labels,
    row_name: str,
    col_name: str,
    title: str,
    filepath: str,
    cmap: str = "viridis",
    fmt: str = ".2f",
):
    fig, ax = plt.subplots(figsize=(len(col_labels) * 1.6 + 1, len(row_labels) * 1.2 + 1))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    ax.set_xlabel(col_name)
    ax.set_ylabel(row_name)
    ax.set_title(title)

    # Annotate each cell
    for (i, j), val in np.ndenumerate(matrix):
        text_color = "white" if val < (matrix.max() + matrix.min()) / 2 else "black"
        ax.text(j, i, format(val, fmt), ha="center", va="center",
                fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"    [OK] Heatmap saved → {filepath}")


# ------------------------------------------------------------------ #
# Full 2-D ablation for one (dataset, device)
# ------------------------------------------------------------------ #

def run_ablation_pop_gen(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    pop_sizes: list,
    n_gens: list,
    out_root: str,
):
    rows = []

    for pop_size, n_gen in itertools.product(pop_sizes, n_gens):
        cell_dir = os.path.join(out_root, f"pop{pop_size}_gen{n_gen}")
        print(
            f"  → pop_size={pop_size:>4d}  n_gen={n_gen:>4d} "
            f"| {dataset} | {device}"
        )

        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            pop_size   = pop_size,
            n_gen      = n_gen,
            output_dir = cell_dir,
        )
        rows.append({
            "pop_size": pop_size,
            "n_gen":    n_gen,
            **metrics,
        })
        print(
            f"    HV={metrics['hv_ratio']:.2f}%  "
            f"GD={metrics['gd']:.4f}  IGD={metrics['igd']:.4f}"
        )

    # --- Summary CSV ---
    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_root, "ablation_pop_gen.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Summary CSV → {csv_path}")

    # --- Heatmaps (rows = pop_size, cols = n_gen) ---
    for metric, cmap, label in [
        ("hv_ratio", "YlGn",    "HV ratio (%)"),
        ("gd",       "YlOrRd",  "GD (normalised)"),
        ("igd",      "YlOrRd",  "IGD (normalised)"),
    ]:
        matrix = (
            df.pivot(index="pop_size", columns="n_gen", values=metric)
              .reindex(index=pop_sizes, columns=n_gens)
              .values.astype(float)
        )
        _save_heatmap(
            matrix,
            row_labels = pop_sizes,
            col_labels = n_gens,
            row_name   = "pop_size",
            col_name   = "n_gen",
            title      = f"{label} | {dataset} | {device}",
            filepath   = os.path.join(out_root, f"ablation_pop_gen_{metric}.png"),
            cmap       = cmap,
        )

    return df


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    print("[INFO] Loading benchmark loaders ...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    for dataset in DATASETS:
        for device in DEVICES:
            print(f"\n{'='*60}")
            print(f"  Ablation: pop_size × n_gen | {dataset} | {device}")
            print(f"{'='*60}")

            out_root = os.path.join(
                BASE_DIR,
                "../results/results_ablation_pop_gen",
                dataset, device,
            )
            os.makedirs(out_root, exist_ok=True)

            df = run_ablation_pop_gen(
                nb201_loader, hw_loader,
                dataset, device,
                pop_sizes = POP_SIZES,
                n_gens    = N_GENS,
                out_root  = out_root,
            )

            # Print best configuration per metric
            print("\n  Best configurations found:")
            best_hv  = df.loc[df["hv_ratio"].idxmax()]
            best_gd  = df.loc[df["gd"].idxmin()]
            best_igd = df.loc[df["igd"].idxmin()]
            print(
                f"    Best HV  → pop={int(best_hv['pop_size'])} "
                f"gen={int(best_hv['n_gen'])}  "
                f"HV={best_hv['hv_ratio']:.2f}%"
            )
            print(
                f"    Best GD  → pop={int(best_gd['pop_size'])} "
                f"gen={int(best_gd['n_gen'])}  "
                f"GD={best_gd['gd']:.4f}"
            )
            print(
                f"    Best IGD → pop={int(best_igd['pop_size'])} "
                f"gen={int(best_igd['n_gen'])}  "
                f"IGD={best_igd['igd']:.4f}"
            )


if __name__ == "__main__":
    main()