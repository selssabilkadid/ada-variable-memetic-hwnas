# experiments/run_adaptive_variable_memetic_nsga2.py
"""
Experiment runner for Adaptive Variance-Guided Memetic NSGA-II (Contribution 3).

Identical in structure to run_variable_memetic_nsga2.py (same datasets, same
devices, same metrics, same output layout), with one addition: UCB1 adaptive
operator selection is enabled by passing ``adaptive_mutation`` to run_memetic().

UCB1 selects among the three mutation operators at each generation based on
their historical reward (hypervolume improvement), trading off exploration
and exploitation via an upper-confidence-bound criterion.

Output layout
-------------
results/results_adaptive_variable_memetic_nsga2/
  <dataset>/
    <device>/
      adaptive_variable_memetic_nsga2_front.csv   # arch_index, accuracy, latency
      comparison.png                              # scatter + metrics bar (vs GT)
      edge_weights.csv                            # variance weights used

results/results_adaptive_variable_memetic_nsga2/ablation/
  <dataset>/
    <device>/
      ablation_K.csv               # K, hv_ratio, gd, igd for each K
      ablation_K.png               # line plot of metrics vs K
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader
from src.metaheuristic.variable_memetic_nsga2 import run_memetic
from src.evaluation.metrics import compute_metrics

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "data", "variance_weights")

# ------------------------------------------------------------------ #
# UCB1 adaptive mutation configuration
# ------------------------------------------------------------------ #

ADAPTIVE_MUTATION_CONFIG = {
    "initial_probs": np.array([0.33, 0.33, 0.34]),
    "update_period": 5,
}


# ------------------------------------------------------------------ #
# Ground-truth loader
# ------------------------------------------------------------------ #

def load_gt(dataset: str, device: str) -> pd.DataFrame | None:
    gt_path = os.path.join(
        BASE_DIR, "../ground_truth", dataset, device, "pareto_front.csv"
    )
    if os.path.isfile(gt_path):
        df = pd.read_csv(gt_path).sort_values("latency")
        print(f"[OK] Ground truth loaded ({len(df)} pts)")
        return df
    return None


# ------------------------------------------------------------------ #
# Bar chart helper — values on top of each bar
# ------------------------------------------------------------------ #

def _bar_with_values(ax, x, values, width, label, color=None, offset=0.0):
    """Draw a bar group and annotate each bar with its value on top."""
    bars = ax.bar(
        x + offset, values, width,
        label = label,
        color = color,
        alpha = 0.85,
    )
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha       = "center",
            va       = "bottom",
            fontsize = 7,
        )


# ------------------------------------------------------------------ #
# Single run
# ------------------------------------------------------------------ #

def run_single(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    K: int,
    output_dir: str,
) -> dict:
    """
    Run adaptive variance-guided memetic NSGA-II for one
    (dataset, device, K) combination.
    """

    results = run_memetic(
        nb201_loader      = nb201_loader,
        hw_loader         = hw_loader,
        dataset           = dataset,
        hw_platform       = device,
        pop_size          = 50,
        n_gen             = 20,
        K                 = K,
        seed              = 42,
        verbose           = True,
        weights_dir       = WEIGHTS_DIR,
        adaptive_mutation = ADAPTIVE_MUTATION_CONFIG,
    )

    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save Pareto front CSV
    # ------------------------------------------------------------------
    df_pareto = pd.DataFrame({
        "arch_index": results["pareto_X"],
        "accuracy":   results["pareto_acc"],
        "latency":    results["pareto_lat"],
    })
    df_pareto.to_csv(
        os.path.join(output_dir, "adaptive_variable_memetic_nsga2_front.csv"),
        index=False,
    )

    # ------------------------------------------------------------------
    # Save edge weights CSV (variance weights used)
    # ------------------------------------------------------------------
    w = results["edge_weights"]
    pd.DataFrame({"edge": range(len(w)), "variance_weight": w}).to_csv(
        os.path.join(output_dir, "edge_weights.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Ground truth + metrics
    # ------------------------------------------------------------------
    df_gt   = load_gt(dataset, device)
    metrics = {"hv_ratio": 0.0, "gd": 0.0, "igd": 0.0}

    if df_gt is not None:
        approx  = df_pareto[["accuracy", "latency"]].values
        true    = df_gt[["accuracy", "latency"]].values
        metrics = compute_metrics(approx, true)
        print(
            f"[OK] HV: {metrics['hv_ratio']:.2f}% | "
            f"GD: {metrics['gd']:.4f} | IGD: {metrics['igd']:.4f}"
        )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    df_all = pd.DataFrame({
        "latency":  results["F"][:, 1],
        "accuracy": -results["F"][:, 0],
    })
    df_pareto_sorted = df_pareto.sort_values("latency")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Adaptive Variance-Guided Memetic NSGA-II | {dataset} | {device} | K={K}"
    )

    # --- Left panel ---
    ax1.scatter(df_all["latency"], df_all["accuracy"],
                alpha=0.3, label="Population")
    ax1.plot(df_pareto_sorted["latency"], df_pareto_sorted["accuracy"],
             "r-s", label="Adaptive memetic front")
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"],
                 "k--o", label="True Pareto")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    # --- Right panel: bars with values on top ---
    if df_gt is not None:
        names  = ["HV (%)", "GD×100", "IGD×100"]
        values = [
            metrics["hv_ratio"],
            metrics["gd"]  * 100,
            metrics["igd"] * 100,
        ]
        x     = np.arange(len(names))
        width = 0.5
        _bar_with_values(ax2, x, values, width,
                         label="Adaptive Variance-Guided Memetic",
                         color="tab:orange", offset=0.0)
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

    print(f"[OK] Results saved → {output_dir}")
    return metrics


# ------------------------------------------------------------------ #
# Ablation over K
# ------------------------------------------------------------------ #

def run_ablation(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    K_values: list,
    ablation_dir: str,
):
    rows = []
    for K in K_values:
        print(f"\n--- Ablation K={K} | {dataset} | {device} ---")
        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            K          = K,
            output_dir = os.path.join(ablation_dir, f"K{K}"),
        )
        rows.append({"K": K, **metrics})

    df_abl = pd.DataFrame(rows)
    os.makedirs(ablation_dir, exist_ok=True)
    df_abl.to_csv(os.path.join(ablation_dir, "ablation_K.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: K budget | {dataset} | {device}")
    for ax, col, label in zip(
        axes,
        ["hv_ratio", "gd", "igd"],
        ["HV ratio (%)", "GD (normalised)", "IGD (normalised)"],
    ):
        ax.plot(df_abl["K"], df_abl[col], marker="o")
        # Annotate each point with its value
        for k_val, metric_val in zip(df_abl["K"], df_abl[col]):
            ax.annotate(
                f"{metric_val:.3f}",
                xy         = (k_val, metric_val),
                xytext     = (0, 6),
                textcoords = "offset points",
                ha         = "center",
                fontsize   = 8,
            )
        ax.set_xlabel("K")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(K_values)

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "ablation_K.png"))
    plt.close()
    print(f"[OK] Ablation saved → {ablation_dir}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(run_ablation_flag: bool = False):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    DATASETS   = ["cifar10", "cifar100", "ImageNet16-120"]
    DEVICES    = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]
    K_DEFAULT  = 4
    K_ABLATION = [2, 4, 6, 8]

    for dataset in DATASETS:
        for device in DEVICES:
            print(
                f"\n=== Adaptive Variance-Guided Memetic NSGA-II | "
                f"dataset={dataset} device={device} ==="
            )

            if run_ablation_flag:
                ablation_dir = os.path.join(
                    BASE_DIR,
                    "../results/results_adaptive_variable_memetic_nsga2/ablation",
                    dataset, device,
                )
                run_ablation(
                    nb201_loader, hw_loader,
                    dataset, device,
                    K_values     = K_ABLATION,
                    ablation_dir = ablation_dir,
                )
            else:
                output_dir = os.path.join(
                    BASE_DIR,
                    "../results/results_adaptive_variable_memetic_nsga2",
                    dataset, device,
                )
                metrics = run_single(
                    nb201_loader, hw_loader,
                    dataset, device,
                    K          = K_DEFAULT,
                    output_dir = output_dir,
                )
                print(f"\n=== SUMMARY ({dataset} | {device}) ===")
                print(f"HV:  {metrics['hv_ratio']:.2f}%")
                print(f"GD:  {metrics['gd']:.4f}")
                print(f"IGD: {metrics['igd']:.4f}")


if __name__ == "__main__":
    main(run_ablation_flag="--ablation" in sys.argv)