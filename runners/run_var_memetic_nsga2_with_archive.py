# runners/run_var_memetic_nsga2_with_archive.py
"""
Experiment runner for Online Variance-Guided Memetic NSGA-II with Archive.

Extends run_variable_memetic_nsga2.py by replacing offline precomputed
variance weights with online weights recomputed every generation from
the current archive members' decision variables.

Two archive variants available via --archive flag:
    fh       — Fixed-Hypergrid Archive (FHArchive)
    crowding — Bounded crowding-distance archive (CrowdingArchive)  [default]

Adaptive mutation (UCB1) can be toggled via --adaptive.


Output layout
-------------
results/results_variable_memetic_nsga2_archive/
  <dataset>/
    <device>/
      <archive_type>/                          # fh | crowding
        variable_memetic_archive_front.csv     # arch_index, accuracy, latency
        archive_members.csv                    # full archive at end of run
        edge_weights.csv                       # final online gene weights
        history.csv                            # gen, pareto_front_size,
                                               #   archive_size, gene_probs
        comparison.png                         # scatter + metrics + weight evolution

results/results_variable_memetic_nsga2_archive/ablation/
  <dataset>/
    <device>/
      ablation_K.csv                           # K, hv_ratio, gd, igd
      ablation_K.png                           # line plot of metrics vs K
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
from src.metaheuristic.variable_memetic_nsga2_with_archive import run_memetic_archive
from src.metaheuristic.archives import FHArchive, CrowdingArchive
from src.evaluation.metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

ADAPTIVE_CONFIG = {
    "initial_probs": np.array([0.33, 0.33, 0.34]),
    "update_period": 5,
}

ARCHIVE_VARIANTS = ["fh", "crowding"]
DATASETS         = ["cifar10", "cifar100", "ImageNet16-120"]
DEVICES          = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]


# ------------------------------------------------------------------ #
# Archive factory
# ------------------------------------------------------------------ #

def make_archive(archive_type: str):
    """Return a fresh archive instance for the given type."""
    if archive_type == "fh":
        return FHArchive(
            ref_acc=0.0, ref_lat=0.0,
            delta_acc=0.5, delta_lat=0.5,
            max_cells=2000, max_sols=5,
        )
    if archive_type == "crowding":
        return CrowdingArchive(max_size=500)
    raise ValueError(f"Unknown archive type: {archive_type!r}. "
                     f"Choose from {ARCHIVE_VARIANTS}")


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
    archive_type: str = "crowding",
    adaptive: bool    = False,
) -> dict:
    """Run online variance-guided memetic NSGA-II for one combination."""

    adaptive_cfg = ADAPTIVE_CONFIG if adaptive else None
    mode_label   = f"{archive_type}" + (" | Adaptive" if adaptive else "")

    results = run_memetic_archive(
        nb201_loader      = nb201_loader,
        hw_loader         = hw_loader,
        dataset           = dataset,
        hw_platform       = device,
        archive           = make_archive(archive_type),
        pop_size          = 50,
        n_gen             = 50,
        K                 = K,
        seed              = 42,
        verbose           = True,
        adaptive_mutation = adaptive_cfg,
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
        os.path.join(output_dir, "variable_memetic_archive_front.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save final online edge weights CSV
    # ------------------------------------------------------------------
    w = results["edge_weights"]
    pd.DataFrame({"edge": range(len(w)), "variance_weight": w}).to_csv(
        os.path.join(output_dir, "edge_weights.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save full archive CSV
    # ------------------------------------------------------------------
    results["archive_df"].to_csv(
        os.path.join(output_dir, "archive_members.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save history CSV (includes gene_probs per generation)
    # ------------------------------------------------------------------
    history_df = pd.DataFrame([
        {
            "generation":        h["generation"],
            "pareto_front_size": h["pareto_front_size"],
            "archive_size":      h["archive_size"],
            **{f"w_e{i}": p for i, p in enumerate(h["gene_probs"])},
        }
        for h in results["history"]
    ])
    history_df.to_csv(os.path.join(output_dir, "history.csv"), index=False)

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
    # Plot — 3 panels: Pareto scatter | metrics bar | weight evolution
    # ------------------------------------------------------------------
    df_all = pd.DataFrame({
        "latency":  results["F"][:, 1],
        "accuracy": -results["F"][:, 0],
    })
    df_pareto_sorted = df_pareto.sort_values("latency")

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    fig.suptitle(
        f"Online Variance Memetic NSGA-II | {mode_label} | "
        f"{dataset} | {device} | K={K}"
    )

    # --- Left: Pareto scatter ---
    ax1.scatter(df_all["latency"], df_all["accuracy"],
                alpha=0.3, label="Population")
    ax1.plot(df_pareto_sorted["latency"], df_pareto_sorted["accuracy"],
             "r-s", label="Memetic front")
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"],
                 "k--o", label="True Pareto")
    ax1.set_xlabel("Latency (ms)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Pareto Front")
    ax1.legend()

    # --- Middle: metrics bar with values on top ---
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
                         label=f"Memetic ({mode_label})",
                         color="tab:purple", offset=0.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.axhline(100, linestyle="--", color="black", linewidth=0.8)
        ax2.set_title("Metrics vs Ground Truth")
        ax2.legend(fontsize=8)
    else:
        ax2.axis("off")

    # --- Right: online gene weight evolution over generations ---
    if not history_df.empty:
        weight_cols = [c for c in history_df.columns if c.startswith("w_e")]
        gens = history_df["generation"]
        for col in weight_cols:
            edge_label = col.replace("w_", "")
            ax3.plot(gens, history_df[col], marker="", label=edge_label)
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Gene weight")
        ax3.set_title("Online Variance Weights over Generations")
        ax3.legend(fontsize=7, ncol=2)
    else:
        ax3.axis("off")

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
    archive_type: str = "crowding",
    adaptive: bool    = False,
):
    rows = []
    for K in K_values:
        print(f"\n--- Ablation K={K} | {dataset} | {device} ---")
        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            K            = K,
            output_dir   = os.path.join(ablation_dir, f"K{K}"),
            archive_type = archive_type,
            adaptive     = adaptive,
        )
        rows.append({"K": K, **metrics})

    df_abl = pd.DataFrame(rows)
    os.makedirs(ablation_dir, exist_ok=True)
    df_abl.to_csv(os.path.join(ablation_dir, "ablation_K.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: K budget | {dataset} | {device} | {archive_type}")
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

def main(
    run_ablation_flag: bool = False,
    archive_type: str       = "crowding",
    adaptive: bool          = False,
):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    K_DEFAULT  = 4
    K_ABLATION = [2, 4, 6, 8]
    suffix     = "_adaptive" if adaptive else ""

    for dataset in DATASETS:
        for device in DEVICES:
            print(
                f"\n=== Online Variance Memetic NSGA-II | "
                f"archive={archive_type} | dataset={dataset} device={device} ==="
            )

            if run_ablation_flag:
                ablation_dir = os.path.join(
                    BASE_DIR,
                    "../results/results_variable_memetic_nsga2_archive/ablation",
                    dataset, device, f"{archive_type}{suffix}",
                )
                run_ablation(
                    nb201_loader, hw_loader,
                    dataset, device,
                    K_values     = K_ABLATION,
                    ablation_dir = ablation_dir,
                    archive_type = archive_type,
                    adaptive     = adaptive,
                )
            else:
                output_dir = os.path.join(
                    BASE_DIR,
                    "../results/results_variable_memetic_nsga2_archive",
                    dataset, device, f"{archive_type}{suffix}",
                )
                metrics = run_single(
                    nb201_loader, hw_loader,
                    dataset, device,
                    K            = K_DEFAULT,
                    output_dir   = output_dir,
                    archive_type = archive_type,
                    adaptive     = adaptive,
                )
                print(f"\n=== SUMMARY ({dataset} | {device} | {archive_type}) ===")
                print(f"HV:  {metrics['hv_ratio']:.2f}%")
                print(f"GD:  {metrics['gd']:.4f}")
                print(f"IGD: {metrics['igd']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        choices=ARCHIVE_VARIANTS,
        default="crowding",
        help="Archive variant to use (default: crowding)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable UCB1 adaptive mutation",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation over K values",
    )
    args = parser.parse_args()

    main(
        run_ablation_flag = args.ablation,
        archive_type      = args.archive,
        adaptive          = args.adaptive,
    )