# runners/run_nsga2_with_archive.py
"""
Experiment runner for NSGA-II with elitist archive.

Two archive variants per (dataset, device):
    fh       — Fixed-Hypergrid Archive (FHArchive)
    crowding — Bounded crowding-distance archive (CrowdingArchive)

Each variant can also be run with adaptive mutation (UCB1) by passing
--adaptive on the command line.

Output layout
-------------
results/results_nsga2_with_archive/
  <dataset>/
    <device>/
      <archive_type>/          # fh | crowding
        nsga2_archive_front.csv     # arch_index, accuracy, latency
        archive_members.csv         # full archive content at end of run
        history.csv                 # gen, pareto_front_size, archive_size
        comparison.png              # scatter + metrics bar (vs ground truth)

results/results_nsga2_with_archive/ablation/
  <dataset>/
    <device>/
      ablation_archive.csv          # archive_type, hv_ratio, gd, igd
      ablation_archive.png          # bar comparison

Run
---
    python runners/run_nsga2_with_archive.py                        # all variants
    python runners/run_nsga2_with_archive.py --archive crowding
    python runners/run_nsga2_with_archive.py --archive fh --adaptive
    python runners/run_nsga2_with_archive.py --ablation
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader
from src.metaheuristic.archives import FHArchive, CrowdingArchive
from src.metaheuristic.nsga2_with_archive import run_nsga2_with_archive
from src.evaluation.metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

POP_SIZE    = 50
GENERATIONS = 50
SEED        = 42

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
    archive_type: str,
    output_dir: str,
    adaptive: bool = False,
) -> dict:
    """Run NSGA-II + archive for one (dataset, device, archive_type) combo."""

    adaptive_cfg = ADAPTIVE_CONFIG if adaptive else None
    mode_label   = f"{archive_type}" + (" | Adaptive" if adaptive else "")

    results = run_nsga2_with_archive(
        nb201_loader      = nb201_loader,
        hw_loader         = hw_loader,
        dataset           = dataset,
        hw_platform       = device,
        archive           = make_archive(archive_type),
        pop_size          = POP_SIZE,
        n_gen             = GENERATIONS,
        seed              = SEED,
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
        os.path.join(output_dir, "nsga2_archive_front.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save full archive CSV
    # ------------------------------------------------------------------
    results["archive_df"].to_csv(
        os.path.join(output_dir, "archive_members.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save history CSV
    # ------------------------------------------------------------------
    pd.DataFrame(results["history"]).to_csv(
        os.path.join(output_dir, "history.csv"), index=False
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
    history_df       = pd.DataFrame(results["history"])
    df_pareto_sorted = df_pareto.sort_values("latency")

    fig = plt.figure(figsize=(18, 5))
    gs  = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    fig.suptitle(f"NSGA-II + Archive ({mode_label}) | {dataset} | {device}")

    # --- Left: Pareto scatter ---
    ax1.plot(df_pareto_sorted["latency"], df_pareto_sorted["accuracy"],
             "r-s", label=f"Archive front ({archive_type})")
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"],
                 "k--o", label="True Pareto")
    ax1.set_xlabel("Latency")
    ax1.set_ylabel("Accuracy")
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
                         label=f"Archive ({archive_type})",
                         color="tab:blue", offset=0.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels(names)
        ax2.axhline(100, linestyle="--", color="black", linewidth=0.8)
        ax2.set_title("Metrics vs Ground Truth")
        ax2.legend(fontsize=8)
    else:
        ax2.axis("off")

    # --- Right: archive size + Pareto front size over generations ---
    if not history_df.empty:
        ax3.plot(history_df["generation"], history_df["archive_size"],
                 "b-", label="Archive size")
        ax3.plot(history_df["generation"], history_df["pareto_front_size"],
                 "r--", label="Pareto front size")
        ax3.set_xlabel("Generation")
        ax3.set_ylabel("Count")
        ax3.set_title("Archive / Front Size over Generations")
        ax3.legend(fontsize=8)
    else:
        ax3.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

    print(f"[OK] Results saved → {output_dir}")
    return metrics


# ------------------------------------------------------------------ #
# Ablation: fh vs crowding for one (dataset, device)
# ------------------------------------------------------------------ #

def run_ablation(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    ablation_dir: str,
    adaptive: bool = False,
):
    rows = []
    for archive_type in ARCHIVE_VARIANTS:
        print(f"\n--- Ablation archive={archive_type} | {dataset} | {device} ---")
        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            archive_type = archive_type,
            output_dir   = os.path.join(ablation_dir, archive_type),
            adaptive     = adaptive,
        )
        rows.append({"archive_type": archive_type, **metrics})

    df_abl = pd.DataFrame(rows)
    os.makedirs(ablation_dir, exist_ok=True)
    df_abl.to_csv(os.path.join(ablation_dir, "ablation_archive.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: Archive type | {dataset} | {device}")
    for ax, col, label in zip(
        axes,
        ["hv_ratio", "gd", "igd"],
        ["HV ratio (%)", "GD (normalised)", "IGD (normalised)"],
    ):
        bars = ax.bar(df_abl["archive_type"], df_abl[col],
                      color=["tab:blue", "tab:orange"], alpha=0.85)
        # Values on top of each bar
        for bar, val in zip(bars, df_abl[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.5,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "ablation_archive.png"))
    plt.close()
    print(f"[OK] Ablation saved → {ablation_dir}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(
    run_ablation_flag: bool  = False,
    archive_types: list[str] = ARCHIVE_VARIANTS,
    adaptive: bool           = False,
):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    for dataset in DATASETS:
        for device in DEVICES:
            print(f"\n=== NSGA-II + Archive | dataset={dataset} device={device} ===")

            if run_ablation_flag:
                ablation_dir = os.path.join(
                    BASE_DIR, "../results/results_nsga2_with_archive/ablation",
                    dataset, device,
                )
                run_ablation(
                    nb201_loader, hw_loader,
                    dataset, device,
                    ablation_dir = ablation_dir,
                    adaptive     = adaptive,
                )

            else:
                for archive_type in archive_types:
                    suffix     = "_adaptive" if adaptive else ""
                    output_dir = os.path.join(
                        BASE_DIR, "../results/results_nsga2_with_archive",
                        dataset, device, f"{archive_type}{suffix}",
                    )
                    metrics = run_single(
                        nb201_loader, hw_loader,
                        dataset, device,
                        archive_type = archive_type,
                        output_dir   = output_dir,
                        adaptive     = adaptive,
                    )
                    print(f"\n=== SUMMARY ({dataset} | {device} | {archive_type}) ===")
                    print(f"HV:  {metrics['hv_ratio']:.2f}%")
                    print(f"GD:  {metrics['gd']:.4f}")
                    print(f"IGD: {metrics['igd']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        choices=ARCHIVE_VARIANTS + ["all"],
        default="all",
        help="Archive variant to run (default: all)",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable UCB1 adaptive mutation",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run ablation over archive types",
    )
    args = parser.parse_args()

    archive_types = ARCHIVE_VARIANTS if args.archive == "all" else [args.archive]

    main(
        run_ablation_flag = args.ablation,
        archive_types     = archive_types,
        adaptive          = args.adaptive,
    )