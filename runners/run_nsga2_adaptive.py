# runners/run_nsga2.py
"""
Experiment runner for NSGA-II (baseline + adaptive).

Output layout
-------------
results/results_nsga2/
  <dataset>/
    <device>/
      baseline/
        nsga2_front.csv        # arch_index, accuracy, latency
        comparison.png         # scatter + metrics bar (vs ground truth)
      adaptive/
        nsga2_front.csv
        comparison.png

results/results_nsga2/ablation/
  <dataset>/
    <device>/
      ablation_adaptive.csv    # mode, hv_ratio, gd, igd
      ablation_adaptive.png    # bar comparison of metrics
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
from src.metaheuristic.nsga2.core.nsga2 import NSGA2
from src.benchmark.hwnas_problem_def import HWNASProblem
from src.evaluation.metrics import compute_metrics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
DEVICES  = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]

POP_SIZE    = 50
GENERATIONS = 50
SEED        = 42


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
# Single run
# ------------------------------------------------------------------ #

def run_single(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    adaptive: bool,
    output_dir: str,
) -> dict:
    """Run NSGA-II (baseline or adaptive) for one (dataset, device) combo."""

    print(f"\n{'='*70}")
    print(f"NSGA-II | dataset={dataset} | device={device} | adaptive={adaptive}")
    print(f"{'='*70}")

    # ----------------------------------------------------------------
    # Problem + algorithm
    # ----------------------------------------------------------------
    problem = HWNASProblem(
        acc_loader=nb201_loader,
        hw_loader=hw_loader,
        dataset=dataset,
        platform=device,
    )

    adaptive_config = None
    if adaptive:
        adaptive_config = {
            "initial_probs": np.array([0.33, 0.33, 0.34]),
            "update_period": 5,
        }

    algo = NSGA2(
        problem=problem,
        pop_size=POP_SIZE,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        num_choices=problem.num_choices,
        seed=SEED,
        adaptive_mutation=adaptive_config,
    )

    pareto_set, pareto_front = algo.run(generations=GENERATIONS, verbose=False)

    # ----------------------------------------------------------------
    # Build Pareto DataFrame  [accuracy, latency]
    # ----------------------------------------------------------------
    acc = -pareto_front[:, 0]
    lat =  pareto_front[:, 1]

    df_pareto = pd.DataFrame({
        "arch_index": np.arange(len(acc)),
        "accuracy":   acc,
        "latency":    lat,
    })

    os.makedirs(output_dir, exist_ok=True)
    df_pareto.to_csv(os.path.join(output_dir, "nsga2_front.csv"), index=False)

    # ----------------------------------------------------------------
    # Ground truth + metrics
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # Plot — scatter + metrics bar (identical layout to memetic runner)
    # ----------------------------------------------------------------
    mode_label  = "Adaptive" if adaptive else "Baseline"
    df_pop      = pd.DataFrame({"latency": lat, "accuracy": acc})
    df_p_sorted = df_pareto.sort_values("latency")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"NSGA-II ({mode_label}) | {dataset} | {device}")

    ax1.scatter(df_pop["latency"], df_pop["accuracy"],
                alpha=0.3, label="Pareto population")
    ax1.plot(df_p_sorted["latency"], df_p_sorted["accuracy"],
             "r-s", label=f"NSGA-II front ({mode_label})")
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"],
                 "k--o", label="True Pareto")
    ax1.set_xlabel("Latency")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    if df_gt is not None:
        names  = ["HV (%)", "GD×100", "IGD×100"]
        values = [
            metrics["hv_ratio"],
            metrics["gd"]  * 100,
            metrics["igd"] * 100,
        ]
        ax2.bar(names, values)
        ax2.axhline(100, linestyle="--", color="black")
        ax2.set_title("Metrics vs Ground Truth")
    else:
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

    print(f"[OK] Results saved → {output_dir}")
    return metrics


# ------------------------------------------------------------------ #
# Ablation: baseline vs adaptive for one (dataset, device)
# ------------------------------------------------------------------ #

def run_ablation(
    nb201_loader,
    hw_loader,
    dataset: str,
    device: str,
    ablation_dir: str,
):
    rows = []
    for adaptive in [False, True]:
        mode = "adaptive" if adaptive else "baseline"
        print(f"\n--- Ablation mode={mode} | {dataset} | {device} ---")
        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            adaptive   = adaptive,
            output_dir = os.path.join(ablation_dir, mode),
        )
        rows.append({"mode": mode, **metrics})

    df_abl = pd.DataFrame(rows)
    os.makedirs(ablation_dir, exist_ok=True)
    df_abl.to_csv(os.path.join(ablation_dir, "ablation_adaptive.csv"), index=False)

    # Bar comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: Baseline vs Adaptive | {dataset} | {device}")
    for ax, col, label in zip(
        axes,
        ["hv_ratio", "gd", "igd"],
        ["HV ratio (%)", "GD (normalised)", "IGD (normalised)"],
    ):
        ax.bar(df_abl["mode"], df_abl[col])
        ax.set_ylabel(label)
        ax.set_title(label)

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "ablation_adaptive.png"))
    plt.close()
    print(f"[OK] Ablation saved → {ablation_dir}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(run_ablation_flag: bool = False):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    for dataset in DATASETS:
        for device in DEVICES:
            print(f"\n=== NSGA-II | dataset={dataset} device={device} ===")

            if run_ablation_flag:
                ablation_dir = os.path.join(
                    BASE_DIR, "../results/results_nsga2/ablation",
                    dataset, device,
                )
                run_ablation(
                    nb201_loader, hw_loader,
                    dataset, device,
                    ablation_dir = ablation_dir,
                )

            else:
                # Default: run both modes and report
                for adaptive in [False, True]:
                    mode       = "adaptive" if adaptive else "baseline"
                    output_dir = os.path.join(
                        BASE_DIR, "../results/results_nsga2",
                        dataset, device, mode,
                    )
                    metrics = run_single(
                        nb201_loader, hw_loader,
                        dataset, device,
                        adaptive   = adaptive,
                        output_dir = output_dir,
                    )
                    print(f"\n=== SUMMARY ({dataset} | {device} | {mode}) ===")
                    print(f"HV:  {metrics['hv_ratio']:.2f}%")
                    print(f"GD:  {metrics['gd']:.4f}")
                    print(f"IGD: {metrics['igd']:.4f}")


if __name__ == "__main__":
    main(run_ablation_flag="--ablation" in sys.argv)