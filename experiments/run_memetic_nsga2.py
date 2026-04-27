# experiments/run_memetic_nsga2.py
"""
Experiment runner for Memetic NSGA-II (Contribution 2).

Mirrors run_nsgai2.py exactly in structure: same datasets, same devices,
same metrics, same output layout.  The only additions are:
  - Loading ground-truth Pareto front indices for KL weight computation.
  - The K hyperparameter (neighbourhood budget).
  - An ablation loop over K values when --ablation is passed.

Output layout
-------------
results_memetic_nsga2/
  <dataset>/
    <device>/
      memetic_nsga2_front.csv      # arch_index, accuracy, latency
      comparison.png               # scatter + metrics bar (vs ground truth)
      kl_weights.csv               # edge weights used for this device

results_memetic_nsga2/ablation/
  <dataset>/
    <device>/
      ablation_K.csv               # K, hv_ratio, gd, igd for K in {2,4,6,8}
      ablation_K.png               # line plot of metrics vs K
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Add project root to Python path so 'src' is findable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader
from src.metaheuristic.memetic_nsga2 import run_memetic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Metrics (identical to run_nsgai2.py)
# ------------------------------------------------------------------ #

def hypervolume_2d(front_arr: np.ndarray, ref_point: tuple) -> float:
    pts = np.column_stack([-front_arr[:, 0], front_arr[:, 1]])
    ref = np.array([-ref_point[0], ref_point[1]])
    pts = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for pt in pts:
        if pt[1] < prev_y:
            hv += (ref[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]
    return hv


def normalize(arr, vmin, vmax):
    return (arr - vmin) / (vmax - vmin + 1e-8)


def gd(approx, true):
    return np.mean([np.min(np.linalg.norm(true - pt, axis=1)) for pt in approx])


def igd(approx, true):
    return np.mean([np.min(np.linalg.norm(approx - pt, axis=1)) for pt in true])


def compute_metrics(approx_front, true_front):
    all_acc = np.concatenate([approx_front[:, 0], true_front[:, 0]])
    all_lat = np.concatenate([approx_front[:, 1], true_front[:, 1]])
    margin  = 0.05
    ref = (
        np.min(all_acc) - margin * abs(np.min(all_acc)),
        np.max(all_lat) + margin * abs(np.max(all_lat)),
    )
    hv_true   = hypervolume_2d(true_front, ref)
    hv_approx = hypervolume_2d(approx_front, ref)
    hv_ratio  = hv_approx / hv_true * 100 if hv_true > 0 else 0.0
    vmin = np.array([np.min(all_acc), np.min(all_lat)])
    vmax = np.array([np.max(all_acc), np.max(all_lat)])
    approx_n = normalize(approx_front, vmin, vmax)
    true_n   = normalize(true_front,   vmin, vmax)
    return {
        "hv_true": hv_true, "hv_approx": hv_approx, "hv_ratio": hv_ratio,
        "gd": gd(approx_n, true_n), "igd": igd(approx_n, true_n),
        "front_size_true": len(true_front), "front_size_approx": len(approx_front),
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
        print(f"[✓] Ground truth loaded ({len(df)} pts)")
        return df
    return None


# ------------------------------------------------------------------ #
# Single run
# ------------------------------------------------------------------ #

def run_single(
    nb201_loader, hw_loader,
    dataset: str, device: str,
    K: int, output_dir: str,
    save_kl_weights: bool = True,
) -> dict:
    """Run memetic NSGA-II for one (dataset, device, K) combination."""

    # Load ground-truth Pareto indices for KL weight computation
    df_gt = load_gt(dataset, device)
    if df_gt is not None:
        pareto_indices_by_device = {device: df_gt["arch_index"].values}
    else:
        # No ground truth: use empty front → KL falls back to uniform weights
        print("[WARN] No ground truth found — using uniform KL weights")
        pareto_indices_by_device = {device: np.array([], dtype=int)}

    results = run_memetic(
        nb201_loader             = nb201_loader,
        hw_loader                = hw_loader,
        pareto_indices_by_device = pareto_indices_by_device,
        dataset                  = dataset,
        hw_platform              = device,
        pop_size                 = 50,
        n_gen                    = 20,
        K                        = K,
        seed                     = 42,
        verbose                  = True,
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
        os.path.join(output_dir, "memetic_nsga2_front.csv"), index=False
    )

    # ------------------------------------------------------------------
    # Save KL weights CSV
    # ------------------------------------------------------------------
    if save_kl_weights and device in results["kl_weights"]:
        w = results["kl_weights"][device]
        pd.DataFrame({"edge": range(len(w)), "kl_weight": w}).to_csv(
            os.path.join(output_dir, "kl_weights.csv"), index=False
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    metrics = {"hv_ratio": 0.0, "gd": 0.0, "igd": 0.0}
    if df_gt is not None:
        approx = df_pareto[["accuracy", "latency"]].values
        true   = df_gt[["accuracy", "latency"]].values
        metrics = compute_metrics(approx, true)
        print(
            f"[✓] HV: {metrics['hv_ratio']:.2f}% | "
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
    fig.suptitle(f"Memetic NSGA-II | {dataset} | {device} | K={K}")

    ax1.scatter(df_all["latency"], df_all["accuracy"], alpha=0.3,
                label="Population")
    ax1.plot(df_pareto_sorted["latency"], df_pareto_sorted["accuracy"],
             "r-s", label="Memetic front")
    if df_gt is not None:
        ax1.plot(df_gt["latency"], df_gt["accuracy"],
                 "k--o", label="True Pareto")
    ax1.set_xlabel("Latency"); ax1.set_ylabel("Accuracy"); ax1.legend()

    if df_gt is not None:
        names  = ["HV (%)", "GD×100", "IGD×100"]
        values = [metrics["hv_ratio"],
                  metrics["gd"]  * 100,
                  metrics["igd"] * 100]
        ax2.bar(names, values)
        ax2.axhline(100, linestyle="--", color="black")
        ax2.set_title("Metrics vs Ground Truth")
    else:
        ax2.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

    return metrics


# ------------------------------------------------------------------ #
# Ablation over K
# ------------------------------------------------------------------ #

def run_ablation(
    nb201_loader, hw_loader,
    dataset: str, device: str,
    K_values: list,
    ablation_dir: str,
):
    """Run one (dataset, device) combination for each K in K_values."""
    rows = []
    for K in K_values:
        print(f"\n--- Ablation K={K} | {dataset} | {device} ---")
        metrics = run_single(
            nb201_loader, hw_loader,
            dataset, device,
            K       = K,
            output_dir     = os.path.join(ablation_dir, f"K{K}"),
            save_kl_weights = (K == K_values[0]),  # save once
        )
        rows.append({"K": K, **metrics})

    df_abl = pd.DataFrame(rows)
    os.makedirs(ablation_dir, exist_ok=True)
    df_abl.to_csv(os.path.join(ablation_dir, "ablation_K.csv"), index=False)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"Ablation: K budget | {dataset} | {device}")
    for ax, col, label in zip(
        axes,
        ["hv_ratio", "gd", "igd"],
        ["HV ratio (%)", "GD (normalised)", "IGD (normalised)"],
    ):
        ax.plot(df_abl["K"], df_abl[col], marker="o")
        ax.set_xlabel("K"); ax.set_ylabel(label); ax.set_title(label)
        ax.set_xticks(K_values)

    plt.tight_layout()
    plt.savefig(os.path.join(ablation_dir, "ablation_K.png"))
    plt.close()
    print(f"[✓] Ablation saved → {ablation_dir}")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(run_ablation_flag: bool = False):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
    DEVICES  = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]
    K_DEFAULT  = 4
    K_ABLATION = [2, 4, 6, 8]

    for dataset in DATASETS:
        for device in DEVICES:
            print(f"\n=== Memetic NSGA-II | dataset={dataset} device={device} ===")

            if run_ablation_flag:
                ablation_dir = os.path.join(
                    BASE_DIR, "../results/results_memetic_nsga2/ablation", dataset, device
                )
                run_ablation(
                    nb201_loader, hw_loader,
                    dataset, device,
                    K_values     = K_ABLATION,
                    ablation_dir = ablation_dir,
                )
            else:
                output_dir = os.path.join(
                    BASE_DIR, "../results/results_memetic_nsga2", dataset, device
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
    import sys
    main(run_ablation_flag="--ablation" in sys.argv)