# experiments/run_nsga2.py

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
from src.metaheuristic.nsga2 import run

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Metrics (YOUR VERSION — slightly improved)
# ------------------------------------------------------------------ #

def hypervolume_2d(front_arr: np.ndarray, ref_point: tuple) -> float:
    pts = np.column_stack([-front_arr[:, 0], front_arr[:, 1]])
    ref = np.array([-ref_point[0], ref_point[1]])

    # sort by f1 (minimization)
    pts = pts[np.argsort(pts[:, 0])]

    hv = 0.0
    prev_y = ref[1]

    for pt in pts:
        if pt[1] < prev_y:
            hv += (ref[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]

    return hv


def normalize(arr: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    return (arr - vmin) / (vmax - vmin + 1e-8)


def gd(approx: np.ndarray, true: np.ndarray) -> float:
    return np.mean([
        np.min(np.linalg.norm(true - pt, axis=1))
        for pt in approx
    ])


def igd(approx: np.ndarray, true: np.ndarray) -> float:
    return np.mean([
        np.min(np.linalg.norm(approx - pt, axis=1))
        for pt in true
    ])


def compute_metrics(approx_front: np.ndarray, true_front: np.ndarray) -> dict:
    approx_arr = approx_front
    true_arr = true_front

    # Reference point (robust)
    all_acc = np.concatenate([approx_arr[:, 0], true_arr[:, 0]])
    all_lat = np.concatenate([approx_arr[:, 1], true_arr[:, 1]])

    margin = 0.05
    ref = (
        np.min(all_acc) - margin * abs(np.min(all_acc)),
        np.max(all_lat) + margin * abs(np.max(all_lat))
    )

    # Hypervolume
    hv_true = hypervolume_2d(true_arr, ref)
    hv_approx = hypervolume_2d(approx_arr, ref)
    hv_ratio = hv_approx / hv_true * 100 if hv_true > 0 else 0.0

    # Normalization
    vmin = np.array([np.min(all_acc), np.min(all_lat)])
    vmax = np.array([np.max(all_acc), np.max(all_lat)])

    approx_n = normalize(approx_arr, vmin, vmax)
    true_n = normalize(true_arr, vmin, vmax)

    return {
        "hv_true": hv_true,
        "hv_approx": hv_approx,
        "hv_ratio": hv_ratio,
        "gd": gd(approx_n, true_n),
        "igd": igd(approx_n, true_n),
        "front_size_true": len(true_arr),
        "front_size_approx": len(approx_arr),
    }


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    print("[INFO] Loading loaders...")

    nb201_loader = NASBench201CSVLoader()
    hw_loader = HWNASLoader()

    DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
    DEVICES = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]

    for dataset in DATASETS:
        for device in DEVICES:

            print(f"\n=== Running NSGA-II | dataset={dataset} device={device} ===")

            results = run(
                nb201_loader=nb201_loader,
                hw_loader=hw_loader,
                dataset=dataset,
                hw_platform=device,
                pop_size=50,
                n_gen=20,
                seed=42,
                verbose=True
            )

            print("\n[✓] NSGA-II finished")

            OUTPUT_DIR = os.path.join(BASE_DIR, "../results/results_nsga2", dataset, device)
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            # ----------------------------------------------------------
            # Dataframes
            # ----------------------------------------------------------
            df_all = pd.DataFrame({
                "arch_index": results["X"],
                "accuracy": -results["F"][:, 0],
                "latency": results["F"][:, 1],
            })

            df_pareto = pd.DataFrame({
                "arch_index": results["pareto_X"],
                "accuracy": results["pareto_acc"],
                "latency": results["pareto_lat"],
            })

            # Save Pareto
            df_pareto.to_csv(os.path.join(OUTPUT_DIR, "nsga2_front.csv"), index=False)

            # ----------------------------------------------------------
            # Load ground truth
            # ----------------------------------------------------------
            gt_path = os.path.join(BASE_DIR, "../ground_truth", dataset, device, "pareto_front.csv")

            df_gt = None
            if os.path.isfile(gt_path):
                df_gt = pd.read_csv(gt_path).sort_values("latency")
                print(f"[✓] Ground truth loaded ({len(df_gt)} pts)")

            # ----------------------------------------------------------
            # Metrics
            # ----------------------------------------------------------
            hv_ratio, gd_val, igd_val = 0.0, 0.0, 0.0

            if df_gt is not None:
                approx = df_pareto[["accuracy", "latency"]].values
                true = df_gt[["accuracy", "latency"]].values

                metrics = compute_metrics(approx, true)

                hv_ratio = metrics["hv_ratio"]
                gd_val = metrics["gd"]
                igd_val = metrics["igd"]

                print(f"[✓] HV: {hv_ratio:.2f}% | GD: {gd_val:.4f} | IGD: {igd_val:.4f}")

            # ----------------------------------------------------------
            # Plot
            # ----------------------------------------------------------
            df_pareto_sorted = df_pareto.sort_values("latency")

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

            # Scatter
            ax1.scatter(df_all["latency"], df_all["accuracy"], alpha=0.3, label="NSGA-II")
            ax1.plot(df_pareto_sorted["latency"], df_pareto_sorted["accuracy"], 'r-s', label="NSGA-II front")

            if df_gt is not None:
                ax1.plot(df_gt["latency"], df_gt["accuracy"], 'k--o', label="True Pareto")

            ax1.set_xlabel("Latency")
            ax1.set_ylabel("Accuracy")
            ax1.legend()

            # Metrics bar
            if df_gt is not None:
                names = ["HV (%)", "GD×100", "IGD×100"]
                values = [hv_ratio, gd_val * 100, igd_val * 100]

                ax2.bar(names, values)
                ax2.axhline(100, linestyle="--", color="black")
                ax2.set_title("Metrics")

            else:
                ax2.axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "comparison.png"))
            plt.close()

            print(f"\n=== SUMMARY ({dataset} | {device}) ===")
            print(f"HV: {hv_ratio:.2f}%")
            print(f"GD: {gd_val:.4f}")
            print(f"IGD: {igd_val:.4f}")


if __name__ == "__main__":
    main()