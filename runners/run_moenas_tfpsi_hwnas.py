# experiments/run_moenas_tfpsi_hwnas.py
"""
Experiment runner: MOENAS-TF-PSI adapted to HW-NAS-Bench.

Three modes per (dataset, device):
    vanilla  — NSGA-II, no local search        (their baseline)
    psi      — NSGA-II + PSI, true objectives  (PSI oracle upper bound)
    tfpsi    — NSGA-II + TF-PSI               (their main contribution)

Run:
    python experiments/run_moenas_tfpsi_hwnas.py              # all modes
    python experiments/run_moenas_tfpsi_hwnas.py --mode tfpsi
    python experiments/run_moenas_tfpsi_hwnas.py --mode vanilla --seed 0
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pymoo.optimize import minimize
from pymoo.termination import get_termination

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))   # .../experiments/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                    # .../hwnas-with-metaheuristics-project/

# Both the project root AND the MOENAS-TF-PSI clone must be on sys.path.
# algorithm_hwnas.py handles the MOENAS path itself, but we add the project
# root here so 'from src...' and 'from moenas_tf_psi_hwnas...' both resolve
# when this script is launched from any working directory (e.g. repo root,
# experiments/, or via python -m).
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader
from moenas_tf_psi_hwnas.problem_hwnas import HWNASProblemMOENAS, index_to_ops
from moenas_tf_psi_hwnas.algorithm_hwnas import NSGA2_PSI, _inject_hash_keys


# ------------------------------------------------------------------ #
# Hyperparameters — match MOENAS-TF-PSI paper settings exactly
# ------------------------------------------------------------------ #

POP_SIZE  = 20     # their reported default
MAX_EVALS = 300    # their reported budget for NAS-Bench-201


# ------------------------------------------------------------------ #
# Metrics  (identical implementation as run_nsgai2.py)
# ------------------------------------------------------------------ #

def hypervolume_2d(front_arr: np.ndarray, ref_point: tuple) -> float:
    pts    = np.column_stack([-front_arr[:, 0], front_arr[:, 1]])
    ref    = np.array([-ref_point[0], ref_point[1]])
    pts    = pts[np.argsort(pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for pt in pts:
        if pt[1] < prev_y:
            hv    += (ref[0] - pt[0]) * (prev_y - pt[1])
            prev_y = pt[1]
    return hv


def normalize(arr, vmin, vmax):
    return (arr - vmin) / (vmax - vmin + 1e-8)


def gd(approx, true):
    return np.mean([np.min(np.linalg.norm(true - pt, axis=1)) for pt in approx])


def igd(approx, true):
    return np.mean([np.min(np.linalg.norm(approx - pt, axis=1)) for pt in true])


def compute_metrics(approx_front: np.ndarray, true_front: np.ndarray) -> dict:
    all_acc = np.concatenate([approx_front[:, 0], true_front[:, 0]])
    all_lat = np.concatenate([approx_front[:, 1], true_front[:, 1]])
    margin  = 0.05
    ref     = (
        np.min(all_acc) - margin * abs(np.min(all_acc)),
        np.max(all_lat) + margin * abs(np.max(all_lat)),
    )
    hv_true   = hypervolume_2d(true_front, ref)
    hv_approx = hypervolume_2d(approx_front, ref)
    hv_ratio  = hv_approx / hv_true * 100 if hv_true > 0 else 0.0
    vmin = np.array([np.min(all_acc), np.min(all_lat)])
    vmax = np.array([np.max(all_acc), np.max(all_lat)])
    return {
        "hv_ratio": hv_ratio,
        "gd":  gd(normalize(approx_front, vmin, vmax),
                  normalize(true_front,   vmin, vmax)),
        "igd": igd(normalize(approx_front, vmin, vmax),
                   normalize(true_front,   vmin, vmax)),
    }


def load_gt(dataset: str, device: str) -> pd.DataFrame | None:
    gt_path = os.path.join(
        BASE_DIR, "../ground_truth", dataset, device, "pareto_front.csv"
    )
    if os.path.isfile(gt_path):
        return pd.read_csv(gt_path).sort_values("latency")
    return None


# ------------------------------------------------------------------ #
# Single run
# ------------------------------------------------------------------ #

def run_one_mode(
    nb201_loader, hw_loader,
    dataset: str, device: str,
    mode: str,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    """
    Run one (dataset, device, mode) combination.

    Returns
    -------
    df_pareto  : DataFrame [arch_index, accuracy, latency]
    n_evals    : new evaluations consumed (cache misses only)
    """
    problem = HWNASProblemMOENAS(
        nb201_loader = nb201_loader,
        hw_loader    = hw_loader,
        dataset      = dataset,
        hw_platform  = device,
    )

    use_psi      = (mode != "vanilla")
    psi_using_zc = (mode == "tfpsi")

    algorithm = NSGA2_PSI(
        problem      = problem,
        use_psi      = use_psi,
        psi_npoints  = 1,           # k=1: their best reported setting
        psi_using_zc = psi_using_zc,
        gamma        = 210,         # their default knee-point threshold
        pop_size     = POP_SIZE,
    )

    # Inject hashKeys into the initial population before the first PSI call
    # (pymoo initialises the population before _advance is ever called)
    termination = get_termination("n_eval", MAX_EVALS)

    print(f"  [{mode:7s}] pop={POP_SIZE}  max_evals={MAX_EVALS}  seed={seed}")

    result = minimize(
        problem,
        algorithm,
        termination,
        seed    = seed,
        verbose = False,
    )

    # ------------------------------------------------------------------
    # Extract Pareto front
    # Two sources: pymoo result.opt AND the elitist archive.
    # The archive may contain solutions accepted by PSI that were later
    # dominated in the population — take the union and re-filter.
    # ------------------------------------------------------------------
    opt_X = result.opt.get("X")              # (n_pareto, 6)
    opt_F = result.opt.get("F")              # (n_pareto, 2)  [err, latency]

    # Add archive members
    arc_F = algorithm.E_Archive_search.pareto_F()   # may be empty
    arc_X = algorithm.E_Archive_search.pareto_X()

    if len(arc_F) > 0:
        all_X = np.vstack([opt_X, np.array(arc_X)])
        all_F = np.vstack([opt_F, arc_F])
    else:
        all_X = opt_X
        all_F = opt_F

    # Re-filter to true Pareto front
    is_pareto = _get_pareto_mask(all_F)
    pareto_X_ops = all_X[is_pareto]           # op-vectors
    pareto_F     = all_F[is_pareto]

    # Convert op-vectors → arch_index for output consistency
    pareto_arch_indices = np.array([
        sum(int(op) * (5 ** e) for e, op in enumerate(x))
        for x in pareto_X_ops
    ])
    pareto_acc = (1.0 - pareto_F[:, 0]) * 100.0   # error → accuracy %
    pareto_lat =  pareto_F[:, 1]

    df_pareto = pd.DataFrame({
        "arch_index": pareto_arch_indices,
        "accuracy":   pareto_acc,
        "latency":    pareto_lat,
    })

    return df_pareto, problem.n_evals


def _get_pareto_mask(F: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated rows in F (minimisation)."""
    n = len(F)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                is_pareto[i] = False
                break
    return is_pareto


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

MODES = ["vanilla", "psi", "tfpsi"]
MODE_LABELS = {
    "vanilla": "NSGA-II",
    "psi":     "NSGA-II + PSI",
    "tfpsi":   "NSGA-II + TF-PSI",
}
MODE_STYLES = {
    "vanilla": "b-s",
    "psi":     "g-^",
    "tfpsi":   "r-o",
}


def main(run_modes: list[str], seed: int = 42):
    print("[INFO] Loading loaders...")
    nb201_loader = NASBench201CSVLoader()
    hw_loader    = HWNASLoader()

    DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]
    DEVICES  = ["fpga", "edgegpu", "edgetpu", "raspi4", "pixel3", "eyeriss"]

    for dataset in DATASETS:
        for device in DEVICES:
            print(f"\n=== MOENAS-TF-PSI (HW) | {dataset} | {device} ===")

            OUT = os.path.join(
                BASE_DIR, "../results_moenas_tfpsi", dataset, device
            )
            os.makedirs(OUT, exist_ok=True)

            df_gt        = load_gt(dataset, device)
            fronts       = {}
            metrics_rows = []

            for mode in run_modes:
                df_pareto, n_evals = run_one_mode(
                    nb201_loader, hw_loader,
                    dataset, device, mode, seed=seed,
                )
                fronts[mode] = df_pareto
                df_pareto.to_csv(
                    os.path.join(OUT, f"{mode}_front.csv"), index=False
                )

                m = {"mode": mode, "n_evals": n_evals,
                     "hv_ratio": 0.0, "gd": 0.0, "igd": 0.0}
                if df_gt is not None:
                    approx = df_pareto[["accuracy", "latency"]].values
                    true   = df_gt[["accuracy", "latency"]].values
                    m.update(compute_metrics(approx, true))

                metrics_rows.append(m)
                print(
                    f"  [{mode:7s}] evals={n_evals:4d}  "
                    f"HV={m['hv_ratio']:6.2f}%  "
                    f"GD={m['gd']:.4f}  IGD={m['igd']:.4f}"
                )

            pd.DataFrame(metrics_rows).to_csv(
                os.path.join(OUT, "metrics.csv"), index=False
            )

            # ----------------------------------------------------------
            # Plot: all modes + ground truth
            # ----------------------------------------------------------
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle(f"MOENAS-TF-PSI (HW) | {dataset} | {device}")

            for mode, df in fronts.items():
                df_s = df.sort_values("latency")
                ax.plot(df_s["latency"], df_s["accuracy"],
                        MODE_STYLES[mode], label=MODE_LABELS[mode])

            if df_gt is not None:
                ax.plot(df_gt["latency"], df_gt["accuracy"],
                        "k--o", label="True Pareto", linewidth=1.5)

            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Accuracy (%)")
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(OUT, "comparison.png"))
            plt.close()

            print(f"  [✓] Saved → {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["vanilla", "psi", "tfpsi", "all"],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes_to_run = MODES if args.mode == "all" else [args.mode]
    main(run_modes=modes_to_run, seed=args.seed)