import os
import json
import argparse
import yaml
import numpy as np
from tqdm import tqdm

from src.benchmark.hwnas_loader import HWNASLoader
from src.evaluation.metrics import compute_pareto_front, hypervolume_2d
from src.evaluation.plotting import plot_pareto_front


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment from YAML config")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # ── Config unpacking ────────────────────────────────────
    seed = cfg["experiment"]["seed"]
    n_evals = cfg["experiment"]["n_evals"]

    dataset = cfg["data"]["dataset"]
    hw_platform = cfg["hardware"]["platform"]

    hw_pickle = cfg["paths"]["hw_pickle"]
    nasbench_pth = cfg["paths"]["nasbench_pth"]

    output_dir = cfg["output"]["dir"]
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(seed)

    # ── Load HW ─────────────────────────────────────────────
    hw_loader = HWNASLoader(data_path=hw_pickle)

    # ── Load accuracy (optional) ────────────────────────────
    USE_ACCURACY = nasbench_pth and os.path.exists(nasbench_pth)

    if USE_ACCURACY:
        from src.benchmark.nasbench201_loader import NASBench201Loader
        acc_loader = NASBench201Loader(pth_path=nasbench_pth)
        print("[INFO] Accuracy enabled.")
    else:
        print("[WARNING] No NAS-Bench-201 file → accuracy = 0.")

    # ── Random Search ───────────────────────────────────────
    all_solutions = []

    for _ in tqdm(range(n_evals), desc="Random Search"):
        idx = hw_loader.random_architecture()

        hw_metrics = hw_loader.get_hw_metrics(idx, dataset=dataset)
        latency = hw_metrics[f"{hw_platform}_latency"]

        accuracy = (
            acc_loader.get_accuracy(idx, dataset)
            if USE_ACCURACY else 0.0
        )

        all_solutions.append({
            "arch_index": idx,
            "accuracy": accuracy,
            "latency": latency,
        })

    # ── Pareto ──────────────────────────────────────────────
    pareto = compute_pareto_front(all_solutions)

    print(f"\nEvaluated: {n_evals}")
    print(f"Pareto size: {len(pareto)}")

    if USE_ACCURACY:
        print(f"Best accuracy: {max(s['accuracy'] for s in pareto):.2f}%")

    print(f"Best latency: {min(s['latency'] for s in pareto):.4f} ms")

    # ── Hypervolume ─────────────────────────────────────────
    if USE_ACCURACY:
        max_latency = max(s["latency"] for s in all_solutions)
        ref_point = (max_latency * 1.1, 0.0)
        hv = hypervolume_2d(pareto, ref_point)
        print(f"Hypervolume: {hv:.4f}")

    # ── Output paths ────────────────────────────────────────
    exp_name = cfg["experiment"]["name"]
    plot_path = os.path.join(output_dir, f"{exp_name}_{dataset}_{hw_platform}.png")
    json_path = os.path.join(output_dir, f"{exp_name}_{dataset}_{hw_platform}.json")

    # ── Plot ───────────────────────────────────────────────
    plot_pareto_front(
        all_solutions=all_solutions,
        pareto_front=pareto,
        title=f"{exp_name} — {dataset} / {hw_platform}",
        save_path=plot_path
    )

    # ── Save JSON ──────────────────────────────────────────
    output = {
        "config": cfg,
        "pareto_front": pareto,
        "stats": {
            "pareto_size": len(pareto),
            "best_accuracy": max(s["accuracy"] for s in pareto) if USE_ACCURACY else None,
            "best_latency": min(s["latency"] for s in pareto),
        }
    }

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {json_path}")
    print(f"Plot:  {plot_path}")


if __name__ == "__main__":
    main()