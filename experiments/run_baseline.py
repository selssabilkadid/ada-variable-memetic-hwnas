import sys
import os
import json
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmark.hwnas_loader import HWNASLoader
from src.evaluation.metrics import compute_pareto_front, hypervolume_2d
from src.evaluation.plotting import plot_pareto_front

# ── Config ────────────────────────────────────────────────────
HW_NAS_PICKLE   = "D:\\HW-NAS-Bench\\HW-NAS-Bench-v1_0.pickle"
NAS201_PTH      = "data/NAS-Bench-201-v1_1-096897.pth"
DATASET         = "cifar10"
HW_PLATFORM     = "raspi4"   # key prefix: raspi4_latency
N_EVALUATIONS   = 500
SEED            = 42
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(SEED)

    # Load HW-NAS-Bench (always available — pickle is in the repo)
    hw_loader = HWNASLoader(data_path=HW_NAS_PICKLE)

    # Load NAS-Bench-201 for accuracy (requires the .pth download)
    USE_ACCURACY = os.path.exists(NAS201_PTH)
    if USE_ACCURACY:
        from src.benchmark.nasbench201_loader import NASBench201Loader
        acc_loader = NASBench201Loader(pth_path=NAS201_PTH)
        print("[Main] Accuracy data available.")
    # Inspect architecture 0 to confirm everything works
        acc_loader.inspect(0, DATASET)
    # Then test get_accuracy on a few indices
        for test_idx in [0, 1, 2]:
            acc = acc_loader.get_accuracy(test_idx, DATASET)
            print(f"  Arch {test_idx} accuracy: {acc:.2f}%")
        print("[Main] Accuracy queries working. Starting search...\n")
    else:
        print("[Main] WARNING: NAS-Bench-201 .pth not found. "
              "Accuracy will be set to 0. Download it to enable.")

    # ── Random Search ────────────────────────────────────────
    all_solutions = []

    for _ in tqdm(range(N_EVALUATIONS), desc="Random Search"):
        idx = hw_loader.random_architecture()

        # Hardware metrics — always available
        hw_metrics = hw_loader.get_hw_metrics(idx, dataset=DATASET)
        latency = hw_metrics[f"{HW_PLATFORM}_latency"]

        # Accuracy — only if .pth is downloaded
        accuracy = acc_loader.get_accuracy(idx, DATASET) if USE_ACCURACY else 0.0

        all_solutions.append({
            "arch_index": idx,
            "accuracy":   accuracy,
            "latency":    latency,
            "hw_metrics": hw_metrics,   # store everything for later analysis
        })

    # ── Pareto Front ─────────────────────────────────────────
    pareto = compute_pareto_front(all_solutions)

    print(f"\nEvaluated:        {N_EVALUATIONS} architectures")
    print(f"Pareto front size: {len(pareto)}")
    if USE_ACCURACY:
        print(f"Best accuracy:     {max(s['accuracy'] for s in pareto):.2f}%")
    print(f"Best latency:      {min(s['latency'] for s in pareto):.4f} ms")

    # ── Hypervolume ───────────────────────────────────────────
    if USE_ACCURACY:
        max_latency = max(s["latency"] for s in all_solutions)
        ref_point = (max_latency * 1.1, 0.0)
        hv = hypervolume_2d(pareto, ref_point)
        print(f"Hypervolume:       {hv:.4f}")

    # ── Plot ──────────────────────────────────────────────────
    plot_pareto_front(
        all_solutions=all_solutions,
        pareto_front=pareto,
        title=f"Random Search — {DATASET} / {HW_PLATFORM}",
        save_path=f"results/pareto_random_{DATASET}_{HW_PLATFORM}.png"
    )

    # ── Save ──────────────────────────────────────────────────
    # Remove hw_metrics from JSON to keep it readable
    output = {
        "config": {
            "dataset": DATASET,
            "hw_platform": HW_PLATFORM,
            "n_evaluations": N_EVALUATIONS,
            "seed": SEED,
        },
        "pareto_front": [
            {"arch_index": s["arch_index"],
             "accuracy": s["accuracy"],
             "latency": s["latency"]}
            for s in pareto
        ],
        "stats": {
            "pareto_size": len(pareto),
            "best_accuracy": max(s["accuracy"] for s in pareto) if USE_ACCURACY else None,
            "best_latency": min(s["latency"] for s in pareto),
        }
    }
    os.makedirs("results", exist_ok=True)
    with open(f"results/random_search_{DATASET}_{HW_PLATFORM}.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nDone. Results saved to results/")