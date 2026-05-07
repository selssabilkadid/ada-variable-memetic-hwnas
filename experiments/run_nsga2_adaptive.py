import os
import numpy as np

from src.metaheuristic.nsga2.core.nsga2 import NSGA2
from src.benchmark.hwnas_problem_def import HWNASProblem
from src.benchmark.nasbench201csv_loader import NASBench201CSVLoader
from src.benchmark.hwnas_loader import HWNASLoader


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DATASETS = ["cifar10", "cifar100", "ImageNet16-120"]

DEVICES = [
    "raspi4",
    "edgegpu",
    "edgetpu",
    "pixel3",
    "fpga",
    "eyeriss",
]

POP_SIZE = 100
GENERATIONS = 50
SEED = 42


# ------------------------------------------------------------
# CORE RUN
# ------------------------------------------------------------

def run_experiment(acc_loader, hw_loader, dataset, device, adaptive: bool):

    print("\n" + "=" * 70)
    print(f"NSGA-II | dataset={dataset} | device={device} | adaptive={adaptive}")
    print("=" * 70)

    # ---- problem -----------------------------------------------
    problem = HWNASProblem(
        acc_loader=acc_loader,
        hw_loader=hw_loader,
        dataset=dataset,
        platform=device,
    )

    # ---- adaptive config ---------------------------------------
    adaptive_config = None
    if adaptive:
        adaptive_config = {
            "initial_probs": np.array([0.33, 0.33, 0.34]),
            "update_period": 5,
        }

    # ---- algorithm ---------------------------------------------
    algo = NSGA2(
        problem=problem,
        pop_size=POP_SIZE,
        n_var=problem.n_var,
        n_obj=problem.n_obj,
        num_choices=problem.num_choices,
        seed=SEED,
        adaptive_mutation=adaptive_config,
    )

    # ---- run ----------------------------------------------------
    pareto_set, pareto_front = algo.run(
        generations=GENERATIONS,
        verbose=False,
    )

    # ---- results ------------------------------------------------
    acc = -pareto_front[:, 0]
    lat = pareto_front[:, 1]

    result = {
        "dataset": dataset,
        "device": device,
        "adaptive": adaptive,
        "pareto_size": len(pareto_set),
        "best_acc": float(np.max(acc)),
        "best_lat": float(np.min(lat)),
    }

    # ---- save ---------------------------------------------------
    out_dir = f"outputs/nsga2/{dataset}/{device}/{'adaptive' if adaptive else 'baseline'}"
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "pareto_set.npy"), pareto_set)
    np.save(os.path.join(out_dir, "pareto_front.npy"), pareto_front)

    print(f"Done | PF size={result['pareto_size']} | "
          f"acc={result['best_acc']:.2f}% | lat={result['best_lat']:.3f}ms")

    return result


# ------------------------------------------------------------
# FULL SWEEP
# ------------------------------------------------------------

def run_full_benchmark(adaptive: bool):

    acc_loader = NASBench201CSVLoader()
    hw_loader = HWNASLoader()

    all_results = []

    for dataset in DATASETS:
        for device in DEVICES:

            res = run_experiment(
                acc_loader=acc_loader,
                hw_loader=hw_loader,
                dataset=dataset,
                device=device,
                adaptive=adaptive,
            )

            all_results.append(res)

    # ---- summary -----------------------------------------------
    print("\n" + "=" * 70)
    print(f"SUMMARY | adaptive={adaptive}")
    print("=" * 70)

    for r in all_results:
        print(
            f"{r['dataset']:15} | {r['device']:10} | "
            f"PF={r['pareto_size']:4d} | "
            f"acc={r['best_acc']:.2f} | lat={r['best_lat']:.3f}"
        )

    return all_results


# ------------------------------------------------------------
# ENTRY POINT
# ------------------------------------------------------------

if __name__ == "__main__":

    print("\n### BASELINE RUN ###")
    run_full_benchmark(adaptive=False)

    print("\n### ADAPTIVE RUN ###")
    run_full_benchmark(adaptive=True)