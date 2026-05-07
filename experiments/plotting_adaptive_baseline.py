import os
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# CONFIG (must match your experiment script)
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

BASE_DIR = "outputs/nsga2"


# ------------------------------------------------------------
# LOADERS
# ------------------------------------------------------------

def load_front(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


# ------------------------------------------------------------
# PLOTTING HELPERS
# ------------------------------------------------------------

def plot_pareto(front, title, save_path):
    acc = -front[:, 0]
    lat = front[:, 1]

    plt.figure(figsize=(6, 5))
    plt.scatter(lat, acc, s=25, alpha=0.8)

    # connect for readability
    idx = np.argsort(lat)
    plt.plot(lat[idx], acc[idx], linewidth=1, alpha=0.6)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_comparison(base_front, adapt_front, title, save_path):
    acc_b, lat_b = -base_front[:, 0], base_front[:, 1]
    acc_a, lat_a = -adapt_front[:, 0], adapt_front[:, 1]

    plt.figure(figsize=(6, 5))

    plt.scatter(lat_b, acc_b, label="Baseline", alpha=0.7)
    plt.scatter(lat_a, acc_a, label="Adaptive", alpha=0.7)

    plt.xlabel("Latency (ms)")
    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------

def main():

    for dataset in DATASETS:
        for device in DEVICES:

            base_dir = os.path.join(BASE_DIR, dataset, device)

            baseline_path = os.path.join(base_dir, "baseline", "pareto_front.npy")
            adaptive_path = os.path.join(base_dir, "adaptive", "pareto_front.npy")

            baseline = load_front(baseline_path)
            adaptive = load_front(adaptive_path)

            print(f"\n[{dataset} | {device}]")

            # skip if missing
            if baseline is None:
                print("  Missing baseline")
                continue
            if adaptive is None:
                print("  Missing adaptive")
                continue

            # --------------------------------------------------------
            # 1. Individual Pareto plots
            # --------------------------------------------------------

            plot_pareto(
                baseline,
                title=f"Baseline Pareto | {dataset} | {device}",
                save_path=os.path.join(base_dir, "plots", "baseline_pareto.png"),
            )

            plot_pareto(
                adaptive,
                title=f"Adaptive Pareto | {dataset} | {device}",
                save_path=os.path.join(base_dir, "plots", "adaptive_pareto.png"),
            )

            # --------------------------------------------------------
            # 2. Comparison plot
            # --------------------------------------------------------

            plot_comparison(
                baseline,
                adaptive,
                title=f"Pareto Comparison | {dataset} | {device}",
                save_path=os.path.join(base_dir, "plots", "comparison.png"),
            )

            print("  Plots saved.")


# ------------------------------------------------------------
# ENTRY
# ------------------------------------------------------------

if __name__ == "__main__":
    main()