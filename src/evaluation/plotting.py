import matplotlib.pyplot as plt
import os


def plot_pareto_front(all_solutions: list, pareto_front: list,
                      title: str = "Pareto Front",
                      save_path: str = None):
    """
    Plots all evaluated solutions and highlights the Pareto front.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # All solutions
    if all_solutions:
        ax.scatter(
            [s["latency"] for s in all_solutions],
            [s["accuracy"] for s in all_solutions],
            alpha=0.3, color="steelblue", s=15, label="All solutions"
        )

    # Pareto front
    pareto_sorted = sorted(pareto_front, key=lambda x: x["latency"])
    ax.scatter(
        [s["latency"] for s in pareto_sorted],
        [s["accuracy"] for s in pareto_sorted],
        color="red", s=40, zorder=5, label="Pareto front"
    )
    ax.plot(
        [s["latency"] for s in pareto_sorted],
        [s["accuracy"] for s in pareto_sorted],
        color="red", linewidth=1, alpha=0.6
    )

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Saved to {save_path}")
    else:
        plt.show()
    plt.close()