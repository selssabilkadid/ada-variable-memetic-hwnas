from __future__ import annotations
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from .style import METRIC_STYLE, LABEL_SIZE, TITLE_SIZE, DPI

def plot_gd(
    metric_values: List[float],             # List of metric values, one per generation.
    title: str = "Generational Distance",
    xlabel: str = "Generation",
    ylabel: str = "GD",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Plot GD across generations."""

    # Styling
    s = METRIC_STYLE["GD"]
    fig, ax = plt.subplots(figsize=figsize)

    # Plot
    generations = np.arange(1, len(metric_values) + 1)
    ax.plot(generations, metric_values, color=s["color"], linewidth=2, marker=s["marker"], markersize=3)
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Export
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"GD plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig
