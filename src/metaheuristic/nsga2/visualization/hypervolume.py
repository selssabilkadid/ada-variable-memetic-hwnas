from __future__ import annotations
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from .style import LABEL_SIZE, TITLE_SIZE, DPI

def plot_hypervolume_indicator(
    front: np.ndarray,                       # (N, 2) array of objectives.
    ref_point: np.ndarray,                   # reference point.
    true_front: Optional[np.ndarray] = None, # optional (P, 2) array of true Pareto-optimal objectives.
    hv_value: Optional[float] = None,        # optional pre-calculated HV value to display.
    title: str = "Hypervolume Indicator",
    xlabel: str = "$f_1$",
    ylabel: str = "$f_2$",
    save_path: Optional[str] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
    color: str = "#3B82F6",               # color of the solution points.
    shade_color: str = "#3B82F6",         # color of the dominated region.
    shade_alpha: float = 0.25,              # transparency of the dominated region.
) -> plt.Figure:
    """Plot the 2D dominated region (hypervolume) bounded by the front and ref point."""

    front = np.asarray(front, dtype=np.float64)
    ref_point = np.asarray(ref_point, dtype=np.float64)

     # Sort by first objective
    sorted_front = front[np.argsort(front[:, 0])]
    fig, ax = plt.subplots(figsize=figsize)

    # Build staircase polygon vertices
    poly_x, poly_y = [ref_point[0]], [ref_point[1]]
    prev_y = ref_point[1]
    for i in range(len(sorted_front)):
        x_i, y_i = sorted_front[i]
        poly_x.append(x_i)
        poly_y.append(prev_y)
        poly_x.append(x_i)
        poly_y.append(y_i)
        prev_y = y_i
    poly_x.append(ref_point[0])
    poly_y.append(prev_y)

    # Fill dominated region
    ax.fill(poly_x, poly_y, color=shade_color, alpha=shade_alpha, label="Hypervolume")

    # Draw boundary line
    bound_x, bound_y = list(poly_x), list(poly_y)
    bound_x.append(sorted_front[0, 0])
    bound_y.append(sorted_front[0, 1])
    ax.plot(bound_x, bound_y, color=shade_color, linewidth=1.2, linestyle="--", alpha=0.6)

    # True Pareto front
    if true_front is not None:
        idx = np.argsort(true_front[:, 0])
        ax.plot(true_front[idx, 0], true_front[idx, 1], "k--", linewidth=1.5,
                label="True Pareto Front", zorder=2)

    # Solutions scatter
    ax.scatter(sorted_front[:, 0], sorted_front[:, 1], c=color, s=40, alpha=0.9,
               edgecolors="white", linewidth=0.8, label="NSGA-II solutions", zorder=3)

    # Reference point 
    ax.plot(ref_point[0], ref_point[1], marker="X", color="#EF4444",
            markersize=10, markeredgewidth=1, linestyle="None", zorder=4)
    ax.plot([ref_point[0], ref_point[0]], [0, ref_point[1]],
            color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.plot([0, ref_point[0]], [ref_point[1], ref_point[1]],
            color="#EF4444", linewidth=0.8, linestyle=":", alpha=0.5)
    ref_handle = mlines.Line2D([], [], color="#EF4444", marker="X", linestyle="None",
                               markersize=10, markeredgewidth=2,
                               label=f"Ref ({ref_point[0]:.2f}, {ref_point[1]:.2f})")
    ax.add_artist(ref_handle)

    # HV value text box
    if hv_value is not None:
        ax.text(0.02, 0.95, f"HV = {hv_value:.4f}", transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))

    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Export
    if save_path:
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")
        print(f"Hypervolume indicator saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()
    return fig
