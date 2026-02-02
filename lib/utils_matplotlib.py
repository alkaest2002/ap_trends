
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Define colors
COLORS: dict[str, str] = {
    "base": "#3A4F43",
    "color_1": "orange",
    "color_2": "#00A2FF",
}


def configure_matplotlib_environment() -> Any:
    """Configure matplotlib environment for consistent plotting style."""
    # Set global matplotlib parameters
    plt.rcParams.update(
        {
            "figure.figsize": (10, 6),
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10,
            "font.size": 12,
            "font.family": "sans-serif",
        }
    )

    return plt, COLORS


def colorize_axes(ax: Axes) -> Axes:
    """Colorize matplotib axes.

    Args:
        ax: Matplotlib Axes object

    """
    ax.tick_params(color=COLORS["base"], labelcolor=COLORS["base"])
    ax.spines[:].set_color(COLORS["base"])
    ax.xaxis.label.set_color(COLORS["base"])
    ax.yaxis.label.set_color(COLORS["base"])

    return ax
