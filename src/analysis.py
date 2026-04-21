"""
Plotting helpers to create visualizations.

"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt

from .oed import FACTOR_DESCRIPTIONS, FACTOR_NAMES, LEVELS, OEDAnalysis


# Use a non-interactive backend so plots work without a display.
matplotlib.use("Agg", force=True)


def _savefig(fig: plt.Figure, path: Optional[str | Path]) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches="tight")


def _level_label(factor: str, level_idx: int) -> str:
    value = LEVELS[factor][level_idx]
    if factor == "B":
        return f"{value}x{value}"
    if factor == "C":
        return f"{value:.1f}"
    if factor == "D":
        a, b, c = value
        return f"{a}-{b}-({c})"
    return str(value)


def plot_range_values(analysis: OEDAnalysis, path: Optional[str | Path] = None) -> plt.Figure:
    """Recreate Figure 4: bar chart of the range value ``R`` per factor."""

    labels = [f"{f}\n{FACTOR_DESCRIPTIONS[f]}" for f in FACTOR_NAMES]
    values = [analysis.factor_summaries[f].range_value for f in FACTOR_NAMES]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color="#4C72B0")
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Range value R")
    ax.set_title("Figure 4. Range values of different factors")
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    fig.tight_layout()
    _savefig(fig, path)
    return fig


def plot_factor_means(
    analysis: OEDAnalysis,
    factor: str,
    title: str,
    path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot mean validation accuracy vs factor level (Figures 5-8)."""

    if factor not in FACTOR_NAMES:
        raise ValueError(f"Unknown factor {factor!r}")

    summary = analysis.factor_summaries[factor]
    x_labels = [_level_label(factor, i) for i in range(3)]
    y = list(summary.means)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(x_labels, y, marker="o", color="#4C72B0")
    for xi, yi in zip(x_labels, y):
        ax.annotate(f"{yi:.4f}", (xi, yi), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel(FACTOR_DESCRIPTIONS[factor])
    ax.set_ylabel("Validation accuracy (mean)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, path)
    return fig


def plot_all_factor_means(
    analysis: OEDAnalysis,
    out_dir: Optional[str | Path] = None,
) -> dict[str, plt.Figure]:
    """Produce Figures 5-8 (one per factor)."""

    titles = {
        "A": "Figure 5. Relationship between number of convolutional layers and validation accuracy",
        "B": "Figure 6. Relationship between size of filter and validation accuracy",
        "C": "Figure 7. Relationship between percentage of dropouts and validation accuracy",
        "D": "Figure 8. Relationship between number of filters and validation accuracy",
    }
    figs: dict[str, plt.Figure] = {}
    for f in FACTOR_NAMES:
        p = None if out_dir is None else Path(out_dir) / f"figure_factor_{f}.png"
        figs[f] = plot_factor_means(analysis, f, titles[f], p)
    return figs


def plot_model_comparison(
    names: Sequence[str],
    accuracies: Sequence[float],
    path: Optional[str | Path] = None,
) -> plt.Figure:
    """Bar chart comparing the three architectures (LeNet-5 / AlexNet / ResNet)."""

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(list(names), list(accuracies), color="#55A868")
    for bar, v in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylabel("Validation accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on small Pokémon dataset")
    fig.tight_layout()
    _savefig(fig, path)
    return fig


__all__ = [
    "plot_range_values",
    "plot_factor_means",
    "plot_all_factor_means",
    "plot_model_comparison",
]
