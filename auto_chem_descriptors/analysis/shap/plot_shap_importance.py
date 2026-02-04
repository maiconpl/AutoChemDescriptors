"""Render global SHAP importance ranking as a standalone figure."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_shap_importance(feature_names: Sequence[str],
                         importance: np.ndarray,
                         indices: Sequence[int],
                         filename: str) -> str:
    fig_height = max(6.2, 1.1 + 0.45 * len(indices))
    fig, ax = plt.subplots(figsize=(9.4, fig_height), dpi=350, facecolor='#f7f8fb')
    _render_bars(ax, feature_names, importance, indices)
    fig.suptitle("Global SHAP importance", fontsize=18, fontweight='bold', y=0.97)
    fig.text(0.5,
             0.92,
             "Top descriptors ranked by mean |SHAP|",
             fontsize=12,
             color='#444444',
             ha='center')
    fig.subplots_adjust(left=0.34, right=0.98, top=0.85, bottom=0.12)
    fig.savefig(filename, dpi=350, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("SHAP importance plot saved to:", filename)
    return filename


def _render_bars(ax: plt.Axes,
                 feature_names: Sequence[str],
                 importance: np.ndarray,
                 indices: Sequence[int]) -> None:
    if len(indices) == 0:
        _render_empty(ax, "No descriptors available for SHAP ranking.")
        return
    values = np.asarray(importance[indices], dtype=float)
    readable_names = [feature_names[idx] for idx in indices]
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_names = [readable_names[idx] for idx in order]
    positions = np.arange(len(ordered_names))
    cmap = cm.get_cmap('viridis')
    colors = cmap(np.linspace(0.15, 0.85, len(ordered_names)))
    ax.set_facecolor('#ffffff')
    bars = ax.barh(positions,
                   ordered_values,
                   color=colors,
                   edgecolor='#1b1b1b',
                   linewidth=0.5)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|", fontsize=12)
    if np.any(np.isfinite(ordered_values)):
        max_value = float(np.nanmax(ordered_values))
        span = max_value if np.isfinite(max_value) and max_value > 0 else 1.0
        ax.set_xlim(0, span * 1.06)
        labels = [f"{val:.3f}" for val in ordered_values]
        ax.bar_label(bars,
                     labels=labels,
                     padding=6,
                     fontsize=9,
                     color='#0b0b0b')
    ax.grid(axis='x', linestyle=(0, (2, 4)), alpha=0.35, color='#6b6b6b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _render_empty(ax: plt.Axes, message: str) -> None:
    ax.axis('off')
    ax.text(0.5,
            0.5,
            message,
            ha='center',
            va='center',
            fontsize=11,
            fontweight='bold')
