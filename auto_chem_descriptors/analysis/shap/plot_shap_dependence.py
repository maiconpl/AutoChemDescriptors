"""Multi-panel SHAP dependence visualization."""

from __future__ import annotations

from typing import Sequence

import math
import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_shap_dependence_grid(feature_names: Sequence[str],
                              features: np.ndarray,
                              shap_values: np.ndarray,
                              targets: np.ndarray,
                              ordered_indices: Sequence[int],
                              max_panels: int,
                              filename: str) -> str:
    if len(ordered_indices) == 0:
        fig, ax = plt.subplots(figsize=(6.0, 4.0), dpi=350)
        ax.axis('off')
        ax.text(0.5, 0.5, "No descriptors available for dependence plots.", ha='center', va='center', fontsize=12)
        fig.savefig(filename, dpi=350, bbox_inches='tight')
        plt.close(fig)
        return filename

    panel_indices = ordered_indices[:max(1, min(max_panels, len(ordered_indices)))]
    n_panels = len(panel_indices)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows,
                             ncols,
                             figsize=(8.5 * ncols, 5.8 * nrows),
                             dpi=350,
                             facecolor='#f8f9fb')
    axes = np.array(axes, ndmin=2)

    for idx, feature_idx in enumerate(panel_indices):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        _render_dependence_panel(
            ax,
            feature_names,
            features,
            shap_values,
            targets,
            feature_idx,
            add_colorbar=(idx == 0),
        )

    # Remove unused axes
    total_axes = nrows * ncols
    for idx in range(n_panels, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    fig.suptitle("SHAP dependence (top descriptors)", fontsize=18, fontweight='bold', y=0.99)
    fig.subplots_adjust(hspace=0.32, wspace=0.22)
    fig.savefig(filename, dpi=350, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("SHAP dependence grid saved to:", filename)
    return filename


def _render_dependence_panel(ax: plt.Axes,
                             feature_names: Sequence[str],
                             features: np.ndarray,
                             shap_values: np.ndarray,
                             targets: np.ndarray,
                             feature_index: int,
                             add_colorbar: bool) -> None:
    x = features[:, feature_index]
    y = shap_values[:, feature_index]
    ax.set_facecolor('#ffffff')
    cmap = cm.get_cmap('viridis')
    norm = None
    if targets is not None and len(targets):
        vmin = float(np.nanmin(targets))
        vmax = float(np.nanmax(targets))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        ax.scatter(x,
                   y,
                   c=targets,
                   cmap=cmap,
                   norm=norm,
                   s=34,
                   alpha=0.85,
                   edgecolors='white',
                   linewidths=0.25)
        if add_colorbar:
            cbar = plt.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.098, pad=0.01)
            cbar.set_label('Target value', fontsize=10)
            cbar.ax.tick_params(labelsize=9)
    else:
        ax.scatter(x,
                   y,
                   color='#0B84A5',
                   s=34,
                   alpha=0.8,
                   edgecolors='white',
                   linewidths=0.25)
    ax.set_xlabel(f"{feature_names[feature_index]} value", fontsize=11)
    ax.set_ylabel("SHAP", fontsize=11)
    ax.set_title(feature_names[feature_index], fontsize=14, loc='left', fontweight='bold')
    ax.axhline(0.0, color='#5a5a5a', linestyle='--', linewidth=0.9)
    ax.grid(alpha=0.35, linestyle=(0, (2, 4)), color='#909090')
    _plot_running_mean(ax, x, y)


def _plot_running_mean(ax: plt.Axes, x: np.ndarray, y: np.ndarray, segments: int = 15) -> None:
    if x.size < 5:
        return
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    bins = np.array_split(np.arange(x_sorted.size), min(segments, x_sorted.size))
    xs, ys = [], []
    for bin_idx in bins:
        if bin_idx.size == 0:
            continue
        xs.append(float(np.mean(x_sorted[bin_idx])))
        ys.append(float(np.mean(y_sorted[bin_idx])))
    ax.plot(xs,
            ys,
            color='#d62728',
            linewidth=1.6,
            marker='o',
            markersize=3.2,
            alpha=0.9,
            label='Local trend')
    ax.legend(loc='best', fontsize=8, frameon=False)
