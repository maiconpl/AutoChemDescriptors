"""Standalone beeswarm visualization for SHAP values."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def plot_shap_beeswarm(feature_names: Sequence[str],
                       features: np.ndarray,
                       shap_values: np.ndarray,
                       indices: Sequence[int],
                       filename: str) -> str:
    fig, ax = plt.subplots(figsize=(8.4, 7.8), dpi=350, facecolor='#f8f9fb')
    _render_beeswarm(ax, feature_names, features, shap_values, indices)
    fig.suptitle("SHAP distribution across molecules", fontsize=17, fontweight='bold', y=0.97)
    fig.savefig(filename, dpi=350, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("SHAP beeswarm plot saved to:", filename)
    return filename


def _render_beeswarm(ax: plt.Axes,
                     feature_names: Sequence[str],
                     features: np.ndarray,
                     shap_values: np.ndarray,
                     indices: Sequence[int]) -> None:
    if len(indices) == 0:
        _render_empty(ax, "Insufficient descriptors to render beeswarm plot.")
        return
    rng = np.random.default_rng(0)
    subset_features = features[:, indices]
    flat_values = subset_features.ravel()
    vmin = np.nanmin(flat_values)
    vmax = np.nanmax(flat_values)
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap('coolwarm')
    ax.set_facecolor('#ffffff')
    x_values = shap_values[:, indices]
    shap_span = float(np.nanmax(np.abs(x_values))) if np.size(x_values) else 1.0
    if not np.isfinite(shap_span) or shap_span == 0.0:
        shap_span = 1.0
    ax.set_xlim(-1.15 * shap_span, 1.15 * shap_span)
    for pos, idx in enumerate(indices):
        shap_column = shap_values[:, idx]
        feature_column = features[:, idx]
        spread = rng.uniform(-0.27, 0.27, size=shap_column.shape[0])
        ax.scatter(shap_column,
                   np.full_like(shap_column, fill_value=pos) + spread,
                   c=feature_column,
                   cmap=cmap,
                   norm=norm,
                   s=26,
                   alpha=0.8,
                   edgecolors='none')
    ax.set_yticks(np.arange(len(indices)))
    ax.set_yticklabels([feature_names[idx] for idx in indices], fontsize=10)
    ax.set_xlabel("SHAP contribution", fontsize=12)
    ax.set_ylabel("Descriptors (ranked by |SHAP|)", fontsize=11)
    ax.set_title("Positive values increase the prediction", fontsize=14, loc='left', fontweight='bold')
    ax.axvline(0.0, color='#5a5a5a', linestyle='--', linewidth=0.9)
    ax.grid(axis='x', linestyle=(0, (2, 3)), alpha=0.35, color='#808080')
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label('Descriptor magnitude', fontsize=10)
    cbar.ax.tick_params(labelsize=9)


def _render_empty(ax: plt.Axes, message: str) -> None:
    ax.axis('off')
    ax.text(0.5,
            0.5,
            message,
            ha='center',
            va='center',
            fontsize=11,
            fontweight='bold')
