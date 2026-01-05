"""Feature ranking visualization for PCAPG."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize


def plot_pcapg_feature_importance(feature_names: Sequence[str],
                                  scores: np.ndarray,
                                  ordered_indices: Sequence[int],
                                  top_k: int,
                                  filename: str) -> str:
    """Render a horizontal bar ranking of the most informative descriptors."""
    selected = list(ordered_indices[:top_k])
    if not selected:
        raise ValueError("PCAPG importance plot requires at least one descriptor.")
    readable_names = [feature_names[idx] for idx in selected]
    values = np.asarray(scores[selected], dtype=float)
    order = np.argsort(values)
    ordered_names = [readable_names[idx] for idx in order]
    ordered_values = values[order]

    height = max(5.5, 1.2 + 0.45 * len(selected))
    fig, ax = plt.subplots(figsize=(11.0, height), dpi=330, facecolor='#f4f5fb')
    cmap = cm.get_cmap('viridis')
    ranks = np.arange(1, len(selected) + 1)
    rank_norm = Normalize(vmin=float(ranks.min()), vmax=float(ranks.max()))
    colors = cmap(rank_norm(ranks))

    bars = ax.barh(np.arange(len(selected)),
                   ordered_values,
                   color=colors,
                   edgecolor='#111111',
                   linewidth=0.6)
    ax.set_facecolor('#ffffff')
    ax.set_xlabel("‖Wᵢ‖₂ (PCAPG loading norm)", fontsize=12)
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle=(0, (3, 3)), alpha=0.35)

    sm = cm.ScalarMappable(norm=rank_norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.01)
    cbar.set_label("Relative importance rank (1 = least informative)", fontsize=11)
    cbar.ax.invert_yaxis()

    max_value = float(np.max(ordered_values))
    ax.set_xlim(0, max_value * 1.1 if max_value > 0 else 1.0)
    ax.bar_label(bars,
                 labels=[f"{value:.4f}" for value in ordered_values],
                 padding=8,
                 fontsize=9,
                 color='#0b0b0b')
    fig.suptitle("PCAPG descriptor relevance", fontsize=19.8, fontweight='bold', y=0.985)
    fig.text(0.5,
             0.9295,
             f"Top {len(selected)} descriptors ranked by L₂ norm of projection rows",
             ha='center',
             fontsize=12.6,
             color='#424242')
    fig.subplots_adjust(left=0.32, right=0.94, top=0.9, bottom=0.08)
    fig.savefig(filename, dpi=330, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("PCAPG importance plot saved to:", filename)
    return filename
