#!/usr/bin/python3
"""
Laplacian Score ranking plot.

Produces a publication-ready bar chart ranking descriptors by LS.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_ls_ranking(feature_names: Sequence[str],
                    scores: np.ndarray,
                    filename: str,
                    top_k: int) -> str:
    names, values = _prepare_top_features(scores, feature_names, top_k)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    _render_ranking(ax, names, values)

    fig.suptitle("Laplacian Score ranking", fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Laplacian Score ranking figure saved to:", filename)
    return filename


def _prepare_top_features(scores: np.ndarray,
                          names: Sequence[str],
                          top_k: int) -> Tuple[Sequence[str], np.ndarray]:
    finite_idx = np.where(np.isfinite(scores))[0]
    if finite_idx.size == 0:
        return [], np.array([])
    sorted_idx = finite_idx[np.argsort(scores[finite_idx])]
    limit = min(len(sorted_idx), max(1, top_k))
    selected = sorted_idx[:limit]
    selected_names = [str(names[idx]) for idx in selected]
    return selected_names, scores[selected]


def _render_ranking(ax: plt.Axes,
                    names: Sequence[str],
                    values: np.ndarray) -> None:
    if not names:
        ax.axis('off')
        ax.text(0.5,
                0.5,
                "Insufficient data to render this panel.",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold')
        return

    order = np.argsort(values)
    ordered_names = [names[idx] for idx in order]
    ordered_scores = values[order]
    positions = np.arange(len(ordered_names))
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0.2, 0.85, len(ordered_scores)))

    ax.barh(positions,
            ordered_scores,
            color=colors,
            edgecolor='#1f1f1f',
            linewidth=0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score (lower preserves the manifold structure)", fontsize=11)
    ax.set_title("Top descriptors by Laplacian Score", loc='left', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    max_score = np.max(ordered_scores) if np.isfinite(ordered_scores).any() else 1.0
    offset = max(0.02 * max_score, 0.02)
    for pos, value in zip(positions, ordered_scores):
        ax.text(value + offset,
                pos,
                f"{value:.4f}",
                va='center',
                ha='left',
                fontsize=8,
                color='#111111')
