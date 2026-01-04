#!/usr/bin/python3
"""
Marginal Laplacian Score ranking plot.

Produces a publication-ready chart highlighting descriptors governed by
distribution tails.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_mls_ranking(feature_names: Sequence[str],
                     scores: np.ndarray,
                     coverage: np.ndarray,
                     filename: str,
                     top_k: int) -> str:
    names, values, cov_values = _prepare_top_features(scores, coverage, feature_names, top_k)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(8.5, 6.5), dpi=300)
    _render_ranking(ax, names, values, cov_values)

    fig.suptitle("Marginal Laplacian Score ranking", fontsize=16, fontweight='bold', y=0.98)

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Marginal Laplacian Score ranking figure saved to:", filename)
    return filename


def _prepare_top_features(scores: np.ndarray,
                          coverage: np.ndarray,
                          names: Sequence[str],
                          top_k: int) -> Tuple[Sequence[str], np.ndarray, np.ndarray]:
    finite_idx = np.where(np.isfinite(scores))[0]
    if finite_idx.size == 0:
        return [], np.array([]), np.array([])
    sorted_idx = finite_idx[np.argsort(scores[finite_idx])]
    limit = min(len(sorted_idx), max(1, top_k))
    selected = sorted_idx[:limit]
    selected_names = [str(names[idx]) for idx in selected]
    return selected_names, scores[selected], coverage[selected] * 100.0


def _render_ranking(ax: plt.Axes,
                    names: Sequence[str],
                    values: np.ndarray,
                    coverage: np.ndarray) -> None:
    if not names:
        ax.axis('off')
        ax.text(0.5,
                0.5,
                "Marginal Laplacian Score is unavailable for this configuration.",
                ha='center',
                va='center',
                fontsize=11,
                fontweight='bold')
        return

    order = np.argsort(values)
    ordered_names = [names[idx] for idx in order]
    ordered_scores = values[order]
    ordered_coverage = coverage[order]
    positions = np.arange(len(ordered_names))
    cmap = plt.get_cmap('magma')
    colors = cmap(np.linspace(0.25, 0.85, len(ordered_scores)))

    ax.barh(positions,
            ordered_scores,
            color=colors,
            edgecolor='#1f1f1f',
            linewidth=0.6)
    ax.set_yticks(positions)
    ax.set_yticklabels(ordered_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Score (lower preserves marginal structure)", fontsize=11)
    ax.set_title("Descriptors prioritized by marginal tails", loc='left', fontsize=14, fontweight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.tick_params(labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    max_score = np.max(ordered_scores) if np.isfinite(ordered_scores).any() else 1.0
    offset = max(0.02 * max_score, 0.02)
    for pos, (score_value, cov_value) in enumerate(zip(ordered_scores, ordered_coverage)):
        label = f"{score_value:.4f}  |  coverage {cov_value:.1f}%"
        ax.text(score_value + offset,
                pos,
                label,
                va='center',
                ha='left',
                fontsize=8,
                color='#111111')
