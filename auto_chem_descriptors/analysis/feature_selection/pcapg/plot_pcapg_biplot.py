"""Projected loading biplot for PCAPG."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm

from .pcapg_axes import select_projection_axes


def plot_pcapg_biplot(embedding: np.ndarray,
                      projection: np.ndarray,
                      ordered_indices: Sequence[int],
                      feature_names: Sequence[str],
                      top_k: int,
                      component_order: np.ndarray | None,
                      filename: str) -> str:
    """Overlay top descriptor loadings as arrows on the manifold coordinates."""
    if projection.ndim != 2:
        raise ValueError("Projection matrix must be 2D.")
    coords, axis_labels, component_pair = select_projection_axes(embedding, component_order)
    comp_a, comp_b = component_pair
    selected = list(ordered_indices[:max(1, min(top_k, len(feature_names)))])
    loadings = projection[:, [comp_a, comp_b]]
    biplot_vectors = loadings[selected]

    vector_norms = np.linalg.norm(biplot_vectors, axis=1)
    max_vector = float(np.max(vector_norms)) if vector_norms.size else 0.0
    if max_vector <= 0:
        raise ValueError("PCAPG biplot requires non-zero descriptor loadings.")

    radius = float(np.max(np.linalg.norm(coords, axis=1))) if coords.size else 1.0
    if not np.isfinite(radius) or radius <= 0:
        radius = 1.0
    scale = 0.65 * radius / max_vector

    fig, ax = plt.subplots(figsize=(9.5, 8.0), dpi=330, facecolor='#f4f6fb')
    ax.set_facecolor('#ffffff')
    scatter = ax.scatter(coords[:, 0],
                         coords[:, 1],
                         s=30,
                         color='#7d91b1',
                         edgecolor='#1f1f1f',
                         linewidth=0.35,
                         alpha=0.7,
                         label='Samples')

    arrow_color = '#d63b61'
    arrow_collection = []
    for vec, feat_idx in zip(biplot_vectors, selected):
        scaled_vec = vec * scale
        arrow = ax.annotate("",
                            xy=(scaled_vec[0], scaled_vec[1]),
                            xytext=(0.0, 0.0),
                            arrowprops=dict(arrowstyle='-|>',
                                            color=arrow_color,
                                            linewidth=1.4,
                                            shrinkA=0,
                                            shrinkB=0,
                                            alpha=0.9))
        arrow_collection.append(arrow)
        label_x = scaled_vec[0] * 1.05
        label_y = scaled_vec[1] * 1.05
        ax.text(label_x,
                label_y,
                feature_names[feat_idx],
                fontsize=9.2,
                color='#2b2b2b',
                weight='bold')

    ax.axhline(0, color='#bbbbbb', linewidth=0.8, linestyle='--')
    ax.axvline(0, color='#bbbbbb', linewidth=0.8, linestyle='--')
    ax.set_xlabel(f"PCAPG component {axis_labels[0]}", fontsize=12)
    ax.set_ylabel(f"PCAPG component {axis_labels[1]}", fontsize=12)
    ax.set_title("PCAPG projected loading biplot", fontsize=18, pad=16)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.45, alpha=0.32)
    legend = ax.legend(handles=[scatter],
                       labels=['Samples'],
                       loc='upper left',
                       frameon=False,
                       fontsize=9)
    legend.get_texts()[0].set_color('#1f1f1f')
    arrow_patch = plt.Line2D([0], [0],
                             color=arrow_color,
                             marker='',
                             linewidth=1.8,
                             label='Descriptor vector')
    ax.legend(handles=[scatter, arrow_patch],
              labels=['Samples', 'Descriptor vector'],
              loc='upper left',
              frameon=False,
              fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=330, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print("PCAPG projected loading biplot saved to:", filename)
    return filename
