"""Plotting helpers for the DBSCAN workflow."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_k_distance_curve(sorted_curve: np.ndarray,
                          eps_used: float,
                          knee_info: Dict[str, float],
                          filename: str) -> None:
    if sorted_curve is None or sorted_curve.size == 0:
        raise ValueError("sorted_curve must include at least one k-distance value.")
    if not np.isfinite(sorted_curve).all():
        raise ValueError("sorted_curve contains NaN or infinite values.")
    if eps_used <= 0:
        raise ValueError("eps_used must be positive for k-distance plotting.")
    if not filename:
        raise ValueError("filename is required for k-distance plot.")

    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = np.arange(1, sorted_curve.size + 1)
    ax.plot(x_axis, sorted_curve, color='#1f77b4', linewidth=2.0, label='k-distance (sorted)')
    ax.fill_between(x_axis, sorted_curve, alpha=0.15, color='#1f77b4')
    ax.axhline(eps_used, color='#d62728', linestyle='--', linewidth=1.6, label=f'eps = {eps_used:.4f}')
    if knee_info:
        knee_x = knee_info.get('index', 0) + 1
        knee_y = knee_info.get('value', float(sorted_curve[min(knee_x - 1, sorted_curve.size - 1)]))
        ax.scatter([knee_x], [knee_y], color='#000000', zorder=5, s=40, label='knee point')
        ax.annotate(f"knee = {knee_y:.4f}",
                    xy=(knee_x, knee_y),
                    xytext=(knee_x + max(5, 0.03 * x_axis.size), knee_y),
                    arrowprops={'arrowstyle': '->', 'color': '#000000', 'lw': 1.0},
                    fontsize=9)
    ax.set_xlabel("Sorted sample index", fontsize=11)
    ax.set_ylabel("k-distance (Tanimoto)", fontsize=11)
    ax.set_title("k-distance curve and eps suggestion", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("DBSCAN k-distance plot saved to:", filename)


def plot_dbscan_clusters(projection: np.ndarray,
                         labels: np.ndarray,
                         classification: Sequence[str],
                         sample_labels: Sequence[str],
                         filename: str) -> None:
    if projection is None or projection.ndim != 2 or projection.shape[1] != 2:
        raise ValueError("projection must be a 2D array with exactly two components for plotting.")
    if labels is None or labels.size != projection.shape[0]:
        raise ValueError("labels must match projection rows.")
    if len(classification) != projection.shape[0]:
        raise ValueError("classification length must match projection rows.")
    if len(sample_labels) != projection.shape[0]:
        raise ValueError("sample_labels length must match projection rows.")
    if not filename:
        raise ValueError("filename is required for the DBSCAN cluster plot.")

    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    max_clusters = max(len([lbl for lbl in unique_labels if lbl != -1]), 1)
    cmap = plt.cm.get_cmap('tab20', max_clusters)
    palette: List = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in np.linspace(0, 1, max_clusters)]
    markers = {'core': 'o', 'border': 's', 'noise': 'X'}
    sizes = {'core': 70, 'border': 55, 'noise': 80}
    classification_array = np.asarray(classification)

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        color = '#7f7f7f' if label == -1 else palette[idx % len(palette)]
        for category in ('core', 'border', 'noise'):
            submask = mask & (classification_array == category)
            if not np.any(submask):
                continue
            sc = ax.scatter(projection[submask, 0],
                            projection[submask, 1],
                            c=color,
                            marker=markers[category],
                            s=sizes[category],
                            alpha=0.85 if label != -1 else 0.7,
                            linewidths=0.6,
                            edgecolors='white',
                            label=f"cluster {label} ({category})" if label != -1 else "noise")
            sc.set_gid(f"cluster-{label}-{category}")

    for point, sample_label in zip(projection, sample_labels):
        ax.annotate(str(sample_label),
                    (point[0], point[1]),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6,
                    color='#1a1a1a',
                    alpha=0.9)

    ax.set_xlabel("Component 1 (local PCA)", fontsize=11)
    ax.set_ylabel("Component 2 (local PCA)", fontsize=11)
    ax.set_title("DBSCAN clustering (PCA projection)", fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.4)
    handles, labels_list = ax.get_legend_handles_labels()
    seen_labels = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels_list):
        if label not in seen_labels:
            seen_labels.add(label)
            filtered_handles.append(handle)
            filtered_labels.append(label)
    if filtered_handles:
        ax.legend(filtered_handles,
                  filtered_labels,
                  loc='upper right',
                  frameon=True,
                  fontsize=8,
                  title="Legend",
                  title_fontsize=9)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("DBSCAN cluster plot saved to:", filename)
