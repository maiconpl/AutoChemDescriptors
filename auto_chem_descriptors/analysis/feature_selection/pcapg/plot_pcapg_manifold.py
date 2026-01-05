"""Topological embedding visualization for PCAPG."""

from __future__ import annotations

from typing import Sequence

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from scipy import linalg
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform


def plot_pcapg_manifold(embedding: np.ndarray,
                        similarity: np.ndarray,
                        labels: Sequence[str],
                        filename: str) -> str:
    """Plot the learned embedding with possibilistic graph overlays."""
    if embedding.ndim != 2 or embedding.shape[1] == 0:
        raise ValueError("PCAPG manifold plot requires at least one latent component.")
    if similarity.shape[0] != embedding.shape[0]:
        raise ValueError("Similarity matrix size must match embedding samples.")

    coords = _refine_layout(embedding, similarity)
    degrees = similarity.sum(axis=1)
    norm = Normalize(vmin=float(np.min(degrees)), vmax=float(np.max(degrees)))
    cmap = get_cmap('viridis')
    node_colors = _resolve_node_colors(degrees, norm, cmap)
    metric_caption = _format_degree_caption(degrees)

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=320, facecolor='#f5f6fa')
    ax.set_facecolor('#ffffff')
    _draw_edges(ax, coords, similarity)
    scatter = ax.scatter(coords[:, 0],
                         coords[:, 1],
                         s=80,
                         c=node_colors,
                         edgecolor='#111111',
                         linewidth=0.6,
                         alpha=0.92)
    for idx, label in enumerate(labels):
        ax.text(coords[idx, 0],
                coords[idx, 1],
                f" {label}",
                fontsize=8,
                color='#1f1f1f',
                ha='left',
                va='center')
    ax.set_xlabel("PCAPG component 1", fontsize=12)
    ax.set_ylabel("PCAPG component 2", fontsize=12)
    ax.grid(True, linestyle=(0, (2, 4)), linewidth=0.5, alpha=0.35)
    ax.set_title("Possibilistic graph-preserving embedding", fontsize=17, pad=12)
    fig.text(0.5,
             0.92,
             metric_caption,
             ha='center',
             fontsize=10,
             color='#444444')

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Graph degree (adaptive typicality)", fontsize=11)

    fig.tight_layout()
    fig.savefig(filename,
                dpi=320,
                bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print("PCAPG manifold topology plot saved to:", filename)
    return filename


def _refine_layout(embedding: np.ndarray,
                   similarity: np.ndarray,
                   iterations: int = 280) -> np.ndarray:
    """Apply a force-directed refinement to minimize edge crossings."""
    n_samples, n_components = embedding.shape
    if n_components >= 2:
        coords = embedding[:, :2].copy()
    else:
        coords = np.pad(embedding[:, :1], ((0, 0), (0, 1)), mode='constant')

    coords = _spectral_initialization(similarity, coords)
    coords -= np.mean(coords, axis=0, keepdims=True)
    std = np.std(coords, axis=0, keepdims=True) + 1e-6
    coords /= std

    sym_sim = 0.5 * (similarity + similarity.T)
    max_weight = float(np.max(sym_sim))
    if max_weight > 0:
        sym_sim /= max_weight

    temperature = 1.0
    for step_idx in range(iterations):
        diff = coords[:, None, :] - coords[None, :, :]
        distance_sq = np.sum(diff**2, axis=2) + 1e-9
        np.fill_diagonal(distance_sq, np.inf)

        repulsive = 0.015 / distance_sq
        rep_force = np.sum(diff * repulsive[:, :, None], axis=1)

        attractive = sym_sim
        attr_force = np.sum(diff * attractive[:, :, None], axis=1)

        displacement = rep_force - attr_force
        step_size = 0.045 * temperature
        coords += step_size * displacement

        max_norm = np.max(np.linalg.norm(coords, axis=1))
        if np.isfinite(max_norm) and max_norm > 0:
            coords /= max_norm

        temperature *= 0.985

    coords -= np.mean(coords, axis=0, keepdims=True)
    max_norm = np.max(np.linalg.norm(coords, axis=1)) + 1e-9
    coords /= max_norm
    return coords


def _spectral_initialization(similarity: np.ndarray,
                             fallback: np.ndarray) -> np.ndarray:
    """Use a Laplacian eigenmap to obtain a low-stress starting layout."""
    sym_sim = 0.5 * (similarity + similarity.T)
    degrees = np.sum(sym_sim, axis=1)
    laplacian = np.diag(degrees) - sym_sim
    deg_diag = np.diag(np.maximum(degrees, 1e-9))
    try:
        eigvals, eigvecs = linalg.eigh(laplacian, deg_diag)
    except linalg.LinAlgError:
        return fallback
    order = np.argsort(eigvals)
    non_trivial = [idx for idx in order if eigvals[idx] > 1e-8]
    if len(non_trivial) < 2:
        return fallback
    basis = eigvecs[:, non_trivial[:2]]
    return basis


def _draw_edges(ax: plt.Axes, coords: np.ndarray, similarity: np.ndarray) -> None:
    weights = 0.5 * (similarity + similarity.T)
    np.fill_diagonal(weights, 0.0)
    if not np.any(weights > 0):
        return

    distances = squareform(pdist(coords))
    mst_edges = _minimum_spanning_edges(distances)
    local_edges = _high_confidence_edges(distances, per_node=2)
    merged = mst_edges.union(local_edges)
    if not merged:
        return

    segments = []
    strength = []
    for i, j in sorted(merged):
        weight = weights[i, j]
        if weight <= 0:
            continue
        segments.append([coords[i], coords[j]])
        strength.append(weight)
    if not segments:
        return

    norm = Normalize(vmin=min(strength), vmax=max(strength))
    lc = LineCollection(segments,
                        linewidths=0.6 + 1.6 * norm(strength),
                        colors=get_cmap('magma')(norm(strength)),
                        alpha=0.28)
    ax.add_collection(lc)


def _minimum_spanning_edges(distances: np.ndarray) -> set[tuple[int, int]]:
    """Extract MST edges on geometric distances to avoid long crossovers."""
    if np.allclose(distances, 0.0):
        return set()
    mst = minimum_spanning_tree(distances).tocoo()
    edges: set[tuple[int, int]] = set()
    for i, j, _ in zip(mst.row, mst.col, mst.data):
        if i == j:
            continue
        a, b = int(min(i, j)), int(max(i, j))
        edges.add((a, b))
    return edges


def _high_confidence_edges(distances: np.ndarray, per_node: int) -> set[tuple[int, int]]:
    """Select a few spatially closest edges per node to highlight local structure."""
    n_samples = distances.shape[0]
    edges: set[tuple[int, int]] = set()
    if per_node <= 0:
        return edges

    for i in range(n_samples):
        order = np.argsort(distances[i])
        count = 0
        for j in order:
            if i == j:
                continue
            edge = (int(min(i, j)), int(max(i, j)))
            if edge in edges:
                continue
            edges.add(edge)
            count += 1
            if count >= per_node:
                break
    return edges


def _resolve_node_colors(degrees: np.ndarray,
                         degree_norm: Normalize,
                         cmap):
    span = float(degree_norm.vmax - degree_norm.vmin) or 1.0
    normalized = (degrees - degree_norm.vmin) / span
    return cmap(np.clip(normalized, 0.0, 1.0))


def _format_degree_caption(degrees: np.ndarray) -> str:
    mean_degree = float(np.mean(degrees))
    std_degree = float(np.std(degrees))
    return f"Mean degree: {mean_degree:.2f}   |   Std dev: {std_degree:.2f}"
