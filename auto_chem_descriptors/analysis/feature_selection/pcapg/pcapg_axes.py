"""Helper utilities for selecting projection axes in PCAPG plots."""

from __future__ import annotations

import numpy as np


def select_projection_axes(embedding: np.ndarray,
                           component_order: np.ndarray | None) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    """Return 2D coordinates and component indices for visualization."""
    if embedding.ndim != 2:
        raise ValueError("Embedding array must be 2D.")
    n_samples, n_components = embedding.shape
    if n_components == 0:
        raise ValueError("Embedding must contain at least one component.")

    order = np.asarray(component_order, dtype=int) if component_order is not None else np.arange(n_components)
    order = order[(order >= 0) & (order < n_components)]
    if order.size == 0:
        order = np.arange(n_components)

    first = int(order[0])
    second = int(order[1]) if order.size > 1 else first

    coords = np.column_stack((embedding[:, first], embedding[:, second]))
    if first == second:
        coords[:, 1] = 0.0

    axis_labels = (first + 1, second + 1)
    component_pair = (first, second)
    return coords.copy(), axis_labels, component_pair
