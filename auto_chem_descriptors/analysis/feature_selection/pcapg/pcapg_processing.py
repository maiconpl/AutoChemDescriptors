"""Numerical core for the PCAPG (PCA + Possibilistic Graph) analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


_EPS = 1e-10


@dataclass
class PCAPGPayload:
    feature_matrix: np.ndarray
    scaled_matrix: np.ndarray
    projection_matrix: np.ndarray
    embedding: np.ndarray
    similarity_matrix: np.ndarray
    laplacian_matrix: np.ndarray
    feature_scores: np.ndarray
    ordered_indices: np.ndarray
    history: Dict[str, Sequence[float]]
    graph_profile: Dict[str, float]
    scaler: StandardScaler | None


def compute_pcapg_payload(descriptors_list: Sequence[Sequence[float]],
                          config: Dict[str, float | int | str]) -> PCAPGPayload:
    """Run the iterative optimization loop and extract summary statistics."""
    feature_matrix = _validate_feature_matrix(descriptors_list)
    scaled_matrix, scaler = _scale_features(feature_matrix,
                                            str(config.get('scaling', 'standard')))
    n_samples, n_features = scaled_matrix.shape
    n_components = _resolve_components(int(config.get('n_components', 6)),
                                       n_samples,
                                       n_features)
    alpha = float(config.get('alpha', 0.7))
    beta = float(config.get('beta', 0.05))
    lambda_reg = float(config.get('lambda_reg', 1.0))
    possibilistic = float(config.get('possibilistic_sharpness', 0.35))
    n_neighbors = int(config.get('n_neighbors', 8))
    stop_tol = float(config.get('tol', 1e-4))
    max_iter = max(3, int(config.get('max_iter', 50)))
    rng = np.random.default_rng(int(config.get('random_state', 42)))

    projection = _orthonormal_matrix(n_features, n_components, rng)
    embedding = scaled_matrix @ projection
    similarity = _initialize_similarity(embedding, n_neighbors, lambda_reg)

    history = {'objective': [], 'reconstruction': []}
    laplacian = _build_laplacian(similarity)

    for iteration in range(max_iter):
        projection = _update_projection(scaled_matrix,
                                        laplacian,
                                        projection,
                                        alpha,
                                        beta)
        embedding = scaled_matrix @ projection
        similarity = _update_similarity(embedding,
                                        n_neighbors,
                                        lambda_reg,
                                        possibilistic)
        laplacian = _build_laplacian(similarity)
        objective = _compute_objective(scaled_matrix,
                                       projection,
                                       laplacian,
                                       similarity,
                                       alpha,
                                       beta,
                                       lambda_reg)
        reconstruction = _compute_reconstruction_error(scaled_matrix, projection)
        history['objective'].append(float(objective))
        history['reconstruction'].append(float(reconstruction))
        if iteration > 0:
            prev = history['objective'][-2]
            curr = history['objective'][-1]
            if abs(prev - curr) <= stop_tol * (1.0 + abs(prev)):
                break

    feature_scores = np.linalg.norm(projection, axis=1)
    ordered_indices = np.argsort(feature_scores)[::-1]
    graph_profile = _summarize_graph(similarity)

    return PCAPGPayload(
        feature_matrix=feature_matrix,
        scaled_matrix=scaled_matrix,
        projection_matrix=projection,
        embedding=embedding,
        similarity_matrix=similarity,
        laplacian_matrix=laplacian,
        feature_scores=feature_scores,
        ordered_indices=ordered_indices,
        history=history,
        graph_profile=graph_profile,
        scaler=scaler,
    )


def _validate_feature_matrix(data: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("PCAPG expects a 2D descriptor matrix (samples x features).")
    if array.shape[0] < 3:
        raise ValueError("At least three samples are required for PCAPG.")
    if array.shape[1] < 2:
        raise ValueError("PCAPG requires at least two descriptors.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Descriptor matrix contains NaN or infinite values.")
    return array


def _scale_features(matrix: np.ndarray,
                    mode: str) -> Tuple[np.ndarray, StandardScaler | None]:
    mode = mode.lower().strip()
    if mode in ('none', 'off', ''):
        return matrix.copy(), None
    if mode not in ('standard', 'zscore', 'z-score'):
        raise ValueError("PCAPG supports 'standard' or 'none' for scaling.")
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    return scaled, scaler


def _resolve_components(requested: int, n_samples: int, n_features: int) -> int:
    limit = min(n_samples - 1, n_features)
    if limit < 2:
        raise ValueError("Insufficient rank for PCAPG: need >=2 latent components.")
    requested = max(2, requested)
    return min(requested, limit)


def _orthonormal_matrix(n_features: int, n_components: int,
                        rng: np.random.Generator) -> np.ndarray:
    basis = rng.normal(size=(n_features, n_components))
    q, _ = np.linalg.qr(basis)
    return q[:, :n_components]


def _initialize_similarity(embedding: np.ndarray,
                           n_neighbors: int,
                           lambda_reg: float) -> np.ndarray:
    distances = pairwise_distances(embedding, metric='euclidean', squared=True)
    return _construct_weight_matrix(distances, n_neighbors, lambda_reg, possibilistic=0.5)


def _update_similarity(embedding: np.ndarray,
                       n_neighbors: int,
                       lambda_reg: float,
                       possibilistic: float) -> np.ndarray:
    distances = pairwise_distances(embedding, metric='euclidean', squared=True)
    return _construct_weight_matrix(distances,
                                    n_neighbors,
                                    lambda_reg,
                                    possibilistic=possibilistic)


def _construct_weight_matrix(distances: np.ndarray,
                             n_neighbors: int,
                             lambda_reg: float,
                             possibilistic: float) -> np.ndarray:
    n_samples = distances.shape[0]
    weights = np.zeros_like(distances)
    k = max(2, min(n_neighbors, n_samples - 1))
    softness = np.clip(possibilistic, 0.05, 0.95)
    for idx in range(n_samples):
        order = np.argsort(distances[idx])[1:k + 1]
        neighbor_dist = distances[idx, order]
        if neighbor_dist.size == 0:
            continue
        scale = neighbor_dist.mean() + lambda_reg + _EPS
        confidences = np.exp(-neighbor_dist / scale)
        penalties = np.maximum(0.0, 1.0 - neighbor_dist / (neighbor_dist[-1] + _EPS))
        possibilistic_mask = penalties ** (1.0 / softness)
        raw = confidences * possibilistic_mask
        if raw.sum() <= _EPS:
            raw = np.ones_like(raw) / raw.size
        else:
            raw /= raw.sum()
        weights[idx, order] = raw
    sym_weights = 0.5 * (weights + weights.T)
    np.fill_diagonal(sym_weights, 0.0)
    return sym_weights


def _build_laplacian(similarity: np.ndarray) -> np.ndarray:
    degrees = np.sum(similarity, axis=1)
    laplacian = np.diag(degrees) - similarity
    return laplacian


def _update_projection(matrix: np.ndarray,
                       laplacian: np.ndarray,
                       current_projection: np.ndarray,
                       alpha: float,
                       beta: float) -> np.ndarray:
    gram = matrix.T @ matrix
    locality = matrix.T @ laplacian @ matrix
    row_norms = np.linalg.norm(current_projection, axis=1) + _EPS
    l21_penalty = np.diag(0.5 / row_norms)
    solver = gram + alpha * locality + beta * l21_penalty
    solver = 0.5 * (solver + solver.T)
    eigenvalues, eigenvectors = np.linalg.eigh(solver)
    order = np.argsort(eigenvalues)[:current_projection.shape[1]]
    projection = eigenvectors[:, order]
    q, _ = np.linalg.qr(projection)
    return q[:, :current_projection.shape[1]]


def _compute_objective(matrix: np.ndarray,
                       projection: np.ndarray,
                       laplacian: np.ndarray,
                       similarity: np.ndarray,
                       alpha: float,
                       beta: float,
                       lambda_reg: float) -> float:
    reconstruction = _compute_reconstruction_error(matrix, projection)
    locality = np.trace(projection.T @ matrix.T @ laplacian @ matrix @ projection)
    sparsity = np.sum(np.sqrt(np.sum(projection**2, axis=1) + _EPS))
    graph_energy = np.sum(similarity**2)
    return reconstruction + alpha * locality + beta * sparsity + lambda_reg * graph_energy


def _compute_reconstruction_error(matrix: np.ndarray,
                                  projection: np.ndarray) -> float:
    reconstruction = matrix @ projection @ projection.T
    residual = matrix - reconstruction
    return float(np.sum(residual**2) / matrix.size)


def _summarize_graph(similarity: np.ndarray) -> Dict[str, float]:
    degrees = similarity.sum(axis=1)
    edge_count = np.count_nonzero(similarity > 0) / 2.0
    density = 0.0
    if similarity.shape[0] > 1:
        density = float(edge_count / (similarity.shape[0] * (similarity.shape[0] - 1) / 2.0))
    return {
        'avg_degree': float(np.mean(degrees)),
        'max_degree': float(np.max(degrees)),
        'min_degree': float(np.min(degrees)),
        'density': density,
        'edges': float(edge_count),
    }
