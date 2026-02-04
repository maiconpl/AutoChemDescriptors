#!/usr/bin/python3
"""
Numerical core for Laplacian Score and Marginal Laplacian Score.

The functions here operate purely on numpy/scipy primitives and are
deliberately agnostic to plotting/reporting so they can be reused by
other layers (graphs, Markdown, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import sparse
from scipy.stats import skew
from sklearn.metrics import pairwise_distances


_EPS = 1e-12


@dataclass
class DescriptorProfile:
    is_binary: bool
    has_negative: bool
    min_value: float
    max_value: float


def compute_laplacian_scores(feature_matrix: np.ndarray,
                             config: Dict[str, object]) -> Dict[str, np.ndarray]:
    """
    Compute Laplacian Score (LS) and Marginal Laplacian Score (MLS) for every descriptor.

    Parameters
    ----------
    feature_matrix
        Array with shape (n_samples, n_features).
    config
        Dictionary with hyperparameters (see `_resolve_config_defaults` in the orchestrator).

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys: `ls_scores`, `mls_scores`, `skewness`,
        `margin_coverage`, `marginal_labels`, and graph metadata.
    """
    matrix = _validate_matrix(feature_matrix)
    n_samples, n_features = matrix.shape

    profile = _inspect_descriptors(matrix)
    metric_choice = _resolve_metric(str(config.get('metric', 'auto')).lower(), profile)
    distances = _compute_distance_matrix(matrix, metric_choice, profile)
    laplacian, degrees, graph_meta = _construct_graph(distances, config, metric_choice)

    centered = _center_features(matrix, degrees)
    ls_scores = _compute_laplacian_score(centered, laplacian, degrees)

    skew_values = skew(matrix, axis=0, bias=False, nan_policy='omit')
    skew_values = np.nan_to_num(skew_values, nan=0.0)

    mode = str(config.get('mode', 'both')).lower()
    compute_mls = mode in ('both', 'mls', 'marginal')
    if compute_mls:
        quantile = float(config.get('quantile', 0.90))
        quantile = min(max(quantile, 0.55), 0.99)
        skew_right = float(config.get('skew_right', 0.5))
        skew_left = float(config.get('skew_left', -0.5))
        mls_scores, margin_coverage, marginal_labels = _compute_marginal_scores(
            matrix,
            centered,
            laplacian,
            degrees,
            skew_values,
            quantile,
            skew_left,
            skew_right,
        )
    else:
        mls_scores = np.full(n_features, np.nan)
        margin_coverage = np.zeros(n_features, dtype=float)
        marginal_labels = np.array(['disabled'] * n_features, dtype='<U16')

    return {
        'ls_scores': ls_scores,
        'mls_scores': mls_scores,
        'skewness': skew_values,
        'margin_coverage': margin_coverage,
        'marginal_labels': marginal_labels,
        'graph_profile': graph_meta,
    }


def _validate_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2:
        raise ValueError("Descriptor matrix must be 2-dimensional.")
    if array.shape[0] < 2:
        raise ValueError("At least two samples are required for Laplacian ranking.")
    if array.shape[1] == 0:
        raise ValueError("Descriptor matrix must contain at least one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Descriptor matrix contains NaN or infinite values.")
    return array


def _inspect_descriptors(matrix: np.ndarray) -> DescriptorProfile:
    rounded = np.round(matrix)
    close_to_binary = np.all(np.logical_or(np.isclose(matrix, 0.0, atol=1e-8),
                                           np.isclose(matrix, 1.0, atol=1e-8)))
    is_binary = close_to_binary and np.array_equal(rounded, np.clip(rounded, 0, 1))
    has_negative = np.any(matrix < 0)
    return DescriptorProfile(
        is_binary=is_binary,
        has_negative=bool(has_negative),
        min_value=float(matrix.min()),
        max_value=float(matrix.max()),
    )


def _resolve_metric(metric: str, profile: DescriptorProfile) -> str:
    if metric in ('tanimoto', 'jaccard'):
        if profile.has_negative:
            print("Warning: Negative descriptors detected; falling back to Euclidean metric.")
            return 'euclidean'
        return 'tanimoto'
    if metric in ('euclidean', 'l2'):
        return 'euclidean'
    # Auto mode: prefer Tanimoto whenever descriptors are non-negative.
    if profile.is_binary:
        return 'tanimoto'
    if not profile.has_negative:
        return 'tanimoto'
    return 'euclidean'


def _compute_distance_matrix(matrix: np.ndarray,
                             metric: str,
                             profile: DescriptorProfile) -> np.ndarray:
    if metric == 'euclidean':
        distances = pairwise_distances(matrix, metric='euclidean')
    elif metric == 'tanimoto' and profile.is_binary:
        distances = pairwise_distances(matrix.astype(bool), metric='jaccard')
    else:
        distances = _generalized_tanimoto(matrix)
    np.fill_diagonal(distances, 0.0)
    return distances


def _generalized_tanimoto(matrix: np.ndarray) -> np.ndarray:
    # Works for non-negative descriptors; caller ensures negatives are handled elsewhere.
    matrix = np.asarray(matrix, dtype=float)
    dot_products = matrix @ matrix.T
    squared_norms = np.sum(matrix * matrix, axis=1)
    denominator = squared_norms[:, None] + squared_norms[None, :] - dot_products
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.divide(dot_products,
                               denominator,
                               out=np.zeros_like(dot_products),
                               where=denominator > _EPS)
    similarity = np.clip(similarity, 0.0, 1.0)
    distances = 1.0 - similarity
    np.fill_diagonal(distances, 0.0)
    return distances


def _construct_graph(distances: np.ndarray,
                     config: Dict[str, object],
                     metric: str) -> Tuple[sparse.csr_matrix, np.ndarray, Dict[str, float]]:
    n_samples = distances.shape[0]
    if n_samples < 2:
        raise ValueError("Cannot build a graph with fewer than two samples.")
    requested_k = int(config.get('k_neighbors', 7))
    k_neighbors = max(1, min(requested_k, n_samples - 1))

    neighbor_idx = np.argpartition(distances, kth=k_neighbors, axis=1)[:, 1:k_neighbors + 1]
    neighbor_dist = distances[np.arange(n_samples)[:, None], neighbor_idx]
    sigma_vector = neighbor_dist[:, -1]

    valid_sigma = sigma_vector[sigma_vector > _EPS]
    sigma_median = float(np.median(valid_sigma)) if valid_sigma.size else 1.0
    sigma_vector = np.where(sigma_vector > _EPS, sigma_vector, sigma_median)

    adaptive_kernel = bool(config.get('adaptive_kernel', True))
    heat_parameter = config.get('heat_parameter')
    if heat_parameter is not None:
        heat_parameter = max(float(heat_parameter), _EPS)

    if not adaptive_kernel:
        if heat_parameter is None:
            heat_parameter = max(sigma_median ** 2, _EPS)
        denom = np.full(neighbor_dist.size, heat_parameter, dtype=float)
    else:
        sigma_i = np.repeat(sigma_vector, k_neighbors)
        sigma_j = sigma_vector[neighbor_idx].ravel()
        denom = np.maximum(sigma_i * sigma_j, _EPS)

    weights = np.exp(- (neighbor_dist.ravel() ** 2) / denom)
    rows = np.repeat(np.arange(n_samples), k_neighbors)
    cols = neighbor_idx.ravel()
    affinity = sparse.csr_matrix((weights, (rows, cols)), shape=(n_samples, n_samples))
    affinity = affinity.maximum(affinity.transpose())

    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    laplacian = sparse.diags(degrees) - affinity

    meta = {
        'metric': metric,
        'k_neighbors': k_neighbors,
        'adaptive_kernel': adaptive_kernel,
        'heat_parameter': float(heat_parameter) if heat_parameter is not None else None,
        'sigma_median': sigma_median,
        'n_samples': n_samples,
    }
    return laplacian.tocsr(), degrees, meta


def _center_features(matrix: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    denominator = float(np.sum(degrees))
    if denominator <= _EPS:
        denominator = matrix.shape[0]
    weighted_mean = (degrees[:, None] * matrix).sum(axis=0) / denominator
    return matrix - weighted_mean


def _compute_laplacian_score(centered: np.ndarray,
                             laplacian: sparse.csr_matrix,
                             degrees: np.ndarray) -> np.ndarray:
    laplacian_projection = laplacian.dot(centered)
    numerator = np.einsum('ij,ij->j', centered, laplacian_projection)
    denominator = np.sum((centered ** 2) * degrees[:, None], axis=0) + _EPS
    scores = numerator / denominator
    return np.clip(scores, 0.0, np.inf)


def _compute_marginal_scores(raw_matrix: np.ndarray,
                             centered: np.ndarray,
                             laplacian: sparse.csr_matrix,
                             degrees: np.ndarray,
                             skew_values: np.ndarray,
                             quantile: float,
                             skew_left: float,
                             skew_right: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = raw_matrix.shape[1]
    mls_scores = np.full(n_features, np.nan)
    coverage = np.zeros(n_features, dtype=float)
    marginal_labels = np.empty(n_features, dtype='<U16')

    for idx in range(n_features):
        classification = _classify_feature(skew_values[idx], skew_left, skew_right)
        mask = _build_margin_mask(raw_matrix[:, idx], classification, quantile)
        marginal_labels[idx] = classification
        if mask is None or mask.sum() < 2:
            coverage[idx] = 0.0
            continue
        coverage[idx] = float(mask.mean())
        masked_vector = centered[:, idx] * mask
        laplacian_response = laplacian.dot(masked_vector)
        numerator = float(masked_vector @ laplacian_response)
        denominator = float(np.sum((masked_vector ** 2) * degrees)) + _EPS
        if denominator <= _EPS:
            continue
        mls_scores[idx] = numerator / denominator
    return np.clip(mls_scores, 0.0, np.inf), coverage, marginal_labels


def _classify_feature(skew_value: float, skew_left: float, skew_right: float) -> str:
    if skew_value >= skew_right:
        return 'right'
    if skew_value <= skew_left:
        return 'left'
    return 'two-sided'


def _build_margin_mask(values: np.ndarray, classification: str, quantile: float) -> np.ndarray | None:
    if np.allclose(values, values[0]):
        return None
    quantile = min(max(quantile, 0.55), 0.99)
    if classification == 'right':
        threshold = np.quantile(values, quantile)
        return (values >= threshold).astype(float)
    if classification == 'left':
        threshold = np.quantile(values, 1.0 - quantile)
        return (values <= threshold).astype(float)
    lower = np.quantile(values, 1.0 - quantile)
    upper = np.quantile(values, quantile)
    mask = np.logical_or(values <= lower, values >= upper)
    if not np.any(mask):
        return None
    return mask.astype(float)
