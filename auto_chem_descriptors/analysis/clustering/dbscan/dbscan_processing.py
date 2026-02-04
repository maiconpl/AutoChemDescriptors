"""Core numerical workflow for DBSCAN clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclass
class DBSCANPayload:
    """Container for the intermediate results produced by DBSCAN processing."""

    fingerprint_matrix: np.ndarray
    binary_matrix: Optional[np.ndarray]
    profile: Dict[str, Any]
    sample_labels: List[str]
    min_samples: int
    metric_mode: str
    warnings: List[str]
    sorted_k_curve: np.ndarray
    kth_distances: np.ndarray
    eps_suggested: float
    eps_used: float
    eps_source: str
    knee_info: Dict[str, Any]
    labels: np.ndarray
    classification: List[str]
    stats: Dict[str, Any]
    projection: np.ndarray
    explained_variance: np.ndarray


def compute_dbscan_payload(descriptors_list: Sequence[Sequence[Any]],
                           analysis: Optional[Dict[str, Any]],
                           config: Dict[str, Any]) -> DBSCANPayload:
    """Execute DBSCAN over the descriptor matrix and collect diagnostics."""

    fingerprints, profile = _prepare_fingerprint_matrix(descriptors_list)
    n_samples, n_bits = fingerprints.shape
    print("DBSCAN analysis: samples =", n_samples, "bits =", n_bits)
    if profile.get('shift_offsets') is not None:
        print("Descriptor columns shifted by minima to enforce non-negativity for generalized Tanimoto similarity.")

    sample_labels = _extract_labels(analysis, n_samples)
    min_samples = _resolve_min_samples(config, n_samples)
    metric_mode = _decide_metric_mode(n_samples, config, profile['is_binary'])

    warnings = _inspect_fingerprints(fingerprints)
    for message in warnings:
        print("Warning:", message)

    binary_view = fingerprints.astype(bool) if profile['is_binary'] else None
    distance_matrix = None
    if metric_mode == 'precomputed':
        _log_distance_memory_estimate(n_samples)
        distance_matrix = _compute_tanimoto_distance_matrix(fingerprints)

    n_jobs_value = _parse_optional_int(config.get('n_jobs'), "analysis['dbscan']['n_jobs']")
    sorted_k_curve, kth_distances = _compute_k_distance_profile(binary_view,
                                                                min_samples,
                                                                distance_matrix,
                                                                n_jobs_value)
    eps_suggested, knee_info = _detect_knee(sorted_k_curve)
    eps_config = config.get('eps')
    eps_used = float(eps_config) if eps_config is not None else eps_suggested
    if eps_used <= 0:
        eps_used = max(eps_suggested, 1e-6)
    eps_source = 'user_provided' if eps_config is not None else 'knee_suggestion'
    print("Metric mode:", metric_mode)
    if eps_config is None:
        print(f"eps not provided; using knee suggestion {eps_suggested:.4f}")
    else:
        print(f"eps provided by user: {eps_used:.4f} (knee suggestion {eps_suggested:.4f})")

    dbscan = _fit_dbscan(binary_view if metric_mode == 'jaccard' else fingerprints,
                         distance_matrix,
                         eps_used,
                         min_samples,
                         metric_mode,
                         config,
                         n_jobs_value)

    labels = dbscan.labels_
    classification = _classify_points(labels,
                                      getattr(dbscan, 'core_sample_indices_', []),
                                      n_samples)
    stats_payload = _summarize_clusters(labels,
                                        classification,
                                        kth_distances,
                                        eps_used,
                                        eps_suggested,
                                        min_samples,
                                        metric_mode,
                                        warnings,
                                        knee_info,
                                        n_bits)

    projection_components = config.get('projection_components')
    projection_count = _parse_optional_int(projection_components,
                                          "analysis['dbscan']['projection_components']")
    projection, explained_variance = _project_fingerprints(fingerprints,
                                                           projection_count)

    return DBSCANPayload(fingerprint_matrix=fingerprints,
                         binary_matrix=binary_view,
                         profile=profile,
                         sample_labels=sample_labels,
                         min_samples=min_samples,
                         metric_mode=metric_mode,
                         warnings=warnings,
                         sorted_k_curve=sorted_k_curve,
                         kth_distances=kth_distances,
                         eps_suggested=eps_suggested,
                         eps_used=eps_used,
                         eps_source=eps_source,
                         knee_info=knee_info,
                         labels=labels,
                         classification=classification,
                         stats=stats_payload,
                         projection=projection,
                         explained_variance=explained_variance)


def _prepare_fingerprint_matrix(data: Sequence[Sequence[Any]]) -> Tuple[np.ndarray, Dict[str, Any]]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("DBSCAN expects a 2D fingerprint matrix.")
    if array.shape[0] < 2:
        raise ValueError("DBSCAN requires at least two samples.")
    if array.shape[1] == 0:
        raise ValueError("Fingerprint matrix must include at least one bit.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Fingerprint matrix contains NaN or infinite values.")
    tolerance = 1e-8
    zero_like = np.isclose(array, 0.0, atol=tolerance)
    one_like = np.isclose(array, 1.0, atol=tolerance)
    is_binary = np.all(zero_like | one_like)
    profile: Dict[str, Any] = {'is_binary': bool(is_binary), 'shift_offsets': None}
    if is_binary:
        normalized = np.where(one_like, 1, 0)
        return normalized.astype(np.uint8), profile
    min_values = np.min(array, axis=0)
    needs_shift = min_values < 0
    shift_offsets = None
    if np.any(needs_shift):
        shift_offsets = np.abs(min_values * needs_shift)
        array = array + shift_offsets
    if np.any(array < 0):
        raise ValueError("DBSCAN fingerprints contain negative values even after shifting; cannot apply Tanimoto distance.")
    if shift_offsets is not None:
        profile['shift_offsets'] = shift_offsets.tolist()
    return array, profile


def _extract_labels(analysis: Optional[Dict[str, Any]], n_samples: int) -> List[str]:
    if isinstance(analysis, dict):
        labels = analysis.get('molecules_label')
        if isinstance(labels, list) and len(labels) == n_samples:
            return [str(label) for label in labels]
    return [f"sample_{idx + 1}" for idx in range(n_samples)]


def _parse_optional_int(value: Any, field_label: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_label} must be an integer.") from exc


def _resolve_min_samples(config: Dict[str, Any], n_samples: int) -> int:
    raw = config.get('min_samples')
    min_samples = _parse_optional_int(raw, "analysis['dbscan']['min_samples']") if raw is not None else 5
    if min_samples is None:
        min_samples = 5
    if min_samples < 2:
        min_samples = 2
    if min_samples > n_samples:
        raise ValueError("min_samples for DBSCAN cannot exceed the number of samples.")
    return min_samples


def _decide_metric_mode(n_samples: int,
                        config: Dict[str, Any],
                        is_binary: bool) -> str:
    mode = str(config.get('metric_mode', 'auto')).lower()
    if mode == 'jaccard' and not is_binary:
        print("Metric mode 'jaccard' requires binary fingerprints; falling back to 'precomputed'.")
        return 'precomputed'
    if mode in ('precomputed', 'jaccard'):
        if mode == 'jaccard' and not is_binary:
            return 'precomputed'
        return mode
    threshold_setting = config.get('precomputed_max_samples')
    threshold = _parse_optional_int(threshold_setting,
                                    "analysis['dbscan']['precomputed_max_samples']") if threshold_setting is not None else 1500
    if not is_binary:
        print("Non-binary descriptors detected; enforcing 'precomputed' metric to compute generalized Tanimoto distance.")
        return 'precomputed'
    if n_samples <= threshold:
        return 'precomputed'
    print("Switching to jaccard metric due to sample size >", threshold)
    return 'jaccard'


def _inspect_fingerprints(matrix: np.ndarray) -> List[str]:
    warnings: List[str] = []
    bit_sums = matrix.sum(axis=1)
    zero_mask = bit_sums == 0
    if np.any(zero_mask):
        warnings.append(f"{int(zero_mask.sum())} fingerprints contain only zeros; Tanimoto distance defaults to 1.0 for pairs of empty vectors.")
    _, counts = np.unique(matrix, axis=0, return_counts=True)
    if np.any(counts > 1):
        warnings.append("Identical fingerprints detected; dense clusters may collapse into single cores.")
    return warnings


def _log_distance_memory_estimate(n_samples: int) -> None:
    bytes_needed = n_samples ** 2 * 8
    gigabytes = bytes_needed / (1024 ** 3)
    print(f"Estimated memory for distance matrix: {gigabytes:.3f} GiB")


def _compute_tanimoto_distance_matrix(matrix: np.ndarray) -> np.ndarray:
    dense = matrix.astype(np.float64, copy=False)
    products = dense @ dense.T
    squared_norms = np.sum(dense * dense, axis=1)
    denominator = squared_norms[:, None] + squared_norms[None, :] - products
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.divide(products,
                               denominator,
                               out=np.zeros_like(products, dtype=np.float64),
                               where=denominator > 0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance


def _compute_k_distance_profile(binary_matrix: Optional[np.ndarray],
                                min_samples: int,
                                distance_matrix: Optional[np.ndarray],
                                n_jobs: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1 for k-distance computation.")
    if distance_matrix is not None:
        sorted_rows = np.sort(distance_matrix, axis=1)
        kth = sorted_rows[:, min_samples - 1]
        return np.sort(kth), kth
    if binary_matrix is None:
        raise ValueError("Binary fingerprint matrix required for neighbor-based k-distance computation.")
    neighbors = NearestNeighbors(metric='jaccard',
                                 algorithm='brute',
                                 n_jobs=n_jobs)
    neighbors.fit(binary_matrix)
    distances, _ = neighbors.kneighbors(binary_matrix, n_neighbors=min_samples)
    kth = distances[:, -1]
    return np.sort(kth), kth


def _detect_knee(sorted_curve: np.ndarray) -> Tuple[float, Dict[str, Any]]:
    if sorted_curve.size == 0:
        raise ValueError("k-distance curve is empty; cannot infer eps.")
    x_coords = np.arange(sorted_curve.size, dtype=float)
    start = np.array([x_coords[0], sorted_curve[0]])
    end = np.array([x_coords[-1], sorted_curve[-1]])
    line_vec = end - start
    norm = np.linalg.norm(line_vec)
    if norm == 0:
        info = {'method': 'max_distance_to_line', 'index': 0, 'value': float(sorted_curve[0]), 'distance': 0.0}
        return float(sorted_curve[0]), info
    line_unit = line_vec / norm
    distances = []
    for idx, value in enumerate(sorted_curve):
        point = np.array([x_coords[idx], value])
        proj_length = np.dot(point - start, line_unit)
        projection = start + proj_length * line_unit
        perpendicular = np.linalg.norm(point - projection)
        distances.append(perpendicular)
    knee_index = int(np.argmax(distances))
    info = {
        'method': 'max_distance_to_line',
        'index': knee_index,
        'value': float(sorted_curve[knee_index]),
        'distance': float(distances[knee_index])
    }
    return float(sorted_curve[knee_index]), info


def _fit_dbscan(matrix: Optional[np.ndarray],
                distance_matrix: Optional[np.ndarray],
                eps: float,
                min_samples: int,
                metric_mode: str,
                config: Dict[str, Any],
                n_jobs: Optional[int]) -> DBSCAN:
    if eps <= 0:
        raise ValueError("eps must be positive for DBSCAN.")
    leaf_size_value = config.get('leaf_size', 30)
    try:
        leaf_size = int(leaf_size_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("analysis['dbscan']['leaf_size'] must be an integer.") from exc
    params = {
        'eps': eps,
        'min_samples': min_samples,
        'leaf_size': leaf_size,
        'n_jobs': n_jobs
    }
    if metric_mode == 'precomputed':
        if distance_matrix is None:
            raise ValueError("distance_matrix is required when metric_mode='precomputed'.")
        dbscan = DBSCAN(metric='precomputed', **params)
        dbscan.fit(distance_matrix)
    else:
        if matrix is None:
            raise ValueError("fingerprint matrix is required when metric_mode='jaccard'.")
        algorithm = str(config.get('algorithm', 'brute')).lower()
        if algorithm not in ('auto', 'ball_tree', 'kd_tree', 'brute'):
            algorithm = 'brute'
        dbscan = DBSCAN(metric='jaccard', algorithm=algorithm, **params)
        dbscan.fit(matrix)
    return dbscan


def _classify_points(labels: np.ndarray,
                     core_indices: Sequence[int],
                     n_samples: int) -> List[str]:
    classification = ['noise'] * n_samples
    core_mask = np.zeros(n_samples, dtype=bool)
    for idx in core_indices:
        if 0 <= idx < n_samples:
            core_mask[idx] = True
    for idx, label in enumerate(labels):
        if label == -1:
            classification[idx] = 'noise'
        elif core_mask[idx]:
            classification[idx] = 'core'
        else:
            classification[idx] = 'border'
    return classification


def _summarize_clusters(labels: np.ndarray,
                        classification: List[str],
                        kth_distances: np.ndarray,
                        eps_used: float,
                        eps_suggested: float,
                        min_samples: int,
                        metric_mode: str,
                        warnings: List[str],
                        knee_info: Dict[str, Any],
                        n_bits: int) -> Dict[str, Any]:
    n_samples = labels.size
    unique_labels = [label for label in sorted(set(labels)) if label != -1]
    cluster_counts = {int(label): int((labels == label).sum()) for label in unique_labels}
    noise_count = int(np.sum(labels == -1))
    core_count = int(sum(1 for item in classification if item == 'core'))
    border_count = int(sum(1 for item in classification if item == 'border'))

    stats = {
        'n_samples': int(n_samples),
        'n_clusters': int(len(unique_labels)),
        'cluster_sizes': cluster_counts,
        'noise_count': noise_count,
        'noise_ratio': float(noise_count / n_samples) if n_samples else 0.0,
        'core_count': core_count,
        'border_count': border_count,
        'min_samples': int(min_samples),
        'eps_suggested': float(eps_suggested),
        'eps_used': float(eps_used),
        'metric_mode': metric_mode,
        'knee': knee_info,
        'k_distance_summary': {
            'median': float(np.median(kth_distances)),
            'max': float(np.max(kth_distances)),
            'min': float(np.min(kth_distances)),
            'mean': float(np.mean(kth_distances))
        },
        'warnings': warnings,
        'n_bits': int(n_bits)
    }
    return stats


def _project_fingerprints(matrix: np.ndarray,
                          requested_components: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    features = matrix.astype(float)
    scaler = StandardScaler(with_mean=True)
    scaled = scaler.fit_transform(features)
    n_requested = requested_components if requested_components is not None else 2
    components = max(2, min(n_requested, scaled.shape[1], scaled.shape[0]))
    pca = PCA(n_components=components)
    projection = pca.fit_transform(scaled)
    if projection.shape[1] < 2:
        projection = np.column_stack((projection[:, 0], np.zeros(projection.shape[0])))
    else:
        projection = projection[:, :2]
    return projection, pca.explained_variance_ratio_
