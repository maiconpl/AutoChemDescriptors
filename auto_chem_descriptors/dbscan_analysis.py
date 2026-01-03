#!/usr/bin/python3
'''
Created on February 12, 2026.

Density-based clustering with DBSCAN tailored for binary chemical fingerprints.
'''

from typing import Any, Dict, List, Optional, Sequence, Tuple

import csv
import json
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dbscan_report import generate_dbscan_report

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_dbscan_analysis(descriptors_list, analysis):
    '''
    Execute DBSCAN over binary fingerprints, exporting diagnostics and plots.
    '''
    config = _extract_dbscan_config(analysis)
    fingerprints, profile = _prepare_fingerprint_matrix(descriptors_list)
    is_binary = profile['is_binary']
    binary_view = fingerprints.astype(bool) if is_binary else None
    n_samples, n_bits = fingerprints.shape

    print("DBSCAN analysis: samples =", n_samples, "bits =", n_bits)
    if profile.get('shift_offsets') is not None:
        print("Descriptor columns shifted by minima to enforce non-negativity for generalized Tanimoto similarity.")

    sample_labels = _extract_labels(analysis, n_samples)
    min_samples = _resolve_min_samples(config, n_samples)
    metric_mode = _decide_metric_mode(n_samples, config, is_binary)

    warnings = _inspect_fingerprints(fingerprints)
    for message in warnings:
        print("Warning:", message)

    print("Metric mode:", metric_mode)
    if metric_mode == 'precomputed':
        _log_distance_memory_estimate(n_samples)
        distance_matrix = _compute_tanimoto_distance_matrix(fingerprints)
    else:
        distance_matrix = None

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
    if eps_config is None:
        print(f"eps not provided; using knee suggestion {eps_suggested:.4f}")
    else:
        print(f"eps provided by user: {eps_used:.4f} (knee suggestion {eps_suggested:.4f})")

    dbscan = _fit_dbscan(binary_view,
                         distance_matrix,
                         eps_used,
                         min_samples,
                         metric_mode,
                         config,
                         n_jobs_value)

    labels = dbscan.labels_
    classification = _classify_points(labels, getattr(dbscan, 'core_sample_indices_', []), n_samples)
    stats_payload = _summarize_clusters(labels,
                                        classification,
                                        kth_distances,
                                        eps_used,
                                        eps_suggested,
                                        min_samples,
                                        metric_mode,
                                        warnings,
                                        knee_info)
    stats_payload['n_bits'] = int(n_bits)

    artifacts = _resolve_artifact_names()
    _save_parameters_file(stats_payload, artifacts['parameters'])
    _save_labels_file(labels, sample_labels, classification, artifacts['labels'])
    _save_stats_file(stats_payload, artifacts['stats'])

    _plot_k_distance_curve(sorted_k_curve,
                           eps_used,
                           knee_info,
                           artifacts['k_distance_plot'])

    projection_setting = config.get('projection_components')
    projection_components = _parse_optional_int(projection_setting, "analysis['dbscan']['projection_components']") if projection_setting is not None else 2
    projection, explained_variance = _project_fingerprints(fingerprints,
                                                           projection_components)
    print("PCA variance for visualization:", np.round(explained_variance, 4))
    _plot_clusters(projection,
                   labels,
                   classification,
                   sample_labels,
                   artifacts['cluster_plot'])

    report_payload = _build_report_payload(stats_payload,
                                           labels,
                                           classification,
                                           sample_labels,
                                           artifacts,
                                           binary_view is not None,
                                           profile,
                                           kth_distances,
                                           explained_variance,
                                           eps_config,
                                           sorted_k_curve)
    report_filename = generate_dbscan_report(report_payload, analysis)
    print("DBSCAN markdown report saved to:", report_filename)

    return stats_payload


def _extract_dbscan_config(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = {}
    if isinstance(analysis, dict):
        config = analysis.get('dbscan', {})
        if isinstance(config, list) and config:
            tail = config[-1]
            config = tail if isinstance(tail, dict) else {}
        elif isinstance(config, bool):
            config = {}
        elif config is None:
            config = {}
    if not isinstance(config, dict):
        raise ValueError("analysis['dbscan'] must be a dictionary or list ending with a dictionary.")
    return config


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
    profile: Dict[str, Any] = {'is_binary': is_binary, 'shift_offsets': None}
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
            return labels
    return [f"sample_{idx+1}" for idx in range(n_samples)]


def _parse_optional_int(value: Any, field_label: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{field_label} must be an integer.")


def _resolve_min_samples(config: Dict[str, Any], n_samples: int) -> int:
    raw = config.get('min_samples')
    min_samples = _parse_optional_int(raw, "analysis['dbscan']['min_samples']") if raw is not None else 5
    if min_samples < 2:
        min_samples = 2
    if min_samples > n_samples:
        raise ValueError("min_samples for DBSCAN cannot exceed the number of samples.")
    return min_samples


def _decide_metric_mode(n_samples: int, config: Dict[str, Any], is_binary: bool) -> str:
    mode = str(config.get('metric_mode', 'auto')).lower()
    if mode == 'jaccard' and not is_binary:
        print("Metric mode 'jaccard' requires binary fingerprints; falling back to 'precomputed'.")
        return 'precomputed'
    if mode in ('precomputed', 'jaccard'):
        if mode == 'jaccard' and not is_binary:
            return 'precomputed'
        return mode
    threshold_setting = config.get('precomputed_max_samples')
    threshold = _parse_optional_int(threshold_setting, "analysis['dbscan']['precomputed_max_samples']") if threshold_setting is not None else 1500
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


def _log_distance_memory_estimate(n_samples: int):
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


def _fit_dbscan(matrix: np.ndarray,
                distance_matrix: Optional[np.ndarray],
                eps: float,
                min_samples: int,
                metric_mode: str,
                config: Dict[str, Any],
                n_jobs: Optional[int]) -> DBSCAN:
    leaf_size_value = config.get('leaf_size', 30)
    try:
        leaf_size = int(leaf_size_value)
    except (TypeError, ValueError):
        raise ValueError("analysis['dbscan']['leaf_size'] must be an integer.")
    params = {
        'eps': eps,
        'min_samples': min_samples,
        'leaf_size': leaf_size,
        'n_jobs': n_jobs
    }
    if metric_mode == 'precomputed':
        dbscan = DBSCAN(metric='precomputed', **params)
        dbscan.fit(distance_matrix)
    else:
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
                        knee_info: Dict[str, Any]) -> Dict[str, Any]:
    n_samples = labels.size
    unique_labels = [label for label in sorted(set(labels)) if label != -1]
    cluster_counts = Counter(labels[labels != -1])
    noise_count = int(np.sum(labels == -1))
    core_count = int(sum(1 for item in classification if item == 'core'))
    border_count = int(sum(1 for item in classification if item == 'border'))

    stats = {
        'n_samples': int(n_samples),
        'n_clusters': int(len(unique_labels)),
        'cluster_sizes': {int(label): int(count) for label, count in cluster_counts.items()},
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
    }
    return stats


def _resolve_artifact_names() -> Dict[str, str]:
    return {
        'parameters': 'dbscan_parameters.json',
        'labels': 'dbscan_labels.csv',
        'stats': 'dbscan_stats.json',
        'k_distance_plot': 'plot_dbscan_k_distance.png',
        'cluster_plot': 'plot_dbscan_clusters.png',
        'report': 'report_dbscan.md',
    }


def _save_parameters_file(stats_payload: Dict[str, Any], filename: str):
    payload = {
        'eps_used': stats_payload['eps_used'],
        'eps_suggested': stats_payload['eps_suggested'],
        'min_samples': stats_payload['min_samples'],
        'metric_mode': stats_payload['metric_mode'],
        'knee': stats_payload['knee']
    }
    with open(filename, 'w', encoding='utf-8') as handler:
        json.dump(payload, handler, indent=2)
    print("DBSCAN parameters saved to:", filename)


def _save_labels_file(labels: np.ndarray,
                      sample_labels: List[str],
                      classification: List[str],
                      filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as handler:
        writer = csv.writer(handler)
        writer.writerow(['sample_index', 'sample_label', 'cluster', 'classification'])
        for idx, (label, sample_label, category) in enumerate(zip(labels, sample_labels, classification)):
            writer.writerow([idx, sample_label, int(label), category])
    print("DBSCAN labels saved to:", filename)


def _save_stats_file(stats_payload: Dict[str, Any], filename: str):
    with open(filename, 'w', encoding='utf-8') as handler:
        json.dump(stats_payload, handler, indent=2)
    print("DBSCAN stats saved to:", filename)


def _build_report_payload(stats_payload: Dict[str, Any],
                          labels: np.ndarray,
                          classification: List[str],
                          sample_labels: List[str],
                          artifacts: Dict[str, str],
                          is_binary: bool,
                          profile: Dict[str, Any],
                          kth_distances: np.ndarray,
                          explained_variance: np.ndarray,
                          eps_config: Optional[float],
                          sorted_k_curve: np.ndarray) -> Dict[str, Any]:
    eps_source = 'user_provided' if eps_config is not None else 'knee_suggestion'
    payload: Dict[str, Any] = {
        'stats': stats_payload,
        'labels': labels.tolist(),
        'classification': classification,
        'sample_labels': sample_labels,
        'artifacts': artifacts,
        'n_bits': stats_payload.get('n_bits') or 0,
        'is_binary': is_binary,
        'profile': profile,
        'kth_distances': kth_distances.tolist(),
        'projection_variance': explained_variance.tolist() if explained_variance is not None else [],
        'eps_source': eps_source,
        'k_distance_curve': sorted_k_curve.tolist(),
    }
    payload['stats']['n_bits'] = payload['n_bits']
    return payload


def _plot_k_distance_curve(sorted_curve: np.ndarray,
                           eps_used: float,
                           knee_info: Dict[str, Any],
                           filename: str):
    fig, ax = plt.subplots(figsize=(7, 5))
    x_axis = np.arange(1, sorted_curve.size + 1)
    ax.plot(x_axis, sorted_curve, color='#1f77b4', linewidth=2.0, label='k-distance (sorted)')
    ax.fill_between(x_axis, sorted_curve, alpha=0.15, color='#1f77b4')
    ax.axhline(eps_used, color='#d62728', linestyle='--', linewidth=1.6, label=f'eps = {eps_used:.4f}')
    if knee_info:
        knee_x = knee_info['index'] + 1
        knee_y = knee_info['value']
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


def _project_fingerprints(matrix: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    features = matrix.astype(float)
    scaler = StandardScaler(with_mean=True)
    scaled = scaler.fit_transform(features)
    components = max(2, min(n_components, scaled.shape[1], scaled.shape[0]))
    pca = PCA(n_components=components)
    projection = pca.fit_transform(scaled)
    if projection.shape[1] < 2:
        projection = np.column_stack((projection[:, 0], np.zeros(projection.shape[0])))
    else:
        projection = projection[:, :2]
    return projection, pca.explained_variance_ratio_


def _plot_clusters(projection: np.ndarray,
                   labels: np.ndarray,
                   classification: List[str],
                   sample_labels: List[str],
                   filename: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    max_clusters = max(len([lbl for lbl in unique_labels if lbl != -1]), 1)
    cmap = plt.cm.get_cmap('tab20', max_clusters)
    palette = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in np.linspace(0, 1, max_clusters)]
    markers = {'core': 'o', 'border': 's', 'noise': 'X'}
    sizes = {'core': 70, 'border': 55, 'noise': 80}
    classification_array = np.array(classification)
    legend_entries = []

    for idx, label in enumerate(unique_labels):
        mask = labels == label
        if not np.any(mask):
            continue
        if label == -1:
            color = '#7f7f7f'
        else:
            color = palette[idx % len(palette)]
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
            legend_entries.append(sc)

    for point, sample_label in zip(projection, sample_labels):
        ax.annotate(sample_label,
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
