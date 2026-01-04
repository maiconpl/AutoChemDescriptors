#!/usr/bin/python3
'''
Created on February 06, 2026.

Independent K-Means analysis with elbow and silhouette diagnostics.
'''

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import math
from time import perf_counter

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from .kmeans_report import generate_kmeans_report

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run_kmeans_analysis(descriptors_list, analysis):

    config = _extract_kmeans_config(analysis)
    feature_matrix = _validate_descriptors(descriptors_list)
    n_samples = feature_matrix.shape[0]
    use_minibatch_flag = bool(config.get('use_minibatch', False))

    print("K-Means analysis: samples =", n_samples, "features =", feature_matrix.shape[1])

    scaled_matrix, _ = _scale_features(feature_matrix, config)
    labels_reference = _extract_labels(analysis, n_samples)

    outlier_info = _inspect_outliers(scaled_matrix)

    k_values = _resolve_k_values(config, n_samples)
    print("Evaluating K range:", k_values)

    metrics, models = _evaluate_k_series(scaled_matrix, k_values, config)

    elbow_k, elbow_reason = _suggest_k_elbow(k_values, [entry['inertia'] for entry in metrics])
    silhouette_k, silhouette_reason = _suggest_k_silhouette(metrics)

    print("Elbow method:", elbow_reason, "-> K", elbow_k)
    print("Silhouette method:", silhouette_reason, "-> K", silhouette_k if silhouette_k > 1 else "undefined")

    target_k = _select_target_k(silhouette_k, elbow_k, n_samples)
    final_labels = models[target_k]['labels']

    silhouette_values = _compute_sample_silhouette(scaled_matrix, final_labels)

    _log_cluster_diagnostics(final_labels, silhouette_values)

    _plot_elbow(k_values, metrics, config.get('elbow_plot', 'plot_kmeans_elbow.png'))
    _plot_silhouette_curve(k_values, metrics, config.get('silhouette_plot', 'plot_kmeans_silhouette.png'))

    projection = _project_for_visualization(scaled_matrix, target_k, final_labels, config.get('projection_components', 2))
    _plot_clusters(projection,
                  final_labels,
                  labels_reference,
                  config.get('cluster_plot', 'plot_kmeans_clusters.png'))

    report_payload = {
        'metrics': metrics,
        'elbow': {'k': elbow_k, 'justification': elbow_reason},
        'silhouette': {'k': silhouette_k, 'justification': silhouette_reason},
        'target_k': target_k,
        'labels': final_labels.tolist(),
        'sample_labels': labels_reference,
        'silhouette_values': silhouette_values,
        'plots': {
            'elbow': config.get('elbow_plot', 'plot_kmeans_elbow.png'),
            'silhouette': config.get('silhouette_plot', 'plot_kmeans_silhouette.png'),
            'clusters': config.get('cluster_plot', 'plot_kmeans_clusters.png'),
        },
        'k_values': k_values,
        'n_samples': n_samples,
        'n_features': feature_matrix.shape[1],
        'random_state': int(config.get('random_state', 42)),
        'scaling': config.get('scaling', 'standard'),
        'use_minibatch': use_minibatch_flag,
        'estimator': 'MiniBatchKMeans' if use_minibatch_flag else 'KMeans',
        'outlier_info': outlier_info,
    }

    report_filename = generate_kmeans_report(report_payload, analysis)
    print("K-Means markdown report saved to:", report_filename)

    return report_payload


def _extract_kmeans_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    config = analysis.get('kmeans')
    if isinstance(config, list) and config:
        config_candidate = config[-1]
        config = config_candidate if isinstance(config_candidate, dict) else {}
    elif isinstance(config, bool):
        config = {}
    elif config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("analysis['kmeans'] must be a dictionary or list ending with a dictionary.")
    return config


def _validate_descriptors(data: Sequence[Sequence[float]]) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("Descriptor matrix for K-Means must be 2D.")
    if array.shape[0] < 2:
        raise ValueError("K-Means requires at least two samples.")
    if array.shape[1] == 0:
        raise ValueError("Descriptor matrix must have at least one feature.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Descriptor matrix contains NaN or infinite values.")
    return array


def _scale_features(array: np.ndarray, config: Dict[str, Any]) -> Tuple[np.ndarray, StandardScaler]:
    scaling = config.get('scaling', 'standard')
    if scaling not in (None, 'standard'):
        raise ValueError("Only 'standard' scaling or None is supported to remain consistent with PCA.")
    scaler = StandardScaler()
    scaler.fit(array)
    scaled = scaler.transform(array)
    return scaled, scaler


def _extract_labels(analysis: Dict[str, Any], n_samples: int) -> List[str]:
    labels = analysis.get('molecules_label')
    if isinstance(labels, list) and len(labels) == n_samples:
        return labels
    return [f"sample_{i+1}" for i in range(n_samples)]


def _inspect_outliers(scaled_matrix: np.ndarray) -> Dict[str, Any]:
    z_scores = np.abs(scaled_matrix)
    extreme_mask = z_scores > 5.0
    info = {
        'extreme_count': int(extreme_mask.sum()),
        'extreme_ratio': float(extreme_mask.sum() / extreme_mask.size) if extreme_mask.size else 0.0,
        'max_zscore': float(np.max(z_scores)) if z_scores.size else 0.0,
    }
    if np.any(extreme_mask):
        print("Warning: Detected", extreme_mask.sum(), "feature values beyond 5 std (ratio =", round(info['extreme_ratio'], 4), ").")
    print("Maximum absolute z-score in scaled features:", round(float(info['max_zscore']), 3))
    return info


def _resolve_k_values(config: Dict[str, Any], n_samples: int) -> List[int]:
    if 'k_values' in config and isinstance(config['k_values'], Iterable):
        k_values = sorted({int(value) for value in config['k_values'] if isinstance(value, (int, float))})
    else:
        k_min = int(config.get('k_min', 2))
        k_max_default = min(n_samples - 1, max(k_min + 4, 3))
        k_max = int(config.get('k_max', k_max_default))
        k_values = list(range(k_min, k_max + 1))
    k_values = [k for k in k_values if 1 < k < n_samples]
    if not k_values:
        raise ValueError("K range for K-Means is empty. Provide valid k_min/k_max or k_values.")
    return k_values


def _evaluate_k_series(features: np.ndarray,
                       k_values: List[int],
                       config: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    metrics: List[Dict[str, Any]] = []
    models: Dict[int, Dict[str, Any]] = {}
    random_state = int(config.get('random_state', 42))
    use_minibatch = bool(config.get('use_minibatch', False))

    for k in k_values:
        estimator = _instantiate_estimator(k, random_state, config, use_minibatch)
        start = perf_counter()
        estimator.fit(features)
        elapsed = perf_counter() - start
        inertia = float(estimator.inertia_)
        labels = estimator.labels_
        silhouette_avg = float('nan')
        has_valid_silhouette = _is_valid_silhouette(labels)
        if has_valid_silhouette:
            silhouette_avg = float(silhouette_score(features, labels))
        metrics.append({
            'k': k,
            'inertia': inertia,
            'silhouette': silhouette_avg,
            'runtime': elapsed,
        })
        models[k] = {'model': estimator, 'labels': labels}
        print("K =", k, "| inertia =", round(inertia, 4), "| silhouette =", round(silhouette_avg, 4) if not math.isnan(silhouette_avg) else "nan", "| time =", round(elapsed, 4), "s")
    return metrics, models


def _instantiate_estimator(k: int,
                           random_state: int,
                           config: Dict[str, Any],
                           use_minibatch: bool):
    init_method = config.get('init', 'k-means++')
    max_iter = int(config.get('max_iter', 300))
    n_init = int(config.get('n_init', 10))
    tol = float(config.get('tol', 1e-4))
    if use_minibatch:
        batch_size = int(config.get('batch_size', 100))
        max_no_improvement = int(config.get('max_no_improvement', 10))
        reassignment_ratio = float(config.get('reassignment_ratio', 0.01))
        return MiniBatchKMeans(n_clusters=k,
                               random_state=random_state,
                               init=init_method,
                               max_iter=max_iter,
                               n_init=n_init,
                               batch_size=batch_size,
                               max_no_improvement=max_no_improvement,
                               reassignment_ratio=reassignment_ratio,
                               tol=tol)
    return KMeans(n_clusters=k,
                  random_state=random_state,
                  init=init_method,
                  max_iter=max_iter,
                  n_init=n_init,
                  tol=tol)


def _is_valid_silhouette(labels: np.ndarray) -> bool:
    unique = np.unique(labels)
    return unique.size > 1 and unique.size < labels.size


def _suggest_k_elbow(k_values: List[int], inertia_values: List[float]) -> Tuple[int, str]:
    points = np.column_stack((k_values, inertia_values))
    start = points[0]
    end = points[-1]
    line_vec = end - start
    norm = np.linalg.norm(line_vec)
    if norm == 0:
        return k_values[0], "inertia curve is flat; defaulting to smallest K"
    line_unit = line_vec / norm
    distances = []
    for point in points:
        vec = point - start
        proj = np.dot(vec, line_unit)
        proj_point = start + proj * line_unit
        distance = np.linalg.norm(point - proj_point)
        distances.append(distance)
    best_index = int(np.argmax(distances))
    justification = f"distance_to_line={distances[best_index]:.4f}"
    return k_values[best_index], justification


def _suggest_k_silhouette(metrics: List[Dict[str, Any]]) -> Tuple[int, str]:
    valid_entries = [entry for entry in metrics if not math.isnan(entry['silhouette'])]
    if not valid_entries:
        return -1, "silhouette undefined for all tested K"
    best = max(valid_entries, key=lambda item: item['silhouette'])
    justification = f"max_silhouette={best['silhouette']:.4f}"
    return best['k'], justification


def _select_target_k(silhouette_k: int, elbow_k: int, n_samples: int) -> int:
    if silhouette_k > 1:
        print("Using silhouette-optimal K:", silhouette_k)
        return silhouette_k
    print("Silhouette unavailable; fallback to elbow K:", elbow_k)
    return elbow_k if elbow_k < n_samples else max(2, n_samples - 1)


def _compute_sample_silhouette(features: np.ndarray, labels: np.ndarray) -> List[float]:
    if not _is_valid_silhouette(labels):
        return [float('nan')] * len(labels)
    values = silhouette_samples(features, labels)
    return [float(value) for value in values]


def _log_cluster_diagnostics(labels: np.ndarray, silhouette_values: List[float]):
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster population distribution:", dict(zip(unique.tolist(), counts.tolist())))
    if counts.min() > 0:
        imbalance = counts.max() / counts.min()
        print("Cluster size imbalance ratio:", round(float(imbalance), 3))
    low_quality = [value for value in silhouette_values if not math.isnan(value) and value < 0.0]
    if low_quality:
        print("Warning: Detected", len(low_quality), "samples with negative silhouette values.")


def _plot_elbow(k_values: List[int],
                metrics: List[Dict[str, Any]],
                filename: str):
    inertia_values = [entry['inertia'] for entry in metrics]
    plt.figure()
    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel("K")
    plt.ylabel("Inércia (WSS)")
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def _plot_silhouette_curve(k_values: List[int],
                           metrics: List[Dict[str, Any]],
                           filename: str):
    silhouette_values = []
    for entry in metrics:
        silhouette_values.append(entry['silhouette'] if not math.isnan(entry['silhouette']) else float('nan'))
    plt.figure()
    plt.plot(k_values, silhouette_values, marker='s', color='darkgreen')
    plt.xlabel("K")
    plt.ylabel("Silhueta média")
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def _project_for_visualization(features: np.ndarray,
                               target_k: int,
                               labels: Sequence[int],
                               n_components: int) -> np.ndarray:
    components = max(2, int(n_components))
    components = min(components, features.shape[1], features.shape[0])
    pca = PCA(n_components=components)
    projection = pca.fit_transform(features)
    print("Explained variance ratio for visualization PCA:", np.round(pca.explained_variance_ratio_, 4))
    return projection[:, :2] if projection.shape[1] >= 2 else np.column_stack((projection[:, 0], np.zeros(projection.shape[0])))


def _plot_clusters(projection: np.ndarray,
                   clusters: Sequence[int],
                   labels_reference: List[str],
                   filename: str):
    plt.figure()
    scatter = plt.scatter(projection[:, 0], projection[:, 1], c=clusters, cmap='tab10', s=70, edgecolors='k', linewidths=0.5)
    plt.xlabel("Componente 1 (PCA local)")
    plt.ylabel("Componente 2 (PCA local)")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(scatter, label="Clusters")
    for point, label in zip(projection, labels_reference):
        plt.annotate(label, (point[0], point[1]), textcoords="offset points", xytext=(2, 2), fontsize=6)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print("Cluster scatter plot saved to:", filename)
