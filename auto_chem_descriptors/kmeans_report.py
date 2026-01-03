#!/usr/bin/python3
'''
Created on February 07, 2026.

Markdown reporting for K-Means clustering diagnostics.
'''

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import csv
import json
import math
from statistics import mean


MetricEntry = Dict[str, Any]
ReportPayload = Dict[str, Any]


def generate_kmeans_report(kmeans_payload: ReportPayload,
                           analysis: Optional[Dict[str, Any]] = None) -> str:
    '''
    Generate CSV artifacts and a Markdown interpretation for the K-Means run.
    '''
    payload = _validate_payload(kmeans_payload)
    analysis_dict = analysis if isinstance(analysis, dict) else {}
    report_settings = analysis_dict.get('kmeans_report', {})

    metrics_filename = report_settings.get('metrics_filename', 'kmeans_metrics.csv')
    suggestions_filename = report_settings.get('suggestions_filename', 'kmeans_suggestions.json')
    labels_filename = report_settings.get('labels_filename', 'kmeans_cluster_labels.csv')
    report_filename = report_settings.get('report_filename', 'report_kmeans.md')

    _write_metrics(payload['metrics'], metrics_filename)
    _write_suggestions(payload['elbow'], payload['silhouette'], payload['target_k'], suggestions_filename)
    _write_labels(payload['labels'], payload['sample_labels'], payload['silhouette_values'], labels_filename)

    lines = _build_markdown_report(payload,
                                   metrics_filename,
                                   suggestions_filename,
                                   labels_filename,
                                   report_settings)

    with open(report_filename, 'w', encoding='utf-8') as handler:
        handler.write("\n".join(lines))

    return report_filename


def _validate_payload(payload: ReportPayload) -> ReportPayload:
    required_keys = ['metrics', 'elbow', 'silhouette', 'target_k', 'labels', 'sample_labels',
                     'silhouette_values', 'plots', 'k_values', 'n_samples', 'n_features',
                     'random_state', 'scaling', 'use_minibatch', 'estimator', 'outlier_info']
    missing = [key for key in required_keys if key not in payload]
    if missing:
        raise ValueError(f"K-Means payload is missing keys: {missing}")
    if not isinstance(payload['metrics'], Iterable):
        raise ValueError("metrics entry must be iterable")
    if len(payload['labels']) != len(payload['sample_labels']):
        raise ValueError("labels and sample_labels must have the same size")
    return payload


def _write_metrics(metrics: Sequence[MetricEntry], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as handler:
        writer = csv.writer(handler)
        writer.writerow(['K', 'inertia', 'silhouette', 'runtime_s'])
        for entry in metrics:
            writer.writerow([entry['k'], entry['inertia'], entry['silhouette'], entry['runtime']])


def _write_suggestions(elbow: Dict[str, Any], silhouette: Dict[str, Any], target_k: int, filename: str):
    suggestions = {
        'elbow': elbow,
        'silhouette': silhouette,
        'adopted_k': {'k': target_k, 'rule': 'silhouette_if_defined_else_elbow'}
    }
    with open(filename, 'w', encoding='utf-8') as handler:
        json.dump(suggestions, handler, indent=2)


def _write_labels(labels: Sequence[int], sample_labels: Sequence[str], silhouette_values: Sequence[float], filename: str):
    with open(filename, 'w', newline='', encoding='utf-8') as handler:
        writer = csv.writer(handler)
        writer.writerow(['sample_index', 'sample_label', 'cluster', 'silhouette'])
        for idx, (label, cluster, silhouette_value) in enumerate(zip(sample_labels, labels, silhouette_values)):
            silhouette_entry = silhouette_value if not math.isnan(silhouette_value) else ''
            writer.writerow([idx, label, cluster, silhouette_entry])


def _build_markdown_report(payload: ReportPayload,
                            metrics_filename: str,
                            suggestions_filename: str,
                            labels_filename: str,
                            report_settings: Dict[str, Any]) -> List[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []

    lines.append("# K-Means Analysis Report")
    lines.append("")
    lines.append(f"- Generated at: {timestamp}")
    lines.append(f"- Samples: {payload['n_samples']}")
    lines.append(f"- Descriptors: {payload['n_features']}")
    lines.append(f"- Estimator: {payload['estimator']} (random_state={payload['random_state']})")
    lines.append(f"- Scaling: {payload['scaling']}")
    lines.append(f"- MiniBatch enabled: {payload['use_minibatch']}")
    lines.append(f"- Metrics table: `{metrics_filename}`")
    lines.append(f"- Cluster labels file: `{labels_filename}`")
    lines.append(f"- Suggested K file: `{suggestions_filename}`")
    lines.append(f"- Elbow plot: `{payload['plots']['elbow']}` | Silhouette plot: `{payload['plots']['silhouette']}` | Cluster plot: `{payload['plots']['clusters']}`")
    lines.append("")

    lines.append("## Diagnostic overview")
    lines.append("| K | Inertia | Silhouette | Runtime (s) |")
    lines.append("| --- | --- | --- | --- |")
    for entry in payload['metrics']:
        silhouette_value = entry['silhouette'] if not math.isnan(entry['silhouette']) else float('nan')
        silhouette_str = f"{silhouette_value:.4f}" if not math.isnan(silhouette_value) else "nan"
        lines.append(f"| {entry['k']} | {entry['inertia']:.4f} | {silhouette_str} | {entry['runtime']:.4f} |")
    lines.append("")
    lines.append("### Interpretation per K")
    lines.extend(_describe_k_metrics(payload['metrics']))
    lines.append("")

    lines.append("## K suggestions")
    lines.append(f"- Elbow: K={payload['elbow']['k']} ({payload['elbow']['justification']})")
    silhouette_k = payload['silhouette']['k'] if payload['silhouette']['k'] > 1 else 'undefined'
    lines.append(f"- Silhouette: K={silhouette_k} ({payload['silhouette']['justification']})")
    lines.append(f"- Adopted K for labeling: {payload['target_k']}")
    lines.append("")

    cluster_stats = _summarize_clusters(payload['labels'], payload['silhouette_values'], payload['sample_labels'])
    lines.append("## Cluster distribution")
    lines.append("| Cluster | Count | Percentage (%) | Average silhouette | Minimum silhouette |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in cluster_stats:
        mean_str = f"{item['silhouette_mean']:.4f}" if item['silhouette_mean'] is not None else "nan"
        min_str = f"{item['silhouette_min']:.4f}" if item['silhouette_min'] is not None else "nan"
        lines.append(f"| {item['cluster']} | {item['count']} | {item['percentage']:.2f} | {mean_str} | {min_str} |")
    lines.append("")
    lines.append("### Cluster comments")
    lines.extend(_describe_cluster_balance(cluster_stats))
    lines.append("")

    if payload['outlier_info'].get('extreme_count', 0) > 0:
        lines.append(f"- Alert: {payload['outlier_info']['extreme_count']} values exceed 5 standard deviations (ratio={payload['outlier_info']['extreme_ratio']:.4f}).")
        lines.append("- Heavy-tailed descriptors distort centroids; consider trimming or robust scaling before clustering.")
    lines.append(f"- Maximum absolute z-score observed: {payload['outlier_info']['max_zscore']:.3f}")
    lines.append("- Extreme values bias spectral cluster balance and depress silhouette coherence.")
    lines.append("")

    low_silhouette = [entry for entry in cluster_stats if entry['silhouette_min'] is not None and entry['silhouette_min'] < 0]
    if low_silhouette:
        lines.append("### Negative silhouette observations")
        for item in low_silhouette:
            lines.append(f"- Cluster {item['cluster']} contains samples with silhouette < 0, indicating overlap or noisy descriptors.")
        lines.append("")

    critical_samples = _find_critical_samples(payload['sample_labels'], payload['labels'], payload['silhouette_values'], threshold=0.15)
    if critical_samples:
        lines.append("### Low-silhouette samples")
        lines.append("| Sample | Cluster | Silhouette |")
        lines.append("| --- | --- | --- |")
        for sample in critical_samples:
            lines.append(f"| `{sample['label']}` | {sample['cluster']} | {sample['silhouette']:.4f} |")
        lines.append("")
        lines.extend(_interpret_low_silhouette_samples(critical_samples))
        lines.append("")

    lines.append("## Additional settings")
    if report_settings:
        for key, value in report_settings.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- No report-specific settings were provided.")

    return lines


def _summarize_clusters(labels: Sequence[int], silhouettes: Sequence[float], sample_labels: Sequence[str]) -> List[Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    total = len(labels)
    for cluster_id in sorted(set(labels)):
        stats[cluster_id] = {'cluster': cluster_id,
                             'count': 0,
                             'percentage': 0.0,
                             'silhouette_mean': None,
                             'silhouette_min': None}
    for idx, cluster_id in enumerate(labels):
        entry = stats[cluster_id]
        entry['count'] += 1
        silhouette_value = silhouettes[idx]
        if not math.isnan(silhouette_value):
            values = entry.setdefault('silhouette_values', [])
            values.append(silhouette_value)
    for entry in stats.values():
        entry['percentage'] = (entry['count'] / total * 100.0) if total else 0.0
        values = entry.get('silhouette_values', [])
        if values:
            entry['silhouette_mean'] = mean(values)
            entry['silhouette_min'] = min(values)
        else:
            entry['silhouette_mean'] = None
            entry['silhouette_min'] = None
        entry.pop('silhouette_values', None)
    return list(stats.values())


def _find_critical_samples(sample_labels: Sequence[str],
                           clusters: Sequence[int],
                           silhouettes: Sequence[float],
                           threshold: float) -> List[Dict[str, Any]]:
    critical: List[Dict[str, Any]] = []
    for label, cluster_id, silhouette_value in zip(sample_labels, clusters, silhouettes):
        if math.isnan(silhouette_value):
            continue
        if silhouette_value < threshold:
            critical.append({'label': label, 'cluster': cluster_id, 'silhouette': silhouette_value})
    return sorted(critical, key=lambda item: item['silhouette'])[:10]


def _describe_k_metrics(metrics: Sequence[MetricEntry]) -> List[str]:
    lines: List[str] = []
    if not metrics:
        lines.append("- No results available for interpretation.")
        return lines
    previous_entry: Optional[MetricEntry] = None
    best_drop: Optional[Tuple[float, int, int]] = None
    valid_silhouettes: List[MetricEntry] = []
    for entry in metrics:
        parts = [f"K={entry['k']}", f"inertia={entry['inertia']:.4f}"]
        if previous_entry is not None and previous_entry['inertia'] > 0:
            drop = (previous_entry['inertia'] - entry['inertia']) / previous_entry['inertia'] * 100.0
            parts.append(f"Delta_WSS={drop:.2f}% vs K={previous_entry['k']}")
            if best_drop is None or drop > best_drop[0]:
                best_drop = (drop, entry['k'], previous_entry['k'])
        silhouette_value = entry['silhouette']
        if not math.isnan(silhouette_value):
            sil_detail = f"silhouette={silhouette_value:.4f}"
            if previous_entry is not None and not math.isnan(previous_entry['silhouette']):
                delta_sil = silhouette_value - previous_entry['silhouette']
                sil_detail += f" ({delta_sil:+.4f} vs previous)"
            parts.append(sil_detail)
            valid_silhouettes.append(entry)
        else:
            parts.append("silhouette undefined")
        parts.append(f"runtime={entry['runtime']:.2f}s")
        lines.append("- " + ", ".join(parts) + ".")
        previous_entry = entry
    if best_drop is not None:
        lines.append(f"- Largest relative WSS drop ({best_drop[0]:.2f}%) occurred between K={best_drop[2]} and K={best_drop[1]}, indicating an elbow there.")
    if valid_silhouettes:
        best = max(valid_silhouettes, key=lambda item: item['silhouette'])
        consistency = "stable" if len(valid_silhouettes) < 2 else _describe_silhouette_trend(valid_silhouettes)
        lines.append(f"- Maximum silhouette at K={best['k']} ({best['silhouette']:.4f}), trend {consistency} across the tested range.")
    else:
        lines.append("- No valid silhouette because all tested K collapsed to single clusters.")
    return lines


def _describe_silhouette_trend(valid_silhouettes: Sequence[MetricEntry]) -> str:
    deltas = []
    previous_value: Optional[float] = None
    for entry in valid_silhouettes:
        current = entry['silhouette']
        if previous_value is not None:
            deltas.append(current - previous_value)
        previous_value = current
    if not deltas:
        return "neutral"
    positives = sum(1 for delta in deltas if delta > 0)
    negatives = sum(1 for delta in deltas if delta < 0)
    if positives == len(deltas):
        return "ascending"
    if negatives == len(deltas):
        return "descending"
    return "fluctuating"


def _describe_cluster_balance(cluster_stats: Sequence[Dict[str, Any]]) -> List[str]:
    if not cluster_stats:
        return ["- No clusters available to summarize."]
    counts = [entry['count'] for entry in cluster_stats]
    min_count = min(counts)
    max_count = max(counts)
    ratio = (max_count / min_count) if min_count else float('inf')
    lines = [f"- Ratio between largest and smallest cluster: {ratio:.2f}."]
    for entry in cluster_stats:
        base = f"- Cluster {entry['cluster']}: {entry['count']} samples ({entry['percentage']:.2f}%), "
        if entry['silhouette_mean'] is not None:
            detail = base + f"average silhouette {entry['silhouette_mean']:.4f}"
        else:
            detail = base + "average silhouette unavailable"
        if entry['silhouette_min'] is not None:
            detail += f", minimum {entry['silhouette_min']:.4f}"
        detail += "."
        lines.append(detail)
        if entry['percentage'] < 10.0:
            lines.append(f"  Cluster {entry['cluster']} covers less than 10% of the samples; verify if it corresponds to instrumental or experimental outliers.")
        if entry['silhouette_mean'] is not None and entry['silhouette_mean'] < 0.2:
            lines.append(f"  Average silhouette < 0.2 indicates poorly separated centroids for cluster {entry['cluster']}.")
    return lines


def _interpret_low_silhouette_samples(samples: Sequence[Dict[str, Any]]) -> List[str]:
    lines: List[str] = []
    if not samples:
        return lines
    worst = samples[0]
    lines.append(f"- Sample `{worst['label']}` shows the lowest silhouette ({worst['silhouette']:.4f}); review its spectrum or scaling for artifacts.")
    average = sum(item['silhouette'] for item in samples) / len(samples)
    lines.append(f"- Mean silhouette across flagged samples: {average:.4f}; negative values suggest chemical overlap between neighboring clusters.")
    return lines
