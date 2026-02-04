#!/usr/bin/python3
'''
Created on February 12, 2026.

Density-based clustering with DBSCAN tailored for binary chemical fingerprints.
'''

from typing import Any, Dict

import csv
import json

from .dbscan_plotting import plot_dbscan_clusters, plot_k_distance_curve
from .dbscan_processing import DBSCANPayload, compute_dbscan_payload
from .dbscan_report import generate_dbscan_report


def run_dbscan_analysis(descriptors_list, analysis):
    '''Execute DBSCAN pipeline orchestrating processing, plotting, and reporting.'''
    config = _extract_dbscan_config(analysis)
    payload = compute_dbscan_payload(descriptors_list, analysis, config)

    artifacts = _resolve_artifact_names()
    _persist_core_artifacts(payload, artifacts)

    plot_k_distance_curve(payload.sorted_k_curve,
                          payload.eps_used,
                          payload.knee_info,
                          artifacts['k_distance_plot'])

    plot_dbscan_clusters(payload.projection,
                         payload.labels,
                         payload.classification,
                         payload.sample_labels,
                         artifacts['cluster_plot'])

    report_payload = _build_report_payload(payload, artifacts)
    report_filename = generate_dbscan_report(report_payload, analysis)
    print("DBSCAN markdown report saved to:", report_filename)

    return payload.stats


def _extract_dbscan_config(analysis: Dict[str, Any] | None) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    if isinstance(analysis, dict):
        config = analysis.get('dbscan', {})
        if isinstance(config, list) and config:
            tail = config[-1]
            config = tail if isinstance(tail, dict) else {}
        elif isinstance(config, bool) or config is None:
            config = {}
    if not isinstance(config, dict):
        raise ValueError("analysis['dbscan'] must be a dictionary or list ending with a dictionary.")
    return config


def _persist_core_artifacts(payload: DBSCANPayload, artifacts: Dict[str, str]) -> None:
    _save_parameters_file(payload.stats, artifacts['parameters'])
    _save_labels_file(payload.labels,
                      payload.sample_labels,
                      payload.classification,
                      artifacts['labels'])
    _save_stats_file(payload.stats, artifacts['stats'])


def _resolve_artifact_names() -> Dict[str, str]:
    return {
        'parameters': 'dbscan_parameters.json',
        'labels': 'dbscan_labels.csv',
        'stats': 'dbscan_stats.json',
        'k_distance_plot': 'dbscan_k_distance_plot.png',
        'cluster_plot': 'dbscan_cluster_plot.png',
        'report': 'dbscan_report.md',
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


def _save_labels_file(labels,
                      sample_labels,
                      classification,
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


def _build_report_payload(payload: DBSCANPayload,
                          artifacts: Dict[str, str]) -> Dict[str, Any]:
    stats_payload = dict(payload.stats)
    stats_payload['n_bits'] = payload.stats.get('n_bits', payload.fingerprint_matrix.shape[1])
    report_payload: Dict[str, Any] = {
        'stats': stats_payload,
        'labels': payload.labels.tolist(),
        'classification': payload.classification,
        'sample_labels': payload.sample_labels,
        'artifacts': artifacts,
        'n_bits': stats_payload['n_bits'],
        'is_binary': payload.profile['is_binary'],
        'profile': payload.profile,
        'kth_distances': payload.kth_distances.tolist(),
        'projection_variance': payload.explained_variance.tolist() if payload.explained_variance is not None else [],
        'eps_source': payload.eps_source,
        'k_distance_curve': payload.sorted_k_curve.tolist(),
    }
    return report_payload
