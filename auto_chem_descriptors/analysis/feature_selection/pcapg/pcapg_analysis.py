"""Public entry point for PCAPG processing + plotting."""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Sequence

import numpy as np

from .pcapg_processing import PCAPGPayload, compute_pcapg_payload
from .pcapg_plotting import render_pcapg_figures


def run_pcapg_analysis(descriptors_list,
                       analysis: Dict[str, Any]) -> Dict[str, Any]:
    config = _extract_pcapg_config(analysis)
    payload = compute_pcapg_payload(descriptors_list, config)
    feature_names = _resolve_feature_names(analysis, payload.feature_matrix.shape[1])

    csv_filename = _export_rankings(feature_names,
                                    payload.feature_scores,
                                    payload.ordered_indices,
                                    str(config.get('csv_filename', 'pcapg_feature_scores.csv')))
    figures = render_pcapg_figures(feature_names, payload, analysis, config)
    _log_console_summary(feature_names, payload.feature_scores)
    return {
        'payload': payload,
        'feature_names': feature_names,
        'csv_filename': csv_filename,
        'figures': figures,
    }


def _extract_pcapg_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    raw = analysis.get('pcapg')
    if isinstance(raw, list) and raw:
        candidate = raw[-1]
        config = candidate if isinstance(candidate, dict) else {}
    elif isinstance(raw, dict):
        config = raw
    elif raw in (True, None):
        config = {}
    else:
        raise ValueError("analysis['pcapg'] must be True, a dict, or a list ending with a dict.")

    defaults = {
        'n_components': 6,
        'alpha': 0.7,
        'beta': 0.05,
        'lambda_reg': 1.0,
        'possibilistic_sharpness': 0.35,
        'n_neighbors': 8,
        'max_iter': 60,
        'tol': 1e-4,
        'scaling': 'standard',
        'top_features': 25,
        'ranking_plot_filename': 'plot_pcapg_importance.png',
        'manifold_plot_filename': 'plot_pcapg_manifold.png',
        'convergence_plot_filename': 'plot_pcapg_convergence.png',
        'csv_filename': 'pcapg_feature_scores.csv',
        'random_state': 42,
    }
    merged = {**defaults, **config}
    return merged


def _resolve_feature_names(analysis: Dict[str, Any], n_features: int) -> List[str]:
    names = analysis.get('descriptor_names')
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        labels = [str(item) for item in names]
        if len(labels) == n_features:
            return labels
    return [f"descriptor_{idx + 1}" for idx in range(n_features)]


def _export_rankings(feature_names: Sequence[str],
                     scores: np.ndarray,
                     ordered_indices: np.ndarray,
                     filename: str) -> str:
    with open(filename, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(["rank", "descriptor", "pcapg_score"])
        for rank, idx in enumerate(ordered_indices, start=1):
            writer.writerow([rank, feature_names[idx], f"{scores[idx]:.8f}"])
    print("PCAPG ranking table saved to:", filename)
    return filename


def _log_console_summary(feature_names: Sequence[str], scores: np.ndarray) -> None:
    order = np.argsort(scores)[::-1]
    top = order[: min(10, order.size)]
    print("\nTop descriptors by PCAPG relevance (higher = more representative):")
    for rank, idx in enumerate(top, start=1):
        print(f"  #{rank:02d} {feature_names[idx]} -> {scores[idx]:.6f}")
