#!/usr/bin/python3
"""
Public entry point for Laplacian/Marginal Laplacian ranking.

Handles configuration parsing, descriptor name resolution, CSV export,
and plotting orchestration.
"""

from __future__ import annotations

import csv
from typing import Any, Dict, List, Sequence

import numpy as np

from .laplacian_processing import compute_laplacian_scores
from .laplacian_plotting import generate_laplacian_plots
from .laplacian_report import generate_laplacian_report


def run_laplacian_score_analysis(descriptors_list, analysis: Dict[str, Any]) -> Dict[str, Any]:
    config = _extract_laplacian_config(analysis)
    descriptor_matrix = np.asarray(descriptors_list, dtype=float)
    n_features = descriptor_matrix.shape[1]
    feature_names = _resolve_feature_names(analysis, n_features)

    payload = compute_laplacian_scores(descriptor_matrix, config)

    csv_filename = _export_ranking_table(feature_names, payload, config)
    plot_filenames = generate_laplacian_plots(feature_names, payload, config)
    report_filename = generate_laplacian_report(feature_names, payload, analysis, config)

    _log_console_summary(feature_names, payload)

    return {
        'csv_filename': csv_filename,
        'plot_filenames': plot_filenames,
        'report_filename': report_filename,
        'ls_scores': payload['ls_scores'],
        'mls_scores': payload['mls_scores'],
        'feature_names': feature_names,
        'graph_profile': payload['graph_profile'],
    }


def _extract_laplacian_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    raw = analysis.get('laplacian_score')
    if isinstance(raw, list) and raw:
        raw = raw[-1] if isinstance(raw[-1], dict) else {}
    elif isinstance(raw, bool):
        raw = {}
    elif raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("analysis['laplacian_score'] must be a dictionary or list ending with a dictionary.")

    defaults = {
        'k_neighbors': 7,
        'metric': 'auto',
        'mode': 'both',
        'quantile': 0.90,
        'skew_right': 0.5,
        'skew_left': -0.5,
        'adaptive_kernel': True,
        'top_descriptors': 20,
        'ls_plot_filename': 'plot_laplacian_ls.png',
        'mls_plot_filename': 'plot_laplacian_mls.png',
        'csv_filename': 'laplacian_scores.csv',
    }
    config = {**defaults, **raw}

    if 'plot_filename' in raw and 'ls_plot_filename' not in raw:
        config['ls_plot_filename'] = raw['plot_filename']
    if 'plot_filename' in raw and 'mls_plot_filename' not in raw:
        base = raw['plot_filename']
        if base.lower().endswith('.png'):
            config['mls_plot_filename'] = base[:-4] + '_mls.png'
        else:
            config['mls_plot_filename'] = base + '_mls'

    return config


def _resolve_feature_names(analysis: Dict[str, Any], n_features: int) -> List[str]:
    candidate = analysis.get('descriptor_names') or analysis.get('descriptors_name')
    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        candidate_list = [str(item) for item in candidate]
        if len(candidate_list) == n_features:
            return candidate_list
    return [f"descriptor_{idx + 1}" for idx in range(n_features)]


def _export_ranking_table(feature_names: Sequence[str],
                          payload: Dict[str, np.ndarray],
                          config: Dict[str, Any]) -> str:
    ls_scores = payload['ls_scores']
    mls_scores = payload['mls_scores']
    skewness = payload['skewness']
    coverage = payload['margin_coverage']
    marginal_labels = payload['marginal_labels']

    ls_rank = _rank_scores(ls_scores)
    mls_rank = _rank_scores(mls_scores)

    csv_filename = str(config.get('csv_filename', 'laplacian_scores.csv'))
    with open(csv_filename, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "descriptor",
            "laplacian_score",
            "laplacian_rank",
            "marginal_score",
            "marginal_rank",
            "skewness",
            "marginal_coverage_percent",
            "marginal_profile",
        ])
        for idx, name in enumerate(feature_names):
            writer.writerow([
                name,
                _format_score(ls_scores[idx]),
                _format_rank(ls_rank[idx]),
                _format_score(mls_scores[idx]),
                _format_rank(mls_rank[idx]),
                f"{skewness[idx]:.4f}",
                f"{coverage[idx] * 100:.2f}",
                marginal_labels[idx],
            ])
    print("Laplacian ranking table saved to:", csv_filename)
    return csv_filename


def _rank_scores(values: np.ndarray) -> np.ndarray:
    ranks = np.full(values.shape, np.nan, dtype=float)
    finite_idx = np.where(np.isfinite(values))[0]
    if finite_idx.size == 0:
        return ranks
    order = finite_idx[np.argsort(values[finite_idx])]
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def _format_score(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return f"{value:.8f}"


def _format_rank(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return str(int(value))


def _log_console_summary(feature_names: Sequence[str],
                         payload: Dict[str, np.ndarray]) -> None:
    ls_scores = payload['ls_scores']
    mls_scores = payload['mls_scores']

    print("\nTop descritores pelo Laplacian Score (menor = melhor):")
    _print_top(feature_names, ls_scores)

    if np.isfinite(mls_scores).any():
        print("\nTop descritores pelo Marginal Laplacian Score (foco nas caudas):")
        _print_top(feature_names, mls_scores)
    else:
        print("\nMarginal Laplacian Score desativado para esta execução.")


def _print_top(feature_names: Sequence[str], scores: np.ndarray, limit: int = 10) -> None:
    finite_idx = np.where(np.isfinite(scores))[0]
    if finite_idx.size == 0:
        print("  (sem dados válidos)")
        return
    order = finite_idx[np.argsort(scores[finite_idx])]
    for rank, idx in enumerate(order[:limit], start=1):
        print(f"  #{rank:02d} {feature_names[idx]} -> {scores[idx]:.6f}")
