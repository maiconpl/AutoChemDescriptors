"""Markdown reporting for SHAP explainability outputs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np


Payload = Dict[str, Any]


def generate_shap_report(payload: Payload,
                         config: Dict[str, Any],
                         analysis: Optional[Dict[str, Any]] = None,
                         figures: Optional[Dict[str, Any]] = None) -> str:
    """Compose the SHAP Markdown narrative and persist it to disk."""

    report_settings = _resolve_report_settings(analysis)
    default_filename = str(config.get('report_filename', 'report_shap_validation.md'))
    report_filename = str(report_settings.get('report_filename', default_filename))

    top_base = int(config.get('top_features', 12) or 12)
    top_k = int(report_settings.get('top_descriptors', top_base))
    local_limit = int(report_settings.get('local_examples', 3))

    lines = _build_report_lines(payload,
                                report_settings,
                                figures or {},
                                top_k,
                                local_limit)

    with open(report_filename, 'w', encoding='utf-8') as handler:
        handler.write("\n".join(lines))

    return report_filename


def _resolve_report_settings(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    candidate = analysis.get('shap_validation_report')
    if isinstance(candidate, dict):
        return candidate
    return {}


def _build_report_lines(payload: Payload,
                        report_settings: Dict[str, Any],
                        figures: Dict[str, Any],
                        top_k: int,
                        local_limit: int) -> List[str]:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    feature_names = list(payload.get('feature_names', []))
    features = np.asarray(payload.get('feature_matrix', []), dtype=float)
    shap_values = np.asarray(payload.get('shap_values', []), dtype=float)
    importance = np.asarray(payload.get('importance', []), dtype=float)
    ordered = np.asarray(payload.get('ordered_indices', []), dtype=int)
    targets = np.asarray(payload.get('targets', []), dtype=float)
    metrics = payload.get('metrics', {}) or {}
    task = str(payload.get('task', 'regression')).lower() or 'regression'
    estimator_name = payload.get('estimator_name') or _estimator_label(payload.get('estimator'))
    scaler_label = 'standard' if payload.get('scaler') is not None else 'none'
    explainer_meta = payload.get('explainer', {}) or {}
    expected_value = payload.get('expected_value')
    target_meta = payload.get('target_meta', {}) or {}
    sample_labels = payload.get('sample_labels', []) or []

    n_samples = shap_values.shape[0] if shap_values.ndim == 2 else features.shape[0]
    n_features = shap_values.shape[1] if shap_values.ndim == 2 else len(feature_names)

    if not feature_names and n_features:
        feature_names = [f"descriptor_{idx + 1}" for idx in range(n_features)]

    if ordered.size == 0 and importance.size:
        sorted_idx = np.argsort(importance)[::-1]
        ordered = sorted_idx

    figure_map = _normalize_figures(figures)
    ranking_rows = _compose_ranking_rows(feature_names,
                                       importance,
                                       shap_values,
                                       features,
                                       ordered,
                                       top_k)
    global_summary = _summarize_vector(np.abs(shap_values).ravel()) if shap_values.size else None
    target_summary = _summarize_vector(targets)
    dependence_rows = _dependence_rows(ranking_rows)
    local_cases = _local_examples(shap_values,
                                  features,
                                  feature_names,
                                  sample_labels,
                                  local_limit)

    lines: List[str] = []
    lines.append('# SHAP Explainability Report')
    lines.append('')
    lines.append(f'- Generated at: {timestamp}')
    lines.append(f'- Samples: {n_samples}')
    lines.append(f'- Descriptors evaluated: {n_features}')
    lines.append(f'- Task: {task.capitalize()} | Surrogate estimator: {estimator_name}')
    lines.append(f'- Feature scaling: {scaler_label} | Explainer mode: {explainer_meta.get("mode", "tree")} (background={explainer_meta.get("background_size", "auto")})')
    if expected_value is not None and math.isfinite(expected_value):
        lines.append(f'- Base value (expected prediction): {expected_value:.6f}')
    if target_meta.get('encoder'):
        encoder = target_meta['encoder']
        mapping = ', '.join(f"{label}->{code}" for label, code in encoder.items())
        lines.append(f'- Target encoding: {mapping}')
    if figure_map:
        figure_summary = ', '.join(f"{key}: {len(paths)} file(s)" for key, paths in figure_map.items())
        lines.append(f'- Figures available: {figure_summary}')
    if report_settings:
        settings_summary = ', '.join(f"{k}={v}" for k, v in report_settings.items())
        lines.append(f'- Report overrides: {settings_summary}')
    lines.append('')

    lines.extend(_methodology_section(task))

    lines.append('## Model performance snapshot')
    if metrics:
        lines.extend(_render_metrics_table(metrics, task))
    else:
        lines.append('Model metrics unavailable; verify that the surrogate estimator finished training.')
    if target_summary:
        lines.append('')
        lines.append('| Statistic | Target values |')
        lines.append('| --- | --- |')
        for label in ('count', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'std'):
            lines.append(f'| {label} | {_format_stat(target_summary, label)} |')
    lines.append('')

    lines.append('## Global descriptor ranking (mean |SHAP|)')
    if ranking_rows:
        lines.append('| Rank | Descriptor | Mean \|SHAP\| | Share (%) | Positive (%) | Value range |')
        lines.append('| --- | --- | --- | --- | --- | --- |')
        for row in ranking_rows:
            value_range = _format_range(row['value_range'])
            lines.append(f"| {row['rank']} | `{row['name']}` | {row['mean_abs']:.6f} | {row['share']:.2f} | {row['positive_pct']:.1f} | {value_range} |")
        champion = ranking_rows[0]
        lines.append('')
        lines.append(f"- `{champion['name']}` explains {champion['share']:.2f}% of the model variance on its own (mean |SHAP| = {champion['mean_abs']:.6f}), making it the first descriptor to keep when trimming the panel.")
        if len(ranking_rows) > 1:
            tail = ranking_rows[-1]
            lines.append(f"- `{tail['name']}` closes the top-{len(ranking_rows)} window with {tail['share']:.2f}% share, indicating diminishing returns after this cutoff.")
    else:
        lines.append('No descriptors provided valid SHAP values; check the descriptor matrix and targets before rerunning.')
    lines.append('')
    lines.extend(_embed_figures('importance', figure_map.get('importance', []), 'Global SHAP importance plot'))
    lines.append('')

    lines.append('## Distribution across molecules (beeswarm perspective)')
    if global_summary:
        lines.append(f"- Median |SHAP|={global_summary['median']:.6f} with IQR {global_summary['q3'] - global_summary['q1']:.6f}; descriptors beyond Q3 drive the swarm spread.")
        lines.append(f"- Extremes span [{global_summary['min']:.6f}, {global_summary['max']:.6f}], so the color-coded swarm should reveal both stabilizing (negative SHAP) and activating (positive SHAP) cases.")
    lines.extend(_embed_figures('beeswarm', figure_map.get('beeswarm', []), 'SHAP beeswarm distribution'))
    lines.append('')

    lines.append('## Dependence sweeps (feature value vs. SHAP)')
    if dependence_rows:
        lines.append('| Descriptor | Corr(value, SHAP) | Value median | SHAP median | Impact span |')
        lines.append('| --- | --- | --- | --- | --- |')
        for row in dependence_rows:
            corr_text = f"{row['corr']:.3f}" if math.isfinite(row['corr']) else 'n/a'
            span_text = f"{row['impact_span']:.6f}" if math.isfinite(row['impact_span']) else 'n/a'
            lines.append(f"| `{row['name']}` | {corr_text} | {row['value_median']:.4f} | {row['shap_median']:.6f} | {span_text} |")
        lines.append('')
        lines.append('- Positive correlations indicate descriptors whose higher values push the prediction upward; negative correlations dampen the response.')
    else:
        lines.append('Dependence rows unavailable; ensure that SHAP values and descriptor matrices share the same shape.')
    lines.extend(_embed_figures('dependence', figure_map.get('dependence', []), 'SHAP dependence grid'))
    lines.append('')

    lines.append('## Local narratives (highest absolute SHAP impacts)')
    if local_cases:
        lines.append('| Sample | Descriptor | SHAP | Value | Interpretation |')
        lines.append('| --- | --- | --- | --- | --- |')
        for case in local_cases:
            lines.append(f"| `{case['sample']}` | `{case['descriptor']}` | {case['impact']:.6f} | {case['value']:.4f} | {case['direction']} |")
        lines.append('')
        lines.append('- Use these samples to validate whether experimental anomalies follow the SHAP direction (positive SHAP pushes predictions higher, negative values pull them lower).')
    else:
        lines.append('Not enough SHAP data to highlight local narratives.')
    lines.append('')

    lines.append('## Generated artifacts')
    lines.append('- Figures:')
    if figure_map:
        for label, paths in figure_map.items():
            for path in paths:
                lines.append(f"  - `{label}` -> `{path}`")
    else:
        lines.append('  - No figures detected in this run.')
    lines.append('- Markdown report: this file.')

    return lines


def _estimator_label(estimator: Any) -> str:
    if estimator is None:
        return 'unknown'
    return type(estimator).__name__


def _methodology_section(task: str) -> List[str]:
    lines = []
    lines.append('## Methodological framing')
    lines.append('- SHAP (SHapley Additive exPlanations) decomposes each prediction into a base value plus descriptor contributions, mirroring the Evaluate -> Rank -> Cut mindset used elsewhere in AutoChemDescriptors.')
    lines.append('- The surrogate estimator learns a predictive shortcut on top of the descriptor matrix; TreeSHAP/KernelSHAP then quantify how each descriptor perturbs the prediction relative to the background distribution.')
    if task == 'classification':
        lines.append('- For classification tasks, positive SHAP values push the model toward the monitored class (column `class_index`), whereas negative values suppress it; the beeswarm colors expose class probabilities across molecules.')
    else:
        lines.append('- For regression tasks, SHAP values share the same units as the predicted target, enabling direct traceability between descriptor swings and property improvements.')
    lines.append('')
    return lines


def _render_metrics_table(metrics: Dict[str, float], task: str) -> List[str]:
    ordered = _ordered_metric_names(task, metrics)
    if not ordered:
        return ['Model metrics computed but no recognized keys to display.']
    lines = ['| Metric | Value |', '| --- | --- |']
    for name in ordered:
        lines.append(f"| {name} | {_format_metric(metrics[name])} |")
    lines.append('')
    return lines


def _ordered_metric_names(task: str, metrics: Dict[str, float]) -> List[str]:
    priority = ['accuracy', 'balanced_accuracy', 'f1', 'roc_auc'] if task == 'classification' else ['r2', 'rmse', 'mae']
    ordered = [name for name in priority if name in metrics]
    extras = [name for name in metrics.keys() if name not in priority]
    return ordered + sorted(extras)


def _format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'n/a'
    return f'{value:.4f}'


def _summarize_vector(values: np.ndarray) -> Optional[Dict[str, float]]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return {
        'count': float(finite.size),
        'min': float(np.min(finite)),
        'q1': float(np.percentile(finite, 25)) if finite.size > 1 else float(finite[0]),
        'median': float(np.median(finite)),
        'q3': float(np.percentile(finite, 75)) if finite.size > 1 else float(finite[0]),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)) if finite.size > 1 else 0.0,
    }


def _format_stat(summary: Optional[Dict[str, float]], label: str) -> str:
    if summary is None:
        return 'n/a'
    value = summary.get(label)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return 'n/a'
    if label == 'count':
        return str(int(value))
    return f'{value:.6f}'


def _compose_ranking_rows(feature_names: Sequence[str],
                          importance: np.ndarray,
                          shap_values: np.ndarray,
                          features: np.ndarray,
                          ordered_indices: np.ndarray,
                          limit: int) -> List[Dict[str, Any]]:
    if importance.size == 0 or len(feature_names) == 0:
        return []
    finite_idx = np.where(np.isfinite(importance))[0]
    if finite_idx.size == 0:
        return []
    if ordered_indices.size == 0:
        ordered_indices = finite_idx[np.argsort(importance[finite_idx])[::-1]]
    denom = float(np.nansum(importance[finite_idx]))
    denom = denom if denom > 0 else 1.0
    rows: List[Dict[str, Any]] = []
    limit = max(1, min(limit, len(feature_names)))
    for rank, idx in enumerate(ordered_indices[:limit], start=1):
        if idx >= len(feature_names):
            continue
        shap_column = shap_values[:, idx] if shap_values.ndim == 2 and shap_values.shape[1] > idx else np.array([])
        value_column = features[:, idx] if features.ndim == 2 and features.shape[1] > idx else np.array([])
        positive_pct = float(np.mean(shap_column > 0) * 100.0) if shap_column.size else float('nan')
        shap_median = float(np.median(shap_column)) if shap_column.size else float('nan')
        impact_span = _impact_span(shap_column)
        corr = _safe_corr(value_column, shap_column)
        row = {
            'rank': rank,
            'name': feature_names[idx],
            'mean_abs': float(importance[idx]),
            'share': float(importance[idx]) / denom * 100.0,
            'positive_pct': positive_pct if math.isfinite(positive_pct) else float('nan'),
            'value_range': (_safe_min(value_column), _safe_max(value_column)),
            'value_median': float(np.median(value_column)) if value_column.size else float('nan'),
            'shap_median': shap_median,
            'corr': corr,
            'impact_span': impact_span,
        }
        rows.append(row)
    return rows


def _safe_min(values: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    finite = values[np.isfinite(values)]
    return float(np.min(finite)) if finite.size else float('nan')


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    finite = values[np.isfinite(values)]
    return float(np.max(finite)) if finite.size else float('nan')


def _impact_span(values: np.ndarray) -> float:
    if values.size == 0:
        return float('nan')
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float('nan')
    p95 = np.percentile(finite, 95)
    p5 = np.percentile(finite, 5)
    return float(p95 - p5)


def _dependence_rows(ranking_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in ranking_rows[:max(1, len(ranking_rows))]:
        rows.append({
            'name': row['name'],
            'corr': row['corr'],
            'value_median': row['value_median'],
            'shap_median': row['shap_median'],
            'impact_span': row['impact_span'],
        })
    return rows


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size == 0 or y.size == 0:
        return float('nan')
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float('nan')
    corr = np.corrcoef(x[mask], y[mask])[0, 1]
    return float(corr)


def _format_range(value_range: tuple) -> str:
    low, high = value_range
    if not math.isfinite(low) or not math.isfinite(high):
        return 'n/a'
    return f'{low:.4f} -> {high:.4f}'


def _local_examples(shap_values: np.ndarray,
                    features: np.ndarray,
                    feature_names: Sequence[str],
                    sample_labels: Sequence[str],
                    limit: int) -> List[Dict[str, Any]]:
    if shap_values.ndim != 2 or shap_values.size == 0:
        return []
    n_samples, n_features = shap_values.shape
    flat = np.abs(shap_values).reshape(-1)
    mask = np.isfinite(flat)
    if not np.any(mask):
        return []
    valid = np.where(mask)[0]
    limit = max(1, min(limit, valid.size))
    order = valid[np.argsort(flat[valid])[::-1][:limit]]
    rows: List[Dict[str, Any]] = []
    for flat_idx in order:
        sample_idx = flat_idx // n_features
        feature_idx = flat_idx % n_features
        shap_value = shap_values[sample_idx, feature_idx]
        feature_value = features[sample_idx, feature_idx] if features.ndim == 2 and features.shape[1] > feature_idx else float('nan')
        label = sample_labels[sample_idx] if sample_idx < len(sample_labels) else f"sample_{sample_idx + 1}"
        direction = 'positive impact (raises target)' if shap_value > 0 else 'negative impact (lowers target)'
        rows.append({
            'sample': label,
            'descriptor': feature_names[feature_idx],
            'impact': float(shap_value),
            'value': float(feature_value) if math.isfinite(feature_value) else float('nan'),
            'direction': direction,
        })
    return rows


def _normalize_figures(figures: Dict[str, Any]) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    for key, value in figures.items():
        if isinstance(value, str):
            normalized[key] = [value]
        elif isinstance(value, Sequence):
            normalized[key] = [str(entry) for entry in value]
    return normalized


def _embed_figures(label: str, files: Sequence[str], caption: str) -> List[str]:
    lines: List[str] = []
    if not files:
        return lines
    for idx, path in enumerate(files, start=1):
        suffix = '' if len(files) == 1 else f' ({idx})'
        lines.append(f'![{caption}{suffix}]({path})')
    lines.append('')
    return lines
