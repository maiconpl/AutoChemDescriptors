#!/usr/bin/python3
"""
Markdown reporter for Laplacian Score and Marginal Laplacian Score outputs.

The goal is to keep narrative/analysis concerns separated from plotting and
numerical computation. The report follows the same spirit as the K-Means
documentation module while focusing on the spectral feature selection context.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import math

import numpy as np


_BALANCED_SKEW_THRESHOLD = 0.1


Payload = Dict[str, Any]


def generate_laplacian_report(feature_names: Sequence[str],
                              payload: Payload,
                              analysis: Optional[Dict[str, Any]] = None,
                              config: Optional[Dict[str, Any]] = None) -> str:
    """Write a Markdown report highlighting LS/MLS insights."""

    report_settings = _resolve_report_settings(analysis)
    report_filename = str(report_settings.get('report_filename', 'report_laplacian_score.md'))
    top_k = int(report_settings.get('top_descriptors',
                                    (config or {}).get('top_descriptors', 20)))

    feature_names = list(feature_names)
    ls_scores = np.asarray(payload.get('ls_scores', []), dtype=float)
    mls_scores = np.asarray(payload.get('mls_scores', []), dtype=float)
    skewness = np.asarray(payload.get('skewness', []), dtype=float)
    coverage = np.asarray(payload.get('margin_coverage', []), dtype=float)
    marginal_labels = np.asarray(payload.get('marginal_labels', []), dtype=str)
    graph_profile = payload.get('graph_profile', {}) or {}

    lines = _build_report_lines(feature_names,
                                ls_scores,
                                mls_scores,
                                skewness,
                                coverage,
                                marginal_labels,
                                graph_profile,
                                config or {},
                                report_settings,
                                top_k)

    with open(report_filename, 'w', encoding='utf-8') as handler:
        handler.write("\n".join(lines))

    return report_filename


def _resolve_report_settings(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    candidate = analysis.get('laplacian_score_report')
    if isinstance(candidate, dict):
        return candidate
    return {}


def _build_report_lines(feature_names: List[str],
                        ls_scores: np.ndarray,
                        mls_scores: np.ndarray,
                        skewness: np.ndarray,
                        coverage: np.ndarray,
                        marginal_labels: np.ndarray,
                        graph_profile: Dict[str, Any],
                        config: Dict[str, Any],
                        report_settings: Dict[str, Any],
                        top_k: int) -> List[str]:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n_features = len(feature_names)
    n_samples = int(graph_profile.get('n_samples', ls_scores.shape[0]))

    ls_summary = _summarize_vector(ls_scores)
    mls_summary = _summarize_vector(mls_scores)
    coverage_summary = _summarize_vector(coverage[np.isfinite(mls_scores)]) if np.isfinite(mls_scores).any() else None

    ls_top = _select_top_features(ls_scores, feature_names, top_k)
    mls_top = _select_top_features(mls_scores, feature_names, top_k, coverage, marginal_labels)

    profile_distribution = _profile_distribution(marginal_labels, n_features)
    skew_highlights = _skew_highlights(feature_names, skewness)

    quantile = float(config.get('quantile', 0.90))
    skew_right = float(config.get('skew_right', 0.5))
    skew_left = float(config.get('skew_left', -0.5))
    mode = str(config.get('mode', 'both')).lower()

    csv_filename = config.get('csv_filename', 'laplacian_scores.csv')
    ls_plot = config.get('ls_plot_filename', 'plot_laplacian_ls.png')
    mls_plot = config.get('mls_plot_filename', 'plot_laplacian_mls.png')

    lines: List[str] = []
    lines.append('# Laplacian Score Feature Ranking Report')
    lines.append('')
    lines.append(f'- Generated at: {timestamp}')
    lines.append(f'- Samples (vertices): {n_samples}')
    lines.append(f'- Descriptors ranked: {n_features}')
    lines.append(f'- Graph metric: {graph_profile.get("metric", "auto")} | k-neighbors: {graph_profile.get("k_neighbors", "?")}')
    kernel_desc = 'adaptive heat kernel' if graph_profile.get('adaptive_kernel', True) else 'global heat kernel'
    lines.append(f'- Kernel: {kernel_desc} | sigma_med: {graph_profile.get("sigma_median", "n/a")} | heat_parameter: {graph_profile.get("heat_parameter", "auto")}')
    lines.append(f'- Laplacian modes computed: {mode.upper()} (quantile={quantile:.2f}, skew thresholds=[{skew_left:.2f}, {skew_right:.2f}])')
    lines.append(f'- Ranking CSV: `{csv_filename}` | LS plot: `{ls_plot}` | MLS plot: `{mls_plot}`')
    lines.append('')

    lines.extend(_methodology_section())

    lines.append('## Spectral graph configuration')
    lines.append(f'- Metric selection resolved to **{graph_profile.get("metric", "auto") }** to honor descriptor sign constraints and binary fingerprints.')
    lines.append(f'- Neighborhood size fixed at **k={graph_profile.get("k_neighbors", "?")}**, striking a balance between manifold locality and graph connectivity for n={n_samples}.')
    lines.append(f'- Adaptive kernel active: {graph_profile.get("adaptive_kernel", True)}; sigma_median={graph_profile.get("sigma_median", "n/a")} ensures locality adapts to dense/sparse chemical regions.')
    if graph_profile.get('heat_parameter') is not None:
        lines.append(f'- Heat parameter injected explicitly ({graph_profile.get("heat_parameter")}); diffusion scale therefore anchored to practitioner prior instead of automatic sigma product.')
    lines.append('')

    lines.append('## Score distribution summary')
    if ls_summary or mls_summary:
        lines.append('| Statistic | Laplacian Score | Marginal Laplacian Score |')
        lines.append('| --- | --- | --- |')
        for label in ('count', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'std'):
            ls_value = _format_stat(ls_summary, label)
            mls_value = _format_stat(mls_summary, label)
            lines.append(f'| {label} | {ls_value} | {mls_value} |')
    else:
        lines.append('No finite scores were produced; verify descriptor variance before re-running the ranking module.')
    lines.append('')

    lines.append('## Evaluate → Rank → Cut (Laplacian Score focus)')
    if ls_summary:
        lines.append('### Evaluation')
        q1 = ls_summary['q1']
        q3 = ls_summary['q3']
        iqr = q3 - q1
        finite_mask = np.isfinite(ls_scores)
        total_valid = int(np.sum(finite_mask))
        below_q1 = int(np.sum(finite_mask & (ls_scores < q1)))
        above_q3 = int(np.sum(finite_mask & (ls_scores > q3)))
        tail_pct_low = (below_q1 / total_valid * 100.0) if total_valid else 0.0
        tail_pct_high = (above_q3 / total_valid * 100.0) if total_valid else 0.0
        lines.append(f"- Median LS = {ls_summary['median']:.6f}; descriptors below this value keep neighboring molecules nearly indistinguishable in the spectral graph.")
        lines.append(f"- Interquartile span (Q1={q1:.6f}, Q3={q3:.6f}) gives an IQR of {iqr:.6f}, quantifying how tightly manifold-preserving signals cluster.")
        lines.append(f"- Mean vs. median delta ({ls_summary['mean']:.6f} − {ls_summary['median']:.6f}) reveals the magnitude of noisy descriptors pulling the average upward.")
        lines.append(f"- Distribution pressure: {below_q1} descriptors ({tail_pct_low:.1f}%) sit below Q1 (excellent smoothness) while {above_q3} ({tail_pct_high:.1f}%) exceed Q3 and are prime candidates for removal.")
        lines.append('')
    lines.append('### Ranking')
    if ls_top:
        lines.append('| Rank | Descriptor | LS |')
        lines.append('| --- | --- | --- |')
        for entry in ls_top:
            lines.append(f"| {entry['rank']} | `{entry['name']}` | {entry['score']:.6f} |")
        highlight = ls_top[0]
        lines.append('')
        lines.append(f"- `{highlight['name']}` delivers the minimum LS ({highlight['score']:.6f}), proving it respects the Evaluate → Rank → Cut mandate by minimizing local scatter in the Tanimoto graph.")
    else:
        lines.append('No valid Laplacian scores were computed; remove constant descriptors or fill missing values before re-running the ranking pipeline.')
    lines.append('')
    lines.append('### Cut recommendation')
    ls_cut = _build_cut_recommendation(ls_scores, feature_names, ls_summary)
    lines.extend(_render_cut_recommendation(ls_cut, score_label='LS'))
    if ls_plot:
        lines.append('')
        lines.append(f"![Laplacian Score ranking chart]({ls_plot})")
    lines.append('')

    lines.append('## Marginal Laplacian Score (rare chemistry emphasis)')
    if np.isfinite(mls_scores).any():
        lines.append(f'- Marginal quantile targets the top {quantile * 100:.1f}% (or symmetric tails) depending on skew classification, with thresholds [{skew_left:.2f}, {skew_right:.2f}].')
        if coverage_summary:
            lines.append(f"- Average marginal coverage: {coverage_summary['mean'] * 100:.2f}% (median {coverage_summary['median'] * 100:.2f}%).")
        if mls_summary:
            lines.append('### Evaluation')
            q1_m = mls_summary['q1']
            q3_m = mls_summary['q3']
            iqr_m = q3_m - q1_m
            finite_mask_m = np.isfinite(mls_scores)
            total_valid_m = int(np.sum(finite_mask_m))
            below_q1_m = int(np.sum(finite_mask_m & (mls_scores < q1_m)))
            above_q3_m = int(np.sum(finite_mask_m & (mls_scores > q3_m)))
            tail_pct_low_m = (below_q1_m / total_valid_m * 100.0) if total_valid_m else 0.0
            tail_pct_high_m = (above_q3_m / total_valid_m * 100.0) if total_valid_m else 0.0
            lines.append(f"- Median MLS = {mls_summary['median']:.6f}; lower-than-median tails preserve minority manifolds, while higher scores fracture them.")
            lines.append(f"- Marginal IQR (Q1={q1_m:.6f}, Q3={q3_m:.6f}) spans {iqr_m:.6f}; descriptors beyond this range typically fail to keep rare scaffolds cohesive.")
            lines.append(f"- Tail census: {below_q1_m} descriptors ({tail_pct_low_m:.1f}%) already excel at rare-chemistry fidelity, whereas {above_q3_m} ({tail_pct_high_m:.1f}%) distort the marginal structure and should be pruned first.")
            lines.append('')
        lines.append('### Ranking')
        lines.append('| Rank | Descriptor | MLS | Coverage (%) | Marginal profile |')
        lines.append('| --- | --- | --- | --- | --- |')
        for entry in mls_top:
            cov_percent = entry.get('coverage', float('nan')) * 100.0
            lines.append(f"| {entry['rank']} | `{entry['name']}` | {entry['score']:.6f} | {cov_percent:.2f} | {entry.get('profile', 'n/a')} |")
        if mls_top:
            champion = mls_top[0]
            champion_cov = champion.get('coverage', float('nan'))
            cov_text = f" covering {champion_cov * 100:.2f}% of samples" if champion_cov is not None and math.isfinite(champion_cov) else ''
            lines.append('')
            lines.append(f"- `{champion['name']}` is the most reliable descriptor for rare-chemistry navigation (MLS={champion['score']:.6f}); it keeps the {champion.get('profile', 'tail')} tail compact{cov_text}.")
        lines.append('')
        lines.append('### Cut recommendation')
        mls_cut = _build_cut_recommendation(mls_scores, feature_names, mls_summary)
        lines.extend(_render_cut_recommendation(mls_cut, score_label='MLS', tail_context=True))
        if mls_plot:
            lines.append('')
            lines.append(f"![Marginal Laplacian Score ranking chart]({mls_plot})")
    else:
        lines.append('MLS mode was disabled or produced no finite scores. Enable `mode="both"` (default) or `"mls"` if tail preservation is required for scaffold hopping scenarios.')
    lines.append('')

    lines.append('## Skewness and marginal profile statistics')
    lines.append('| Profile | Count | Percentage (%) | Interpretation |')
    lines.append('| --- | --- | --- | --- |')
    for entry in profile_distribution:
        lines.append(f"| {entry['profile']} | {entry['count']} | {entry['percentage']:.2f} | {entry['comment']} |")
    lines.append('')
    if skew_highlights['right']:
        lines.append('- Right-skew exemplars (extreme high-activity tails): ' + ', '.join(f"`{name}` ({value:.2f})" for name, value in skew_highlights['right']))
    if skew_highlights['left']:
        lines.append('- Left-skew exemplars (low-value rarity): ' + ', '.join(f"`{name}` ({value:.2f})" for name, value in skew_highlights['left']))
    if skew_highlights['balanced']:
        lines.append('- Balanced descriptors feeding two-sided margins: ' + ', '.join(f"`{name}` (|skew|<{abs(value):.2f})" for name, value in skew_highlights['balanced']))
    lines.append('')

    lines.append('## Generated artifacts')
    lines.append(f'- Ranking CSV: `{csv_filename}` consolidates LS/MLS, skewness, coverage, and marginal category for all {n_features} descriptors.')
    lines.append(f'- Figures: `{ls_plot}` (core manifold fidelity) and `{mls_plot}` (tail sensitivity).')
    lines.append(f"- Markdown report: `{report_settings.get('report_filename', 'report_laplacian_score.md')}` (this file).")
    if report_settings:
        lines.append('- Report settings override detected: ' + ', '.join(f"`{k}`={v}" for k, v in report_settings.items()))
    else:
        lines.append('- Report relied exclusively on defaults triggered by `analysis["laplacian_score"]`.')

    return lines


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
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'n/a'
    if label == 'count':
        return str(int(value))
    return f'{value:.6f}'


def _select_top_features(scores: np.ndarray,
                         feature_names: Sequence[str],
                         limit: int,
                         coverage: Optional[np.ndarray] = None,
                         labels: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    finite_idx = np.where(np.isfinite(scores))[0]
    if finite_idx.size == 0:
        return []
    order = finite_idx[np.argsort(scores[finite_idx])]
    entries: List[Dict[str, Any]] = []
    for rank, idx in enumerate(order[:max(1, limit)], start=1):
        entry = {'rank': rank, 'name': str(feature_names[idx]), 'score': float(scores[idx])}
        if coverage is not None and coverage.size > idx:
            entry['coverage'] = float(coverage[idx])
        if labels is not None and len(labels) > idx:
            entry['profile'] = str(labels[idx])
        entries.append(entry)
    return entries


def _profile_distribution(labels: np.ndarray, total_features: int) -> List[Dict[str, Any]]:
    if labels.size == 0:
        return [{'profile': 'unknown', 'count': total_features, 'percentage': 100.0, 'comment': 'No marginal labels exported.'}]
    counter = Counter(labels.tolist())
    entries: List[Dict[str, Any]] = []
    for profile in ('right', 'left', 'two-sided', 'disabled'):
        count = counter.get(profile, 0)
        if count == 0 and profile == 'disabled' and len(counter) == 1:
            count = total_features
        percentage = (count / total_features * 100.0) if total_features else 0.0
        comment = _profile_comment(profile)
        entries.append({'profile': profile, 'count': count, 'percentage': percentage, 'comment': comment})
    other = counter.keys() - {'right', 'left', 'two-sided', 'disabled'}
    for profile in sorted(other):
        count = counter[profile]
        percentage = (count / total_features * 100.0) if total_features else 0.0
        entries.append({'profile': profile, 'count': count, 'percentage': percentage, 'comment': 'custom classification from MLS routine'})
    return entries


def _profile_comment(profile: str) -> str:
    mapping = {
        'right': 'Tail defined by descriptors with positive skew (high-value rarity).',
        'left': 'Tail defined by negative skew (low-value rarity).',
        'two-sided': 'Symmetric or near-symmetric descriptors contribute both tails.',
        'disabled': 'MLS disabled or insufficient variance for marginal extraction.',
    }
    return mapping.get(profile, 'custom marginal classification')


def _skew_highlights(feature_names: Sequence[str], skewness: np.ndarray, limit: int = 3) -> Dict[str, List[tuple]]:
    highlights = {'right': [], 'left': [], 'balanced': []}
    if skewness.size == 0:
        return highlights
    # Right skew
    right_idx = np.argsort(-skewness)
    for idx in right_idx[:limit]:
        if skewness[idx] <= 0:
            break
        highlights['right'].append((feature_names[idx], float(skewness[idx])))
    left_idx = np.argsort(skewness)
    for idx in left_idx[:limit]:
        if skewness[idx] >= 0:
            break
        highlights['left'].append((feature_names[idx], float(skewness[idx])))
    if highlights['right'] or highlights['left']:
        balanced_mask = np.abs(skewness) < 0.1
        balanced_idx = np.where(balanced_mask)[0][:limit]
        for idx in balanced_idx:
            highlights['balanced'].append((feature_names[idx], float(skewness[idx])))
    return highlights


def _methodology_section() -> List[str]:
    lines = []
    lines.append('## Methodological framing')
    lines.append('- Laplacian Score (LS) is a **filter method**: Evaluate → Rank → Cut. Each descriptor is examined independently, ordered from lower to higher LS (lower preserves manifold structure), and only then do we decide what to keep or drop.')
    lines.append('- The manifold structure is captured through a k-NN graph built with a heat kernel operating on the chemical similarity metric (Tanimoto for fingerprints or Euclidean for continuous descriptors).')
    lines.append('- The numerator $f^T L f$ penalizes descriptors that oscillate inside neighborhoods, while the denominator normalizes by global scatter to avoid rewarding constant columns.')
    lines.append('- Marginal Laplacian Score (MLS) accentuates descriptors that keep rare chemistry (distribution tails) coherent—critical for scaffold hopping and activity cliff investigations in imbalanced datasets.')
    lines.append('')
    return lines
def _build_cut_recommendation(scores: np.ndarray,
                              feature_names: Sequence[str],
                              summary: Optional[Dict[str, float]]) -> Dict[str, Any]:
    finite = np.isfinite(scores)
    if not np.any(finite):
        return {'status': 'undefined'}
    finite_scores = scores[finite]
    threshold = float(np.percentile(finite_scores, 75)) if finite_scores.size > 1 else float(finite_scores[0])
    cut_mask = scores <= threshold
    kept = [(feature_names[idx], scores[idx]) for idx in np.where(cut_mask & finite)[0]]
    dropped = [(feature_names[idx], scores[idx]) for idx in np.where(~cut_mask & finite)[0]]
    return {
        'status': 'ok',
        'threshold': threshold,
        'median': summary['median'] if summary else None,
        'kept': kept,
        'dropped': dropped,
    }


def _render_cut_recommendation(cut_info: Dict[str, Any],
                               score_label: str,
                               tail_context: bool = False) -> List[str]:
    if cut_info.get('status') != 'ok':
        return ['- Cut policy could not be derived (scores missing or undefined).']
    lines: List[str] = []
    threshold = cut_info['threshold']
    lines.append(f"- Proposed cutoff: keep descriptors with {score_label} ≤ {threshold:.6f} (Q3-based).")
    if cut_info.get('median') is not None:
        lines.append(f"- Median {score_label} = {cut_info['median']:.6f}; descriptors above the cutoff exceed the Evaluate → Rank → Cut interquartile expectation.")
    kept_ratio = len(cut_info['kept']) / (len(cut_info['kept']) + len(cut_info['dropped']) or 1)
    lines.append(f"- Action: retain {len(cut_info['kept'])} descriptors ({kept_ratio * 100:.1f}%) and discard {len(cut_info['dropped'])} ({(1 - kept_ratio) * 100:.1f}%).")
    if cut_info['dropped']:
        descriptor_list = ', '.join(f"`{name}` ({value:.4f})" for name, value in cut_info['dropped'])
        reason = 'they disrupt tail coherence needed for rare chemistry.' if tail_context else 'they inject excessive local scatter.'
        lines.append(f"- Drop rationale: {descriptor_list} because {reason}")
    if cut_info['kept']:
        keep_list = ', '.join(f"`{name}` ({value:.4f})" for name, value in cut_info['kept'])
        merit = 'their MLS scores stay within the tail smoothness band.' if tail_context else 'they minimize LS variance across neighbor pairs.'
        lines.append(f"- Keep rationale: {keep_list} since {merit}")
    return lines
