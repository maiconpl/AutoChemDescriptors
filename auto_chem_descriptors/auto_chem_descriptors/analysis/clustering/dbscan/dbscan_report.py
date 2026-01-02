#!/usr/bin/python3
'''
Created on February 15, 2026.

Markdown reporting for DBSCAN density-clustering diagnostics.
'''

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import math

ReportPayload = Dict[str, Any]
ClusterRow = Dict[str, Any]


def generate_dbscan_report(payload: ReportPayload,
                           analysis: Optional[Dict[str, Any]] = None) -> str:
    '''Generate a Markdown summary for a DBSCAN run.'''
    validated = _validate_payload(payload)
    report_settings = _extract_report_settings(analysis)
    report_filename = report_settings.get('report_filename', 'report_dbscan.md')

    lines = _build_report_lines(validated, report_settings)

    with open(report_filename, 'w', encoding='utf-8') as handler:
        handler.write("\n".join(lines))

    return report_filename


def _validate_payload(payload: ReportPayload) -> ReportPayload:
    required = ['stats', 'labels', 'classification', 'sample_labels', 'artifacts',
                'n_bits', 'is_binary', 'profile', 'kth_distances', 'projection_variance',
                'eps_source', 'k_distance_curve']
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"DBSCAN report payload is missing keys: {missing}")

    stats = payload['stats']
    expected_keys = ['n_samples', 'n_clusters', 'cluster_sizes', 'noise_count',
                     'noise_ratio', 'core_count', 'border_count', 'min_samples',
                     'eps_suggested', 'eps_used', 'metric_mode', 'knee',
                     'k_distance_summary', 'warnings']
    missing_stats = [key for key in expected_keys if key not in stats]
    if missing_stats:
        raise ValueError(f"DBSCAN stats payload is missing keys: {missing_stats}")

    labels = payload['labels']
    classification = payload['classification']
    sample_labels = payload['sample_labels']
    kth_distances = payload['kth_distances']

    n_samples = stats['n_samples']
    if not (len(labels) == len(classification) == len(sample_labels) == len(kth_distances) == n_samples):
        raise ValueError("labels, classification, sample_labels, and kth_distances must match n_samples")

    return payload


def _extract_report_settings(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    settings = analysis.get('dbscan_report')
    if settings is None:
        return {}
    if isinstance(settings, list) and settings:
        tail = settings[-1]
        settings = tail if isinstance(tail, dict) else {}
    elif isinstance(settings, bool):
        settings = {}
    if not isinstance(settings, dict):
        raise ValueError("analysis['dbscan_report'] must be a dictionary or list ending with a dictionary.")
    return settings


def _build_report_lines(payload: ReportPayload, report_settings: Dict[str, Any]) -> List[str]:
    stats = payload['stats']
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    descriptor_mode = "binary bitvectors" if payload['is_binary'] else "dense fingerprints (shifted to non-negative values)"
    shift_note = payload['profile'].get('shift_offsets')

    lines: List[str] = []
    lines.append("# DBSCAN Density Analysis Report")
    lines.append("")
    lines.append("DBSCAN groups points by reachability inside an eps-radius neighborhood that must contain `min_samples` observations. Points meeting this density criterion become `core`, their immediate but underpopulated neighbors are `border`, and isolated observations are labeled `noise`. Because it optimizes connectivity instead of spherical variance, DBSCAN exposes elongated or irregular structures while automatically flagging sparse outliers.")
    lines.append("")
    lines.append("When reviewed alongside the K-Means pipeline, the density perspective validates whether centroid-based partitions are truly supported by local neighborhoods. Agreement between the two methods indicates a fairly homogeneous descriptor space; disagreement highlights scaffolds that K-Means condensed into a single centroid or molecules that fail to reach critical connectivity. This report captures that complementary view in a modular fashion.")
    lines.append("")
    lines.append(f"- Generated at: {timestamp}")
    lines.append(f"- Samples: {stats['n_samples']} | Fingerprint bits: {payload['n_bits']}")
    lines.append(f"- Descriptor view: {descriptor_mode}")
    if shift_note:
        lines.append("- Offsets were applied to enforce non-negative columns before computing Tanimoto distances.")
    lines.append(f"- Min samples: {stats['min_samples']} | Metric mode: {stats['metric_mode']}")
    lines.append(f"- eps suggested (knee): {stats['eps_suggested']:.5f} | eps used: {stats['eps_used']:.5f} ({payload['eps_source']})")
    lines.append(f"- Cluster count: {stats['n_clusters']} | Noise ratio: {stats['noise_ratio']*100:.2f}%")
    artifacts = payload['artifacts']
    lines.append(f"- Artifacts: parameters `{artifacts['parameters']}`, stats `{artifacts['stats']}`, labels `{artifacts['labels']}`")
    lines.append(f"- Plots: k-distance `{artifacts['k_distance_plot']}`, clusters `{artifacts['cluster_plot']}`")
    lines.append("")

    lines.extend(_render_reachability_section(payload))
    lines.append("")

    cluster_rows = _build_cluster_inventory(payload['labels'],
                                           payload['classification'],
                                           payload['sample_labels'])
    lines.extend(_render_cluster_section(stats, cluster_rows))
    lines.append("")

    lines.extend(_render_classification_section(stats))
    lines.append("")

    noise_limit = _parse_positive_int(report_settings.get('noise_limit'), 12)
    lines.extend(_render_sample_listing(payload, 'noise', noise_limit, title="Noise molecules"))

    border_limit = _parse_positive_int(report_settings.get('border_limit'), 10)
    lines.extend(_render_sample_listing(payload, 'border', border_limit, title="Borderline molecules"))

    sparse_limit = _parse_positive_int(report_settings.get('sparse_limit'), 8)
    lines.extend(_render_sparse_tail(payload, sparse_limit))

    lines.extend(_render_projection_section(payload['projection_variance']))

    lines.extend(_render_warning_section(stats['warnings']))

    lines.extend(_render_settings_section(report_settings))

    lines.extend(_render_extended_interpreter(payload, cluster_rows))
    lines.extend(_render_generic_decision_guide())

    return lines


def _render_reachability_section(payload: ReportPayload) -> List[str]:
    stats = payload['stats']
    summary = stats['k_distance_summary']
    knee = stats['knee'] or {}
    eps_used = stats['eps_used']
    eps_suggested = stats['eps_suggested']
    diff = eps_used - eps_suggested
    percent = (diff / eps_suggested * 100) if eps_suggested else float('inf')

    lines: List[str] = []
    lines.append("## Reachability diagnostics")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Suggested eps (knee) | {eps_suggested:.5f} |")
    lines.append(f"| Adopted eps | {eps_used:.5f} |")
    lines.append(f"| Delta eps | {diff:+.5f} ({percent:+.2f}%) |")
    lines.append(f"| Knee detector | {knee.get('method', 'n/a')} @ index {knee.get('index', 'n/a')} |")
    lines.append(f"| Median k-distance | {summary['median']:.5f} |")
    lines.append(f"| Mean k-distance | {summary['mean']:.5f} |")
    lines.append(f"| Min k-distance | {summary['min']:.5f} |")
    lines.append(f"| Max k-distance | {summary['max']:.5f} |")
    lines.append("")

    interpretation = _describe_reachability(stats)
    lines.extend(interpretation)
    lines.append("")
    return lines


def _describe_reachability(stats: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    summary = stats['k_distance_summary']
    eps_used = stats['eps_used']
    eps_suggested = stats['eps_suggested']
    med = summary['median']
    spread = summary['max'] - summary['min']

    lines.append("### Interpretation")
    if eps_used > med * 1.25:
        lines.append("- eps sits well above the median k-distance, allowing border points on elongated scaffolds to coalesce instead of being flagged as noise.")
    elif eps_used < med * 0.9:
        lines.append("- eps is below the median k-distance, enforcing conservative clusters and aggressive noise labeling.")
    else:
        lines.append("- eps closely tracks the median k-distance, yielding balanced sensitivity to dense regions and sparse extremities.")

    if spread < 0.02 * max(summary['max'], 1e-6):
        lines.append("- The k-distance curve is flat, indicating nearly uniform densities; DBSCAN will behave similarly to single-link clustering.")
    else:
        lines.append("- The k-distance spread is wide, so density thresholds discriminate sharply between well-populated scaffolds and sparse molecules.")

    if stats['noise_ratio'] > 0.35:
        lines.append("- More than one-third of the data fall outside dense neighborhoods, signaling either genuine novelty or descriptors that need scaling.")
    elif stats['noise_ratio'] < 0.05:
        lines.append("- Noise is minimal; descriptors form well-connected manifolds amenable to centroid-based models.")

    return lines


def _build_cluster_inventory(labels: Sequence[int],
                             classification: Sequence[str],
                             sample_labels: Sequence[str]) -> List[ClusterRow]:
    total = len(labels)
    rows: List[ClusterRow] = []
    for cluster_id in sorted({label for label in labels if label != -1}):
        members = [idx for idx, value in enumerate(labels) if value == cluster_id]
        if not members:
            continue
        core_count = sum(1 for idx in members if classification[idx] == 'core')
        border_count = sum(1 for idx in members if classification[idx] == 'border')
        examples = [sample_labels[idx] for idx in members[:3]]
        rows.append({
            'cluster': cluster_id,
            'count': len(members),
            'percentage': (len(members) / total * 100.0) if total else 0.0,
            'core': core_count,
            'border': border_count,
            'core_ratio': core_count / len(members) if members else 0.0,
            'border_ratio': border_count / len(members) if members else 0.0,
            'examples': examples,
        })
    return sorted(rows, key=lambda item: item['count'], reverse=True)


def _render_cluster_section(stats: Dict[str, Any], cluster_rows: List[ClusterRow]) -> List[str]:
    lines: List[str] = []
    lines.append("## Cluster inventory")
    if not cluster_rows:
        lines.append("- No clusters beyond noise were identified. Consider lowering `min_samples` or increasing eps to capture tenuous scaffolds.")
        return lines

    lines.append("| Cluster | Count | Share (%) | Core | Border | Core ratio | Examples |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in cluster_rows:
        examples = ", ".join(f"`{name}`" for name in row['examples']) if row['examples'] else "n/a"
        lines.append(f"| {row['cluster']} | {row['count']} | {row['percentage']:.2f} | {row['core']} | {row['border']} | {row['core_ratio']*100:.1f}% | {examples} |")
    lines.append("")
    lines.extend(_describe_cluster_balance(stats, cluster_rows))
    lines.append("")
    return lines


def _describe_cluster_balance(stats: Dict[str, Any], cluster_rows: List[ClusterRow]) -> List[str]:
    lines: List[str] = []
    lines.append("### Notes")
    if not cluster_rows:
        lines.append("- All points defaulted to noise; DBSCAN requires denser neighborhoods for this descriptor set.")
        return lines

    largest = cluster_rows[0]
    if largest['percentage'] > 45:
        lines.append(f"- Cluster {largest['cluster']} concentrates {largest['percentage']:.1f}% of the data, so downstream models should treat it as dominant chemistry.")
    else:
        lines.append(f"- Cluster {largest['cluster']} holds {largest['percentage']:.1f}% of the samples, indicating a balanced distribution across clusters.")

    tail = cluster_rows[-1]
    if tail['count'] <= stats['min_samples']:
        lines.append(f"- Cluster {tail['cluster']} barely exceeds `min_samples` ({tail['count']} molecules), making it sensitive to eps adjustments.")

    core_fraction = stats['core_count'] / stats['n_samples'] if stats['n_samples'] else 0.0
    if core_fraction > 0.7:
        lines.append("- Most molecules are core points, reinforcing strongly connected motifs in descriptor space.")
    elif core_fraction < 0.4:
        lines.append("- Core points are scarce relative to border members, so consider lowering `min_samples` or revisiting descriptor scaling.")

    return lines


def _render_classification_section(stats: Dict[str, Any]) -> List[str]:
    total = stats['n_samples']
    noise = stats['noise_count']
    core = stats['core_count']
    border = stats['border_count']
    lines: List[str] = []
    lines.append("## Reachability inventory")
    lines.append("| Category | Count | Share (%) |")
    lines.append("| --- | --- | --- |")
    for label, count in (("Core", core), ("Border", border), ("Noise", noise)):
        share = (count / total * 100.0) if total else 0.0
        lines.append(f"| {label} | {count} | {share:.2f} |")
    lines.append("")
    if noise > border and noise > core:
        lines.append("- Noise dominates the classification; samples often fail to reach the connectivity threshold.")
    elif border > core:
        lines.append("- Border points outnumber cores, so clusters are filamented rather than compact.")
    else:
        lines.append("- Core points anchor most clusters, with border molecules forming cohesive shells.")
    lines.append("")
    return lines


def _render_sample_listing(payload: ReportPayload,
                           category: str,
                           limit: int,
                           title: str) -> List[str]:
    lines: List[str] = []
    samples = _collect_samples_by_category(payload, category)
    if not samples:
        return lines
    lines.append(f"## {title}")
    lines.append("| Sample | Cluster | k-distance |")
    lines.append("| --- | --- | --- |")
    for entry in samples[:limit]:
        lines.append(f"| `{entry['label']}` | {entry['cluster']} | {entry['distance']:.5f} |")
    if len(samples) > limit:
        lines.append(f"| ... | ... | ... (showing {limit} of {len(samples)}) |")
    lines.append("")
    if category == 'noise':
        lines.append("- These molecules lie outside any eps-connected region and act as density-driven outliers.")
    elif category == 'border':
        lines.append("- Border molecules reach a core neighbor but lack `min_samples` support, marking scaffold peripheries.")
    lines.append("")
    return lines


def _collect_samples_by_category(payload: ReportPayload, category: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for label, cluster_id, classification, distance in zip(payload['sample_labels'],
                                                          payload['labels'],
                                                          payload['classification'],
                                                          payload['kth_distances']):
        if classification != category:
            continue
        results.append({
            'label': label,
            'cluster': cluster_id,
            'distance': distance if isinstance(distance, float) else float(distance)
        })
    return sorted(results, key=lambda item: item['distance'], reverse=(category != 'border'))


def _render_sparse_tail(payload: ReportPayload, limit: int) -> List[str]:
    entries = []
    for label, cluster_id, distance in zip(payload['sample_labels'],
                                           payload['labels'],
                                           payload['kth_distances']):
        entries.append({'label': label, 'cluster': cluster_id, 'distance': distance})
    entries.sort(key=lambda item: item['distance'], reverse=True)
    lines: List[str] = []
    if not entries:
        return lines
    lines.append("## Sparse k-distance tail")
    lines.append("| Sample | Cluster | k-distance |")
    lines.append("| --- | --- | --- |")
    for entry in entries[:limit]:
        lines.append(f"| `{entry['label']}` | {entry['cluster']} | {entry['distance']:.5f} |")
    lines.append("")
    lines.append("- Large k-distances signal molecules that only reach distant neighbors; monitor them when adjusting eps or refining descriptors.")
    lines.append("")
    return lines


def _render_projection_section(variance: Sequence[float]) -> List[str]:
    lines: List[str] = []
    if not variance:
        return lines
    lines.append("## Visualization PCA summary")
    explained = [f"PC{i+1}: {value*100:.2f}%" for i, value in enumerate(variance[:2])]
    lines.append("- Projection variance (first two PCs): " + ", ".join(explained))
    if sum(variance[:2]) < 0.5:
        lines.append("- Less than 50% of the variance is captured in the 2D plot; interpret separations qualitatively rather than quantitatively.")
    else:
        lines.append("- The 2D projection retains most of the variance, so relative positions align with density neighborhoods.")
    lines.append("")
    return lines


def _render_warning_section(warnings: Sequence[str]) -> List[str]:
    lines: List[str] = []
    if not warnings:
        return lines
    lines.append("## Fingerprint warnings")
    for message in warnings:
        lines.append(f"- {message}")
    lines.append("- Address the warnings above to avoid degenerate distances or inflated noise counts.")
    lines.append("")
    return lines


def _render_settings_section(report_settings: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append("## Report settings")
    if report_settings:
        for key, value in report_settings.items():
            lines.append(f"- `{key}`: {value}")
    else:
        lines.append("- Default report settings were used.")
    lines.append("")
    return lines


def _parse_positive_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        value_int = int(value)
        return value_int if value_int > 0 else default
    except (TypeError, ValueError):
        return default


def _render_extended_interpreter(payload: ReportPayload, cluster_rows: List[ClusterRow]) -> List[str]:
    stats = payload['stats']
    n_samples = stats['n_samples'] or 0
    core_ratio = stats['core_count'] / n_samples if n_samples else 0.0
    border_ratio = stats['border_count'] / n_samples if n_samples else 0.0
    noise_ratio = stats['noise_ratio'] if isinstance(stats['noise_ratio'], (int, float)) else 0.0
    eps_used = stats['eps_used']
    eps_suggested = stats['eps_suggested']
    eps_delta_pct = ((eps_used - eps_suggested) / eps_suggested * 100.0) if eps_suggested else float('inf')
    k_distances = payload['kth_distances']
    sample_labels = payload['sample_labels']
    classification = payload['classification']
    variance = payload.get('projection_variance') or []
    retained_variance = sum(variance[:2]) * 100 if variance else None

    border_samples = [label for label, cls in zip(sample_labels, classification) if cls == 'border']
    max_distance_entry = None
    if k_distances:
        max_idx = int(max(range(len(k_distances)), key=lambda idx: k_distances[idx]))
        max_distance_entry = {
            'label': sample_labels[max_idx],
            'distance': k_distances[max_idx],
            'cluster': payload['labels'][max_idx]
        }

    lines: List[str] = []
    lines.append("## Density interpreter")
    lines.append("This section situates the DBSCAN results relative to PCA projections and centroid-based clustering, then derives chemoinformatics conclusions specific to this run.")
    lines.append("")

    lines.append("### PCA vs. K-Means vs. DBSCAN")
    if retained_variance is not None:
        lines.append(f"- **PCA (the map):** The first two principal components retain {retained_variance:.2f}% of the variance, so the cluster plot mirrors the descriptor space faithfully.")
    else:
        lines.append("- **PCA (the map):** Variance retention for the scatter plot is unavailable; interpret spatial separations qualitatively.")
    lines.append("- **K-Means (the partition):** Centroid-based grouping assumes spherical, equal-variance clouds. Use it to stress-test whether DBSCAN's single density component can be artificially split by forcing K ≥ 2.")
    lines.append("- **DBSCAN (the density lens):** Connectivity overrides centroids; molecules remain grouped as long as reachability paths exist, enabling detection of elongated or manifold-shaped series.")
    lines.append("")

    lines.append("### Direct density verdict")
    if stats['n_clusters'] == 1 and noise_ratio < 0.05:
        lines.append("- The library behaves as a continuous family: DBSCAN detected a single cluster with negligible noise, so structural gradations stay connected across the descriptor manifold.")
    elif stats['n_clusters'] > 1 and noise_ratio < 0.2:
        lines.append("- Multiple dense clusters emerge without excessive noise, indicating chemotypes that separate cleanly under the chosen reachability threshold.")
    elif noise_ratio >= 0.2:
        lines.append("- Noise dominates significant portions of the data; descriptors either encode multiple isolated motifs or require a larger eps to bridge sparse regions.")
    lines.append(f"- Core ratio: {core_ratio*100:.1f}% | Border ratio: {border_ratio*100:.1f}% | Noise ratio: {noise_ratio*100:.1f}%.")
    if border_samples:
        formatted = ", ".join(f"`{name}`" for name in border_samples[:5])
        lines.append(f"- Border molecules (peripheral yet connected): {formatted}{' ...' if len(border_samples) > 5 else ''}.")
    else:
        lines.append("- No border molecules were detected; all points either anchor dense regions or stand as noise.")
    if max_distance_entry:
        lines.append(f"- Most isolated neighbor: `{max_distance_entry['label']}` (cluster {max_distance_entry['cluster']}) with k-distance {max_distance_entry['distance']:.4f}; lowering eps below this value would turn it into noise unless another dense core appears.")
    lines.append("")

    lines.append("### Epsilon discrepancy")
    lines.append(f"- Knee suggestion: {eps_suggested:.5f} | Adopted eps: {eps_used:.5f} | Delta: {eps_delta_pct:+.1f}%.")
    if eps_delta_pct > 500:
        lines.append("- eps sits far above the knee, intentionally relaxing the density requirement to keep peripheral scaffolds connected.")
    elif eps_delta_pct < -20:
        lines.append("- eps is substantially tighter than the knee suggestion, prioritizing dense cores at the cost of larger noise fractions.")
    else:
        lines.append("- eps aligns with the knee recommendation, balancing sensitivity to dense nuclei and tolerance to sparse peripheries.")
    lines.append("- Monitor how the noise ratio shifts if eps approaches the knee; dramatic swings reveal cliff effects in the k-distance distribution.")
    lines.append("")

    lines.append("### Chemoinformatics discussion")
    if stats['n_clusters'] == 1 and border_samples:
        lines.append("- The dataset forms a structural monolith with a graded periphery. Border molecules often demarcate the applicability-domain frontier while still reinforcing QSAR training density.")
    elif stats['n_clusters'] > 1:
        lines.append("- Distinct density islands correspond to chemotypes or scaffold series; evaluate each cluster separately for SAR trends and synthetic priorities.")
    else:
        lines.append("- Absence of clusters implies descriptors encode either a sparse library or an overly strict parameterization; revisit eps/min_samples before drawing SAR conclusions.")
    lines.append("- Cross-check whether PCA scatter, DBSCAN labels, and any available K-Means partitions agree. Consensus indicates high confidence that the geometry reflects true chemical continuity rather than projection artifacts.")
    lines.append("- When noise is negligible, all samples remain eligible for local regression or similarity-driven design without pruning.")
    lines.append("")
    return lines


def _render_generic_decision_guide() -> List[str]:
    lines: List[str] = []
    lines.append("## DBSCAN decision playbook")
    lines.append("Use this checklist whenever interpreting a new density report without prior knowledge of the molecules.")
    lines.append("")

    lines.append("### Checklist")
    lines.append("1. **Expansion factor:** Compare eps_used to the knee suggestion and to the median k-distance. Staying between the two keeps reachability realistic; deltas above ~1000% indicate inherently fragmented spaces.")
    lines.append("2. **Density health:** Target core ratios above 80% with noise below 10% for congeners. Noise above 20% or clusters smaller than `min_samples` point to diversified screening libraries.")
    lines.append("3. **Sparse tail scan:** Inspect the last few entries of the k-distance tail. Large jumps (e.g., 0.15 → 0.40) flag molecules that define the applicability boundary.")
    lines.append("4. **Cross-method sanity check:** Ensure PCA plots and any centroid-based clusters do not contradict DBSCAN labels; disagreements often trace back to scaling or descriptor mismatches.")
    lines.append("")

    lines.append("### Diagnostic matrix")
    lines.append("| Report pattern | Diagnosis | Suggested action |")
    lines.append("| --- | --- | --- |")
    lines.append("| Single cluster + 0% noise | Chemical monolith | Lower eps incrementally to probe hidden subfamilies. |")
    lines.append("| Multiple clusters + low noise | Defined chemotypes | Preserve eps/min_samples; clusters represent actionable series. |")
    lines.append("| Many clusters + high noise | Fragmented space | Increase eps or reduce min_samples to reconnect sparsely populated motifs. |")
    lines.append("| Single cluster + high noise | Core with satellites | Focus SAR on Cluster 0 and treat noise as exploratory outliers. |")
    lines.append("")

    lines.append("### Parameter tuning guide")
    lines.append("- **To uncover more groups:** Decrease eps to break tenuous similarity bridges.")
    lines.append("- **To enforce stricter cores/outliers:** Increase min_samples so only heavily populated motifs survive.")
    lines.append("- **To merge fragmented regions:** Raise eps gradually (increments ≈0.05 in Tanimoto space) until reachability paths appear.")
    lines.append("")

    lines.append("### Closing reminder")
    lines.append("PCA provides the artistic view, K-Means offers the administrative partition, and DBSCAN delivers the physical reality of who reaches whom. Treat DBSCAN as the density validator before formalizing QSAR or generative conclusions.")
    lines.append("")
    return lines
