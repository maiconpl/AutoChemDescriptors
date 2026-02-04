"""Markdown report generator for the PCAPG feature selection workflow."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np

from .pcapg_processing import PCAPGPayload
from .pcapg_axes import select_projection_axes


FigureMap = Dict[str, List[str]]


def generate_pcapg_report(feature_names: Sequence[str],
                          payload: PCAPGPayload,
                          config: Dict[str, Any],
                          analysis: Optional[Dict[str, Any]] = None,
                          figures: Optional[Dict[str, str | None]] = None,
                          csv_filename: Optional[str] = None) -> str:
    """Compose the PCAPG Markdown summary and persist it to disk."""

    report_settings = _resolve_report_settings(analysis)
    default_filename = str(config.get('report_filename', 'report_pcapg.md'))
    report_filename = str(report_settings.get('report_filename', default_filename))
    top_default = int(config.get('top_features', 25) or 25)
    top_k = int(report_settings.get('top_descriptors', top_default))

    figure_map = _normalize_figures(figures)
    sample_labels = _resolve_sample_labels(analysis, payload.feature_matrix.shape[0])
    lines = _build_report_lines(feature_names,
                                payload,
                                config,
                                report_settings,
                                figure_map,
                                csv_filename,
                                top_k,
                                sample_labels)

    with open(report_filename, 'w', encoding='utf-8') as handler:
        handler.write("\n".join(lines))

    return report_filename


def _resolve_report_settings(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(analysis, dict):
        return {}
    candidate = analysis.get('pcapg_report')
    if isinstance(candidate, dict):
        return candidate
    return {}


def _build_report_lines(feature_names: Sequence[str],
                        payload: PCAPGPayload,
                        config: Dict[str, Any],
                        report_settings: Dict[str, Any],
                        figures: FigureMap,
                        csv_filename: Optional[str],
                        top_k: int,
                        sample_labels: Sequence[str]) -> List[str]:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    feature_names = list(feature_names)
    n_samples, n_features = payload.feature_matrix.shape
    n_components = payload.projection_matrix.shape[1]
    scaling_mode = 'standard' if payload.scaler is not None else str(config.get('scaling', 'none'))

    scores = np.asarray(payload.feature_scores, dtype=float)
    ordered = np.asarray(payload.ordered_indices, dtype=int)
    descriptor_matrix = np.asarray(payload.feature_matrix, dtype=float)
    embedding = np.asarray(payload.embedding, dtype=float)
    similarity = np.asarray(payload.similarity_matrix, dtype=float)
    history = payload.history or {}
    graph_profile = payload.graph_profile or {}

    score_summary = _summarize_vector(scores)
    ranking_rows = _ranking_rows(feature_names,
                                 scores,
                                 ordered,
                                 descriptor_matrix,
                                 embedding,
                                 top_k)
    coverage_pct = ranking_rows[-1]['cum_share'] if ranking_rows else 0.0

    objective = np.asarray(history.get('objective', []), dtype=float)
    reconstruction = np.asarray(history.get('reconstruction', []), dtype=float)
    obj_stats = _objective_summary(objective, reconstruction)

    graph_stats = _graph_weight_summary(similarity, graph_profile)
    embedding_profile = _embedding_energy_profile(payload.component_variances,
                                                  payload.component_order)
    energy_ratio = _energy_ratio(payload.scaled_matrix, embedding)

    alpha = float(config.get('alpha', 0.7))
    beta = float(config.get('beta', 0.05))
    lambda_reg = float(config.get('lambda_reg', 1.0))
    gamma = float(config.get('possibilistic_sharpness', 0.35))
    k_neighbors = int(config.get('n_neighbors', 8))
    tol = float(config.get('tol', 1e-4))
    max_iter = int(config.get('max_iter', 60))
    random_state = config.get('random_state', 42)

    csv_label = csv_filename or config.get('csv_filename', 'pcapg_feature_scores.csv')

    lines: List[str] = []
    lines.append('# PCAPG Feature Selection Report')
    lines.append('')
    lines.append(f'- Generated at: {timestamp}')
    lines.append(f'- Samples: {n_samples} | Descriptors evaluated: {n_features}')
    lines.append(f'- Latent components: {n_components} | Scaling applied: {scaling_mode}')
    lines.append(f'- Hyperparameters: α={alpha:.3f}, β={beta:.3f}, λ={lambda_reg:.3f}, γ={gamma:.3f}, k={k_neighbors}')
    lines.append(f'- Optimization: tol={tol:.1e}, max_iter={max_iter}, iterations used={obj_stats["iterations"]}')
    lines.append(f'- Random state: {random_state} | Ranking CSV: `{csv_label}`')
    if figures:
        figure_summary = ', '.join(f"{key}: {len(paths)} file(s)" for key, paths in figures.items())
        lines.append(f'- Figures available: {figure_summary}')
    if report_settings:
        overrides = ', '.join(f"{k}={v}" for k, v in report_settings.items())
        lines.append(f'- Report overrides: {overrides}')
    lines.append('')

    lines.extend(_methodology_section())

    lines.append('## Optimization diagnostics')
    lines.append('| Metric | Value |')
    lines.append('| --- | --- |')
    lines.append(f'| Initial objective | {_format_float(obj_stats["initial_objective"])} |')
    lines.append(f'| Final objective | {_format_float(obj_stats["final_objective"])} |')
    lines.append(f'| Relative drop (%) | {_format_float(obj_stats["relative_drop"])} |')
    lines.append(f'| Final reconstruction error | {_format_float(obj_stats["final_reconstruction"])} |')
    lines.append(f'| Energy captured by embedding (%) | {_format_float(energy_ratio * 100.0)} |')
    lines.append(f'| Graph density | {_format_float(graph_stats["density"])} |')
    lines.append(f'| Average degree | {_format_float(graph_stats["avg_degree"])} |')
    lines.append(f'| Active edges | {int(graph_stats["active_edges"])} |')
    lines.append('')
    drop_text = _format_percentage(obj_stats['relative_drop'])
    lines.append(f'- Objective stabilized after {obj_stats["iterations"]} iterations; the {drop_text} drop confirms that the alternating minimization converged.')
    if graph_stats['weight_q2'] is not None:
        lines.append(f'- Learned affinities remain possibilistic: median edge weight = {graph_stats["weight_q2"]:.4f} with upper quartile {graph_stats["weight_q3"]:.4f}, preventing noisy bridges among molecules.')
    if figures.get('convergence'):
        for path in figures['convergence']:
            lines.append('')
            lines.append('- *Figure:* The convergence panel charts the objective (blue) and reconstruction error (red) trajectories, making the stabilization plateau visually explicit.')
            lines.append(f'![PCAPG convergence diagnostics]({path})')
    lines.append('')

    lines.append('## Descriptor ranking window')
    if ranking_rows:
        lines.append('| Rank | Descriptor | ‖Wᵢ‖₂ | Share (%) | Cum. share (%) | Value span | Corr(comp₁) | z-score |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- | --- |')
        for row in ranking_rows:
            lines.append(
                f"| {row['rank']} | `{row['name']}` | {row['score']:.6f} | {row['share']:.2f} | "
                f"{row['cum_share']:.2f} | {row['value_span']} | "
                f"{_format_corr(row['corr_comp1'])} | {row['zscore']:.2f} |"
            )
        lines.append('')
        champion = ranking_rows[0]
        lines.append(f"- `{champion['name']}` controls {champion['share']:.2f}% of the PCAPG energy budget (‖Wᵢ‖₂={champion['score']:.6f}); it is the first descriptor to keep when trimming redundancy.")
        lines.append(f"- Top-{len(ranking_rows)} descriptors already explain {coverage_pct:.2f}% of the projection norm, enabling a {100 - coverage_pct:.2f}% cut without noticeable reconstruction loss.")
        if score_summary:
            lines.append(f"- Score dispersion: median={score_summary['median']:.6f}, IQR={score_summary['q3'] - score_summary['q1']:.6f}; descriptors beyond Q3 behave as noisy manifolds and should be pruned.")
    else:
        lines.append('No descriptor ranking was produced; check whether the descriptor matrix passed validation and try again.')
    if figures.get('feature_importance'):
        for path in figures['feature_importance']:
            lines.append('')
            lines.append('- *Figure:* The descriptor relevance bar plot orders ‖Wᵢ‖₂ scores from least to most informative, highlighting the sparse loading spectrum referenced above.')
            lines.append(f'![PCAPG feature importance]({path})')
    lines.append('')
    lines.extend(_render_biplot_section(figures,
                                        ranking_rows,
                                        embedding,
                                        payload.projection_matrix,
                                        payload.component_order,
                                        sample_labels))
    lines.append('')

    lines.append('## Graph topology and manifold preservation')
    lines.append('| Statistic | Value |')
    lines.append('| --- | --- |')
    lines.append(f'| Avg degree | {_format_float(graph_stats["avg_degree"])} |')
    lines.append(f'| Max degree | {_format_float(graph_stats["max_degree"])} |')
    lines.append(f'| Min degree | {_format_float(graph_stats["min_degree"])} |')
    lines.append(f'| Density | {_format_float(graph_stats["density"])} |')
    lines.append(f'| Active edge ratio (%) | {_format_float(graph_stats["edge_ratio"] * 100.0)} |')
    if graph_stats['weight_q1'] is not None:
        lines.append(f'| Edge weight quartiles | Q1={graph_stats["weight_q1"]:.4f}, Q2={graph_stats["weight_q2"]:.4f}, Q3={graph_stats["weight_q3"]:.4f} |')
    lines.append('')
    lines.append('- Possibilistic weighting suppresses outliers: low-degree samples fall below the adaptive γ exponent, preventing them from corrupting local manifolds.')
    lines.append(f'- Minimum spanning and local edges (see manifold plot) reveal {graph_stats["active_edges"]:.0f} trusted interactions out of {graph_stats["possible_edges"]:.0f} possible pairs.')
    if figures.get('manifold'):
        for path in figures['manifold']:
            lines.append('')
            lines.append('- *Figure:* The manifold layout overlays the learned possibilistic graph on the 2-D embedding; node colors encode adaptive typicality while curated edges expose high-confidence chemical neighborhoods.')
            lines.append(f'![PCAPG manifold projection]({path})')
    lines.append('')

    lines.append('## Embedding energy profile')
    if embedding_profile:
        lines.append('| Component | Variance | Share (%) |')
        lines.append('| --- | --- | --- |')
        for entry in embedding_profile:
            lines.append(f"| PCAPG component {entry['component']} | {entry['variance']:.6f} | {entry['share']:.2f} |")
        lines.append('')
        lines.append('- Components with higher variance share carry the chemical families detected by the learned graph; consider retaining dimensions until the cumulative share exceeds ~90%.')
    else:
        lines.append('Embedding statistics unavailable; ensure at least one latent component was computed.')
    lines.append('')

    lines.extend(_figure_insights(similarity,
                                  graph_stats,
                                  sample_labels,
                                  obj_stats,
                                  ranking_rows,
                                  coverage_pct))
    lines.append('')

    lines.append('## Generated artifacts')
    lines.append(f'- Ranking CSV: `{csv_label}`')
    if figures:
        for key, paths in figures.items():
            for path in paths:
                lines.append(f"- Figure ({key}): `{path}`")
    else:
        lines.append('- Figures: none generated.')
    lines.append('- Markdown report: this file.')

    return lines


def _normalize_figures(figures: Optional[Dict[str, str | None]]) -> FigureMap:
    figure_map: FigureMap = {}
    if not isinstance(figures, dict):
        return figure_map
    for key, value in figures.items():
        if value:
            figure_map.setdefault(key, []).append(str(value))
    return figure_map


def _methodology_section() -> List[str]:
    lines: List[str] = []
    lines.append('## Methodological framing')
    lines.append('- PCAPG unites PCA reconstruction with possibilistic graph learning, ensuring that the selected descriptors preserve both global variance and adaptive local topology.')
    lines.append('- The learned graph is re-estimated at every iteration, so noisy AutoChemDescriptors signals receive near-zero affinity and cannot dominate the manifold.')
    lines.append('- A row-wise L₂,₁ penalty on the projection forces descriptors with redundant information to vanish, yielding an interpretable subset rather than abstract latent components.')
    lines.append('')
    return lines


def _figure_insights(similarity: np.ndarray,
                     graph_stats: Dict[str, float],
                     sample_labels: Sequence[str],
                     obj_stats: Dict[str, float],
                     ranking_rows: List[Dict[str, Any]],
                     coverage_pct: float) -> List[str]:
    lines: List[str] = []
    lines.append('## Automated figure insights')
    lines.append(f"- **Possibilistic graph-preserving embedding:** Typicality spans "
                 f"[{_format_float(graph_stats['min_degree'])}, {_format_float(graph_stats['max_degree'])}] "
                 f"with average {_format_float(graph_stats['avg_degree'])}. "
                 f"{_manifold_commentary(similarity, sample_labels)}")
    lines.append(f"- **Convergence diagnostics:** Alternating minimization used {obj_stats['iterations']} iterations; "
                 f"objective drop of {_format_percentage(obj_stats['relative_drop'])} "
                 f"and final reconstruction {_format_float(obj_stats['final_reconstruction'])} confirm numerical stability.")
    if ranking_rows:
        champion = ranking_rows[0]
        tail = ranking_rows[-1]
        lines.append(f"- **Descriptor relevance:** `{champion['name']}` dominates the loading spectrum "
                     f"(share {champion['share']:.2f}%), while `{tail['name']}` closes the window. "
                     f"The plotted set retains {coverage_pct:.2f}% of the projection norm, signalling a sparse, "
                     f"chemically focused panel.")
    else:
        lines.append("- **Descriptor relevance:** Ranking unavailable; check descriptor variance before interpreting the bar chart.")
    return lines


def _render_biplot_section(figures: FigureMap,
                           ranking_rows: List[Dict[str, Any]],
                           embedding: np.ndarray,
                           projection: np.ndarray,
                           component_order: np.ndarray,
                           sample_labels: Sequence[str]) -> List[str]:
    lines: List[str] = []
    lines.append('## Structural driver analysis (biplot)')
    biplot_paths = figures.get('biplot', [])
    if biplot_paths:
        for path in biplot_paths:
            lines.append('- *Figure:* The projected loading biplot overlays the manifold coordinates and the descriptor vectors so sample and variable spaces converge in a single view.')
            lines.append(f'![PCAPG projected loading biplot]({path})')
    else:
        lines.append('- Biplot figure unavailable for this run; verify that the descriptor loadings contain non-zero variance before regenerating.')

    if ranking_rows:
        biplot_table = _biplot_table(ranking_rows,
                                     embedding,
                                     projection,
                                     component_order,
                                     sample_labels)
        if biplot_table:
            lines.append('')
            lines.append('| Descriptor | Vector length | Axis alignment | Cluster affinity |')
            lines.append('| --- | --- | --- | --- |')
            for row in biplot_table:
                lines.append(f"| `{row['descriptor']}` | {row['length']:.3f} | {row['alignment']} | {row['cluster_hint']} |")
            lines.append('')
        else:
            lines.append('- Descriptor loadings could not be summarized; inspect the projection matrix for singular columns.')
    else:
        lines.append('- Descriptor ranking unavailable; rerun the PCAPG module after validating descriptor variance.')

    lines.append('Sample–variable linkage: the biplot fuses the sample scores and descriptor loadings. Points that sit near the tip of a vector exhibit elevated values for that descriptor, so cluster formation becomes immediately traceable to chemical variables instead of separate correlation tables.')
    lines.append('Orthogonality vs. redundancy: the angular disposition of the vectors double-checks the $L_{2,1}$ penalty. Acute angles reveal residual correlation (descriptors that still rise and fall together), whereas orthogonal vectors flag chemically independent families.')
    lines.append('Axis/variance diagnostics: vector lengths validate that the plotted PCAPG components carry real chemical variance. When arrows remain long, the axes in the manifold plot—and therefore the corresponding rows of W—capture genuine structural differences rather than numerical noise.')
    lines.append('')
    return lines


def _ranking_rows(feature_names: Sequence[str],
                  scores: np.ndarray,
                  ordered: np.ndarray,
                  matrix: np.ndarray,
                  embedding: np.ndarray,
                  top_k: int) -> List[Dict[str, Any]]:
    if ordered.size == 0 and scores.size:
        ordered = np.argsort(scores)[::-1]
    selected = [idx for idx in ordered.tolist() if 0 <= idx < len(feature_names)]
    selected = selected[:max(1, top_k)]
    total = float(np.sum(scores)) or 1.0
    mean = float(np.mean(scores))
    std = float(np.std(scores))
    embedding_component = embedding[:, 0] if embedding.ndim == 2 and embedding.shape[1] > 0 else None

    rows: List[Dict[str, Any]] = []
    cumulative = 0.0
    for rank, feature_idx in enumerate(selected, start=1):
        score = float(scores[feature_idx])
        share = (score / total) * 100.0
        cumulative += share
        values = matrix[:, feature_idx]
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        corr = _correlation(values, embedding_component)
        zscore = (score - mean) / std if std > 0 else 0.0
        rows.append({
            'rank': rank,
            'index': int(feature_idx),
            'name': feature_names[feature_idx],
            'score': score,
            'share': share,
            'cum_share': cumulative,
            'value_span': f"{vmin:.4f} → {vmax:.4f}",
            'corr_comp1': corr,
            'zscore': zscore,
        })
    return rows


def _summarize_vector(values: np.ndarray) -> Optional[Dict[str, float]]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return {
        'count': float(finite.size),
        'min': float(np.min(finite)),
        'q1': float(np.percentile(finite, 25)),
        'median': float(np.percentile(finite, 50)),
        'q3': float(np.percentile(finite, 75)),
        'max': float(np.max(finite)),
        'mean': float(np.mean(finite)),
        'std': float(np.std(finite)),
    }


def _objective_summary(objective: np.ndarray,
                       reconstruction: np.ndarray) -> Dict[str, float]:
    iterations = int(objective.size) if objective.size else 0
    initial = float(objective[0]) if objective.size else math.nan
    final = float(objective[-1]) if objective.size else math.nan
    drop = (initial - final) if objective.size else math.nan
    relative = (drop / abs(initial) * 100.0) if objective.size and abs(initial) > 1e-12 else math.nan
    final_recon = float(reconstruction[-1]) if reconstruction.size else math.nan
    return {
        'iterations': iterations,
        'initial_objective': initial,
        'final_objective': final,
        'relative_drop': relative,
        'final_reconstruction': final_recon,
    }


def _graph_weight_summary(similarity: np.ndarray,
                          profile: Dict[str, Any]) -> Dict[str, float]:
    n_samples = similarity.shape[0]
    upper = similarity[np.triu_indices(n_samples, 1)]
    positives = upper[upper > 0]
    possible_edges = n_samples * (n_samples - 1) / 2.0 if n_samples > 1 else 0.0
    active_edges = float(np.count_nonzero(upper > 0))
    edge_ratio = (active_edges / possible_edges) if possible_edges else 0.0
    degree_vector = similarity.sum(axis=1)
    weight_q1 = float(np.percentile(positives, 25)) if positives.size else None
    weight_q2 = float(np.percentile(positives, 50)) if positives.size else None
    weight_q3 = float(np.percentile(positives, 75)) if positives.size else None

    return {
        'avg_degree': float(np.mean(degree_vector)),
        'max_degree': float(np.max(degree_vector)) if degree_vector.size else math.nan,
        'min_degree': float(np.min(degree_vector)) if degree_vector.size else math.nan,
        'density': float(profile.get('density', edge_ratio)),
        'active_edges': active_edges,
        'possible_edges': float(possible_edges),
        'edge_ratio': float(edge_ratio),
        'weight_q1': weight_q1,
        'weight_q2': weight_q2,
        'weight_q3': weight_q3,
    }


def _embedding_energy_profile(variances: np.ndarray,
                              order: np.ndarray) -> List[Dict[str, float]]:
    if variances.size == 0:
        return []
    total = float(np.sum(variances)) or 1.0
    order = np.asarray(order, dtype=int)
    order = order[(order >= 0) & (order < variances.size)]
    if order.size == 0:
        order = np.arange(variances.size)
    profile: List[Dict[str, float]] = []
    for idx in order:
        variance = float(variances[idx])
        profile.append({
            'component': idx + 1,
            'variance': variance,
            'share': float(variance / total * 100.0),
        })
    return profile


def _energy_ratio(matrix: np.ndarray, embedding: np.ndarray) -> float:
    total_energy = float(np.linalg.norm(matrix)**2) or 1.0
    embedding_energy = float(np.linalg.norm(embedding)**2)
    return embedding_energy / total_energy


def _correlation(values: np.ndarray,
                 component: Optional[np.ndarray]) -> float:
    if component is None:
        return math.nan
    col = np.asarray(values, dtype=float).ravel()
    comp = np.asarray(component, dtype=float).ravel()
    if col.size != comp.size:
        return math.nan
    if np.std(col) == 0 or np.std(comp) == 0:
        return math.nan
    corr = np.corrcoef(col, comp)[0, 1]
    return float(corr)


def _format_float(value: float) -> str:
    if value is None or not math.isfinite(value):
        return 'n/a'
    return f'{value:.6f}'


def _format_percentage(value: float) -> str:
    if value is None or not math.isfinite(value):
        return 'n/a'
    return f'{value:.2f}%'


def _format_corr(value: float) -> str:
    if value is None or not math.isfinite(value):
        return 'n/a'
    return f'{value:.3f}'


def _biplot_table(ranking_rows: List[Dict[str, Any]],
                  embedding: np.ndarray,
                  projection: np.ndarray,
                  component_order: np.ndarray,
                  sample_labels: Sequence[str]) -> List[Dict[str, Any]]:
    if projection.ndim != 2 or projection.shape[0] == 0:
        return []
    coords, axis_labels, component_pair = select_projection_axes(embedding, component_order)
    comp_a, comp_b = component_pair
    if comp_a >= projection.shape[1] or comp_b >= projection.shape[1]:
        return []
    loadings = projection[:, [comp_a, comp_b]]
    table: List[Dict[str, Any]] = []
    for row in ranking_rows:
        idx = row.get('index')
        if idx is None or idx >= loadings.shape[0]:
            continue
        vector = loadings[idx]
        length = float(np.linalg.norm(vector))
        alignment = _alignment_label(vector, axis_labels)
        cluster_hint = _cluster_hint(coords, vector, sample_labels)
        table.append({
            'descriptor': row['name'],
            'length': length,
            'alignment': alignment,
            'cluster_hint': cluster_hint,
        })
    return table


def _alignment_label(vector: np.ndarray,
                     axis_labels: tuple[int, int]) -> str:
    eps = 1e-8
    def _symbol(value: float) -> str:
        if abs(value) < eps:
            return '≈0'
        return '+' if value > 0 else '−'

    comp_x = f"C{axis_labels[0]} {_symbol(vector[0])}"
    comp_y = f"C{axis_labels[1]} {_symbol(vector[1])}"
    return f"{comp_x}, {comp_y}"


def _cluster_hint(coords: np.ndarray,
                  vector: np.ndarray,
                  sample_labels: Sequence[str]) -> str:
    length = float(np.linalg.norm(vector))
    if coords.size == 0 or length <= 1e-9:
        return "No dominant samples"
    unit = vector / length
    projections = coords @ unit
    high_idx = int(np.argmax(projections))
    low_idx = int(np.argmin(projections))
    high_label = sample_labels[high_idx] if high_idx < len(sample_labels) else f"sample_{high_idx + 1}"
    low_label = sample_labels[low_idx] if low_idx < len(sample_labels) else f"sample_{low_idx + 1}"
    return f"high in `{high_label}`, low in `{low_label}`"


def _manifold_commentary(similarity: np.ndarray,
                         sample_labels: Sequence[str]) -> str:
    if similarity.size == 0:
        return "Graph metrics unavailable."
    degrees = similarity.sum(axis=1)
    if degrees.size == 0:
        return "Graph metrics unavailable."
    high_idx = np.argsort(degrees)[-3:][::-1]
    low_idx = np.argsort(degrees)[:3]
    typical = _format_sample_group(high_idx, sample_labels)
    atypical = _format_sample_group(low_idx, sample_labels)
    return (f"High-typicality molecules such as {typical} anchor dense clusters, "
            f"whereas {atypical} remain softly connected, signalling potential experimental noise.")


def _format_sample_group(indices: np.ndarray,
                         sample_labels: Sequence[str]) -> str:
    names = []
    for idx in indices:
        if 0 <= int(idx) < len(sample_labels):
            names.append(sample_labels[int(idx)])
    if not names:
        return "unlabeled samples"
    if len(names) == 1:
        return f"`{names[0]}`"
    if len(names) == 2:
        return f"`{names[0]}` and `{names[1]}`"
    return ", ".join(f"`{name}`" for name in names[:-1]) + f", `{names[-1]}`"


def _resolve_sample_labels(analysis: Optional[Dict[str, Any]],
                           n_samples: int) -> List[str]:
    if isinstance(analysis, dict):
        candidate = analysis.get('molecules_label') or analysis.get('molecules_labels')
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            text = [str(item) for item in candidate]
            if len(text) == n_samples:
                return text
    return [f"sample_{idx + 1}" for idx in range(n_samples)]
