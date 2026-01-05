"""Plot orchestration for the PCAPG workflow."""

from __future__ import annotations

from typing import Dict, List, Sequence

from .plot_pcapg_feature_importance import plot_pcapg_feature_importance
from .plot_pcapg_manifold import plot_pcapg_manifold
from .plot_pcapg_convergence import plot_pcapg_convergence
from .plot_pcapg_biplot import plot_pcapg_biplot


def render_pcapg_figures(feature_names: Sequence[str],
                         payload,
                         analysis: Dict[str, object],
                         config: Dict[str, object]) -> Dict[str, str | None]:
    """Generate all publication-grade figures for PCAPG."""
    scores = payload.feature_scores
    ordered = payload.ordered_indices
    embedding = payload.embedding
    similarity = payload.similarity_matrix
    history = payload.history

    top_features = min(int(config.get('top_features', 25)), len(feature_names))
    labels = _resolve_sample_labels(analysis, embedding.shape[0])

    figures: Dict[str, str | None] = {}
    figures['feature_importance'] = plot_pcapg_feature_importance(
        feature_names,
        scores,
        ordered,
        top_features,
        str(config.get('ranking_plot_filename', 'plot_pcapg_importance.png')),
    )

    figures['manifold'] = plot_pcapg_manifold(
        embedding,
        similarity,
        labels,
        payload.component_order,
        str(config.get('manifold_plot_filename', 'plot_pcapg_manifold.png')),
    )

    figures['biplot'] = plot_pcapg_biplot(
        embedding,
        payload.projection_matrix,
        ordered,
        feature_names,
        top_features,
        payload.component_order,
        str(config.get('biplot_plot_filename', 'plot_pcapg_biplot.png')),
    )

    figures['convergence'] = plot_pcapg_convergence(
        history,
        str(config.get('convergence_plot_filename', 'plot_pcapg_convergence.png')),
    )

    return figures


def _resolve_sample_labels(analysis: Dict[str, object], n_samples: int) -> List[str]:
    labels = analysis.get('molecules_label') or analysis.get('molecules_labels')
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        text = [str(label) for label in labels]
        if len(text) == n_samples:
            return text
    return [f"sample_{idx + 1}" for idx in range(n_samples)]
