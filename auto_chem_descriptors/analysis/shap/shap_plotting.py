"""Orchestrate the publication-grade SHAP visualizations."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .plot_shap_importance import plot_shap_importance
from .plot_shap_beeswarm import plot_shap_beeswarm
from .plot_shap_dependence import plot_shap_dependence_grid


def plot_shap_diagnostics(payload: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, List[str] | str]:
    feature_names = payload['feature_names']
    shap_values = payload['shap_values']
    features = payload['feature_matrix']
    importance = payload['importance']
    ordered = payload['ordered_indices']
    targets = payload['targets']

    top_features = max(3, min(int(config.get('top_features', 12)), len(feature_names)))
    top_indices = ordered[:top_features]

    importance_file = plot_shap_importance(
        feature_names,
        importance,
        top_indices,
        str(config.get('importance_plot_filename', 'plot_shap_importance.png')),
    )

    beeswarm_file = plot_shap_beeswarm(
        feature_names,
        features,
        shap_values,
        top_indices,
        str(config.get('beeswarm_plot_filename', 'plot_shap_beeswarm.png')),
    )

    dependence_count = max(1, int(config.get('dependence_plots', 2)))
    dependence_filename = str(config.get('dependence_plot_filename', 'plot_shap_dependence.png'))
    dependence_file = plot_shap_dependence_grid(
        feature_names,
        features,
        shap_values,
        targets,
        top_indices,
        dependence_count,
        dependence_filename,
    )

    return {
        'importance': importance_file,
        'beeswarm': beeswarm_file,
        'dependence': dependence_file,
    }
