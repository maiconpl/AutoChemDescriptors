"""Entry point for SHAP explainability analysis."""

from __future__ import annotations

from typing import Any, Dict

from .shap_processing import compute_shap_analysis_payload
from .shap_plotting import plot_shap_diagnostics
from .shap_report import generate_shap_report


def run_shap_analysis(descriptors_list, analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Coordinate SHAP processing and high-quality visualizations."""
    config = _extract_shap_config(analysis)
    payload = compute_shap_analysis_payload(descriptors_list, config, analysis)
    figure_filenames = plot_shap_diagnostics(payload, config)
    report_filename = generate_shap_report(payload, config, analysis, figure_filenames)
    print("SHAP explainability artifacts saved to:")
    for key, value in figure_filenames.items():
        if isinstance(value, list):
            for entry in value:
                print(f"  - figure {key}: {entry}")
        else:
            print(f"  - figure {key}: {value}")
    print(f"  - report: {report_filename}")
    return {
        'figures': figure_filenames,
        'payload': payload,
        'report_filename': report_filename,
    }


def _extract_shap_config(analysis: Dict[str, Any]) -> Dict[str, Any]:
    config = analysis.get('shap_validation')
    if isinstance(config, list) and config:
        candidate = config[-1]
        config = candidate if isinstance(candidate, dict) else {}
    elif isinstance(config, bool):
        config = {} if config else {}
    elif config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError("analysis['shap_validation'] must be a dictionary or list ending with a dictionary.")
    return config
