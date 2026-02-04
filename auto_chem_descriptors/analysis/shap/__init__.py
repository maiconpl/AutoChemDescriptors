"""SHAP-based validation pipeline for AutoChemDescriptors."""

from .shap_analysis import run_shap_analysis

__all__ = [
    "run_shap_analysis",
]

