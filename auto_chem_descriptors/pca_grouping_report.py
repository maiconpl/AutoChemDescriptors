#!/usr/bin/python3
'''
Created on December 11, 2025

@author: maicon
Last modification by MPL: 11/12/2025.
'''

from datetime import datetime
from typing import List, Sequence, Tuple

import numpy as np


def generate_pca_grouping_report(x_pca: Sequence[Sequence[float]],
                                 explained_variance_ratio: Sequence[float],
                                 labels: Sequence[str],
                                 analysis: dict,
                                 molecular_encoding: str,
                                 descriptors_list: Sequence[Sequence[float]],
                                 output_filename: str = "report_PCA_grouping.md",
                                 top_k: int = 3) -> str:
    '''
    Build a Markdown report summarizing the PCA grouping analysis.
    '''

    if analysis is None:
        analysis = {}

    scores = np.asarray(x_pca)
    if scores.size == 0:
        raise ValueError("PCA scores are empty. Nothing to report.")

    n_molecules = scores.shape[0]
    n_components = scores.shape[1]

    if not labels or len(labels) != n_molecules:
        labels = [f"Molecule {i + 1}" for i in range(n_molecules)]

    explained_variance_ratio = np.asarray(explained_variance_ratio)
    cumulative_variance = np.cumsum(explained_variance_ratio) * 100.0
    explained_variance_percent = explained_variance_ratio * 100.0

    descriptor_size = len(descriptors_list[0]) if descriptors_list else 0

    component_highlights = {}
    max_components_to_show = min(2, n_components)
    for component_index in range(max_components_to_show):
        component_scores = scores[:, component_index]
        component_highlights[component_index] = _get_projection_extremes(component_scores,
                                                                         labels,
                                                                         top_k)

    lines: List[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# PCA Grouping Report")
    lines.append("")
    lines.append(f"- Generated at: {timestamp}")
    lines.append(f"- Molecular encoding: {molecular_encoding}")
    lines.append(f"- Number of molecules: {n_molecules}")
    lines.append(f"- Descriptor length: {descriptor_size}")
    lines.append(f"- Requested PCA components: {analysis.get('pca_grouping', ['-', n_components])[1]}")
    lines.append("")

    lines.append("## Component Summary")
    lines.append("| Component | Explained variance (%) | Cumulative variance (%) |")
    lines.append("| --- | --- | --- |")
    for idx in range(n_components):
        lines.append(f"| F{idx + 1} | {explained_variance_percent[idx]:.2f} | {cumulative_variance[idx]:.2f} |")
    lines.append("")

    lines.append("## Projection Highlights")
    for component_index in range(max_components_to_show):
        positive, negative = component_highlights[component_index]
        lines.append(f"### F{component_index + 1}")
        lines.append("| Rank | Positive molecule | Score | Negative molecule | Score |")
        lines.append("| --- | --- | --- | --- | --- |")
        max_rows = max(len(positive), len(negative))
        for rank in range(max_rows):
            pos_name, pos_score = positive[rank] if rank < len(positive) else ("-", 0.0)
            neg_name, neg_score = negative[rank] if rank < len(negative) else ("-", 0.0)
            lines.append(f"| {rank + 1} | {pos_name} | {pos_score:.4f} | {neg_name} | {neg_score:.4f} |")
        lines.append("")

    lines.append("## Analysis Metadata")
    for key, value in analysis.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")

    with open(output_filename, "w") as file_handler:
        file_handler.write("\n".join(lines))

    return output_filename


def _get_projection_extremes(component_scores: np.ndarray,
                             labels: Sequence[str],
                             top_k: int) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    '''
    Identify molecules with the strongest positive and negative loadings for a component.
    '''

    n_points = component_scores.shape[0]
    k = min(top_k, n_points)

    sorted_indices = np.argsort(component_scores)
    negative_indices = sorted_indices[:k]
    positive_indices = sorted_indices[::-1][:k]

    positive = [(labels[idx], component_scores[idx]) for idx in positive_indices]
    negative = [(labels[idx], component_scores[idx]) for idx in negative_indices]

    return positive, negative
