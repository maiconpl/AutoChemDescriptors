#!/usr/bin/python3
'''
Created on December 12, 2025

@author: maicon
Last modification by MPL: 12/12/2025.
'''

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DescriptorDetail = Dict[str, Any]


DESCRIPTOR_CONTEXT: Dict[str, Dict[str, str]] = {
    "FpDensityMorgan01": {
        "property": "Morgan fingerprint density (radius 1)",
        "high": "higher substitution immediately around the core, pointing to richly functionalized scaffolds.",
        "low": "lower local substitution, typical of lean aromatic or aliphatic motifs.",
        "short_high": "dense first-shell substitution",
        "short_low": "lean first-shell motifs",
    },
    "FpDensityMorgan02": {
        "property": "Morgan fingerprint density (radius 2)",
        "high": "branching two bonds away that favors decorated side chains.",
        "low": "linear or weakly substituted neighborhoods past the first bond.",
        "short_high": "radius-2 branching",
        "short_low": "linear radius-2 shells",
    },
    "FpDensityMorgan03": {
        "property": "Morgan fingerprint density (radius 3)",
        "high": "extended substitution networks and long conjugated backbones.",
        "low": "compact frameworks lacking long-range branching.",
        "short_high": "long-range substitution",
        "short_low": "compact frameworks",
    },
    "MaxAbsPartialCharge": {
        "property": "maximum absolute partial charge",
        "high": "strong charge separation or polar/ionic bonds.",
        "low": "uniform charge distribution across the scaffold.",
        "short_high": "polar/ionic motifs",
        "short_low": "uniform charge",
    },
    "MaxPartialCharge": {
        "property": "most positive partial charge",
        "high": "electron-poor centers induced by withdrawing substituents.",
        "low": "absence of strongly electrophilic atoms.",
        "short_high": "electron-poor centers",
        "short_low": "weak electrophiles",
    },
    "MinAbsPartialCharge": {
        "property": "minimum absolute partial charge",
        "high": "pervasive polarity where even the mildest site carries charge.",
        "low": "near-neutral pockets that moderate charge localization.",
        "short_high": "widespread polarity",
        "short_low": "neutral pockets",
    },
    "MinPartialCharge": {
        "property": "most negative partial charge",
        "high": "softly donating atoms with moderated electron density.",
        "low": "strongly negative centers such as phenolate or anionic heteroatoms.",
        "short_high": "moderate donors",
        "short_low": "strong anionic sites",
    },
    "ExactMolWt": {
        "property": "exact molecular weight",
        "high": "heavier scaffolds, often due to halogens or extended conjugation.",
        "low": "lighter, hydrogen-rich frameworks.",
        "short_high": "heavier scaffolds",
        "short_low": "light frameworks",
    },
    "NumRadicalElectrons": {
        "property": "number of radical electrons",
        "high": "open-shell character or persistent radicals.",
        "low": "closed-shell species.",
        "short_high": "open-shell character",
        "short_low": "closed-shell cores",
    },
    "NumValenceElectrons": {
        "property": "total valence electrons",
        "high": "heteroatom-rich or highly conjugated systems.",
        "low": "electron-poor skeletons dominated by C/H atoms.",
        "short_high": "electron-rich systems",
        "short_low": "electron-poor skeletons",
    },
    "MolVolume": {
        "property": "molecular volume",
        "high": "bulky, space-filling molecules or long side chains.",
        "low": "compact or planar scaffolds.",
        "short_high": "bulky volume",
        "short_low": "compact volume",
    },
    "HeavyAtomMolWt": {
        "property": "heavy-atom molecular weight",
        "high": "high heavy-atom content (halogens or heavy heteroatoms).",
        "low": "frameworks dominated by lighter heteroatoms or carbons.",
        "short_high": "heavy-atom content",
        "short_low": "light-atom frameworks",
    },
}


def generate_pca_heatmap_report(pca_components: Sequence[Sequence[float]],
                                explained_variance_ratio: Sequence[float],
                                descriptor_names: Sequence[str],
                                analysis: Optional[Dict[str, Any]] = None,
                                output_filename: str = "report_PCA_heatmap.md",
                                top_k: int = 4) -> str:
    '''
    Build a Markdown report interpreting PCA heatmap contributions.
    '''

    analysis_dict = analysis if isinstance(analysis, dict) else {}

    components = _validate_components(pca_components)
    n_components, n_descriptors = components.shape

    descriptor_names_list = _validate_descriptor_names(descriptor_names, n_descriptors)
    explained_variance = _validate_explained_variance(explained_variance_ratio, n_components)

    report_settings = analysis_dict.get('pca_heatmap_report', {})
    requested_components = analysis_dict.get('pca_heatmap', ['-', n_components])

    top_k_value = int(report_settings.get('top_descriptors', top_k))
    top_k_value = max(1, min(top_k_value, n_descriptors))
    output_filename = report_settings.get('output_filename', output_filename)
    heatmap_filename = report_settings.get('plot_filename', 'plot_PCA_heatmap.png')

    component_details = _collect_component_details(components,
                                                   descriptor_names_list,
                                                   top_k_value)

    lines: List[str] = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# PCA Heatmap Report")
    lines.append("")
    lines.append(f"- Generated at: {timestamp}")
    lines.append(f"- Number of PCA components: {n_components}")
    lines.append(f"- Number of descriptors: {n_descriptors}")
    lines.append(f"- Source heatmap: `{heatmap_filename}`")
    lines.append(f"- Requested PCA components: {requested_components[1] if len(requested_components) > 1 else requested_components}")
    lines.append("")

    lines.append("## Descriptor Contribution Overview")
    lines.append("| Component | Explained variance (%) | Dominant descriptors |")
    lines.append("| --- | --- | --- |")
    for idx, details in enumerate(component_details):
        descriptor_label = ", ".join(f"`{item['name']}`" for item in details)
        lines.append(f"| F{idx + 1} | {explained_variance[idx] * 100:.2f} | {descriptor_label} |")
    lines.append("")

    lines.append("## Component interpretations")
    for idx, details in enumerate(component_details):
        lines.append(f"### F{idx + 1} ({explained_variance[idx] * 100:.2f} %)")
        lines.append("| Rank | Descriptor | Loading | Chemical implication |")
        lines.append("| --- | --- | --- | --- |")
        for item in details:
            lines.append(f"| {item['rank']} | `{item['name']}` | {item['loading']:.4f} | {item['implication']} |")
        lines.append("")
        lines.extend(_build_component_summary(details, idx + 1))
        lines.append("")

    if analysis_dict:
        lines.append("## Analysis metadata")
        for key, value in analysis_dict.items():
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    with open(output_filename, "w", encoding="utf-8") as handler:
        handler.write("\n".join(lines))

    return output_filename


def _validate_components(pca_components: Sequence[Sequence[float]]) -> np.ndarray:
    components = np.asarray(pca_components, dtype=float)
    if components.size == 0:
        raise ValueError("PCA components are empty. Nothing to describe.")
    if components.ndim != 2:
        raise ValueError("PCA components must be a 2D array.")
    return components


def _validate_descriptor_names(descriptor_names: Sequence[str],
                               expected_size: int) -> List[str]:
    if descriptor_names is None:
        raise ValueError("Descriptor names are required to describe the heatmap.")
    names = list(descriptor_names)
    if len(names) != expected_size:
        raise ValueError("Descriptor names length does not match PCA components.")
    return names


def _validate_explained_variance(explained_variance_ratio: Sequence[float],
                                 expected_size: int) -> np.ndarray:
    explained = np.asarray(explained_variance_ratio, dtype=float)
    if explained.size != expected_size:
        raise ValueError("Explained variance array should match the number of components.")
    return explained


def _collect_component_details(components: np.ndarray,
                               descriptor_names: List[str],
                               top_k: int) -> List[List[DescriptorDetail]]:
    details: List[List[DescriptorDetail]] = []
    for component_row in components:
        sorted_indices = np.argsort(np.abs(component_row))[::-1][:top_k]
        component_details: List[DescriptorDetail] = []
        for rank, descriptor_index in enumerate(sorted_indices, start=1):
            descriptor_name = descriptor_names[descriptor_index]
            loading = float(component_row[descriptor_index])
            implication, summary = _descriptor_implication(descriptor_name, loading)
            component_details.append({
                "rank": rank,
                "name": descriptor_name,
                "loading": loading,
                "implication": implication,
                "summary": summary,
            })
        details.append(component_details)
    return details


def _descriptor_implication(descriptor_name: str, loading: float) -> Tuple[str, str]:
    context = DESCRIPTOR_CONTEXT.get(descriptor_name, {})
    property_text = context.get("property", descriptor_name)

    if loading >= 0:
        detail = context.get("high", f"higher {property_text} values drive this component upward.")
        summary = context.get("short_high", property_text)
        implication = f"Positive loading: {detail}"
    else:
        detail = context.get("low", f"lower {property_text} values drive this component upward.")
        summary = context.get("short_low", property_text)
        implication = f"Negative loading: {detail}"

    return implication, summary


def _build_component_summary(details: List[DescriptorDetail],
                             component_number: int) -> List[str]:
    positives = [item for item in details if item["loading"] >= 0]
    negatives = [item for item in details if item["loading"] < 0]

    summary_lines: List[str] = []
    if positives:
        summary_lines.append(
            f"- **High F{component_number} scores** are driven by {_join_descriptor_summaries(positives)}."
        )
    if negatives:
        summary_lines.append(
            f"- **Low F{component_number} scores** reflect {_join_descriptor_summaries(negatives)}."
        )
    if not summary_lines:
        summary_lines.append("- Loadings selected for this component are near zero.")

    return summary_lines


def _join_descriptor_summaries(items: List[DescriptorDetail]) -> str:
    descriptor_labels = ", ".join(f"`{item['name']}`" for item in items)
    summary_phrases = _unique_ordered(item.get("summary") for item in items)
    if summary_phrases:
        return f"{descriptor_labels} (signaling {'; '.join(summary_phrases)})"
    return descriptor_labels


def _unique_ordered(strings: Iterable[Optional[str]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for entry in strings:
        if entry and entry not in seen:
            ordered.append(entry)
            seen.add(entry)
    return ordered
