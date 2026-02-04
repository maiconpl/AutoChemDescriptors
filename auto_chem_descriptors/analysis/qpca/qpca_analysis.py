#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon & clayton
Last modification by MPL: 22/01/2026.
'''

from typing import Any, Dict, List
from .QPCA import QPCA
#from .plot_qpca_dispersion import plot_qpca_dispersion
from .plot_qpca_grouping import plot_qpca_grouping
#from .plot_qpca_heatmap import plot_qpca_heatmap

from sklearn.preprocessing import StandardScaler

def run_qpca_analysis(descriptors_list: List[Any],
                      molecular_encoding: List[Any],
                      analysis: Dict[str, Any],
                      ) -> Dict[str, Any]:

    """Coordinate qPCA processing and high-quality visualizations."""

    print("qPCA explainability artifacts saved to...")
    print(descriptors_list)
    print(molecular_encoding)
    print(analysis)

    n_components = analysis['qpca']['n_components']
    feature_map_type = analysis['qpca']['feature_map']
    entanglement = analysis['qpca']['entanglement']

    #heatmap = analysis['qpca']['heatmap']
    #grouping = analysis['qpca']['grouping']
    #dispersion = analysis['qpca']['dispersion']

    X = descriptors_list

    feature_dimension = len(X[0]) # the number of qubits in the circuit

    print("feature_dimension:", feature_dimension)

    if feature_map_type == "ZZFeatureMap" or feature_map_type == "ZZ":

        from qiskit.circuit.library import ZZFeatureMap
        feature_map = ZZFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement=entanglement)

    elif feature_map_type == "ZFeatureMap" or feature_map_type == "Z":

        from qiskit.circuit.library import ZFeatureMap
        feature_map = ZFeatureMap(feature_dimension=feature_dimension, reps=2)

    elif feature_map_type == "PauliFeatureMap" or feature_map_type.lower() == "pauli":

        from qiskit.circuit.library import PauliFeatureMap
        #feature_map = PauliFeatureMap(feature_dimension=2, reps=2, entanglement=entanglement)
        feature_map = PauliFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement=entanglement)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    qpca = QPCA(n_components=n_components, feature_map=feature_map)

    X_qpca, explained_variance_ratio, eigenvectors = qpca.transform(X_scaled)

    print("X_qpca:", X_qpca)
    print("explained_variance_ratio:", explained_variance_ratio)
    print("eigenvectors:", explained_variance_ratio)

    #plot_qpca_dispersion(X_qpca, components_percentage_sorted, analysis)
    plot_qpca_grouping(X_qpca, explained_variance_ratio, analysis)
    #plot_qpca_heatmap(X_qpca, explained_variance_ratio, analysis)
    #plot_qpca_heatmap(X_qpca, eigenvectors, analysis)
