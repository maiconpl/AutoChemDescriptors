#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon & clayton
Last modification by MPL: 22/01/2026.
'''

from typing import Any, Dict, List
from .QKPCA import QKPCA
#from .plot_qkpca_dispersion import plot_qkpca_dispersion
from .plot_qkpca_grouping import plot_qkpca_grouping
#from .plot_qkpca_heatmap import plot_qkpca_heatmap

from sklearn.preprocessing import StandardScaler
from .print_feature_map import print_feature_map

def run_qkpca_analysis(descriptors_list: List[Any],
                      molecular_encoding: List[Any],
                      analysis: Dict[str, Any],
                      ) -> Dict[str, Any]:

    """Coordinate qPCA processing and high-quality visualizations."""

    print("qPCA explainability artifacts saved to...")
    print(descriptors_list)
    print(molecular_encoding)
    print(analysis)

    n_components = analysis['qkpca']['n_components']
    feature_map_type = analysis['qkpca']['feature_map']
    entanglement = analysis['qkpca']['entanglement']

    #heatmap = analysis['qkpca']['heatmap']
    #grouping = analysis['qkpca']['grouping']
    #dispersion = analysis['qkpca']['dispersion']

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
        feature_map = PauliFeatureMap(feature_dimension=feature_dimension, reps=2, entanglement=entanglement)

    print_feature_map(feature_map, feature_map_type, entanglement)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    qkpca = QKPCA(n_components=n_components, feature_map=feature_map)

    X_qkpca, explained_variance_ratio, eigenvectors = qkpca.transform(X_scaled)

    print("\n--- Begin: Quantum kernel PCA information ---")
    print('QKPCA: FQK, ' + feature_map_type + ', entanglement: ' + entanglement)
    print("qkpca n_components:", n_components)
    print("X_qkpca:", X_qkpca)
    print("qkpca explained_variance_ratio:", explained_variance_ratio)
    #print("eigenvectors:", explained_variance_ratio)
    print("--- End: Quantum kernel PCA information ---\n")

    #plot_qkpca_dispersion(X_qkpca, components_percentage_sorted, analysis)
    plot_qkpca_grouping(X_qkpca, explained_variance_ratio, analysis)
    #plot_qkpca_heatmap(X_qkpca, explained_variance_ratio, analysis)
    #plot_qkpca_heatmap(X_qkpca, eigenvectors, analysis)
