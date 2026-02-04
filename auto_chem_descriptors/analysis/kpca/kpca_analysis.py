#!/usr/bin/python3
'''
Created on Februrary 1, 2026.

@author: maicon & clayton
Last modification by MPL: 01/02/2026.
'''

from typing import Any, Dict, List
from .plot_kpca_grouping import plot_kpca_grouping

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import KernelPCA

def run_kpca_analysis(descriptors_list: List[Any],
                      molecular_encoding: List[Any],
                      analysis: Dict[str, Any],
                      ) -> Dict[str, Any]:

    """Coordinate kPCA processing and high-quality visualizations."""

    print("kPCA explainability artifacts saved to...")

    print(descriptors_list)
    print(molecular_encoding)
    print(analysis)

    n_components = analysis['kpca']['n_components']
    kernel = analysis['kpca']['kernel']
    gamma = analysis['kpca']['gamma']

    X = descriptors_list

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    if kernel == "rbf":
       K = rbf_kernel(X_scaled, gamma=gamma)

    if kernel == "linear":
       K = linear_kernel(X_scaled)

    print("K:", K)

    #K_test = rbf_kernel(X_scaled, X_scaled, gamma=10.0)

    kernel_pca = KernelPCA(n_components=None, kernel="precomputed") #, fit_inverse_transform=False)
    #kernel_pca = KernelPCA(n_components=None, kernel=kernel, gamma=gamma)

    X_kernel_pca = kernel_pca.fit(K).transform(K)
    #kernel_pca.fit(K)
    #X_kernel_pca = kernel_pca.transform(K)
    #X_kernel_pca = kernel_pca.transform(K)

    #X_kernel_pca = kernel_pca.fit_transform(K)

    #kernel_pca.fit(X_scaled)
    #X_kernel_pca = kernel_pca.transform(X_scaled)

    print("X_kernel_pca:", X_kernel_pca)

    tmp01 = 0.0
    explained_variance_ratio = []
    eigenvectors = []
    eigenvectors = kernel_pca.eigenvectors_

    print("xxxx eigenvectors:", kernel_pca.eigenvectors_)
    #print("xxxx components:", kernel_pca.components_) # this does not work for kernelPCA
    print("xxxx eigenvalues:", kernel_pca.eigenvalues_)

    #eigenvalues_list_sorted = sorted(transformer.eigenvalues_.tolist(), reverse=True)
    eigenvalues_list_sorted = sorted(kernel_pca.eigenvalues_.tolist(), reverse=True)

    for i in range(n_components):
        tmp01 = eigenvalues_list_sorted[i]/sum(eigenvalues_list_sorted)
        explained_variance_ratio.append( tmp01 )
        tmp01 = 0.0

    print("xxxxx explained_variance_ratio:", explained_variance_ratio)

    #print("X_kpca:", X_kpca)
    #print("explained_variance_ratio:", explained_variance_ratio)
    #print("eigenvectors:", explained_variance_ratio)

    plot_kpca_grouping(X_kernel_pca, explained_variance_ratio, analysis)
