#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon & clayton
Last modification by MPL: 26/12/2025 to adjust the figure legend.
Last modification by MPL: 10/12/2025.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from pca_heatmap_report import generate_pca_heatmap_report

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt


def plot_pca_heatmap(descriptors_list, analysis):

    X = descriptors_list
    n_components = analysis['pca_grouping'][1]

    #print ("Matrix X:")
    #for i in X:
    #    print (i)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    features_index = ["FpDensityMorgan01", "FpDensityMorgan02", "FpDensityMorgan03", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "ExactMolWt", "NumRadicalElectrons", "NumValenceElectrons", "MolVolume", "HeavyAtomMolWt"]

    n_components = analysis['pca_grouping'][1]
    components_name_list = []
    tmp_string = "Comp."
    for i in range(n_components):
        components_name_list.append(tmp_string + " " + str(i + 1))


    #plt.matshow(pca.components_, cmap='viridis')
    plt.matshow(pca.components_[0:n_components], cmap='viridis')
    plt.yticks([i for i in range(n_components)], components_name_list, size=10)
    #plt.yticks([0, 1, 2], ["First component", "Second component", "Third component"], size=12)
    plt.colorbar()
    plt.xticks(range(0,len(X[0])), features_index, rotation=18, ha='left', size=10)

    plt.savefig('plot_PCA_heatmap.png', bbox_inches='tight', dpi=300)

    report_filename = generate_pca_heatmap_report(pca.components_,
                                                  pca.explained_variance_ratio_,
                                                  features_index,
                                                  analysis)

    print("PCA heatmap report saved to:", report_filename)

    plt.close()
