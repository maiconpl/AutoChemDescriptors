#!/usr/bin/python3
'''
Created on January 22, 2026.

@author: maicon & clayton
Last modification by MPL: 23/01/2026.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt

def plot_qpca_heatmap(X_qpca, explained_variance_ratio, analysis):

    n_components = analysis['qpca']['n_components']

    features_index = ["FpDensityMorgan01", "FpDensityMorgan02", "FpDensityMorgan03", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "ExactMolWt", "NumRadicalElectrons", "NumValenceElectrons", "MolVolume", "HeavyAtomMolWt"]

    components_name_list = []
    tmp_string = "Comp."
    for i in range(n_components):
        components_name_list.append(tmp_string + " " + str(i + 1))

    print("mmm0:", explained_variance_ratio)
    print("mmm1:", explained_variance_ratio[0:n_components])

    #plt.matshow(X_qpca[0:n_components], cmap='viridis')
    plt.matshow(explained_variance_ratio[0:n_components], cmap='viridis')
    plt.yticks([i for i in range(n_components)], components_name_list, size=10)
    plt.colorbar()
    plt.xticks(range(0, len(features_index)), features_index, rotation=18, ha='left', size=10)

    plt.savefig('plot_qPCA_heatmap.png', bbox_inches='tight', dpi=300)

    plt.close()
