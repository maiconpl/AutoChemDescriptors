#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon
Last modification by MPL: 10/12/2025.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0, 1, 2], ["First component", "Second component", "Third component"], size=12)
    #plt.yticks([0, 1], ["First component", "Second component"], size=11)
    plt.colorbar()
    plt.xticks(range(0,len(X[0])), features_index, rotation=18, ha='left', size=12)
    #plt.xlabel("Features", size=15)
    #plt.ylabel("Principal Components", size=15)

    plt.savefig('plot_PCA_heatmap.png', bbox_inches='tight', dpi=300)
    plt.close()
