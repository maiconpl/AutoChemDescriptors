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

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt

def myplot(score, coeff, labels, X_pca, analysis):

    import random
    random.seed(42)

    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'v', '<', '>', '*', '*', 'o']
    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    edgecolors=['none', 'face', 'k', 'b']

    y = ['k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'b', 'g', 'r', 'c', 'm', 'y', 'k', 'r', 'm', 'b']

    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]

    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())

    molecules_label = analysis['molecules_label']

    for i in range( len(xs)):

        marker = random.choice(markers)
        color = random.choice(colors)
        edgecolor = random.choice(edgecolors)

        plt.scatter(xs[i] * scalex, ys[i] * scaley, c=color, s=80, label=molecules_label[i], marker=marker, edgecolors=edgecolor)

    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 1.0)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'b', ha = 'center', va = 'center', size=10)
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'b', ha = 'center', va = 'baseline', size=8, rotation=0)

def plot_pca_dispersion(descriptors_list, analysis):

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

    labels = ["FpDensityMorgan01", "FpDensityMorgan02", "FpDensityMorgan03", "MaxAbsPartialCharge", "MaxPartialCharge", "MinAbsPartialCharge", "MinPartialCharge", "ExactMolWt", "NumRadicalElectrons", "NumValenceElectrons", "MolVolume", "HeavyAtomMolWt"]

    plt.xlabel("F1 (" + str( round(float(pca.explained_variance_ratio_[0]*100), 2) ) + " %)", size=15)
    plt.ylabel("F2 (" + str( round(float(pca.explained_variance_ratio_[1]*100), 2) ) + " %)", size=15)
    plt.grid()

    #Call the function. Use only the 2 PCs.
    myplot(X_pca[:,0:2], np.transpose(pca.components_[0:2, :]), labels, X_pca, analysis)

    if "legend_bbox_to_anchor" in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    elif "legend_bbox_to_anchor" in analysis and "legend_size" not in analysis and "legend_ncol" not in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol= 4 )

    elif "legend_bbox_to_anchor" not in analysis and "legend_size" in analysis and "legend_ncol" not in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)

    elif "legend_bbox_to_anchor" not in analysis and "legend_size" not in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    elif "legend_bbox_to_anchor" in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=4)

    elif "legend_bbox_to_anchor" in analysis and "legend_size" not in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    elif "legend_bbox_to_anchor" not in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        print("zoi")
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))
 
    else: # default
        lgd = plt.legend(loc='upper center', prop={'size':6}, bbox_to_anchor=(0.5, -0.16), fancybox=True, shadow=True, ncol=4)

    plt.axvline(x=0, color='k', linestyle="--")
    plt.axhline(y=0, color='k', linestyle="--")

    plt.savefig('plot_PCA_dispersion.png', bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.close()
