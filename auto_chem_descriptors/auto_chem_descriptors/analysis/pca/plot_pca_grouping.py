#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon & clayton
Last modification by MPL: 26/12/2025 to adjust the figure legend.
Last modification by MPL: 10/12/2025.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt

def plot_pca_grouping(descriptors_list, molecular_encoding, analysis):

    import random
    random.seed(42)

    X = descriptors_list
    n_components = analysis['pca_grouping'][1]
    print ("Matrix X:")
    for i in X:
        print (i)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled=scaler.transform(X)

    colors = analysis['molecules_color']
    labels = analysis['molecules_label']

    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    X_pca = pca.transform(X_scaled)

    print ("Each component weight:", pca.explained_variance_ratio_)
    print ("Sum of the components weight:", sum(pca.explained_variance_ratio_))

    plt.xlabel("F1 (" + str( round(float(pca.explained_variance_ratio_[0]*100), 2) ) + " %)", size=15)
    plt.ylabel("F2 (" + str( round(float(pca.explained_variance_ratio_[1]*100), 2) ) + " %)", size=15)

    n_samples = len(X)
    print("size X_pca:", len(X_pca), n_samples)
    
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'v', '<', '>', '*', '*', 'o']

    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    edgecolors=['none', 'face', 'k', 'b']

    for i in range( len(X_pca[:,0]) ):

        #color = (random.random(), random.random(), random.random())
        marker = random.choice(markers)
        color = random.choice(colors)
        edgecolor = random.choice(edgecolors)

        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color, s=80, label=labels[i], marker=marker, edgecolors=edgecolor)

    if "legend_bbox_to_anchor" in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    #elif "legend_bbox_to_anchor" in analysis: # custom by user
    elif "legend_bbox_to_anchor" in analysis and "legend_size" not in analysis and "legend_ncol" not in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol= 4 )

    #elif "legend_size" in analysis: # custom by user
    elif "legend_bbox_to_anchor" not in analysis and "legend_size" in analysis and "legend_ncol" not in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=4)

    #elif "legend_ncol" in analysis: # custom by user
    elif "legend_bbox_to_anchor" not in analysis and "legend_size" not in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    #elif "legend_bbox_to_anchor" in analysis and "legend_size" in analysis: # custom by user
    elif "legend_bbox_to_anchor" in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=4)

    #elif "legend_bbox_to_anchor" in analysis and "legend_ncol" in analysis: # custom by user
    elif "legend_bbox_to_anchor" in analysis and "legend_size" not in analysis and "legend_ncol" in analysis: # custom by user
        lgd = plt.legend(loc='upper center', prop={'size': 6}, bbox_to_anchor=analysis["legend_bbox_to_anchor"], fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))

    #elif "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
    elif "legend_bbox_to_anchor" not in analysis and "legend_size" in analysis and "legend_ncol" in analysis: # custom by user
        print("zoi")
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))
 
    else: # default
        lgd = plt.legend(loc='upper center', prop={'size':6}, bbox_to_anchor=(0.5, -0.16), fancybox=True, shadow=True, ncol=4)

    plt.axvline(x=0, color='k', linestyle="--")
    plt.axhline(y=0, color='k', linestyle="--")

    plt.savefig('plot_PCA_grouping.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.close()
