#!/usr/bin/python3
'''
Created on December 10, 2025.

@author: maicon
Last modification by MPL: 10/12/2025.
'''

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.feature_selection import SelectPercentile
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from pca_grouping_report import generate_pca_grouping_report

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
    
    #markers = ['o', 's', '^', 'D', 'x', '+', '*', 'p', 'h', 'v', '<', '>']
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'v', '<', '>', '*', '*', 'o']

    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    edgecolors=['none', 'face', 'k', 'b']

    for i in range( len(X_pca[:,0]) ):

        #color = (random.random(), random.random(), random.random())
        marker = random.choice(markers)
        color = random.choice(colors)
        edgecolor = random.choice(edgecolors)

        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color, s=80, label=labels[i], marker=marker, edgecolors=edgecolor)
        #if i <= 4:
        #    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], s=80, label=labels[i])
        ##if i > 4 and i < 11:
        #if i > 4 and i < n_samples - 1:
        #    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], s=80, label=labels[i], marker='^')
        #if i == n_samples - 1:
        #    #plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], s=80, label=labels[i], marker=r'$\clubsuit$')
        #    plt.scatter(X_pca[i, 0], X_pca[i, 1], c=colors[i], s=80, label=labels[i], marker='*')

    #plt.legend(loc='upper center', prop={'size':9})
    #plt.legend(loc='upper right', prop={'size':9})
    #plt.legend(loc='upper right', prop={'size':7}, bbox_to_anchor=(1.2, 1.0))
    print("kkkk 03")
    lgd = plt.legend(loc='upper right', prop={'size':7}, bbox_to_anchor=(1.2, 1.0))

    plt.axvline(x=0, color='k', linestyle="--")
    plt.axhline(y=0, color='k', linestyle="--")

    '''
    # BEGIN draw an elipse
    a = -1.2
    b = -1.4
    h = -1.8
    k = -1.2
    x = np.linspace(-3.3, 3.0, 400)
    y = np.linspace(-3.3, 3.0, 400)
    x, y = np.meshgrid(x, y)
    plt.contour(x, y,((x -h)**2/a**2 + (y - k)**2/b**2), [1], colors='k')
    # END draw an elipse
    '''

    plt.savefig('plot_PCA_grouping.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)

    report_filename = generate_pca_grouping_report(X_pca,
                                                   pca.explained_variance_ratio_,
                                                   labels,
                                                   analysis,
                                                   molecular_encoding,
                                                   descriptors_list)
    print("PCA grouping report saved to:", report_filename)

    plt.close()
