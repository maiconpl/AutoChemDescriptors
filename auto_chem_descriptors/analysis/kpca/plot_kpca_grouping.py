#!/usr/bin/python3
'''
Created on Februrary 1, 2026.

@author: maicon & clayton
Last modification by MPL: 01/02/2026.
'''

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt

def plot_kpca_grouping(X_pca, explained_variance_ratio, analysis):

    import random
    random.seed(42)

    colors = analysis['molecules_color']
    labels = analysis['molecules_label']

    #print ("Each component weight:", explained_variance_ratio)
    #print ("Sum of the components weight:", sum(explained_variance_ratio))

    plt.xlabel("F1 (" + str( round(float(explained_variance_ratio[0]*100), 2) ) + " %)", size=15)
    plt.ylabel("F2 (" + str( round(float(explained_variance_ratio[1]*100), 2) ) + " %)", size=15)

    n_samples = len(X_pca)
    #print("size X_pca:", len(X_pca), n_samples)
    
    markers = ['o', 's', '^', 'D', '*', 'p', 'h', 'v', '<', '>', '*', '*', 'o']

    colors = ['k', 'b', 'g', 'r', 'c', 'm', 'y']
    edgecolors=['none', 'face', 'k', 'b']

    kernel = analysis["kpca"]["kernel"]
    gamma = analysis["kpca"]["gamma"]
    plt.title('KPCA: kernel: ' + kernel + ', gamma: ' + str(gamma))

    for i in range( len(X_pca[:,0]) ):

        marker = random.choice(markers)
        color = random.choice(colors)
        edgecolor = random.choice(edgecolors)

        plt.scatter(X_pca[i, 0], X_pca[i, 1], c=color, s=80, label=labels[i], marker=marker, edgecolors=edgecolor)

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
        lgd = plt.legend(loc='upper center', prop={'size': int(analysis["legend_size"])}, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=int(analysis["legend_ncol"]))
 
    else: # default
        lgd = plt.legend(loc='upper center', prop={'size':6}, bbox_to_anchor=(0.5, -0.16), fancybox=True, shadow=True, ncol=4)

    plt.axvline(x=0, color='k', linestyle="--")
    plt.axhline(y=0, color='k', linestyle="--")

    plt.savefig('plot_kPCA_grouping.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
    plt.close()
