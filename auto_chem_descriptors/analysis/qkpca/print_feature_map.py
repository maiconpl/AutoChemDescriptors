#!/usr/bin/python3
'''
Created on Februrary 04, 2026.

@author: maicon & clayton
Last modification by MPL: 04/02/2026.
'''

import matplotlib.pyplot as plt

def print_feature_map(feature_map, feature_map_type, entanglement):

    decomposed_feature_map = feature_map.decompose()

    if feature_map_type == "ZFeatureMap" or feature_map_type == "Z":
       filename = "feature_map_" + feature_map_type + ".png"

    else:
       filename = "feature_map_" + feature_map_type + "_" + entanglement + ".png"

    decomposed_feature_map.draw(output='mpl', filename=filename, fold=300)
    plt.close('all')
