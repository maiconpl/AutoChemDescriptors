#!/usr/bin/python3
'''
Created on December 18, 2025.

@author: maicon & clayton
Last modification by MPL: 24/12/2025 to print the dscriptors of different molecules sizes from xyz.
Last modification by MPL: 23/12/2025 to print the dscriptors of different molecules sizes.
Last modification by MPL: 18/12/2025.
'''

# Otherwise, does not work, it is mandatory:
import matplotlib
matplotlib.use('Agg') # or 'Qt5Agg', 'TkAgg', etc.
import matplotlib.pyplot as plt
import random

def plot_dscribe(descriptors_list, descriptors_type, analysis, is_debug_true=False):

    X = descriptors_list

    if is_debug_true == True:
       print ("Matrix X:")
       for i in X:
           print (i)

    colors = analysis['molecules_color']
    labels = analysis['molecules_label']

    for i in range(len(X)):

        counter_list = [i for i in range(len(X[i]))]
        color = random.choice(colors)

        if is_debug_true == True:
           print("counter_list and X:", counter_list, X[i])

        plt.plot(counter_list, X[i], label=labels[i], color=color, linewidth=1.1)

        lgd = plt.legend(loc='upper right', prop={'size':7}, bbox_to_anchor=(1.2, 1.0))

        plt.xlabel("Arbitrary scale")
        plt.ylabel(descriptors_type + " values")

        if i > 9:
           plt.savefig('plot_dscribe_desc_' + str(i) + '.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)
        else:
           plt.savefig('plot_dscribe_desc_0' + str(i) + '.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=300)

        plt.close()
