'''
Created on December 7, 2025

@author: maicon
Last modification by MPL: 07/12/2025 in Betim.
'''

def libraries_information_auto_chem_descriptors():

    print("\nBEGIN: main libraries verion:")
    import platform
    print("Python version:", platform.python_version())

    import rdkit
    print("RDKit version:", rdkit.__version__)

    import pyscf
    print ("PySCF:", pyscf.__version__)

    import numpy as np
    print ("Numpy:", np.__version__)

    import scipy
    print ("Scipy:", scipy.__version__)

    import matplotlib as mpl
    print ("Matplotlib:", mpl.__version__)

    #import sklearn
    #print ("Sklearn:", sklearn.__version__)

    #import dscribe
    #print ("Dscribe verion:", dir(dscribe))

    import ase
    print ("ASE:", ase.__version__)

    print("END: main libraries version.\n")
