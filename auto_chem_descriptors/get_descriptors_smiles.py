'''
Created on December 08, 2025

@author: maicon
Last modification by MPL: 08/12/2025 to implement the multiprocess to run RDKit to get the descriptors, in parallell.
'''

from get_rdkit_calculations import get_rdkit_calculations
from utils import smile_molecule_representation

def get_descriptors_smiles(n_jobs, molecules_coded_list, is_debug_true):

    descriptors = []
    from rdkit import Chem
    from get_atom_information import getAtomTypeCounter, getAtomType
    from rdkit.Chem import AllChem

    print("molecules_coded_list:", molecules_coded_list)

    # Here, we are using this just to get the "molX.png" files.
    smile_molecule_representation(n_molecules=len(molecules_coded_list), molecules_coded_list=molecules_coded_list, is_debug_true=is_debug_true)

    print("asdf:  asdf:", get_rdkit_calculations(molecules_coded_list, n_jobs=n_jobs) )
    descriptors = get_rdkit_calculations(molecules_coded_list, n_jobs=n_jobs)

    print("xyz_new from mull:", descriptors)

    return descriptors
