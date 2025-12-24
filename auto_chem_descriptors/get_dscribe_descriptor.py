#!/usr/bin/python3
'''
Created on Februrary 10, 2020

@author: maicon
Last modification by MPL: 24/12/2025 (to get the Dscribe descriptor with the same vector size for different molecules).
Last modification by MPL: 22/02/2024 (after installing Dscribe 2.1.x).
Last modification by MPL: 17/09/2020.
'''

from ase import Atom, Atoms
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import SineMatrix
from dscribe.descriptors import EwaldSumMatrix
from dscribe.descriptors import MBTR
from dscribe.descriptors import SOAP
from dscribe.descriptors import ACSF

from get_atom_information import getAtomTypeCounter

import numpy as np

def get_dscribe_descriptor(atomSymbols, atomSymbolsOfAllMolecules, atoms_xyz, system_type, descriptor_type):

    print("into get_dscribe_descriptor atomSymbols:", atomSymbols)
    print("into get_dscribe_descriptor atomSymbolsOfAllMolecules:", atomSymbolsOfAllMolecules)
    print("into get_dscribe_descriptor atoms_xyz:", atoms_xyz)

    if system_type == "cluster":

       atoms = Atoms("".join(atomSymbols), positions = atoms_xyz)

    #if self.is_debug_true == True:
       #print ("atoms:", atoms)

    if descriptor_type == "coulombMatrix":

       cm= CoulombMatrix(
       #n_atoms_max=len(self.atomSymbols),
       #n_atoms_max=len(atom_index_config)
       #n_atoms_max=int(len(atoms)/2)
       n_atoms_max=len(atoms)
       )

    if descriptor_type == "sineMatrix":

       cm = SineMatrix(
            #n_atoms_max=len(self.atomSymbols),
            #n_atoms_max=len(config),
            #n_atoms_max = alCounter,
            n_atoms_max=len(atom_index_config),
            permutation="sorted_l2",
            sparse=False,
            flatten=True
       )

    if descriptor_type == "ewaldMatrix":

       cm = EwaldSumMatrix(
            #n_atoms_max=len(self.atomSymbols)
            n_atoms_max=len(atom_index_config)
       )

    if descriptor_type == "mbtr":

        #atomsType = getAtomType(atomSymbols)
        atomsType = atomSymbolsOfAllMolecules
        #print ("AtomsType and atomsSymbols", atomsType, atomSymbols)

        # Instantiating MBTR class in new Dscribe version: 2.1.x.
        cm = MBTR(
                species=atomsType,
                geometry= {"function": "inverse_distance"},
                grid= {"min": 0, "max": 0.5, "sigma": 0.01, "n": 200},
                weighting= {"function": "exp", "scale": 0.5, "threshold": 1e-3},
                periodic=False,
                sparse=False
        )

    if descriptor_type == "soap":

        #atomsType = getAtomType(atomSymbols)
        atomsType = atomSymbolsOfAllMolecules
        #print ("AtomsType and atomsSymbols", atomsType, atomSymbols)

        cm = SOAP(
                species=atomsType, 
                periodic=False, 
                r_cut=5, 
                n_max=8, 
                l_max=6,
                average="inner",
                sparse=False)

    if descriptor_type == "acsf":

        #atomsType = getAtomType(atomSymbols)
        atomsType = atomSymbolsOfAllMolecules

        cm = ACSF(species=atomsType,
                   rcut=6.0,
                   g2_params=[[1, 1], [1, 2], [1, 3]],
                   g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)

    #if self.is_debug_true == True:

    #   for i in range(len(atoms.get_positions())):
    #       print (atoms.get_chemical_symbols()[i], atoms.get_positions()[i])

    ''' 24/12/25: we do not need it anymore.
    if descriptor_type == "soap":
       descriptor_list = np.ndarray.tolist( cm.create(atoms) )[0] # IMPORTANT

    else:
       descriptor_list = np.ndarray.tolist( cm.create(atoms) )
    '''

    #if self.is_debug_true == True:
    #    print ("descriptor_list", descriptor_list)

    descriptor_list = np.ndarray.tolist( cm.create(atoms) )

    return descriptor_list
