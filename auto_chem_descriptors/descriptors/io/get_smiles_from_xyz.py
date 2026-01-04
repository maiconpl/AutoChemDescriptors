'''
Created on December 22, 2025

@author: maicon & clayton
Last modification by MPL: 23/12/2025 to save XYZ structures in a single file.)
https://github.com/rdkit/rdkit/discussions/7918 (about Chem.MolFromXYZFile since, from
our example it was not providing the proper SMILES from XYZ coordinates.)
'''

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import os
from ...utils import smiles_checker

def get_smiles_from_xyz(atoms_string):

    #script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.abspath(os.getcwd())

    molecules_smiles_list = []

    n_molecules = len(atoms_string)

    tmp_string = ""
    for i in range(n_molecules):

        raw_mol = None

        if i > 9:
           tmp_file_name = "tmp_file_" + str(i) + ".xyz"

        elif i < 10:
           tmp_file_name = "tmp_file_0" + str(i) + ".xyz"

        tmp_file = open(tmp_file_name, 'w')

        tmp_string = atoms_string[i].split(";")
 
        n_atoms = len(tmp_string) - 1
        tmp_file.write(str(n_atoms))
        tmp_file.write("\n#\n")
        for iItem in range(len(tmp_string)):
            print(tmp_string)
            tmp_file.write(tmp_string[iItem])
            if iItem < n_atoms - 1:
               tmp_file.write("\n")

        tmp_file.close()
        xyz_file = tmp_file_name
        file_path = os.path.join(script_dir, xyz_file)
        raw_mol = Chem.MolFromXYZFile(file_path)

        editable_mol = Chem.Mol(raw_mol)
        rdDetermineBonds.DetermineConnectivity(editable_mol, charge=0)

        smiles = Chem.MolToSmiles(editable_mol)

        print('\nBegin checking the smiles got from XYZ:')
        print('string01' + str(": ") + smiles + '  ' + smiles_checker(smiles))
        print('End checking the smiles got from XYZ.\n')

        #print(f"SMILES: {smiles}", smiles)
        molecules_smiles_list.append(smiles)

    return molecules_smiles_list
