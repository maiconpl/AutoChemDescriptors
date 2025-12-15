#!/usr/bin/python3
'''
Created on December 6, 2025

@author: maicon
Last modification by MPL: 13/12/25 to handle smiles erro.
Last modification by MPL: 07/12/25.
'''

#from rdkit import Chem
from rdkit.Chem.Draw import MolToFile

def smiles_checker(smiles_string):

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles_string, sanitize=False)

    if mol is None:
       return "Invalid SMILES (Syntax error)"
    else:
       return "SMILES is syntactically valid"

def get_coordinates_ordered_by_atoms_group(atoms_type_ordered, atoms_symbols, atoms_xyz):

    # atoms_symbols (not ordened), i.e.: ['C', 'O', 'C', 'C', 'O', 'O', 'C', 'Cl', 'C', 'C', 'Cl', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']
    # atoms_type, i.e.: ['C', 'O', 'Cl', 'H']
    # ['C8', 'O3', 'Cl2', 'H6']

    atoms_symbols_ordered = []
    atoms_xyz_ordered = []

    for i in range(len(atoms_type_ordered)):
        for j in range(len(atoms_symbols)):

            if atoms_symbols[j] == atoms_type_ordered[i]:
               print("zoi:", atoms_symbols[j], *atoms_xyz[j])
               atoms_symbols_ordered.append(atoms_symbols[j])
               atoms_xyz_ordered.append(atoms_xyz[j])

    return atoms_symbols_ordered, atoms_xyz_ordered

def smile_molecule_representation(n_molecules, molecules_coded_list, is_debug_true):

    mol_list = [] # to be returned

    for iMain in range(n_molecules):

        from rdkit import Chem
        mol01 = None

        string01 = molecules_coded_list[iMain]
        print("string01:", string01)

        mol01 = Chem.MolFromSmiles(string01)

        if iMain <= 0 and iMain <=9:
           MolToFile(mol01, "mol0" + str(iMain + 1) + ".png")
        else:
           MolToFile(mol01, "mol" + str(iMain + 1) + ".png")

        # 2-Create from smiles code the descriptors
        mol_list.append( mol01 )

    return mol_list
