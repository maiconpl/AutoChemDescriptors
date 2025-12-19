'''
Created on December 06, 2025

@author: maicon & clayton
Last modification by MPL: 17/12/2025 to implement the analysis and debug.; )
Last modification by MPL: 11/12/2025 to implement the analysis and debug.; )
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw

#import random # to fix the seed to get similar coordinates

descriptors = []

def get_coordinates_from_smiles(n_molecules, molecules_coded_list, is_force_field_true):

    #random.seed(42) # to fix the seed to get similar coordinates

    atoms_to_be_optimized_string_list = []
    mol_list = []
    mol_list = molecules_coded_list

    for iMain in range(n_molecules):

        xyz = None
        xyz_new = None
        mol = None

        # 2-Create from smiles code the descriptors
        mol=mol_list[iMain]

        print("\nmolecule similes code" + " '" + Chem.MolToSmiles(mol) + "':")

        # 3-Create from smiles code the descriptors get molecule coordinates

        Chem.AllChem.EmbedMolecule(mol) # Why is this important? 06/12/25 , this affects the results, why??
        mol = AllChem.AddHs(mol) # make sure to add explicit hydrogens

        from rdkit.Chem import Draw
        Draw.MolToImage(mol)

        # Generate 3D coordinates
        AllChem.EmbedMolecule(mol)
        xyz = Chem.AllChem.MolToXYZBlock(mol) # initial coords

        file_name01 = "unoptimized_" + str(iMain) + ".xyz"
        with open(file_name01,'w') as outf:
            outf.write(xyz)

        if is_force_field_true == True:
           # Perform UFF optimization
           ff = None
           ff = AllChem.UFFGetMoleculeForceField(mol)
           ff.Initialize()
           ff.Minimize(energyTol=1e-7,maxIts=100000)

           xyz = Chem.AllChem.MolToXYZBlock(mol)
           print(xyz)
           file_name02 = "optimized_ff_" + str(iMain) + ".xyz"
           with open(file_name02,'w') as outf:
               outf.write(xyz)

        xyz_new = xyz.split()[1:]
        print("\nxyz_new (non-optimized):", xyz_new)

        atoms_to_be_optimized_string = "" # from RDKit force field
        for i in range(0, len(xyz_new), 4):
           #print("kk", xyz_new[i])
           #print("kk", xyz_new[i: i + 4])
           atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[i: i + 4]) + "; "
           #print( )

        #print("zaza " + str( Chem.MolToSmiles(mol_list[iMain]) ) + ":", atoms_to_be_optimized_string)
        atoms_to_be_optimized_string_list.append(atoms_to_be_optimized_string)

        del mol

    return atoms_to_be_optimized_string_list # return the string to be used in PySCF input
