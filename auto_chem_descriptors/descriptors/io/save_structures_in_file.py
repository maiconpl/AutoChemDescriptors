#!/usr/bin/python3

'''
Created on December 22, 2025

@author: maicon & clayton
Last modification by MPL: 23/12/2025 to save XYZ structures in a single file.)
'''

def save_structures_in_file(atoms_symbols_list, atoms_xyz_list, smiles_list):

    #print('into save_structures_in_file, smiles_list:', smiles_list)
    #print('into save_structures_in_file, atoms_symbols_list:', atoms_symbols_list)
    #print('into save_structures_in_file, atoms_xyz_list:', atoms_xyz_list)
 
    file_write_xyz_name = "molecules_coordinates_after_calculations.xyz"
    file_write_xyz = open(file_write_xyz_name, 'w')

    n_molecules = len(atoms_symbols_list)

    for iMolecule in range(n_molecules):
        n_atoms = len(atoms_symbols_list[iMolecule])

        #print(n_atoms)

        file_write_xyz.write(str(n_atoms))
        file_write_xyz.write("\n")

        if len(smiles_list) > 0:
           #print("mm:", smiles_list[iMolecule])
           file_write_xyz.write( str(smiles_list[iMolecule]))

        else:
           print("")
           file_write_xyz.write("#")

        file_write_xyz.write("\n")

        for jAtom in range(n_atoms):
            #print("mm:", atoms_symbols_list[iMolecule][jAtom], atoms_xyz_list[iMolecule][jAtom])
            file_write_xyz.write(str(atoms_symbols_list[iMolecule][jAtom]) +  "   " + "  ".join(str(item) for item in atoms_xyz_list[iMolecule][jAtom] ))
            file_write_xyz.write("\n")

    file_write_xyz.close()
