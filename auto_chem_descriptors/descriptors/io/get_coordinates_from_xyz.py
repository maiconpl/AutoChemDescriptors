'''
Created on December 22, 2025

@author: maicon & clayton
Last modification by MPL: 24/12/2025 to save XYZ structures in a single file.)
'''

def get_coordinates_from_xyz(file_name):

    atoms_to_be_optimized_string_list = [] # MPL

    all_molecules_xyz_list = []
    all_molecules_atomic_symbols_list = []

    tmp_xyz_list = []
    tmp_atomic_symbols_list = []

    tmp01_list = []
    tmp02_list = []
    
    fileIn = open(file_name, "r")

    inputLines = fileIn.readlines()

    n_atoms_list=[]

    begin=0
    end=0
    for i in range(0,len(inputLines)):

        if i == begin:

           n_atoms_list.append(float(inputLines[i].split()[0]))
           tmp_n_atoms=(float(inputLines[i].split()[0]))
           #print("tmp_n_atoms:",i, tmp_n_atoms)
           end= end + tmp_n_atoms + 2

           #print("##")
           for j in range(int(begin) + 2,int(end)):
               #print("mm.:", inputLines[j])

               tmp01_list = [float(inputLines[j].split()[1]), float(inputLines[j].split()[2]), float(inputLines[j].split()[3])]
               tmp02_list = [str(inputLines[j].split()[0])]

               tmp_xyz_list.append(tmp01_list)
               tmp_atomic_symbols_list.append(tmp02_list)

               tmp01_list = []
               tmp02_list = []

           begin=begin + tmp_n_atoms + 2

           all_molecules_xyz_list.append(tmp_xyz_list)
           all_molecules_atomic_symbols_list.append( [item for sublist in tmp_atomic_symbols_list for item in sublist]  )

           tmp_xyz_list = []
           tmp_atomic_symbols_list=[]

    fileIn.close()
    #print("n_atoms_list:", n_atoms_list)
    #print("n_atoms_list:", all_molecules_xyz_list, len(all_molecules_xyz_list))
    #print("n_atoms_list:", all_molecules_atomic_symbols_list, len(all_molecules_atomic_symbols_list))

    xyz_new = [] # MPL
    for i in range(len(all_molecules_xyz_list)):
        xyz_new = [] # MPL
        #print("NN,", i, all_molecules_xyz_list[i], len(all_molecules_xyz_list[i]))
        for j in range(len(all_molecules_xyz_list[i])):
            xyz_new.append(str(all_molecules_atomic_symbols_list[i][j]))
            xyz_new = xyz_new + ['  '.join( str(item) for item in all_molecules_xyz_list[i][j] ).split() ][0]

        print("\nxyz_new (non-optimized):", xyz_new)

        atoms_to_be_optimized_string = "" # from RDKit force field
        for iMolecule in range(0, len(xyz_new), 4):
           atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[iMolecule: iMolecule + 4]) + "; "

        atoms_to_be_optimized_string_list.append(atoms_to_be_optimized_string)

    #print('bbasdfasdf:', atoms_to_be_optimized_string_list, len(atoms_to_be_optimized_string_list))

    return atoms_to_be_optimized_string_list # return the string to be used in PySCF input
