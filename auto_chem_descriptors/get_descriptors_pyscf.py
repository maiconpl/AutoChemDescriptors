'''
Created on December 06, 2025

@author: maicon
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from utils import get_coordinates_ordered_by_atoms_group, smile_molecule_representation
from get_describe_descriptor import get_describe_descriptor

from get_pyscf_calculations import get_pyscf_calculations

from get_coordinates_from_smiles import get_coordinates_from_smiles

descriptors = []

def get_descriptors_pyscf(n_jobs, n_molecules, molecules_coded_list, descriptors_type, calculator_controller, is_debug_true):

    from rdkit import Chem
    from get_atom_information import getAtomTypeCounter, getAtomType
    from rdkit.Chem import AllChem

    mol_list = []
    mol_list = smile_molecule_representation(n_molecules, molecules_coded_list, is_debug_true=False)

    is_force_field_true = calculator_controller['is_force_field_true']

    atoms_to_be_optimized_string = get_coordinates_from_smiles(n_molecules, molecules_coded_list=mol_list, is_force_field_true=is_force_field_true)

    print(" atoms_to_be_optimized_string in get:",  atoms_to_be_optimized_string)

    #xyz_new = pyscf_calculator(atoms_to_be_optimized_string, maxsteps)

    maxsteps = 1
    #xyz_new = get_pyscf_calculations(atoms_to_be_optimized_string, maxsteps, n_jobs=n_jobs)
    xyz_new = get_pyscf_calculations(atoms_to_be_optimized_string, calculator_controller, n_jobs=n_jobs)

    print("xyz_new from mull:", xyz_new)

    for iMol in range(len(xyz_new)):

        atoms_to_be_optimized_string = "" # from pyscf
        atoms_symbols=[]
        atoms_xyz = []
        tmp_list01 = []
        atoms_xyz_tmp01 = []

        for j in range(0, len(xyz_new[iMol]), 4):
           print("kkkk", xyz_new[iMol][j])
           atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[iMol][j + 1: j + 4]) + "; "
           tmp_list01 =  xyz_new[iMol][j + 1: j + 4]
           atoms_symbols.append(xyz_new[iMol][j])
           for iCoordinates in tmp_list01:
               atoms_xyz_tmp01.append(float(iCoordinates))
           atoms_xyz.append(atoms_xyz_tmp01)
           atoms_xyz_tmp01 = []
           print( )

        print("zaza final:", atoms_to_be_optimized_string)
        print("zaza atoms_xyz:", atoms_xyz)
        print("zaza atoms_symbols:", atoms_symbols)

        print (getAtomTypeCounter(atoms_symbols))
        print (getAtomType(atoms_symbols))
        atoms_type = getAtomType(atoms_symbols)

        atoms_type_counter=getAtomTypeCounter(atoms_symbols)
        atoms_symbols_ordered, atoms_xyz_ordered = get_coordinates_ordered_by_atoms_group(atoms_type_ordered=atoms_type, atoms_symbols=atoms_symbols, atoms_xyz=atoms_xyz)

        print("zaza atoms_xyz_ordered:", atoms_xyz_ordered)
        print("zaza atoms_symbols_ordered:", atoms_symbols_ordered)
        print("zaza atoms_type_counter:", atoms_type_counter, "".join(atoms_type_counter))

        if descriptors_type == "MBTR":
               descriptor = get_describe_descriptor(atoms_symbols_ordered, atoms_xyz_ordered, system_type="cluster", descriptor_type="mbtr")

        descriptors.append(descriptor)

    return descriptors
