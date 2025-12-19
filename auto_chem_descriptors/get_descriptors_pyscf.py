'''
Created on December 06, 2025

@author: maicon
Last modification by MPL: 17/12/2025 to implement the view/output and deal with the debug."
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from utils import get_coordinates_ordered_by_atoms_group, smile_molecule_representation
from get_dscribe_descriptor import get_dscribe_descriptor

from get_pyscf_calculations import get_pyscf_calculations

from get_coordinates_from_smiles import get_coordinates_from_smiles
from utils import smiles_checker

descriptors = []

def get_descriptors_pyscf(n_jobs, n_molecules, molecules_coded_list, descriptors_type, calculator_controller, is_debug_true):

    from rdkit import Chem
    from get_atom_information import getAtomTypeCounter, getAtomType
    from rdkit.Chem import AllChem

    mol_list = []
    mol_list = smile_molecule_representation(n_molecules, molecules_coded_list, is_debug_true=True)

    is_force_field_true = calculator_controller['is_force_field_true']

    atoms_to_be_optimized_string = get_coordinates_from_smiles(n_molecules, molecules_coded_list=mol_list, is_force_field_true=is_force_field_true)

    if is_debug_true == True:
       print("atoms_to_be_optimized_string:",  atoms_to_be_optimized_string)

    xyz_new = get_pyscf_calculations(atoms_to_be_optimized_string, calculator_controller, n_jobs=n_jobs)

    if is_debug_true == True:
       print("xyz_new as list of strings:", xyz_new)

    for iMol in range(len(xyz_new)):

        atoms_to_be_optimized_string = "" # from pySCF
        atoms_symbols=[]
        atoms_xyz = []
        tmp_list01 = []
        atoms_xyz_tmp01 = []

        for j in range(0, len(xyz_new[iMol]), 4):
           #print("kkkk", xyz_new[iMol][j])
           atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[iMol][j + 1: j + 4]) + "; "
           tmp_list01 =  xyz_new[iMol][j + 1: j + 4]
           atoms_symbols.append(xyz_new[iMol][j])
           for iCoordinates in tmp_list01:
               atoms_xyz_tmp01.append(float(iCoordinates))
           atoms_xyz.append(atoms_xyz_tmp01)
           atoms_xyz_tmp01 = []
           print( )

        if is_debug_true == True:
           print("atoms_to_be_optimized_string (final):", atoms_to_be_optimized_string)
           print("atoms_xyz list of floats:", atoms_xyz)
           print("atoms_symbols:", atoms_symbols)

        print ("number of atoms type (list):", getAtomTypeCounter(atoms_symbols))
        print ("atoms type (list)", getAtomType(atoms_symbols))

        atoms_type = getAtomType(atoms_symbols)

        atoms_type_counter=getAtomTypeCounter(atoms_symbols)
        atoms_symbols_ordered, atoms_xyz_ordered = get_coordinates_ordered_by_atoms_group(atoms_type_ordered=atoms_type, atoms_symbols=atoms_symbols, atoms_xyz=atoms_xyz, is_debug_true=is_debug_true)

        if is_debug_true == True:
           print("atoms_xyz_ordered (list):", atoms_xyz_ordered)
           print("atoms_symbols_ordered (list):", atoms_symbols_ordered)

        print("atoms_type_counter and string:", atoms_type_counter, "".join(atoms_type_counter))

        if descriptors_type == "MBTR":
               descriptor = get_dscribe_descriptor(atoms_symbols_ordered, atoms_xyz_ordered, system_type="cluster", descriptor_type="mbtr")

        descriptors.append(descriptor)

    return descriptors
