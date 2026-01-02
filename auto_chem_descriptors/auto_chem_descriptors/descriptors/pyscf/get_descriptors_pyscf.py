'''
Created on December 06, 2025

@author: maicon
Last modification by MPL: 28/12/2025 to get the properties as descriptors (i.e. HOMO, LUMO, band-gap, electronegativity, hardness, xyz dipole moment) from PySCF calculations.)
Last modification by MPL: 24/12/2025 to save XYZ structures in a single file.)
Last modification by MPL: 17/12/2025 to implement the view/output and deal with the debug."
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from ...utils import get_coordinates_ordered_by_atoms_group, smile_molecule_representation
from ..dscribe.get_dscribe_descriptor import get_dscribe_descriptor

from .get_pyscf_calculations import get_pyscf_calculations

from ..io.get_coordinates_from_smiles import get_coordinates_from_smiles
from ..io.get_coordinates_from_xyz import get_coordinates_from_xyz

from ..io.get_smiles_from_xyz import get_smiles_from_xyz

from ...utils import smiles_checker
from ..io.save_structures_in_file import save_structures_in_file

descriptors = []

def get_descriptors_pyscf(n_jobs, n_molecules, molecules_coded_list, descriptors_type, calculator_controller, is_debug_true):

    from rdkit import Chem
    from ..io.get_atom_information import getAtomTypeCounter, getAtomType
    from rdkit.Chem import AllChem

    atoms_symbols_ordered_list = []
    atoms_xyz_ordered_list = []
    qc_descriptors_list = []
    xyz_new = []

    molecules_coded_from_xyz_list = []

    if len(molecules_coded_list) > 0:

       mol_list = []
       mol_list = smile_molecule_representation(n_molecules, molecules_coded_list, is_debug_true=True)

       is_force_field_true = calculator_controller['is_force_field_true']
       atoms_to_be_optimized_string = get_coordinates_from_smiles(n_molecules, molecules_coded_list=mol_list, is_force_field_true=is_force_field_true)

       print('atoms_to_be_optimized_string from pyscf:', atoms_to_be_optimized_string, len(atoms_to_be_optimized_string))

    elif len(molecules_coded_list) == 0:

        atoms_to_be_optimized_string = get_coordinates_from_xyz('input.xyz')

        molecules_coded_from_xyz_list = get_smiles_from_xyz(atoms_to_be_optimized_string)

        print('molecules_coded_from_xyz_list:', molecules_coded_from_xyz_list)

        #print('atoms_to_be_optimized_string from xyz:', atoms_to_be_optimized_string, len(atoms_to_be_optimized_string))

    if is_debug_true == True:
       print("atoms_to_be_optimized_string:",  atoms_to_be_optimized_string)

    if descriptors_type == "QC":
        xyz_new_and_qc_descriptors_tuple = get_pyscf_calculations(atoms_to_be_optimized_string, calculator_controller, n_jobs=n_jobs)

        print("xyz_new_and_qc_descriptors_tuple:", xyz_new_and_qc_descriptors_tuple)

        for iMolecule in range(len(xyz_new_and_qc_descriptors_tuple)):
            xyz_new.append(xyz_new_and_qc_descriptors_tuple[iMolecule][0])
            qc_descriptors_list.append( xyz_new_and_qc_descriptors_tuple[iMolecule][1])

        if is_debug_true == True:
           print("qc_descriptors_list from QC:", qc_descriptors_list)
           print("xyz_new as list of strings from QC:", xyz_new)

    elif descriptors_type == "MBTR" or descriptors_type == "SOAP":
        xyz_new = get_pyscf_calculations(atoms_to_be_optimized_string, calculator_controller, n_jobs=n_jobs)

    if is_debug_true == True:
       print("xyz_new as list of strings:", xyz_new)

    # get all atoms_symbols in a list (important to use the Dscribe to get the same descriptors size independent of the molecules)
    all_atoms_symbols_list=[]
    for iMol in range(len(xyz_new)):
        for j in range(0, len(xyz_new[iMol]), 4):
           all_atoms_symbols_list.append(xyz_new[iMol][j])

    print('all_atoms_symbols_list:', all_atoms_symbols_list)

    atomSymbolsOfAllMolecules = getAtomType(all_atoms_symbols_list)
    print("atoms type global (list)", atomSymbolsOfAllMolecules)

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

        if is_debug_true == True:
           print("atoms_to_be_optimized_string (final):", atoms_to_be_optimized_string)
           print("atoms_xyz list of floats:", atoms_xyz)
           print("atoms_symbols:", atoms_symbols)

        print ("\nnumber of atoms type (list):", getAtomTypeCounter(atoms_symbols))
        print ("atoms type (list)", getAtomType(atoms_symbols))

        atoms_type = getAtomType(atoms_symbols)

        atoms_type_counter=getAtomTypeCounter(atoms_symbols)
        atoms_symbols_ordered, atoms_xyz_ordered = get_coordinates_ordered_by_atoms_group(atoms_type_ordered=atoms_type, atoms_symbols=atoms_symbols, atoms_xyz=atoms_xyz, is_debug_true=is_debug_true)

        if is_debug_true == True:
           print("atoms_xyz_ordered (list):", atoms_xyz_ordered)
           print("atoms_symbols_ordered (list):", atoms_symbols_ordered)

        print("atoms_type_counter and string:", atoms_type_counter, "".join(atoms_type_counter))

        if descriptors_type == "MBTR" or descriptors_type == 'SOAP':
               descriptor = get_dscribe_descriptor(atoms_symbols_ordered,
                                                   atomSymbolsOfAllMolecules,
                                                   atoms_xyz_ordered,
                                                   system_type="cluster",
                                                   descriptor_type=descriptors_type.lower())

        if descriptors_type == "QC":
               descriptor = qc_descriptors_list[iMol]

        descriptors.append(descriptor)

        # Add the all ordered atoms_symbols and the atoms_xyz obtained from the above loop.

        atoms_symbols_ordered_list.append(atoms_symbols_ordered)
        atoms_xyz_ordered_list.append(atoms_xyz_ordered) 

    # save xyz in a single file
    if len(molecules_coded_list) > 0:
       save_structures_in_file(atoms_symbols_ordered_list, atoms_xyz_ordered_list, smiles_list=molecules_coded_list)

    elif len(molecules_coded_list) == 0:
       n_molecules = len(molecules_coded_from_xyz_list)
       smile_molecule_representation(n_molecules, molecules_coded_list=molecules_coded_from_xyz_list, is_debug_true=True)
       save_structures_in_file(atoms_symbols_ordered_list, atoms_xyz_ordered_list, smiles_list=molecules_coded_from_xyz_list)

    return descriptors, molecules_coded_from_xyz_list
