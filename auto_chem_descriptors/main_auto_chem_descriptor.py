#!/usr/bin/python3
'''
Created on December 03, 2025

@author: maicon & clayton
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from datetime import datetime

# Initial time:
t0 = datetime.now()

from get_descriptors_pyscf import  get_descriptors_pyscf
from get_descriptors_smiles import  get_descriptors_smiles

from software_information_auto_chem_descriptors import software_information_auto_chem_descriptors

software_information_auto_chem_descriptors()

from plot_pca_grouping import plot_pca_grouping
from plot_pca_heatmap import plot_pca_heatmap
from plot_pca_dispersion import plot_pca_dispersion

def main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller,
                              analysis=None,
                              is_debug_true=None):

    #########################
    ###### BEGIN MAIN #######
    #########################

    print("analysis", analysis)

    if is_debug_true==None:
       is_debug_true = False
    elif is_debug_true == True:
       is_debug_true = True
    elif is_debug_true == False:
       is_debug_true = False

    is_force_field_true = False

    molecular_encoding = input_flow_controller['molecular_encoding']
    descriptors_type = input_flow_controller['descriptors_type']
    
    if len(calculator_controller) != 0:
       is_force_field_true = calculator_controller['is_force_field_true']

    descriptors_list = []
    n_molecules = len(molecules_coded_list)

    if descriptors_type == "SMILES":
       descriptors_list = get_descriptors_smiles(n_jobs, molecules_coded_list, is_debug_true)

    if descriptors_type == "MBTR":
       descriptors_list = get_descriptors_pyscf(n_jobs, n_molecules, molecules_coded_list, descriptors_type, calculator_controller, is_debug_true)

    print("descriptor:", descriptors_list)

    ## BEGIN: HERE COMES THE ANALYSIS ##

    print("\nDesciptors list:")
    for iPrint in descriptors_list:
        print(*iPrint)

    if 'pca_grouping' in analysis:
        print("\nPCA grouping analysis:\n")
        plot_pca_grouping(descriptors_list, molecular_encoding, analysis)

    if 'pca_heatmap' in analysis and descriptors_type == "SMILES":
        print("\nPCA heatmap:\n")
        plot_pca_heatmap(descriptors_list, analysis)

    if 'pca_dispersion' in analysis and descriptors_type == "SMILES":
        print("\nPCA dispersion:\n")
        plot_pca_dispersion(descriptors_list, analysis)

    ## END: HERE COMES THE ANALYSIS ##

    ######################
    ###### END MAIN ######
    ######################

    # Main libraries version:
    print("\n")
    from libraries_information_auto_chem_descriptors import libraries_information_auto_chem_descriptors
    libraries_information_auto_chem_descriptors()

    # Measuring the execution time
    delta_t = datetime.now() - t0

    # Measuring the execution time
    delta_t = datetime.now() - t0

    #Final time
    print ("Execution time:", delta_t)

    # Date and time of execution
    print ("Date and time of execution:", t0.strftime("%Y-%m-%d, %H:%M"))
