#!/usr/bin/python3
'''
Created on December 03, 2025

@author: maicon & clayton
Last modification by MPL: 24/12/2025 to import the structures from XYZ files.; )
Last modification by MPL: 17/12/2025 to implement the analysis and debug.; )
Last modification by MPL: 17/12/2025 to implement the output from print and deal with debug.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from datetime import datetime

# Initial time:
t0 = datetime.now()

from get_descriptors_pyscf import  get_descriptors_pyscf
from get_descriptors_smiles import  get_descriptors_smiles

from software_information_auto_chem_descriptors import software_information_auto_chem_descriptors

from plot_dscribe import plot_dscribe
from plot_pca_grouping import plot_pca_grouping
from plot_pca_heatmap import plot_pca_heatmap
from plot_pca_dispersion import plot_pca_dispersion
import csv

def main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller,
                              analysis=None,
                              is_debug_true=None):

    #########################
    ###### BEGIN MAIN #######
    #########################

    software_information_auto_chem_descriptors()

    print("--------------------------")
    print("------ BEGIN OUTPUT -------")
    print("--------------------------\n")

    print("Begin input prints:")
    print("input_flow_controller:", input_flow_controller)
    print("molecules_coded_list:", molecules_coded_list)
    print("calculator_controlle:", calculator_controller)
    print("analysis:", analysis)
    print("End input prints.")

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

    ## BEGIN: DESCRIPTORS ##

    print("\nBegin descriptors " + '"'+ str(descriptors_type) + '"' + " information:")
    if descriptors_type == "SMILES":
       descriptors_list = get_descriptors_smiles(n_jobs, molecules_coded_list)#, is_debug_true)

    if descriptors_type == "MBTR" or descriptors_type == "SOAP":
       descriptors_list, molecules_coded_from_xyz_list = get_descriptors_pyscf(n_jobs, n_molecules, molecules_coded_list, descriptors_type, calculator_controller, is_debug_true)

       if len(molecules_coded_list) == 0: # in case of getting the smiles from xyz
       # redefine analysis dicitionary
          analysis['molecules_label'] = molecules_coded_from_xyz_list 

    print("descriptor:", descriptors_list)

    ## END: DESCRIPTORS ##

    ## BEGIN: WRITING DESCRIPTORS ##
     
    if descriptors_type == "SMILES":

       file_write_txt_name = 'descriptors_from_rdkit.txt'
       file_write_csv_name = 'descriptors_from_rdkit.csv'

       file_write_txt = open(file_write_txt_name, 'w')
       file_write_csv = open(file_write_csv_name, mode='w', newline='')
       csv_writer = csv.writer(file_write_csv)

       descriptors_name_from_rdkit = [
            "FpDensityMorgan1", 
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "MaxAbsPartialCharge",
            "MaxPartialCharge",
            "MinAbsPartialCharge",
            "MinPartialCharge",
            "ExactMolWt",
            "NumRadicalElectrons",
            "NumValenceElectrons",
            "ComputeMolVolume",
            "HeavyAtomMolWt",
       ]

       # write header
       file_write_txt.write("# ".join(str(i) for i in descriptors_name_from_rdkit)  + "\n")
       csv_writer.writerow(descriptors_name_from_rdkit)
       print("\nDesciptors list" + "(" + "'" + str(len(descriptors_list)) + "'"  + " molecules/substances; " + "'" + str(len(descriptors_list[0])) + "'"  + " features" + ")" + ":")
       print ("#", *descriptors_name_from_rdkit)

    if descriptors_type == "MBTR" or descriptors_type == "SOAP":

       file_write_txt_name = 'descriptors_from_' + str(descriptors_type) + '.txt'
       file_write_csv_name = 'descriptors_from_'+ str(descriptors_type) + '.csv'

       file_write_txt = open(file_write_txt_name, 'w')
       file_write_csv = open(file_write_csv_name, mode='w', newline='')
       csv_writer = csv.writer(file_write_csv)

       descriptors_name_from_first_principles = [" ".join(str(i) for i in range(len(descriptors_list[0])))]

       # write header
       file_write_txt.write("# ".join(str(i) for i in descriptors_name_from_first_principles)  + "\n")
       csv_writer.writerow(descriptors_name_from_first_principles)
       print("\nDesciptors list" + "(" + "'" + str(len(descriptors_list)) + "'"  + " molecules/substances; " + "'" + str(len(descriptors_list[0])) + "'"  + " features" + ")" + ":")

    for iPrint in descriptors_list:
        file_write_txt.write(" ".join(str(i) for i in iPrint)  + "\n")
        csv_writer.writerows([iPrint])
        print(*iPrint, "number of features:", len(iPrint))

    file_write_txt.close()
    file_write_csv.close()

    print("\nEnd descriptors " + '"'+ str(descriptors_type) + '"' + " information.")

    ## BEGIN: WRITING DESCRIPTORS ##

    ## BEGIN: ANALYSIS ##

    if 'dscribe_plot' in analysis and descriptors_type != "SMILES":
        plot_dscribe(descriptors_list, descriptors_type, analysis)

    if 'pca_grouping' in analysis:
        print("\nPCA grouping analysis:\n")
        plot_pca_grouping(descriptors_list, molecular_encoding, analysis)

    if 'pca_heatmap' in analysis and descriptors_type == "SMILES":
        print("\nPCA heatmap:\n")
        plot_pca_heatmap(descriptors_list, analysis)

    if 'pca_dispersion' in analysis and descriptors_type == "SMILES":
        print("\nPCA dispersion:\n")
        plot_pca_dispersion(descriptors_list, analysis)

    ## END: ANALYSIS ##

    print("\n------------------------")
    print("------ END OUTPUT ------")
    print("------------------------")

    ######################
    ###### END MAIN ######
    ######################

    # Main libraries version:
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
