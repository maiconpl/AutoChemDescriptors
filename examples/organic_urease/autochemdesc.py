from main_auto_chem_descriptor import main_auto_chem_descriptor

if __name__ == '__main__':

    # ----------
    #BEGIN INPUT
    # ----------


    is_debug_true = False

    n_jobs=3

    input_flow_controller = {

            #'molecular_encoding': "Selfies",
            'molecular_encoding': "SMILES",
            #'descriptors_type': "MBTR",
            #'descriptors_type': "mbtr",
            'descriptors_type': "SMILES",
    }

    molecules_coded_list = [
                 "c1(OCC(O)(=O))c(Cl)cc(Cl)cc1",
                 "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", # cafein
                 "c1ccc(Cl)cc1(Cl)",
                 "c1ccc(Cl)cc1(Cl)",
                ]

    calculator_controller = {}

    n_components=3
    analysis = {
    "pca_grouping": [True, n_components],

    "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y', 
                       'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

    "molecules_label": molecules_coded_list,
    }

    main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller, analysis)#, is_debug_true=True)
