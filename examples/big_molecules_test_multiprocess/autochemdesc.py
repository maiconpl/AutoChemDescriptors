from main_auto_chem_descriptor import main_auto_chem_descriptor

if __name__ == '__main__':

    # ----------
    #BEGIN INPUT
    # ----------

    #is_debug_true = False
    is_debug_true = True

    n_jobs=2

    input_flow_controller = {

            #'molecular_encoding': "Selfies",
            'molecular_encoding': "SMILES",
            #'descriptors_type': "MBTR",
            'descriptors_type': "SOAP",
            #'descriptors_type': "mbtr",
            #'descriptors_type': "SMILES",
    }

    molecules_coded_list = [
                 "c1(OCC(O)(=O))c(Cl)cc(Cl)cc1",
                 "c1(OCC(O)(=O))c(Cl)cc(Cl)cc1",
                 "c1(OCC(O)(=O))ccc(Cl)cc1(Cl)",
                 "c1(OCC(O)(=O))ccc(Cl)cc1(Cl)",
                 "c1ccc(Cl)cc1(Cl)",
                 #"c1ccc(Cl)cc1(Cl)",
                 #"CN1C=NC2=C1C(=O)N(C)C(=O)N2C", # cafein
                 #"CN1C=NC2=C1C(=O)N(C)C(=O)N2C", # cafein
                 #"CC(=O)NC1=CC=C(O)C=C1", #paracetamol
                 #"CC(=O)NC1=CC=C(O)C=C1", #paracetamol
                 #"CN1CCCC1C2=CN=CC=C2", #nicotine
                 #"CN1CCCC1C2=CN=CC=C2", #nicotine
                ]

    calculator_controller = {
    'is_force_field_true': False, # from RDKit, to get the pre-optimized XYZ.
    #'is_force_field_true': True, # from RDKit, to get the pre-optimized XYZ.
    'program': 'pyscf',
    'basis': 'sto-3g',
    'maxsteps': 3,
    #'maxsteps': 30,
    }

    analysis = {}

    n_components=4
    analysis = {
    "dscribe_plot": [True],
    "pca_grouping": [True, n_components],

    "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y',
                       'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

    "molecules_label": molecules_coded_list,
    }

    main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller, analysis)#, is_debug_true=True)
