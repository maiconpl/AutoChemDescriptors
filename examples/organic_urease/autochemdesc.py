from main_auto_chem_descriptor import main_auto_chem_descriptor

if __name__ == '__main__':

    # ----------
    #BEGIN INPUT
    # ----------


    is_debug_true = False

    n_jobs=2

    input_flow_controller = {

            #'molecular_encoding': "Selfies",
            'molecular_encoding': "SMILES",
            #'descriptors_type': "MBTR",
            #'descriptors_type': "mbtr",
            'descriptors_type': "SMILES",
    }

    molecules_coded_list = [
                     "c1ccccc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O)ccc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1ccc(O)cc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O)ccc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(O)c(O)cc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O)c(O(C))cc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O)c(O(C))cc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(F)ccc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(F)ccc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(S(C))ccc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O(C))ccc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(O2)c(O(C2))cc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1cc(O2)c(O(C2))cc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(S(C))ccc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1cc(O)c(O)cc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "C1CCCCC1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "C1CCCCC1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1ccc(O(C))cc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1c(O(C))c(O)c(O(C))cc1(C1c(C(=O)O(CC))c(C)NC(=O)N1)",
                     "c1ccccc1(C1c(C(=O)O(CC))c(C)NC(=S)N1)",
                     "c1(C)c(C(=O)O(CC))C(CCC)NC(=S)N1",
                     "c1(C)c(C(=O)O(CC))C(CCC)NC(=O)N1",
                ]

    calculator_controller = {}

    n_components=5
    analysis = {
    "pca_heatmap": [True, n_components],
    "pca_grouping": [True, n_components],
    "pca_dispersion": [True, n_components],

    "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y', 
                       'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

    "molecules_label": molecules_coded_list,
    "legend_bbox_to_anchor": (0.5, -0.180),
    #"legend_size": 12,
    "legend_ncol": 2, # number of columns of the legend
    }

    main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller, analysis)#, is_debug_true=True)
