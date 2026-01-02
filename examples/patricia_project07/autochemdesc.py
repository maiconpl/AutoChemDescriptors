from main_auto_chem_descriptor import main_auto_chem_descriptor

if __name__ == '__main__':

    # ----------
    # BEGIN INPUT
    # ----------

    is_debug_true = False

    n_jobs = 2

    input_flow_controller = {

        # 'molecular_encoding': "Selfies",
        'molecular_encoding': "SMILES",
        # 'descriptors_type': "MBTR",
        # 'descriptors_type': "mbtr",
        'descriptors_type': "SMILES",
    }

    molecules_coded_list = [
        "COc1ccccc1O",  # 1
        "COc1ccccc1OCC(=O)O",  # 2
        "COc1ccccc1OCC(=O)Nc1ccccc1",  # 3
        "COc1ccccc1OCC(=O)Nc1ccccc1F",  # 4
        "COc1ccccc1OCC(=O)Nc1cccc(F)c1",  # 5
        "COc1ccccc1OCC(=O)Nc1ccc(F)cc1",  # 6
        "COc1ccccc1OCC(=O)Nc1ccccc1Cl",  # 7
        "COc1ccccc1OCC(=O)Nc1cccc(Cl)c1",  # 8
        "COc1ccccc1OCC(=O)Nc1ccc(Cl)cc1",  # 9
        "COc1ccccc1OCC(=O)Nc1ccccc1Br",  # 10
        "COc1ccccc1OCC(=O)Nc1cccc(Br)c1",  # 11
        "COc1ccccc1OCC(=O)Nc1ccc(Br)cc1",  # 12
        "COc1ccccc1OCC(=O)Nc1ccccc1[N+](=O)[O-]",  # 13
        "COc1ccccc1OCC(=O)Nc1cccc([N+](=O)[O-])c1",  # 14
        "COc1ccccc1OCC(=O)Nc1ccc([N+](=O)[O-])cc1",  # 15
        "COc1ccccc1OCC(=O)Nc1ccccc1C",  # 16
        "COc1ccccc1OCC(=O)Nc1cccc(C)c1",  # 17
        "COc1ccccc1OCC(=O)Nc1cccc(C)c1",  # 18
        "COc1ccccc1OCC(=O)Nc1cccc2ccccc12",  # 19
    ]
# E1.png,COc1ccccc1O,0.8836916332166905
# E2.png,COc1ccccc1OCC(=O)O,0.8983917425258536
# E3.png,COc1ccccc1OCC(=O)Nc1ccccc1,0.9191498071277909
# E4.png,COc1ccccc1OCC(=O)Nc1ccccc1F,0.9154075725373523
# E5.png,COc1ccccc1OCC(=O)Nc1cccc(F)c1,0.9014868911883573
# E6.png,COc1ccccc1OCC(=O)Nc1ccc(F)cc1,0.9057028210878992
# E7.png,COc1ccccc1OCC(=O)Nc1ccccc1Cl,0.9100731755768702
# E8.png,COc1ccccc1OCC(=O)Nc1cccc(Cl)c1,0.9037308556218153
# E9.png,COc1ccccc1OCC(=O)Nc1ccc(Cl)cc1,0.9104481272260029
# E10.png,COc1ccccc1OCC(=O)Nc1ccccc1Br,0.8978656128111406
# E11.png,COc1ccccc1OCC(=O)Nc1cccc(Br)c1,0.9110157459253232
# E12.png,COc1ccccc1OCC(=O)Nc1ccc(Br)cc1,0.9059949661575893
# E13.png,COc1ccccc1OCC(=O)Nc1ccccc1[N+](=O)[O-],0.9104868206074117
# E14.png,COc1ccccc1OCC(=O)Nc1cccc([N+](=O)[O-])c1,0.9159857769313288
# E15.png,COc1ccccc1OCC(=O)Nc1ccc([N+](=O)[O-])cc1,0.9130673215598787
# E16.png,COc1ccccc1OCC(=O)Nc1ccccc1C,0.9089714452916253
# E17.png,COc1ccccc1OCC(=O)Nc1cccc(C)c1,0.9019378754647928
# E18.png,COc1ccccc1OCC(=O)Nc1ccc(C)cc1,0.8889553183475573

    calculator_controller = {}

    n_components = 4

    analysis = {

        "pca_heatmap": [True, n_components],
        "pca_grouping": [True, n_components],
        "pca_dispersion": [True, n_components],
        "kmeans": {
            "k_min": 2,
            "k_max": 8,
            "random_state": 42,
            "use_minibatch": False,
            "projection_components": 2
        },
        "dbscan": {
            "min_samples": 8,
            "metric_mode": "auto",
            "precomputed_max_samples": 1200,
            "eps": 0.20,          # optional: omit to rely on knee detection
            "n_jobs": -1,         # optional: set None to use scikit-learn default
            # optional: any of {"auto","ball_tree","kd_tree","brute"}
            "algorithm": "auto"
        },

        "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y',
                            'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

        "molecules_label": [
            'Guaiacol', 'GACO1', 'GAO00',
            'GAA21', 'GAA31', 'GAA41',
            'GAA22', 'GAA32', 'GAA42',
            'GAA23', 'GAA33', 'GAA43',
            'GAA24', 'GAA34', 'GAA44',
            'GAA25', 'GAA35', 'GAA45', 'GANT']
    }

    main_auto_chem_descriptor(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller, analysis)  # , is_debug_true=True)
