#from main_auto_chem_descriptor import main_auto_chem_descriptor
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from auto_chem_descriptors.main.main_auto_chem_descriptors import main_auto_chem_descriptors

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
    'method': 'RHF', 
    'basis': 'sto-3g',
    'maxsteps': 3,
    'properties': False,
    #'maxsteps': 30,
    }

    analysis = {}

    n_components=4
    analysis = {

    "dscribe_plot": [True],
    "pca_grouping": [True, n_components],

    "kmeans": {
           "k_min": 2,
           "k_max": 8,
           "random_state": 42,
           "use_minibatch": False,
           "projection_components": 2
    },

    "dbscan": {
           "min_samples": 4,
           "metric_mode": "auto",
           "precomputed_max_samples": 1200,
           "eps": 0.35,              # optional; omit to adopt the knee suggestion
           "n_jobs": -1,             # optional; set None to use scikit-learn default
           "algorithm": "brute"      # optional; accepts {'auto','ball_tree','kd_tree','brute'}
    
    },

    "kmeans_report": {
           "report_filename": "report_kmeans.md",
           "metrics_filename": "kmeans_metrics.csv",
           "suggestions_filename": "kmeans_suggestions.json",
           "labels_filename": "kmeans_cluster_labels.csv"
    },

    "molecules_color": ['b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y',
                       'b', 'g', 'r', 'c', 'm', 'b', 'g', 'r', 'c', 'm', 'y'],

    "molecules_label": molecules_coded_list,
    "legend_bbox_to_anchor": (0.5, -0.180),
    #"legend_size": 12,
    "legend_ncol": 2, # number of columns of the legend
    }

    main_auto_chem_descriptors(n_jobs,
                              input_flow_controller,
                              molecules_coded_list,
                              calculator_controller, analysis)#, is_debug_true=True)
