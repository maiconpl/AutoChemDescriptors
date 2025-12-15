'''
Created on December 07, 2025

@author: maicon & clayton
Last modification by MPL: 13/12/2025 to implement the view and debug output.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from multiprocessing import Pool
from rdkit_calculator import rdkit_calculator

def get_rdkit_calculations(molecules_coded_list, n_jobs):

    print("into  get_rdkit_calculations:", molecules_coded_list)

    # --- 3. Parallel Execution with Pool ---
    num_processors = n_jobs # Use the number of CPUs you want to dedicate

    with Pool(num_processors) as p:
         # map applies run_pyscf_calc to every item in mol_definitions
         results = p.map(rdkit_calculator, molecules_coded_list)

    return results
