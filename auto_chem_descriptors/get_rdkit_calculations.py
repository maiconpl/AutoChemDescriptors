'''
Created on December 07, 2025

@author: maicon
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from multiprocessing import Pool
from rdkit_calculator import rdkit_calculator

#def get_rdkit_calculations(atoms_to_be_optimized_string, n_jobs):
def get_rdkit_calculations(molecules_coded_list, n_jobs):

    print("into  get_rdkit_calculations:", molecules_coded_list)

    # --- 3. Parallel Execution with Pool ---
    num_processors = n_jobs # Use the number of CPUs you want to dedicate

    #print(f"Starting parallel calculations on {len(mol_definitions)} molecules using {num_processors} processes...")

    #if __name__ == '__main__':
    with Pool(num_processors) as p:
            # map applies run_pyscf_calc to every item in mol_definitions
            #results = p.map(run_pyscf_calc, mol_definitions)
         #results = p.map(pyscf_calculator, atoms_to_be_optimized_string)
         results = p.map(rdkit_calculator, molecules_coded_list)

    return results
    #xyz_new = pyscf_calculator(atoms_to_be_optimized_string, maxsteps)
