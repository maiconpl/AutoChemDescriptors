'''
Created on December 07, 2025

@author: maicon
Last modification by MPL: 28/12/2025 to implement the properties from the optimized geometry: total energy, HOMO, LUMO, band-gap, electronegativiy, hardness and dipole moment.
Last modification by MPL: 08/12/2025 to implement another argument in pyscf_calculations using multiprocessing. 
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from multiprocessing import Pool
from functools import partial
from .pyscf_calculator import pyscf_calculator

def get_pyscf_calculations(atoms_to_be_optimized_string, calculator_controller, n_jobs):

    # --- 3. Parallel Execution with Pool ---
    num_processors = n_jobs # Use the number of CPUs you want to dedicate

    #print(f"Starting parallel calculations on {len(mol_definitions)} molecules using {num_processors} processes...")

    #print("asdf atoms_to_be_optimized_string:", atoms_to_be_optimized_string)

    # As we are passing two parameters in "pyscf_calculator" (i.e. pyscf_calculator(atoms_to_be_optimized_string, maxsteps)), we need to manage the data to be in the format:
    # [('C  -0.342296  -0.664351  0.032460; O  0.845714  -1.415796  0.018408;...), 1)]
    atoms_to_be_optimized_string_list_of_tuples = []

    for i in range(len(atoms_to_be_optimized_string)):
        #atoms_to_be_optimized_string_list_of_tuples.append( (atoms_to_be_optimized_string[i], maxsteps) )
        atoms_to_be_optimized_string_list_of_tuples.append( (atoms_to_be_optimized_string[i], calculator_controller) )

    #print("asdf atoms_to_be_optimized_string_of_tuples:", atoms_to_be_optimized_string_list_of_tuples)

    partial_pyscf_calculator = partial(pyscf_calculator)#, maxsteps=maxsteps)

    #if __name__ == '__main__':
    with Pool(num_processors) as p:
            # map applies run_pyscf_calc to every item in mol_definitions
            #results = p.map(run_pyscf_calc, mol_definitions)
         #results = p.map(pyscf_calculator, atoms_to_be_optimized_string)
         #results = p.starmap(partial_pyscf_calculator, atoms_to_be_optimized_string)

         if calculator_controller["properties"] == False:
            results = p.starmap(partial_pyscf_calculator, atoms_to_be_optimized_string_list_of_tuples)
            return results

         elif calculator_controller["properties"] == True:
            list_of_results = []
            list_of_results = p.starmap(partial_pyscf_calculator, atoms_to_be_optimized_string_list_of_tuples)
            return list_of_results
