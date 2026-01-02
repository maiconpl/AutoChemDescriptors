'''
Created on December 07, 2025

@author: maicon & clayton
Last modification by MPL: 13/12/2025 to implement the view and debug output.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

from multiprocessing import Pool
from typing import Iterable, List

from .rdkit_calculator import rdkit_calculator

def _run_serial(molecules_coded_list: Iterable[str]) -> List[list]:
    print("Falling back to serial RDKit descriptor computation.")
    return [rdkit_calculator(smiles) for smiles in molecules_coded_list]


def get_rdkit_calculations(molecules_coded_list, n_jobs):

    print("into  get_rdkit_calculations:", molecules_coded_list)

    try:
        num_processors = int(n_jobs)
    except (TypeError, ValueError):
        num_processors = 1

    if num_processors <= 1:
        return _run_serial(molecules_coded_list)

    try:
        with Pool(num_processors) as p:
            results = p.map(rdkit_calculator, molecules_coded_list)
        return results
    except (OSError, PermissionError) as exc:
        print("Warning: multiprocessing unavailable (", exc, ") — running RDKit descriptors serially.")
        return _run_serial(molecules_coded_list)
