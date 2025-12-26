'''
Created on December 07, 2025

@author: maicon & clayton
Last modification by MPL: 26/12/2025 to implement the DFT flags as well.
Last modification by MPL: 09/12/2025 to implement another argument in pyscf_calculations using multiprocessing.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

# 07/05/25

# Call PySCF calculator
# pip install --prefer-binary pyscf
# pip install -U pyberny

from pyscf import gto, scf, dft

# pyberny
from pyscf.geomopt.berny_solver import optimize

def pyscf_calculator(atoms_to_be_optimized_string, calculator_controller):

    maxsteps = calculator_controller['maxsteps']
    basis = calculator_controller['basis']
    method = calculator_controller['method']

    print("\natoms_to_be_optimized_string in 'pyscf_calculator':", atoms_to_be_optimized_string)

    mol = gto.M(atom=atoms_to_be_optimized_string, basis=basis)

    if method == "RHF":
       print("\nThe quantum chemistry method is '" + method + "'.")
       mf = scf.RHF(mol)

    elif method == "DFT":
       print("\nThe quantum chemistry method is '" + method + "'.")

       xc = calculator_controller['xc']

       mf = dft.RKS(mol)
       mf.xc = xc
       mf = mf.newton() # second-order algortihm
       mf.kernel()

    mol_eq = optimize(mf, maxsteps=maxsteps)
    print("atoms_optimized_string in 'pyscf_calculator':", mol_eq.tostring())

    xyz_new = mol_eq.tostring().split()

    return xyz_new # as string
