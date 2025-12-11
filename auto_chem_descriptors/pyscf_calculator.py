'''
Created on December 07, 2025

@author: maicon
Last modification by MPL: 09/12/2025 to implement another argument in pyscf_calculations using multiprocessing.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

# 07/05/25

# Call PySCF calculator
# pip install --prefer-binary pyscf
# pip install -U pyberny

from pyscf import gto, scf

# pyberny
from pyscf.geomopt.berny_solver import optimize

#def pyscf_calculator(atoms_to_be_optimized_string):#, maxsteps):
#def pyscf_calculator(atoms_to_be_optimized_string, maxsteps):
def pyscf_calculator(atoms_to_be_optimized_string, calculator_controller):

    maxsteps = calculator_controller['maxsteps']
    basis = calculator_controller['basis']

    print("atoms_to_be_optimized_string in calc:", atoms_to_be_optimized_string)

    #mol = gto.M(atom=atoms_to_be_optimized_string, basis='sto-3g')
    mol = gto.M(atom=atoms_to_be_optimized_string, basis=basis)
    mf = scf.RHF(mol)

    mol_eq = optimize(mf, maxsteps=maxsteps)
    #mol_eq = optimize(mf, maxsteps=10)
    print("ooo kkk:", mol_eq.tostring())

    xyz_new = mol_eq.tostring().split()

    return xyz_new # as string
