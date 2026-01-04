'''
Created on December 07, 2025

@author: maicon & clayton
Last modification by MPL: 28/12/2025 to implement the properties from the optimized geometry: total energy, HOMO, LUMO, band-gap, electronegativiy, hardness and dipole moment.
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
    properties = calculator_controller['properties']

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

    if properties == False:

       mol_eq = optimize(mf, maxsteps=maxsteps)
       print("atoms_optimized_string in 'pyscf_calculator':", mol_eq.tostring())

       xyz_new = mol_eq.tostring().split()

       return xyz_new # as string

    elif properties == True:

       list_of_results = []
       qc_descriptors = [] # quantum chemistry descriptors

       mol_eq = optimize(mf, maxsteps=maxsteps)
       print("atoms_optimized_string in 'pyscf_calculator':", mol_eq.tostring())

       xyz_new = mol_eq.tostring().split()

       atoms_optimized_string = "" # from pySCF

       for j in range(0, len(xyz_new), 4):
           atoms_optimized_string = atoms_optimized_string + "  ".join(xyz_new[j: j + 4]) + "; "

       print("atoms_optimized_string:", atoms_optimized_string)

       if method == "RHF":

          # 3. New SCF after local optimization
          print("\nThe quantum chemistry method is '" + method + "'. Properties are obtained from optimized geometry.")
          mol_opt = gto.M(atom=atoms_optimized_string, basis=basis)
          mf_opt = scf.RHF(mol_opt).run()

       elif method == "DFT":
          print("\nThe quantum chemistry method is '" + method + "'. Properties are obtained from optimized geometry.")

          mol_opt = gto.M(atom=atoms_optimized_string, basis=basis)
          xc = calculator_controller['xc']

          mf_opt = dft.RKS(mol_opt)
          mf_opt.xc = xc
          mf_opt = mf.newton() # second-order algortihm
          mf_opt.kernel()

       # Extract HOMO and LUMO energies
       # mf.mo_energy is an array of molecular orbital energies (eigenvalues)

       total_energy = mf_opt.e_tot

       qc_descriptors.append(total_energy)

       mo_energies = mf_opt.mo_energy
       homo_idx = mol_opt.nelec[0] - 1  # Index of the highest occupied molecular orbital
       lumo_idx = mol_opt.nelec[0]      # Index of the lowest unoccupied molecular orbital

       homo_energy = mo_energies[homo_idx]
       qc_descriptors.append(homo_energy)

       lumo_energy = mo_energies[lumo_idx]
       qc_descriptors.append(lumo_energy)

       # Calculate Electronegativity (in the same unit as the energies, typically Hartree)

       electronegativity = -(homo_energy + lumo_energy) / 2.0
       qc_descriptors.append(electronegativity)

       chemical_hardness = (lumo_energy - homo_energy) / 2.0
       qc_descriptors.append(chemical_hardness)

       bg = lumo_energy - homo_energy # bg = band-gap
       qc_descriptors.append(bg)

       dipole_moment = mf_opt.dip_moment()
       for iValue in dipole_moment:
           qc_descriptors.append(iValue)

       print(f"\nTotal energy (Hartree): {total_energy}")
       print(f"HOMO Energy (Hartree): {homo_energy}")
       print(f"LUMO Energy (Hartree): {lumo_energy}")
       print(f"BG (Hartree): {bg}")
       print(f"Mulliken Electronegativity (Hartree): {electronegativity}")
       print(f"Chemical Hardness (Ha): {chemical_hardness}")
       print(f"Dipole moment (Debye): {dipole_moment}")
       print(f"qc_descriptors: {qc_descriptors}")

       list_of_results = [xyz_new, qc_descriptors]
       return list_of_results
