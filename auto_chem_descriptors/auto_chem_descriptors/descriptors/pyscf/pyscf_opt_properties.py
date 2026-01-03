from pyscf import gto, scf
from pyscf.geomopt.berny_solver import optimize

# 1. Define the molecule and run an SCF calculation
mol = gto.M(atom='H 0.758602  0.000000  0.204284; H 0.358602  0.000000  -0.504284; O 0.0 0.0 0.0', basis='6-31g*')
mf = scf.RHF(mol).run()

maxsteps = 100
mol_eq = optimize(mf, maxsteps=maxsteps)

# 2. Get the final coordinates
xyz_new = mol_eq.tostring().split()
print("xyz_new:", xyz_new)

final_coords = mol_eq.atom_coords()
print("Final Coordinates (Angstroms):")
print(final_coords)

atoms_optimized_string = "" # from pySCF

for j in range(0, len(xyz_new), 4):
    atoms_optimized_string = atoms_optimized_string + "  ".join(xyz_new[j: j + 4]) + "; "

print("atoms_optimized_string:", atoms_optimized_string)

# 3. New SCF after local optimization
mol_opt = gto.M(atom=atoms_optimized_string, basis='6-31g*')
mf_opt = scf.RHF(mol_opt).run()

# 4. Extract HOMO and LUMO energies
# mf.mo_energy is an array of molecular orbital energies (eigenvalues)
# mol = mol_eq

total_energy = mf_opt.e_tot
mo_energies = mf_opt.mo_energy
homo_idx = mol_opt.nelec[0] - 1  # Index of the highest occupied molecular orbital
lumo_idx = mol_opt.nelec[0]      # Index of the lowest unoccupied molecular orbital

homo_energy = mo_energies[homo_idx]
lumo_energy = mo_energies[lumo_idx]

# 4. Calculate Electronegativity (in the same unit as the energies, typically Hartree)

electronegativity = -(homo_energy + lumo_energy) / 2.0

chemical_hardness = (lumo_energy - homo_energy) / 2.0

bg = lumo_energy - homo_energy

dipole_moment = mf_opt.dip_moment()

print(f"\nTotal energy (Hartree): {total_energy}")
print(f"HOMO Energy (Hartree): {homo_energy}")
print(f"LUMO Energy (Hartree): {lumo_energy}")
print(f"BG (Hartree): {bg}")
print(f"Mulliken Electronegativity (Hartree): {electronegativity}")
print(f"Chemical Hardness (Ha): {chemical_hardness}")
print(f"Dipole moment (Debye): {dipole_moment}")
