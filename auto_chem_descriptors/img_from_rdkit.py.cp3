# 03/12/25 

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import MolToFile
from rdkit.Chem import Draw
from rdkit.Chem import AllChem

from rdkit.Chem import rdMolDescriptors
from get_atom_information import getAtomTypeCounter, getAtomType
from utils import get_coordinates_ordered_by_atoms_group

#from dscribe.descriptors import CoulombMatrix

from get_describe_descriptor import get_describe_descriptor
#from dscribe.descriptors import SineMatrix
#from dscribe.descriptors import EwaldSumMatrix
#from dscribe.descriptors import MBTR
#from dscribe.descriptors import SOAP
#from dscribe.descriptors import ACSF

# BEGIN 01 (2,4-Dichlorophenoxyacetic acid) #
mol01 = Chem.MolFromSmiles("c1(OCC(O)(=O))c(Cl)cc(Cl)cc1")
#mol01 = Chem.MolFromSmiles("c1(OCC(O)(=O))ccc(Cl)cc1(Cl)")
#mol01 = Chem.MolFromSmiles("c1ccc(Cl)cc1(Cl)")
#mol01 = Chem.MolFromSmiles("c1ccccc1(OCC(O)(=O))") # OK
#mol01 = Chem.MolFromSmiles("c1ccccc1(OCC(O)C(=O))") # OK
#mol01 = Chem.MolFromSmiles("c1ccccc1(OCC(=O)O(c1c(C)ccc(C)c1))") # OK
#mol01 = Chem.MolFromSmiles("c1ccccc1(OCC(=O)O(c1c(C(C)C)ccc(C)c1))") 
#mol01 = Chem.MolFromSmiles("c1ccccc1(OCC(=O)O(c1c(C(C)C)ccc(C)c1))")

####

# 1-Create smiles code
#mol01 = Chem.MolFromSmiles("c1ccccc1(C=C(N(=O)=O))") # ok

#print(dir(mol01))
MolToFile(mol01, "mol01.png")

# 2-Create from smiles code the descriptors
mol=mol01

#print(mol01.view())

AllChem.EmbedMolecule(mol)
print (Descriptors.FpDensityMorgan1(mol), Descriptors.FpDensityMorgan2(mol), Descriptors.FpDensityMorgan3(mol), Descriptors.MaxAbsPartialCharge(mol, force=False), Descriptors.MaxPartialCharge(mol, force=False), Descriptors.MinAbsPartialCharge(mol, force=False), Descriptors.MinPartialCharge(mol, force=False), Descriptors.ExactMolWt(mol), Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol), AllChem.ComputeMolVolume(mol), Descriptors.HeavyAtomMolWt(mol))

# 3-Create from smiles code the descriptors get molecule coordinates

from rdkit import Chem
from rdkit.Chem import AllChem
# from ase.visualize import view
# from ase.visualize.plot import plot_atoms

# Load SMILES as mol obj
#mol = Chem.MolFromSmiles('[Zn--]([NH3+])([NH3+])([NH3+])N1C=C(Cl)N=C1')
mol = AllChem.AddHs(mol) # make sure to add explicit hydrogens

#
from rdkit.Chem import Draw
Draw.MolToImage(mol)

# Generate 3D coordinates
AllChem.EmbedMolecule(mol)
xyz = Chem.AllChem.MolToXYZBlock(mol) # initial coords

with open('unoptimized.xyz','w') as outf:
    outf.write(xyz)

# Perform UFF optimization
ff = AllChem.UFFGetMoleculeForceField(mol)
ff.Initialize()
ff.Minimize(energyTol=1e-7,maxIts=100000)

xyz = Chem.AllChem.MolToXYZBlock(mol)
print(xyz)
with open('B.xyz','w') as outf:
    outf.write(xyz)

xyz_new = xyz.split()[1:]
print("xyz_new:", xyz_new)

atoms_to_be_optimized_string = ""
for i in range(0, len(xyz_new), 4):
   print("kk", xyz_new[i])
   print("kk", xyz_new[i: i + 4])
   #if i < len(xyz_new)/4:
   atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[i: i + 4]) + "; "
   #if i == len(xyz_new)/4 - 1:
   #   atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[i: i + 4])
   #atoms_to_be_optimized_string.append(  )  = []
   print( )

print("zaza:", atoms_to_be_optimized_string)

# pip install --prefer-binary pyscf
# pip install -U pyberny
# 4- Local optimization https://pyscf.org/user/geomopt.html
from pyscf import gto, scf
#mol = gto.M(atom='N 0 0 0; N 0 0 1.2', basis='ccpvdz')
#mol = gto.M(atom=atoms_to_be_optimized_string, basis='ccpvdz')
mol = gto.M(atom=atoms_to_be_optimized_string, basis='sto-3g')
mf = scf.RHF(mol)

# geometric
#from pyscf.geomopt.geometric_solver import optimize
#mol_eq = optimize(mf, maxsteps=100)
#print(mol_eq.tostring())

# pyberny
from pyscf.geomopt.berny_solver import optimize
#mol_eq = optimize(mf, maxsteps=100)
mol_eq = optimize(mf, maxsteps=1)
print("ooo kkk:", mol_eq.tostring())

# 5- https://singroup.github.io/dscribe/latest/

from dscribe.descriptors import CoulombMatrix

# Setting up the CM descriptor

#methanol.set_cell([[10.0, 0.0, 0.0],
#    [0.0, 10.0, 0.0],
#    [0.0, 0.0, 10.0],
#    ])
#
#cm = CoulombMatrix(n_atoms_max=6)

#xyz_new = mol_eq.tostring()
#xyz_new = mol_eq.tostring().split()[1:]
xyz_new = mol_eq.tostring().split()
atoms_to_be_optimized_string = ""
atoms_symbols=[]
atoms_xyz = []
tmp_list01 = []
atoms_xyz_tmp01 = []
for i in range(0, len(xyz_new), 4):
   print("kk", xyz_new[i])
   #print("kk", xyz_new[i: i + 4])
   #if i < len(xyz_new)/4:
   atoms_to_be_optimized_string = atoms_to_be_optimized_string + "  ".join(xyz_new[i + 1: i + 4]) + "; "
   tmp_list01 =  xyz_new[i + 1: i + 4]
   atoms_symbols.append(xyz_new[i])
   for iCoordinates in tmp_list01:
       atoms_xyz_tmp01.append(float(iCoordinates))
   atoms_xyz.append(atoms_xyz_tmp01)
   atoms_xyz_tmp01 = []
   print( )

print("zaza final:", atoms_to_be_optimized_string)
print("zaza atoms_xyz:", atoms_xyz)
print("zaza atoms_symbols:", atoms_symbols)

#atomSymbols = ['B', 'B', 'N', 'C', 'N', 'N', 'B', 'B', 'C'] # for test only.

print (getAtomTypeCounter(atoms_symbols))
print (getAtomType(atoms_symbols))
atoms_type = getAtomType(atoms_symbols)

atoms_type_counter=getAtomTypeCounter(atoms_symbols)
atoms_symbols_ordered, atoms_xyz_ordered = get_coordinates_ordered_by_atoms_group(atoms_type_ordered=atoms_type, atoms_symbols=atoms_symbols, atoms_xyz=atoms_xyz)

print("zaza atoms_xyz_ordered:", atoms_xyz_ordered)
print("zaza atoms_symbols_ordered:", atoms_symbols_ordered)
print("zaza atoms_type_counter:", atoms_type_counter, "".join(atoms_type_counter))

descriptor = get_describe_descriptor(atoms_symbols_ordered, atoms_xyz_ordered, system_type="cluster", descriptor_type="mbtr")

#descriptor = get_describe_descriptor(atoms_symbols, atoms_xyz, system_type="cluster", descriptor_type="mbtr")

print("descriptor:", descriptor)
