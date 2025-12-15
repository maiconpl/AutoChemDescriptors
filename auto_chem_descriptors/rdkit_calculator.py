'''
Created on December 08, 2025

@author: maicon
Last modification by MPL: 08/12/2025.
'''

def rdkit_descriptor_function(mol):

    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import Descriptors

    Chem.AllChem.EmbedMolecule(mol)

    #print (Descriptors.FpDensityMorgan1(mol), Descriptors.FpDensityMorgan2(mol), Descriptors.FpDensityMorgan3(mol), Descriptors.MaxAbsPartialCharge(mol, force=False), Descriptors.MaxPartialCharge(mol, force=False), Descriptors.MinAbsPartialCharge(mol, force=False), Descriptors.MinPartialCharge(mol, force=False), Descriptors.ExactMolWt(mol), Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol), AllChem.ComputeMolVolume(mol), Descriptors.HeavyAtomMolWt(mol))

    descriptor = [Descriptors.FpDensityMorgan1(mol), Descriptors.FpDensityMorgan2(mol), Descriptors.FpDensityMorgan3(mol), Descriptors.MaxAbsPartialCharge(mol, force=False), Descriptors.MaxPartialCharge(mol, force=False), Descriptors.MinAbsPartialCharge(mol, force=False), Descriptors.MinPartialCharge(mol, force=False), Descriptors.ExactMolWt(mol), Descriptors.NumRadicalElectrons(mol), Descriptors.NumValenceElectrons(mol), AllChem.ComputeMolVolume(mol), Descriptors.HeavyAtomMolWt(mol)]

    #print("descriptor into descriptor_function:", descriptor)

    return descriptor

def rdkit_calculator(molecules_coded):

    from rdkit import Chem
    print("molecules_to_be_created in rdkit:", molecules_coded)

    descriptor = rdkit_descriptor_function( Chem.MolFromSmiles(molecules_coded) )

    return descriptor # float number in a list
