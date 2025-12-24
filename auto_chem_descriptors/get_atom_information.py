'''
Created on December 03, 2025

@author: maicon & clayton
Last modification by MPL: 24/12/2025 to handle the Dscribe descriptor size for all and different molecules.
Last modification by MPL: 07/12/2025 to implement the multiprocess to run PySCF in parallell. I run the Pampulha's lake running race. ; )
'''

def getAtomType(atomSymbols):

    atomsType = []
    atomCounterList = []
    count01 = 0

    # just amazing: python getting the different types of atoms in a list
    atomsType.append(atomSymbols[0])
    for i in range(len(atomSymbols)):
        if atomSymbols[i] not in atomsType:
           atomsType.append(atomSymbols[i])
           continue

    return atomsType

def getAtomTypeCounter(atomSymbols):

    #atomSymbols = ['B', 'B', 'N', 'C', 'N', 'N', 'B', 'B', 'C'] # for test only.
    atomsType = []
    atomCounterList = []
    count01 = 0 

    # just amazing: python getting the different types of atoms in a list
    atomsType.append(atomSymbols[0])
    for i in range(len(atomSymbols)):
        if atomSymbols[i] not in atomsType:
           atomsType.append(atomSymbols[i])
           continue

    for i in range(len(atomsType)):
        for j in range(len(atomSymbols)):

            if atomsType[i] == atomSymbols[j]:

               count01 = count01 + 1 

        atomCounterList.append(count01)
        count01 = 0 

    tmp01 = []

    for iAtomCounter in atomCounterList:
        if iAtomCounter == 1:
           tmp01.append("")

        else:

           tmp01.append(iAtomCounter)

    atom_type_counter_list = [str(i) + str(j) for i, j in zip(atomsType, tmp01)]

    return atom_type_counter_list
