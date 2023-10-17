import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import sys

import os.path
from math import ceil
from pickle import TRUE
from typing import List, Dict, Optional
from matplotlib import image


Si = pd.read_csv('train01.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

Si_smiles=Si['smiles'].values.tolist()
MW=[]

tbdms=0
tms=0

tbdms_arr=[]
tms_arr=[]
count_Si=[]
br=0
c=0
cl=0
fl=0
io=0
ni=0
oxy=0
sul=0
sil=0
Num_atoms=0
for i in range(len(Si_smiles)):
    if '[Si](C)(C)C(C)(C)C' in Si_smiles[i] or 'CC(C)(C)[Si](C)(C)' in Si_smiles[i]:
        tbdms+=1
    elif '[Si](C)(C)C' in Si_smiles[i] or 'C[Si](C)(C)' in Si_smiles[i]:
        tms+=1
    else:
        print(Si_smiles[i])
    count_Si.append(Si_smiles[i].count("Si"))

    mol = Chem.MolFromSmiles(Si_smiles[i])
    Num_atoms+=len(mol.GetAtoms())
    # if mol is None:
    #     MW.append("could not calculate")
    # else:
    #     MW.append(Descriptors.ExactMolWt(Chem.MolFromSmiles(Si_smiles[i])))

    if 'Br' in Si_smiles[i]:
        br+=Si_smiles[i].count("Br")
        #br+=1
    if 'Si' in Si_smiles[i]:
        sil+=Si_smiles[i].count("Si")
    if 'F' in Si_smiles[i]:
        fl+=Si_smiles[i].count("F")
    if 'I' in Si_smiles[i]:
        io+=Si_smiles[i].count("I")
    if 'N' in Si_smiles[i]:
        ni+=Si_smiles[i].count("N")
    if 'O' in Si_smiles[i]:
        oxy+=Si_smiles[i].count("O")
    if 'Cl' in Si_smiles[i]:
        cl+=Si_smiles[i].count("Cl")
    if not 'Si' in Si_smiles[i] and 'S' in Si_smiles[i]:
        sul+=Si_smiles[i].count("S")
    if not 'Cl' in Si_smiles[i] and 'C' in Si_smiles[i]:
        c+=Si_smiles[i].count("C")

print(Num_atoms)
print(br)
print(tms)
print(tbdms)
print(len(Si_smiles))
n=tms/len(Si_smiles)
print(n)
n=tbdms/len(Si_smiles)
print(n)

n=(br/Num_atoms)*100
print(n)
n=(sil/Num_atoms)*100
print(n)
n=(fl/Num_atoms)*100
print(n)
n=(io/Num_atoms)*100
print(n)
n=(ni/Num_atoms)*100
print(n)
n=(oxy/Num_atoms)*100
print(n)
n=(sul/Num_atoms)*100
print(n)
n=(cl/Num_atoms)*100
print(n)
n=(c/Num_atoms)*100
print(n)
#C[Si](C)C
# v=pd.DataFrame(MW)
# v.to_csv("MW.csv",index=False)

v=pd.DataFrame(count_Si)
v.to_csv("count_Si.csv",index=False)