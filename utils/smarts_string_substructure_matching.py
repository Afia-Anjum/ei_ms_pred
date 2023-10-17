import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Descriptors


m = pd.read_csv("raw_further_added.csv")
m = m.iloc[-50000:]
print(m)

smiles_list=m['SMILES'].values.tolist()
names_list=m['name'].values.tolist()
formula_list=m['formula'].values.tolist()
peaks_list=m['peaks'].values.tolist()
num_peaks_list=m['num_peaks'].values.tolist()
inchikey_list=m['inchikey'].values.tolist()


Alkane_pat=Chem.MolFromSmarts('[CX4]')
Alkene_pat1=Chem.MolFromSmarts('[$([CX2](=C)=C)]')
Alkene_pat2=Chem.MolFromSmarts('[$([CX3]=[CX3])]')
Alkyne_pat=Chem.MolFromSmarts('[$([CX2]#C)]')
Aldehyde_pat=Chem.MolFromSmarts('[CX3H1](=O)[#6]')
Ketone_pat=Chem.MolFromSmarts('[#6][CX3](=O)[#6]')
Carbonyl_nitrogen_pat=Chem.MolFromSmarts('[OX1]=CN')
Carbonyl_carbon_pat=Chem.MolFromSmarts('[CX3](=[OX1])C')
Ester_pat=Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')
Azole_pat=Chem.MolFromSmarts('[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]')



#m_Si=m[m['SMILES'].str.contains("Si")]
#m_Si.to_csv("Si.csv", encoding='utf-8', index=False)



MW=[]
for i in range(len(smiles_list)):
    mol = Chem.MolFromSmiles(smiles_list[i])
    #if mol is None:
    #    print(smiles_list[i])
    #else:
    MW.append(Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles_list[i])))



Alkane=[]
for i in range(len(smiles_list)):
    mol = Chem.MolFromSmiles(smiles_list[i])
    patt=Alkene_pat1
    hit_ats = list(mol.GetSubstructMatch(patt))
    print(hit_ats)

print(Alkane)
print(names_list[20])

#substructures
query = Carbonyl_carbon_pat
#query = Chem.MolFromSmiles(parent_comp_smile)
mol_list=[]
for i in range(len(smiles_list)):
    mol_list.append(Chem.MolFromSmiles(smiles_list[i]))
#mol_list.remove(None)
#mol_list.remove(None)
#match_list = [mol.GetSubstructMatch(query) for mol in mol_list]
match_list2 = [mol.HasSubstructMatch(query) for mol in mol_list]
p=pd.DataFrame(match_list2)
p.to_csv("Car_C.csv", encoding='utf-8', index=False)
m.to_csv("Car_C_2.csv", encoding='utf-8', index=False)
#mol.HasSubstructMatch(query)
#print(match_list2)
#print(MolsToGridImage(mols=mol_list, molsPerRow=4, highlightAtomLists=match_list,maxMols=100))
