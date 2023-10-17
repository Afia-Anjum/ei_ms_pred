from rdkit import Chem
rdkit.Chem.rdmolops.DeleteSubstructs('CCOC','OC')
m = Chem.MolFromSmiles('CCOC')
m1 = Chem.MolFromSmiles('OC')
rdkit.Chem.rdmolops.DeleteSubstructs(m,m1)
Chem.MolToSmiles(rdkit.Chem.rdmolops.DeleteSubstructs(m,m1))
