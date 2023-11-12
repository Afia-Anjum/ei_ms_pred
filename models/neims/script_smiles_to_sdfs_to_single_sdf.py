import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PandasTools
import os

pp = pd.read_csv('Aldehyde.txt', names=['Smiles']) 
PandasTools.AddMoleculeColumnToFrame(pp,'Smiles','Molecule') # pp = doesn't work for me
#print(pp)
os.chdir('Aldehyde_sdfs')
PandasTools.WriteSDF(pp, 'pp_out_Aldehyde.sdf', molColName='Molecule', properties=None)


import os
fn = open("pp_out_Aldehyde.sdf","r")
fnew = fn.read()
fs = fnew.split('$$$$\n')
fs.pop()
for i in range(len(fs)):
    f_sdf=open("Aldehyde"+str(i+1)+".sdf",'w')
    f_sdf.write(fs[i])
    f_sdf.close()