import os
import os.path
import pandas as pd
import pickle

os.chdir('D:\\NIST20\\MSSEARCH\\pythonProject\\test_metrics_code\\')

parent_ion_prediction = pd.read_csv('y_parent_ion_pred_nist23.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
parent_ion_prediction_MW=parent_ion_prediction['MW'].values.tolist()
round_MW=[]
for i in range(len(parent_ion_prediction_MW)):
    round_MW.append(round(float(parent_ion_prediction_MW[i])))
with open("spec_ref_low_nist23", "rb") as fp:
    a = pickle.load(fp)
#a.pop(35)
actual_mol_ion_intensity=[]
for i in range(len(a)):
    #############COMMENT IT OUT FOR YOUR OWN MODELS###################
    mol_ref = a[i]
    for j in range(len(mol_ref)):
        mol_ref[j].pop(0)
    spec_ref_list = [item for sublist in mol_ref for item in sublist]
    actual_mol_ion_intensity.append(spec_ref_list[round_MW[i]])
    print(spec_ref_list[round_MW[i]])

pd.DataFrame(actual_mol_ion_intensity).to_csv("nist23_ref_Mplus.csv",index=False)