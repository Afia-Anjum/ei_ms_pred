import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import pickle
from random import sample

raw_Aldehydes = pd.read_csv('smiles_Aldehydes.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
raw_Aldehydes['molecule']=raw_Aldehydes['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
from rdkit import Chem
raw_Aldehydes['molecule']=raw_Aldehydes['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
raw_Aldehydes['Morgan_fp']=raw_Aldehydes['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=250,useFeatures=True,useChirality=True))
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import MolsToGridImage
from rdkit.Chem import Descriptors
raw_Aldehydes['Morgan_fp']=raw_Aldehydes['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=250,useFeatures=True,useChirality=True))
import rdkit
raw_Aldehydes['Morgan_fp']=raw_Aldehydes['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=250,useFeatures=True,useChirality=True))
from rdkit import Chem.rdMolDescriptors
from rdkit.Chem import rdMolDescriptors
raw_Aldehydes['Morgan_fp']=raw_Aldehydes['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=250,useFeatures=True,useChirality=True))
dataframs_list=[]
round_MW_list=raw_Aldehydes['round_MW'].values.tolist()
round_MW=[]
for i in range(len(round_MW_list)):
    round_MW.append(round(float(round_MW_list[i])))
raw_final_high_mbyz_spec['round_MW_final']=round_MW


raw_final_high_mbyz_spec['round_MW']=final_MW
raw_final_high_mbyz_spec['parent_ion_intensity']=M_intensity
raw_final_high_mbyz_spec['molecule']=raw_final_high_mbyz_spec['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
raw_final_high_mbyz_spec['Morgan_fp']=raw_final_high_mbyz_spec['molecule'].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x,radius=4,nBits=250,useFeatures=True,useChirality=True))
dataframs_list=[]
raw_final_high_mbyz_spec.shape[0]
for i in range(raw_final_high_mbyz_spec.shape[0]):
    array=np.array(raw_final_high_mbyz_spec['Morgan_fp'][i])
    datafram_i=pd.DataFrame(array)
    datafram_i=datafram_i.T
    dataframs_list.append(datafram_i)
concatenated_fp_only=pd.concat(dataframs_list, ignore_index=True)
raw_final_high_mbyz_spec_concat_final=concatenated_fp_only.join(raw_final_high_mbyz_spec,how="outer")
raw_final_high_mbyz_spec_concat_final=raw_final_high_mbyz_spec_concat_final.drop(['molecule','Morgan_fp'],axis=1)
raw_final_high_mbyz_spec_concat_final.to_csv("raw_final_high_mbyz_spec_morganFP_final.csv")


## for ALdehydes 
raw_Aldehydes_concat_final=concatenated_fp_only.join(raw_Aldehydes,how="outer")
raw_Aldehydes_concat_final=raw_Aldehydes_concat_final.drop(['molecule','Morgan_fp','round_MW'],axis=1)
raw_Aldehydes_concat_final=raw_Aldehydes_concat_final.drop(['Unnamed: 0',"SMILES"], axis=1)
raw_Aldehydes_concat_final.to_csv("raw_Aldehydes_concat_final_morganFP.csv")
raw_Aldehydes_concat_final_morganFP = pd.read_csv('raw_Aldehydes_concat_final_morganFP.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
raw_Aldehydes_concat_final_morganFP=raw_Aldehydes_concat_final_morganFP.drop(['Unnamed: 0',"SMILES"], axis=1)

from sklearn.preprocessing import MinMaxScaler
scaler=StandardScaler()
minmax=MinMaxScaler()
pipe_ss_mm=make_pipeline(StandardScaler(),MinMaxScaler())
raw_Aldehydes_concat_final_morganFP_transformed=pipe_ss_mm.fit_transform(raw_Aldehydes_concat_final_morganFP.iloc[:,:raw_Aldehydes_concat_final_morganFP.shape[1]])
raw_Aldehydes_concat_final_morganFP_cleaned=pd.DataFrame(raw_Aldehydes_concat_final_morganFP_transformed)
SVR_model = pickle.load(open("finalized_model.sav", 'rb'))
Aldehydes_parent_ion_intensity_pred=SVR_model.predict(raw_Aldehydes_concat_final_morganFP_cleaned)
X_test=np.transpose(raw_Aldehydes_concat_final_morganFP_cleaned)
y_test_pred=Aldehydes_parent_ion_intensity_pred.reshape(1,36)
c = np.concatenate((X_test, y_test_pred))
c=np.transpose(c)
print(pipe_ss_mm.inverse_transform(c))