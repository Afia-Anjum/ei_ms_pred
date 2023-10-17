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

final_high_mbyz_spec = pd.read_csv('raw_final_mbyz_high_spec_set.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')
final_high_peaks=final_high_mbyz_spec['peaks'].values.tolist()
final_high_num_peaks=final_high_mbyz_spec['num_peaks'].values.tolist()
flag_or_not=[]

for i in range(len(final_high_peaks)):
    tmp = final_high_peaks[i].split(";")
    for j in range(len(tmp)):
        tmp[j]=tmp[j].strip()
    tmp.pop()
    flag=0
    for j in range(len(tmp)):
        tmp2 = tmp[j].split()
        if int(tmp2[0]) > 1000:
            flag=1
            break
    if flag==0:
        flag_or_not.append(1)
    else:
        flag_or_not.append(0)

bflag=pd.DataFrame(flag_or_not)
bflag.to_csv("flag.csv",index=False)


high_mz_synthetic_spectra_list=[]
for i in range(len(final_high_num_peaks)):
    tmp = final_high_peaks[i].split(";")
    for j in range(len(tmp)):
        tmp[j]=tmp[j].strip()
    tmp.pop()
    j=0
    while(len(tmp)):
        tmp2 = tmp[j].split()
        if int(tmp2[0])>130:
            break
        else:
            tmp=tmp[1:len(tmp)]
    tmp3 = "; ".join(tmp)
    tmp3=tmp3+";"
    high_mz_synthetic_spectra_list.append(tmp3)
frame=pd.DataFrame(high_mz_synthetic_spectra_list)
frame.to_csv("cropped_peaks.csv",index=False)