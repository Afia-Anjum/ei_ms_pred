import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import joblib
import pickle
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.utils import shuffle

from random import sample

raw_final_spec_morganFP = pd.read_csv('raw_final_morganFP_final.csv',sep=',', error_bad_lines=False, index_col=False, dtype='unicode')

raw_final_spec_morganFP['MW'].to_csv("MW.csv",index=False)
raw_final_spec_morganFP['parent_ion_intensity'].to_csv("intensity.csv",index=False)

exit()
raw_final_spec_morganFP=raw_final_spec_morganFP.drop(['Unnamed: 0',"SMILES","peaks","MW"], axis=1)

raw_final_spec_morganFP = raw_final_spec_morganFP.sample(frac = 1)

raw_final_spec_morganFP.to_csv("raw_final_spec_morganFP_shuffled.csv",index=False)

from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler(feature_range=(0, 1))
raw_final_spec_morganFP_transformed=minmax.fit_transform(raw_final_spec_morganFP.iloc[:25000,:raw_final_spec_morganFP.shape[1]-1])
raw_final_spec_morganFP_cleaned=pd.DataFrame(raw_final_spec_morganFP_transformed)
raw_final_spec_morganFP_cleaned_y=raw_final_spec_morganFP.iloc[:25000,-1]
raw_final_spec_morganFP_cleaned_y = raw_final_spec_morganFP_cleaned_y.astype(float)
raw_final_spec_morganFP_cleaned_X=raw_final_spec_morganFP_cleaned.iloc[:25000,:raw_final_spec_morganFP_cleaned.shape[1]]

gsc = GridSearchCV(
        estimator=SVR(kernel='rbf'),
        param_grid={
            'C': [0.1, 1, 100, 1000],
            'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10],
            'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

def splitting_data(X,y,test_size,random_state):
    return train_test_split(X,y,test_size=test_size, random_state=random_state, shuffle=True)

X_train_valid, X_test, y_train_valid, y_test= splitting_data(X=raw_final_spec_morganFP_cleaned_X, y=raw_final_spec_morganFP_cleaned_y, test_size=0.1, random_state=0)
X_train, X_valid,y_train, y_valid= splitting_data(X=X_train_valid, y=y_train_valid, test_size=0.2, random_state=0)

###Training portion
#grid_result = gsc.fit(X_train, y_train)
#best_params = grid_result.best_params_
#best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
#                   coef0=0.1, shrinking=True,
#                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

#best_svr = SVR(kernel='rbf', C=1000.0, epsilon=0.5, gamma=0.1,
#                   coef0=0.1, shrinking=True,
#                   tol=0.001, cache_size=200, verbose=False, max_iter=-1)

#When no longer training:
#best_svr.fit(X_train, y_train)

#filename="finalized_model20.sav"
#pickle.dump(best_svr, open(filename,'wb'))

best_svr = pickle.load(open("finalized_model20.sav", 'rb'))

y_test_pred=best_svr.predict(X_test)
pd.DataFrame(y_test_pred).to_csv("Y_pred.csv",index=False)
pd.DataFrame(y_test).to_csv("Y_actual.csv",index=False)
