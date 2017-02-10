# CS181 P1
#
# read train data set; first N_train lines are for training. 
# Next N_test lines are for testing.

# try PLS regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition
from sklearn.cross_decomposition import PLSRegression

import time

#make a smaller train and test dataset for debugging
N_train = 10000
N_test = 2000
file = 'train_updated_sq_small.csv'
ind_cols = 2 #extra columns in updated dataset

"""
Read in train and test as Pandas DataFrames
"""

df_train = pd.read_csv(file)
#df_test = pd.read_csv("test.csv")
print('csv load completed')
t0 = time.time()
df_train_cropped = df_train[:(N_train+N_test)]

#print(df_train_cropped.head())
Y_train_val = df_train_cropped[:N_train].gap.values
X_train = df_train_cropped[:N_train]
#print(X_train.head())

Y_test_val = df_train_cropped[(N_train):(N_train+N_test)].gap.values
X_test = df_train_cropped[(N_train):(N_train+N_test)]

X_train = X_train.drop(['smiles', 'gap'], axis=1)
X_train = X_train.iloc[:,ind_cols:]
X_train_val = X_train.values

X_test = X_test.drop(['smiles', 'gap'], axis=1)
X_test = X_test.iloc[:,ind_cols:]
print(X_test.head())
X_test_val = X_test.values

#---------------------------------------------------------------
#n_vals = range(2,40,2)

print('Running PLS')

#for i in n_vals:
    #print(i)
i = 30
pls = PLSRegression(n_components=i, scale=False)
pls.fit(X_train_val,Y_train_val)
Y_pred = pls.predict(X_test_val)
#score = cross_validation.cross_val_score(pls, X_reduced, y, cv=kf_10, scoring='mean_squared_error').mean()
RMSE = np.sqrt(np.sum(np.dot((Y_pred.T-Y_test_val), (Y_pred.T-Y_test_val).T))/N_test) 

#plt.plot(n_vals, RMSE, linestyle='--', marker='o', color='b')
#plt.title(file+', N_train = '+ str(N_train)+', N_test = ' + str(N_test))
#plt.xlabel('Number of principal components in PLS regression')
#plt.ylabel('RMSE')
#plt.xlim((n_vals[0]-1, n_vals[-1]+1))
#plt.ylim((min(RMSE), max(RMSE)))
#plt.show()

t1 = time.time()
print('Time = '+str((t1-t0)))
print(str(N_train)+ ' training lines, '+str(N_test)+' testing lines')
print('RMSE = '+str(RMSE))
#plt.savefig('PLS_plots/'+file+'_N_train_'+ str(N_train)+'_N_test_' + str(N_test)+'.pdf')
