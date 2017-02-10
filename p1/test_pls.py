# CS181 P1
# Yuliya Dovzhenko
# read train data set; first N_train lines are for training. 
# Next N_test lines are for testing.
# normalize the columns
# try PLS regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model, decomposition
from sklearn.cross_decomposition import PLSRegression

import time

#make a smaller train and test dataset for debugging
N_train = 40000
N_test = 8000
file = 'train_updated_sq_inv.csv'
ind_cols =1 #extra columns in updated dataset


df_train = pd.read_csv(file)
#df_test = pd.read_csv("test.csv")
print('csv load completed')
t0 = time.time()

df_train_cropped = df_train[:(N_train+N_test)]
Y_train_val = df_train_cropped[:N_train].gap.values
Y_test_val = df_train_cropped[(N_train):(N_train+N_test)].gap.values
df_train_cropped = df_train_cropped.drop(['smiles', 'gap'], axis=1).values

max_abs_scaler = MinMaxScaler()
X_train = df_train_cropped[:N_train]
X_train = X_train[:,ind_cols:]
X_train_val = max_abs_scaler.fit_transform(X_train)

X_test = df_train_cropped[(N_train):(N_train+N_test)]
X_test = X_test[:,ind_cols:]
X_test_val = max_abs_scaler.transform(X_test)


X_test = df_train_cropped[(N_train):(N_train+N_test)]

print(X_train_val[:5,:])
print(X_test_val[:5,:])
#---------------------------------------------------------------
# Predict
n_vals = range(5,60,5)
RMSE = np.array([])
print('Running PLS')

for i in n_vals:
    print(i)
    pls = PLSRegression(n_components=i, scale=False)
    pls.fit(X_train_val,Y_train_val)
    Y_pred = pls.predict(X_test_val)
    #score = cross_validation.cross_val_score(pls, X_reduced, y, cv=kf_10, scoring='mean_squared_error').mean()
    RMSE = np.append(RMSE,np.sqrt(np.sum(np.dot((Y_pred.T-Y_test_val), (Y_pred.T-Y_test_val).T))/N_test))

plt.plot(n_vals, RMSE, linestyle='--', marker='o', color='b')
plt.title(file+', N_train = '+ str(N_train)+', N_test = ' + str(N_test))
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('RMSE')
plt.xlim((n_vals[0]-1, n_vals[-1]+1))
plt.ylim((min(RMSE), max(RMSE)))
#plt.show()

t1 = time.time()
print('Time = '+str((t1-t0)))
print(str(N_train)+ ' training lines, '+str(N_test)+' testing lines')
print('RMSE = '+str(RMSE))
plt.savefig('PLS_plots/'+file+'_N_train_'+ str(N_train)+'_N_test_' + str(N_test)+'.pdf')
