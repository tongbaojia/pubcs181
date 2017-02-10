# CS181 P1
#
# read train data set; first N_train lines are for training. 
# Next N_test lines are for testing.

# try MLP Regressor

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPRegressor
import numpy as np 
from sklearn.model_selection import train_test_split

import time

N_layers = 3;
N_nodes = 40;
MAX_iter = 1000;
Alpha = 0.000001 #regularization loss

#make a smaller train and test dataset for debugging
N_train = 10000
N_test =2000
file = 'train_updated_sq_small.csv'
ind_cols = 2 #extra columns in updated dataset

df_train = pd.read_csv(file)
#df_test = pd.read_csv("test.csv")
print('csv load completed')
t0 = time.time()
print(df_train.head())

df_train_cropped = df_train[:(N_train+N_test)]
Y_train_val = df_train_cropped[:N_train].gap.values
Y_test_val = df_train_cropped[(N_train):(N_train+N_test)].gap.values

df_train_cropped = df_train_cropped.drop(['smiles', 'gap'], axis=1)
df_train_cropped = normalize(df_train_cropped)
X_test = df_train_cropped[(N_train):(N_train+N_test)]
X_train = df_train_cropped[:N_train]
X_train = X_train[:,ind_cols:]
X_train_val = X_train

X_test = X_test[:,ind_cols:]
X_test_val = X_test

#---------------------------------------------------------------
# Predict and validate

layer_sizes = np.ones(N_layers, dtype = np.int)*N_nodes;

#solver= 'sgd', learning_rate = 'adaptive',activation = 'tanh',
mlp = MLPRegressor(hidden_layer_sizes=layer_sizes,  max_iter = MAX_iter, alpha = Alpha, early_stopping = False,  verbose = True, tol = 0.00000001, batch_size = 10, momentum = 0.7)
mlp.fit(X_train_val, Y_train_val)
Y_pred = mlp.predict(X_test_val)

RMSE = np.sqrt(np.sum(np.dot((Y_pred.T-Y_test_val), (Y_pred.T-Y_test_val).T))/N_test) 

t1 = time.time()
print(str(N_layers)+' layers, '+str(N_nodes)+ ' nodes, '+str(MAX_iter)+' iterations, '+str(Alpha)+' reg. loss')
print(str(N_train)+ ' training lines, '+str(N_test)+' testing lines')
print('Time = '+str((t1-t0)))
print('RMSE = '+str(RMSE))