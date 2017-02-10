# CS181 P1
# read train data set; first N_train lines are for training. 
# Next N_test lines are for testing.

# try PLS regression
import os
os.environ['RDBASE']='C:\\ProgramData\\Anaconda2\\envs\\my-rdkit-env\\Lib\\site-packages\\rdkit'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import numpy as np
import matplotlib.pyplot as plt

import rdkit as rd
from rdkit import Chem
#from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

from sklearn import linear_model, decomposition
from sklearn.cross_decomposition import PLSRegression

import time

#make a smaller train and test dataset for debugging
N_train = 20000
N_test = 4000
file = 'train.csv'
ind_cols = 0 #extra columns in updated dataset


lbls = ['n_hatoms', 'n_atoms', 'mol_wt',\
    'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',\
    'NumRadicalElectrons']
fns = [Chem.rdchem.Mol.GetNumHeavyAtoms, Chem.rdchem.Mol.GetNumAtoms, Chem.Descriptors.MolWt,\
    Chem.Descriptors.HeavyAtomMolWt, Chem.Descriptors.ExactMolWt, Chem.Descriptors.NumValenceElectrons,\
    Chem.Descriptors.NumRadicalElectrons]

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


#add some features
def add_features(dataframe, labels, functions):
   if labels != ['none']:
   #if True:
        N = dataframe.shape[0]
        N_start = int(dataframe.index[0])
        smiles = dataframe[['smiles']]
        smiles = smiles[:].values
        for label, function in zip(labels, functions):
            M = dataframe.shape[1]
            dataframe.insert(M, label, np.zeros([N,1]))
            print(label)
            for n in range(0, N):
                #print(n)
                mol= Chem.MolFromSmiles(smiles[n,0])
                dataframe.set_value(N_start+n, label, function(mol))
    
add_features(X_test, lbls, fns)
#X_test.head()
add_features(X_train, lbls, fns)



X_train = X_train.drop(['smiles', 'gap'], axis=1)
X_train = X_train.iloc[:,ind_cols:]
X_train_val = X_train.values

X_test = X_test.drop(['smiles', 'gap'], axis=1)
X_test = X_test.iloc[:,ind_cols:]
X_test_val = X_test.values

#---------------------------------------------------------------
n_vals = range(2,40,2)

print('Running PLS')
RMSE = np.empty(0)

for i in n_vals:
    print(i)
    pls = PLSRegression(n_components=i, scale=False)
    pls.fit(X_train_val,Y_train_val)
    Y_pred = pls.predict(X_test_val)
    #score = cross_validation.cross_val_score(pls, X_reduced, y, cv=kf_10, scoring='mean_squared_error').mean()
    rmse = np.sqrt(np.sum(np.dot((Y_pred.T-Y_test_val), (Y_pred.T-Y_test_val).T))/N_test) 
    RMSE = np.append(RMSE, rmse)

plt.plot(n_vals, RMSE, linestyle='--', marker='o', color='b')
plt.title(file+', N_train = '+ str(N_train)+', N_test = ' + str(N_test))
plt.xlabel('Number of principal components in PLS regression')
plt.ylabel('RMSE')
plt.xlim((n_vals[0]-1, n_vals[-1]+1))
#plt.ylim((min(RMSE), max(RMSE)))
#plt.show()

t1 = time.time()
print('Time = '+str((t1-t0)))
plt.savefig('PLS_plots/'+file+'_rdkit_N_train_'+ str(N_train)+'_N_test_' + str(N_test)+'.pdf')
