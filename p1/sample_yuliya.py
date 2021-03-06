# CS181 P1
#

# read train data set; first N_train lines are for training. 
# Next N_test lines are for testing.
# Add features from Chem
# compute RMSE

import os
os.environ['RDBASE']='C:\\ProgramData\\Anaconda2\\envs\\my-rdkit-env\\Lib\\site-packages\\rdkit'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import rdkit as rd
from rdkit import Chem
#from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

#make a smaller train and test dataset for debugging
N_train = 10000
N_test = 2000
file = 'train_small.csv'


# make lists of functions and labels to make features
#lbls = ['none']
#fns  = ['none']

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
#print('csv load completed')

df_train_cropped = df_train[:(N_train+N_test)]
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
X_train_val = X_train.values

X_test = X_test.drop(['smiles', 'gap'], axis=1)
X_test_val = X_test.values

print('running Linear regression...')
LR = LinearRegression()
LR.fit(X_train_val, Y_train_val)
LR_pred = LR.predict(X_test)

print('running Random forest regression...')
RF = RandomForestRegressor()
RF.fit(X_train_val, Y_train_val)
RF_pred = RF.predict(X_test)

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

write_to_file("sample1.csv", LR_pred)
write_to_file("sample2.csv", RF_pred)

#Compute RMSE
print(fns)
print(lbls)

RMSE_LR = np.sqrt(np.sum(np.dot((LR_pred-Y_test_val), (LR_pred-Y_test_val)))/N_test)
print(('LR RMSE = ' +str(RMSE_LR)))

RMSE_RF = np.sqrt(np.sum(np.dot((RF_pred-Y_test_val), (RF_pred-Y_test_val)))/N_test)
print(('RF RMSE = '  +str(RMSE_RF)))