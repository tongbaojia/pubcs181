# Tony Tong; baojia.tong@cern.ch
import os, argparse, sys, math, time
#for parallel processing!
import multiprocessing as mp
import rdkit as rd
#other necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from rdkit import Chem
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as MSE
from sklearn.neural_network import BernoulliRBM
#load all things~
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import make_pipeline

"""
Read in train and test as Pandas DataFrames
"""
print "Done with loading!"

#define functions
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir",  default="here")
    parser.add_argument("--output",    action="store_true")
    parser.add_argument("--detail",    default=True)
    return parser.parse_args()


def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")


def add_features(dataframe, labels=['n_atoms'], functions=[Chem.rdchem.Mol.GetNumHeavyAtoms]):
    N = dataframe.shape[0]
    N_start = int(dataframe.index[0])
    smiles = dataframe[['smiles']]
    smiles = smiles[:].values
    for label, function in zip(labels, functions):
        M = dataframe.shape[1]
        dataframe.insert(M, label, np.zeros([N,1]))
    #extras
    dataframe.insert(M, "Bonds_type_mean", np.zeros([N,1]))
    dataframe.insert(M, "Bonds_type_rms", np.zeros([N,1]))
    for n in range(0, N):
        mol= Chem.MolFromSmiles(smiles[n,0])
        for label, function in zip(labels, functions):
            dataframe.set_value(N_start+n, label, function(mol))
        ##loop through the bonds
        temp_bonds_type = []
        for bond in mol.GetBonds():
            temp_bonds_type.append(Chem.rdchem.Bond.GetBondTypeAsDouble(bond))
        dataframe.set_value(N_start+n, "Bonds_type_mean", np.mean(temp_bonds_type))
        dataframe.set_value(N_start+n, "Bonds_type_rms", np.std(temp_bonds_type))


def clean_data(df, output=False):
    #drop gap values
    if not output:
        df  = df.drop(['gap'], axis=1)
    #drop smiles strings, add in length

    ##add in other features
    lbls = ['n_Bonds', 'n_HeavyAtoms', 'n_NumAtoms', 'n_NumConformers']
    fns  = [ Chem.rdchem.Mol.GetNumBonds, Chem.rdchem.Mol.GetNumHeavyAtoms, Chem.rdchem.Mol.GetNumAtoms, Chem.rdchem.Mol.GetNumConformers]
    add_features(df, lbls, fns)
    """
    Example Feature Engineering
    this calculates the length of each smile string and adds a feature column with those lengths
    Note: this is NOT a good feature and will result in a lower score!
    """
    smiles_len = np.vstack(df.smiles.astype(str).apply(lambda x: len(x)))
    df['smiles_len'] = pd.DataFrame(smiles_len)
    df  = df.drop(['smiles'], axis=1)

    print "shape of the dataset is: ", df.shape
    return df

def filter_data(df_train, df_test, Y_train):
    test = pd.DataFrame(Y_train, columns=["gap"])
    for i, col in enumerate(df_train.columns):
        correlation = df_train[col].corr(test["gap"])
        #print correlation
        if np.isnan(correlation):
            # plt.plot(df_train[col], Y_train, 'o' , label='train')
            # plt.xlabel(col)
            # plt.ylabel("gap")
            # plt.savefig('Plot/Drop_nan_' + col + '-gap.png', bbox_inches='tight')
            # plt.clf()
            df_train = df_train.drop(col, axis=1)
            df_test = df_test.drop(col, axis=1)
            #print "Drop: ", col
        # elif math.fabs(correlation) < 0.1:
        #     # plt.plot(df_train[col], Y_train, 'o' , label='train')
        #     # plt.xlabel(col)
        #     # plt.ylabel("gap")
        #     # plt.savefig('Plot/Drop_value_' + col + '-gap.png', bbox_inches='tight')
        #     # plt.clf()
        #     df_train = df_train.drop(col, axis=1)
        #     df_test = df_test.drop(col, axis=1)
        #     print "Drop: ", col, correlation
        else:
            plt.plot(df_train[col], Y_train, 'o' , label='train')
            plt.xlabel(col)
            plt.ylabel("gap")
            plt.savefig('Plot/Kept_' + col + '-gap.png', bbox_inches='tight')
            plt.clf()
            pass
    return df_train, df_test


def plot_data(df, Y_train):

    ##let's look at some plots...
    plt.plot(df['smiles_len'], Y_train, 'o' , label='train')
    plt.xlabel("smile length")
    plt.ylabel("gap")
    plt.savefig('Plot/smile-gap.png', bbox_inches='tight')
    plt.clf()

    ##let's look at another plot
    var  = range(0, df.shape[1])
    corr = []
    test = pd.DataFrame(Y_train, columns=["gap"])
    for i, col in enumerate(df.columns):
        if i > len(var):
            continue
        #print df[col].corr(df["gap"])
        #correlation = np.cov(df[col], test["gap"])
        correlation = df[col].corr(test["gap"])
        corr.append(-2 if np.isnan(correlation) else correlation)

    #print len(var), len(corr)
    # plt.plot(var, corr, 'o', label='correlations')
    # plt.xlabel("variable")
    # plt.ylabel("correlation with gap")
    # plt.savefig('Plot/Var-gap.png', bbox_inches='tight')
    # plt.clf()


def modeling(X_train, Y_train, X_test, Y_test, model=LinearRegression(), leg="LR"):
    
    model.fit(X_train, Y_train)
    model_pred = model.predict(X_test)
    print(leg + ": --- %s seconds ---" % (time.time() - analysis_time))
    
    # RF = RandomForestRegressor()
    # RF.fit(X_train, Y_train)
    # RF_pred = RF.predict(X_test)
    
    #print len(model_pred), len(Y_test)
    plt.plot(model_pred, Y_test, 'o', label='correlations')
    plt.xlabel("prediction")
    plt.ylabel("observation")
    rms = np.sqrt(MSE(model_pred, Y_test))
    print leg, ": RMS score is--", rms/1000.0
    plt.title('RMS:%.3f' % (rms/1000.0))
    plt.savefig('Plot/' + leg +'_pred.png', bbox_inches='tight')
    plt.clf()
    return model_pred


def analysis():
    global analysis_time
    analysis_time = time.time()

    NPartdata = int(1000000 * 0.1) #takes 1sec to load; 1% of the total size
    df_train = pd.read_csv("train.csv", nrows=NPartdata) ##total length is 1,000,000, 47sec loading
    df_test  = pd.read_csv("train.csv", skiprows=NPartdata, nrows=NPartdata/5) #pd.read_csv("test.csv")
    if ops.output:
        df_test = pd.read_csv("test.csv", nrows=10) 
    print("Load: --- %s seconds ---" % (time.time() - analysis_time))

    ##first rename the test sample
    df_test.columns = df_train.columns

    #print df_train.info()
    #print df_train.head()
    #print df_test.head()
    #print df_train.dtypes
    #print df_test.dtypes
    #print df_train.index
    #print df_test.index

    #store gap values
    ##for other classifiers to work; need to convert back!
    Y_train = df_train.gap.values * 1000
    Y_test  = df_test.gap.values * 1000
    Y_train = np.round(Y_train).astype(int)
    Y_test = np.round(Y_test).astype(int) 
    #print Y_train

    ##clean the framework, add in features
    df_train =  clean_data(df_train)
    df_test  =  clean_data(df_test)
    #df_result = clean_data(df_test)

    ##filter the useless variables
    df_train, df_test = filter_data(df_train, df_test, Y_train)

    #study...
    plot_data(df_train, Y_train)
    ##row where testing examples start
    test_idx = df_train.shape[0]
    #DataFrame with all train and test examples so we can more easily apply feature engineering on
    #df_all = pd.concat((df_train, df_test), axis=0)
    #df_all.head()
    trainvals = df_train.values
    X_train   = trainvals[:test_idx]
    testvals  = df_test.values
    X_test    = testvals[:test_idx]

    print "Train features:", X_train.shape, "Train gap:", Y_train.shape
    print "Test features:", X_test.shape, "Test gap:", Y_test.shape

    ##build models
    legs = ["Nearest Neighbors", 
            #"Linear SVM", 
            #"RBF SVM", 
            ##"Gaussian Process",
            #"Decision Tree Classify", 
            "Decision Tree Regress", 
            "Random Forest", 
            #"Neural Net", 
            #"AdaBoost",
            #"Naive Bayes", 
            #"QDA",
            #"BDT",
            "Poly 2, Ridge",
            #"Poly 2, Lasso",
            #"Poly 2, ARD",
            #"Poly 3",
            ]
    classifiers = [
        KNeighborsClassifier(),
        #SVC(kernel="linear", C=0.025), ##very slow
        #SVC(gamma=2, C=1), ##super slow
        ##GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), #super slow
        #DecisionTreeClassifier(max_depth=20),
        DecisionTreeRegressor(max_depth=20),
        RandomForestClassifier(),
        #MLPClassifier(alpha=0.1, random_state=1), #rather slow
        #AdaBoostClassifier(), ##slow
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
        #AdaBoostClassifier( DecisionTreeClassifier(max_depth=7), n_estimators=600, learning_rate=1.5, algorithm="SAMME"),
        make_pipeline(PolynomialFeatures(2), sk.linear_model.Ridge()),
        #make_pipeline(PolynomialFeatures(2), sk.linear_model.Lasso()),
        #make_pipeline(PolynomialFeatures(2), sk.linear_model.ARDRegression()),
        #make_pipeline(PolynomialFeatures(3), sk.linear_model.Ridge()),
        ]

    ##do the real work!!!
    print " ################RESULTS################# "
    modeling(X_train, Y_train, X_test, Y_test, model=LinearRegression(), leg="LR")
    #modeling(X_train, Y_train, X_test, Y_test, model=sk.linear_model.Ridge(alpha = .5), leg="Ridge")
    #modeling(X_train, Y_train, X_test, Y_test, model=svm.SVC(gamma=0.001, C=100.), leg="SVM")
    #modeling(X_train, Y_train, X_test, Y_test, model=RandomForestRegressor(), leg="RF")
    #modeling(X_train, Y_train, X_test, Y_test, model=rbm, leg="NR")
    #outputmodel = []
    for i, model in enumerate(classifiers):
        modeling(X_train, Y_train, X_test, Y_test, model=model, leg=legs[i])
    print " ################DONE################# "
    

    #write_to_file("random1.csv", LR_pred) #sample1.csv
    #write_to_file("random2.csv", RF_pred) #sample2.csv

def main():
    #start time
    start_time = time.time()
    global ops
    ops = options()
    #setup basics
    #analysis()

    ##for slimming the dataset
    df_train = pd.read_csv("train.csv", nrows=1000) ##total length is 1,000,000, 47sec loading
    df_test = pd.read_csv("test.csv", nrows=1000) ##total length is 1,000,000, 47sec loading

    df_train_col_gap  = df_train['gap']
    df_train_col_smile  = df_train['smiles']
    df_train  = df_train.drop(['smiles'], axis=1)
    df_train  = df_train.drop(['gap'], axis=1)
    df_test_col_gap  = df_test['Id']
    df_test_col_smile  = df_test['smiles']
    df_test  = df_test.drop(['Id'], axis=1)
    df_test  = df_test.drop(['smiles'], axis=1)
    Y_train  = df_train_col_gap.values
    df_train, df_test = filter_data(df_train, df_test, Y_train)
    df_train['gap'] = df_train_col_gap
    df_train['smiles'] = df_train_col_smile
    df_test['Id'] = df_test_col_gap
    df_test['smiles']= df_test_col_smile
    df_train.to_csv("train_slim.csv")
    df_test.to_csv("test_slim.csv")

    # #parallel compute!
    # print " Running %s jobs on %s cores" % (len(inputtasks), mp.cpu_count()-1)
    # npool = min(len(inputtasks), mp.cpu_count()-1)
    # pool = mp.Pool(npool)
    # pool.map(dumpRegion, inputtasks)
    # for i in inputtasks:
    #     dumpRegion(i)
    #dumpRegion(inputtasks[0])
    print("--- %s seconds ---" % (time.time() - start_time))


#####################################
if __name__ == '__main__':
    main()
