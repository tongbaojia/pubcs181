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
from rdkit.Chem import Descriptors
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
    M = dataframe.shape[1]
    N_start = int(dataframe.index[0])
    smiles = dataframe.smiles.values
    # for label, function in zip(labels, functions):
    #     dataframe.insert(M, label, np.zeros([N,1]))
    #extras
    #dataframe.insert(M, "Bonds_type_mean", np.zeros([N,1]))
    #dataframe.insert(M, "Bonds_type_rms", np.zeros([N,1]))
    # bond_types = [
    # "AROMATIC",
    # "DATIVE",
    # "DATIVEL",
    # "DATIVEONE",
    # "DATIVER",
    # "DOUBLE",
    # "FIVEANDAHALF",
    # "FOURANDAHALF",
    # "HEXTUPLE",
    # "HYDROGEN",
    # "IONIC",
    # "ONEANDAHALF",
    # "OTHER",
    # "QUADRUPLE",
    # "QUINTUPLE",
    # "SINGLE",
    # "THREEANDAHALF",
    # "THREECENTER",
    # "TRIPLE",
    # "TWOANDAHALF",
    # "UNSPECIFIED",
    # "ZERO",
    # ]
    # rd_bond_types = [
    #     Chem.rdchem.BondType.AROMATIC,
    #     Chem.rdchem.BondType.DATIVE,
    #     Chem.rdchem.BondType.DATIVEL,
    #     Chem.rdchem.BondType.DATIVEONE,
    #     Chem.rdchem.BondType.DATIVER,
    #     Chem.rdchem.BondType.DOUBLE,
    #     Chem.rdchem.BondType.FIVEANDAHALF,
    #     Chem.rdchem.BondType.FOURANDAHALF,
    #     Chem.rdchem.BondType.HEXTUPLE,
    #     Chem.rdchem.BondType.HYDROGEN,
    #     Chem.rdchem.BondType.IONIC,
    #     Chem.rdchem.BondType.ONEANDAHALF,
    #     Chem.rdchem.BondType.OTHER,
    #     Chem.rdchem.BondType.QUADRUPLE,
    #     Chem.rdchem.BondType.QUINTUPLE,
    #     Chem.rdchem.BondType.SINGLE,
    #     Chem.rdchem.BondType.THREEANDAHALF,
    #     Chem.rdchem.BondType.THREECENTER,
    #     Chem.rdchem.BondType.TRIPLE,
    #     Chem.rdchem.BondType.TWOANDAHALF,
    #     Chem.rdchem.BondType.UNSPECIFIED,
    #     Chem.rdchem.BondType.ZERO,
    # ]
    # for bond_type in bond_types:
    #     dataframe.insert(M, "Bond_Type_" + bond_type, np.zeros([N,1]))

    #Rings
    # dataframe.insert(M, "Ring_N", np.zeros([N,1]))
    # dataframe.insert(M, "Ring_Natom", np.zeros([N,1]))
    # dataframe.insert(M, "Ring_Nbond", np.zeros([N,1]))

    for n in range(0, N):
        if n%10000 == 0:
            print "Done ", n, " events!"
        mol = Chem.MolFromSmiles(smiles[n])
        # for label, function in zip(labels, functions):
        #     dataframe.set_value(N_start+n, label, function(mol))
        ##loop through the bonds
        temp_bonds_type = []
        for bond in mol.GetBonds():
            bondtype = Chem.rdchem.Bond.GetBondTypeAsDouble(bond)
            temp_bonds_type.append(bondtype)
        for i, typevalue in enumerate(rd_bond_types):
            dataframe.set_value(N_start+n, "Bond_Type_" + bond_types[i], temp_bonds_type.count(i))

        # ringinfo = mol.GetRingInfo()
        # dataframe.set_value(N_start+n, "Ring_N", ringinfo.NumRings())
        # ring_natom = 0
        # ring_nbond = 0
        # for j in range(ringinfo.NumRings()):
        #     ring_natom += ringinfo.NumAtomRings(j)
        #     ring_nbond += ringinfo.NumBondRings(j)
        # dataframe.set_value(N_start+n, "Ring_Natom", ring_natom)
        # dataframe.set_value(N_start+n, "Ring_Nbond", 
        #     ring_nbond)
        #dataframe.set_value(N_start+n, "Bonds_type_mean", np.mean(temp_bonds_type))
        #dataframe.set_value(N_start+n, "Bonds_type_rms", np.std(temp_bonds_type))


def modify_data(df):
    # lbls = ['n_Bonds', 'n_HeavyAtoms', 'n_NumAtoms', 'n_NumConformers', 
    # 'mol_wt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons', 'NumRadicalElectrons']
    # fns  = [ Chem.rdchem.Mol.GetNumBonds, Chem.rdchem.Mol.GetNumHeavyAtoms, 
    # Chem.rdchem.Mol.GetNumAtoms, Chem.rdchem.Mol.GetNumConformers, Chem.Descriptors.MolWt, 
    # Chem.Descriptors.HeavyAtomMolWt, Chem.Descriptors.ExactMolWt, Chem.Descriptors.NumValenceElectrons, Chem.Descriptors.NumRadicalElectrons]

    lbls = [] 
    fns = [] 
    add_features(df, lbls, fns)
    return df


def clean_data(df, output=False):
    #check sign..
    #weird = df.ix[df["gap"] < 0]
    #print weird.smiles.values
    # for i in df.smiles.values:
    #     if "c12" in i:
    #         print "check:", i
    #drop gap values
    if not output:
        df  = df.drop(['gap'], axis=1)
    #drop smiles strings, add in length

    ##add in other features
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
        #print col, correlation
        if np.isnan(correlation):
            plt.plot(df_train[col], Y_train, 'o' , label='train')
            plt.xlabel(col)
            plt.ylabel("gap")
            plt.savefig('Plot/Drop_nan_' + col + '-gap.png', bbox_inches='tight')
            plt.clf()
            df_train = df_train.drop(col, axis=1)
            df_test = df_test.drop(col, axis=1)
            #print "Drop: ", col
        elif math.fabs(correlation) < 0.1:
            plt.plot(df_train[col], Y_train, 'o' , label='train')
            plt.xlabel(col)
            plt.ylabel("gap")
            plt.savefig('Plot/Drop_value_' + col + '-gap.png', bbox_inches='tight')
            plt.clf()
            df_train = df_train.drop(col, axis=1)
            df_test = df_test.drop(col, axis=1)
            print "Drop: ", col, correlation
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
    print leg, ": RMSE is--", rms/1000.0
    plt.title(leg + ': RMSE:%.3f' % (rms/1000.0))
    plt.savefig('Plot/' + leg +'_pred.png', bbox_inches='tight')
    plt.clf()
    return model_pred


def analysis():
    global analysis_time
    analysis_time = time.time()
    NPartdata = int(1000000) #takes 1sec to load; 1% of the total size
    df_train = pd.read_csv("train_slim3.csv", nrows=NPartdata) ##total length is 1,000,000, 47sec loading
    df_test  = pd.read_csv("train_slim3.csv", skiprows=NPartdata, nrows=NPartdata/5) #pd.read_csv("test.csv")
    if ops.output:
        df_test = pd.read_csv("test_slim3.csv") 
    print("Load: --- %s seconds ---" % (time.time() - analysis_time))

    ##first rename the test sample
    if ops.output:
        df_test['gap'] = df_test['Id']
        df_test = df_test.drop(['Id'], axis=1)
    else:
        df_test.columns = df_train.columns

    #print df_train.info()
    #print df_train.columns.values
    #print df_test.columns.values
    #print df_train.dtypes
    #print df_test.dtypes
    #print df_train.index
    #print df_test.index

    #store gap values
    ##for other classifiers to work; need to convert back!
    Y_train = df_train.gap.values * 1000
    Y_train = np.round(Y_train).astype(int)
    if not ops.output:
        Y_test  = df_test.gap.values * 1000
        Y_test = np.round(Y_test).astype(int) 
    else:
        Y_test = Y_train[:df_test.shape[0]]
    #print Y_train

    ##clean the framework, add in features
    df_train =  clean_data(df_train)
    df_test  =  clean_data(df_test)
    #df_result = clean_data(df_test)

    ##filter the useless variables
    #df_train, df_test = filter_data(df_train, df_test, Y_train)

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
    legs = [#"Nearest Neighbors", 
            #"Linear SVM", 
            #"RBF SVM", 
            ##"Gaussian Process",
            #"Decision Tree Classify", 
            #"Decision_Tree_dep10", 
            #"Decision_Tree_dep20", 
            "Decision_Tree_dep30",  
            #"Decision_Tree_dep30_leaf100000", 
            #"Decision_Tree_dep30_leaf1000000",
            #"Decision_Tree_dep40", 
            #"Decision_Tree_dep20_Ridge", 
            #"Random Forest", 
            #"Neural Net", 
            #"AdaBoost",
            #"Naive Bayes", 
            #"QDA",
            #"BDT_dep20_est100",
            #"BDT_dep20_est200",
            #"BDT_dep20_est300",
            #"Poly 2, Ridge",
            #"Poly 2, Lasso",
            #"Poly 2, ARD",
            #"Poly 3",
            ]
    classifiers = [
        #KNeighborsClassifier(),
        #SVC(kernel="linear", C=0.025), ##very slow
        #SVC(gamma=2, C=1), ##super slow
        ##GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), #super slow
        #DecisionTreeClassifier(max_depth=20),
        #DecisionTreeRegressor(max_depth=10),
        #DecisionTreeRegressor(max_depth=20),
        DecisionTreeRegressor(max_depth=30),
        #DecisionTreeRegressor(max_depth=30, max_leaf_nodes = 100000),
        #DecisionTreeRegressor(max_depth=30, max_leaf_nodes = 1000000),
        #DecisionTreeRegressor(max_depth=40),
        #make_pipeline(DecisionTreeRegressor(max_depth=20), sk.linear_model.Ridge()),
        #RandomForestClassifier(),
        #MLPClassifier(alpha=0.1, solver="sgd", learning_rate="invscaling", momentum=0.9, nesterovs_momentum=True, learning_rate_init=0.2, max_iter=2000), #rather slow
        #AdaBoostClassifier(), ##slow
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
        #AdaBoostClassifier( DecisionTreeClassifier(max_depth=20), n_estimators=100, learning_rate=1, algorithm="SAMME.R"),
        #AdaBoostClassifier( DecisionTreeClassifier(max_depth=20), n_estimators=200, learning_rate=1, algorithm="SAMME.R"),
        #AdaBoostClassifier( DecisionTreeClassifier(max_depth=20), n_estimators=300, learning_rate=1, algorithm="SAMME.R"),
        #make_pipeline(PolynomialFeatures(2), sk.linear_model.Ridge()),
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
    if ops.output:
        classifiers[0].fit(X_train, Y_train)
        result = classifiers[0].predict(X_test)/1000.0
        write_to_file("random1.csv", result) #sample1.csv
    else:
        outputmodel = []
        for i, model in enumerate(classifiers):
            outputmodel.append(modeling(X_train, Y_train, X_test, Y_test, model=model, leg=legs[i]))
        print " ################DONE################# "
    

def main():
    #start time
    start_time = time.time()
    global ops
    ops = options()
    #setup basics
    analysis()

    ##for slimming the dataset
    # df_train = pd.read_csv("train_updated2.csv") ##total length is 1,000,000, 47sec loading
    # df_test = pd.read_csv("test_updated2.csv") ##total length is 1,000,000, 47sec loading
    # df_train_col_gap  = df_train['gap']
    # df_train_col_smile  = df_train['smiles']
    # df_train  = df_train.drop(['smiles'], axis=1)
    # df_train  = df_train.drop(['gap'], axis=1)
    # df_test_col_gap  = df_test['Id']
    # df_test_col_smile  = df_test['smiles']
    # df_test  = df_test.drop(['Id'], axis=1)
    # df_test  = df_test.drop(['smiles'], axis=1)
    # Y_train  = df_train_col_gap.values
    # df_train, df_test = filter_data(df_train, df_test, Y_train)
    # df_train['gap'] = df_train_col_gap
    # df_train['smiles'] = df_train_col_smile
    # df_test['Id'] = df_test_col_gap
    # df_test['smiles']= df_test_col_smile
    # df_train.to_csv("train_slim3.csv", index=False)
    # df_test.to_csv("test_slim3.csv", index=False)

    ##for adding features to dataset
    # df_train = pd.read_csv("train_slim2.csv") ##total length is 1,000,000, 47sec loading
    # df_train = modify_data(df_train)
    # df_train.to_csv("train_updated2.csv", index=False)
    # df_test = pd.read_csv("test_slim2.csv") ##total length is 1,000,000, 47sec loading
    # df_test = modify_data(df_test)
    # df_test.to_csv("test_updated2.csv", index=False)


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
