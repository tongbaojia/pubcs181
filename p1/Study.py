# Tony Tong; baojia.tong@cern.ch
import os, argparse, sys, math, time
#for parallel processing!
import multiprocessing as mp
import rdkit as rd
#other necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

"""
Read in train and test as Pandas DataFrames
"""
print "Done with loading!"

#define functions
def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputdir",  default="here")
    parser.add_argument("--inputroot", default="sum")
    parser.add_argument("--detail",    default=True)
    return parser.parse_args()

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")

def analysis():
    
    analysis_time = time.time()
    df_train = pd.read_csv("train.csv", nrows=99999) #pd.read_csv("test.csv")
    df_test = pd.read_csv("test.csv", nrows=9999) #pd.read_csv("train.csv")
    print("Load: --- %s seconds ---" % (time.time() - analysis_time))

    #print df_train.head()
    #print df_test.head()
    #print df_train.dtypes
    #print df_test.dtypes
    print df_train.index
    print df_test.index

    #store gap values
    Y_train = df_train.gap.values
    #row where testing examples start
    test_idx = df_train.shape[0]
    #delete 'Id' column
    df_test = df_test.drop(['Id'], axis=1)
    #delete 'gap' column
    df_train = df_train.drop(['gap'], axis=1)

    #DataFrame with all train and test examples so we can more easily apply feature engineering on
    df_all = pd.concat((df_train, df_test), axis=0)
    df_all.head()
    """
    Example Feature Engineering

    this calculates the length of each smile string and adds a feature column with those lengths
    Note: this is NOT a good feature and will result in a lower score!
    """
    smiles_len = np.vstack(df_all.smiles.astype(str).apply(lambda x: len(x)))
    df_all['smiles_len'] = pd.DataFrame(smiles_len)

    #Drop the 'smiles' column
    df_all = df_all.drop(['smiles'], axis=1)
    vals = df_all.values
    X_train = vals[:test_idx]
    X_test = vals[test_idx:]

    print "Train features:", X_train.shape
    print "Train gap:", Y_train.shape
    print "Test features:", X_test.shape

    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    LR_pred = LR.predict(X_test)
    print("LR: --- %s seconds ---" % (time.time() - analysis_time))


    RF = RandomForestRegressor()
    RF.fit(X_train, Y_train)
    RF_pred = RF.predict(X_test)
    print("RF: --- %s seconds ---" % (time.time() - analysis_time))

    write_to_file("random1.csv", LR_pred) #sample1.csv
    write_to_file("random2.csv", RF_pred) #sample2.csv

def main():
    #start time
    start_time = time.time()
    global ops
    ops = options()
    #setup basics
    analysis()

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
