#First, perform SVD to reduce dimensions of the data
#Then,predict by projecting user eigenvectors onto artist eigenvectors

import numpy as np
import csv
import json
import random
import scipy as sp
from scipy.sparse import linalg
import matplotlib.pyplot as plt


train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'user_median.csv'

# Load the training data.
prob_train = 0.95
num_lines = 50000000

train_data = {}
test_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    count = 0
    for row in train_csv:
        count+=1
        user   = row[0]
        artist = row[1]
        plays  = row[2]
        
        #decide if test or train
        if random.random() < prob_train:
            if not user in train_data:
                train_data[user] = {}
            train_data[user][artist] = int(plays)        
        else:
            if not user in test_data:
                test_data[user] = {}
            test_data[user][artist] = int(plays)
        if count >num_lines:
            break
#print(count)
#get user means to rescale
user_total = {}
user_means = {}
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        user_plays.append(plays)
    user_total[user] = np.sum(np.array(user_plays))
    user_means[user] = np.mean(np.array(user_plays))
    user_medians[user] = np.median(np.array(user_plays))
#To make a sparse matrix, index users and artists sequentially using data in artists, profiles

user_file = 'profiles.csv'
artist_file = 'artists.csv'

user_num = {}
with open(user_file, 'r') as user_fh:
    user_csv = csv.reader(user_fh, delimiter=',', quotechar='"')
    next(user_csv, None)
    count = 0
    for row in user_csv:
        user   = row[0]
        if not user in user_num:
            user_num[user] = count     
            count+=1
            #print(count)
artist_num = {}
with open(artist_file, 'r') as artist_fh:
    artist_csv = csv.reader(artist_fh, delimiter=',', quotechar='"')
    next(artist_csv, None)
    count = 0
    for row in artist_csv:
        artist  = row[0]
        if not artist in artist_num:
            artist_num[artist] = count     
            count+=1
import scipy
from scipy.sparse import *
train_mat = lil_matrix((len(user_num),len(artist_num)), dtype=np.float)

for user in train_data:
    user_id = user_num[user]
    for artist in train_data[user]:
        artist_id = artist_num[artist]                      
        train_mat[user_id, artist_id] = train_data[user][artist]/np.float(user_total[user])

#now fill in user means as a starting point for the prediction
for user in test_data:
    user_id = user_num[user]
    for artist in test_data[user]:
        artist_id = artist_num[artist]                      
        train_mat[user_id, artist_id] = np.float(user_medians[user])/np.float(user_total[user])
        
train_mat = scipy.sparse.csr_matrix(train_mat)
print train_mat.shape


k_vals = np.array([1400])
acc = np.zeros(k_vals.shape)
for i in range(k_vals.size):
    #Perform SVD-like decomposition of the training data
    U,S,V=scipy.sparse.linalg.svds(train_mat, k = k_vals[i])
    #do the same to the test data

    test_mat = lil_matrix((len(user_num),len(artist_num)), dtype=np.float)
    pred_mat = lil_matrix((len(user_num),len(artist_num)), dtype=np.float)

    for user in test_data:
        user_id = user_num[user]
        user_row = U[user_id,:]
        for artist in test_data[user]:
            artist_id = artist_num[artist]                      
            test_mat[user_id, artist_id] = test_data[user][artist]
            pred_mat[user_id, artist_id] = np.matmul(np.matmul(user_row, np.diag(S)), V[:,artist_id])*np.float(user_total[user])
            #predict number of plays

    # print test_mat.shape
    # print pred_mat.shape
    #Check accuracy
    num_entries = sum(len(v) for v in test_data.itervalues())
    acc[i] = np.sum(np.absolute(np.subtract(pred_mat,test_mat)))/num_entries
    print(acc)
