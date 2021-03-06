#First, perform SVD to reduce dimensions of the data
#Then, predict using average values over g nearest neighbors

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


from scipy.spatial import KDTree

group_size_vals = np.array([2,4,8])
k_vals = np.array([6])
acc = np.zeros(group_size_vals.shape)
U,S,V=scipy.sparse.linalg.svds(train_mat, k = k_vals[0])
test_mat = lil_matrix((len(user_num),len(artist_num)), dtype=np.float)
pred_mat = lil_matrix((len(user_num),len(artist_num)), dtype=np.float)
for i in range(group_size_vals.size):
    group_size = group_size_vals[i]
    #Perform SVD-like decomposition of the training data
    #U,S,V=scipy.sparse.linalg.svds(train_mat, k = k_vals[i])
    
    tree = KDTree(U)
    
    for user in test_data:
        #print(user_id)
        user_id = user_num[user]
        if user_id%100 == 0:       
            print(user_id)
        dists, top_inds = tree.query(U[user_id,:], k = group_size)
        #find closest neighboring users
        sum_user = train_mat[top_inds[1:], :].sum(0)
        #dists =  np.linalg.norm( np.subtract(U[user_id,:], U), axis = 1)
        #top_inds = np.argpartition(dists, group_size)[:group_size]
        #group_avg = np.mean(U[top_inds, :], axis = 0)         
        for artist in test_data[user]:
            artist_id = artist_num[artist]                      
            test_mat[user_id, artist_id] = test_data[user][artist]
            sum_pred = sum_user[0,artist_id]
            #if sum_pred>0:
            pred_mat[user_id, artist_id] =np.true_divide(sum_pred,group_size)*np.float(user_total[user])
            #pred_mat[user_id, artist_id] = (user_medians[user]+(group_size)*pred_mat[user_id, artist_id])/(group_size+1.0)
            #pred_mat[user_id, artist_id] =np.true_divide(sum_pred,(train_mat[top_inds, artist_id]!=0).sum(0))*np.float(user_total[user])
                #pred_mat[user_id, artist_id] = (user_medians[user]+pred_mat[user_id, artist_id])/2.0
            #else:
                #pred_mat[user_id, artist_id] = user_medians[user]
            #pred_mat[user_id, artist_id] =np.true_divide(train_mat[top_inds, artist_id].sum(0),(train_mat[top_inds, artist_id]!=0).sum(0))*np.float(user_total[user])
            #pred_mat[user_id, artist_id] = np.matmul(np.matmul(group_avg, np.diag(S)), V[:,artist_id])*np.float(user_total[user])
            #predict number of plays

    # print test_mat.shape
    # print pred_mat.shape
    #Check accuracy
    num_entries = sum(len(v) for v in test_data.itervalues())
    acc[i] = np.sum(np.absolute(np.subtract(pred_mat,test_mat)))/num_entries
    print(acc)
    

plt.clf()
fig = plt.figure()
plt.plot(group_size_vals,acc)
plt.xlabel('Number of nearest neighbors')
plt.ylabel('Mean absolute error')
plt.title('k = '+str(k_vals[0]))
plt.show()
fig.savefig('SVD_group_'+str(group_size_vals[-1])+'.png')
    