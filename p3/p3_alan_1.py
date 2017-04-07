#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 13:37:14 2017

@author: alanlegoallec
"""

import numpy as np
import csv
import pandas as pd

pd_artists = pd.read_pickle("artists.pd")
pd_mergetrain = pd.read_pickle("train.pd")
train = pd_mergetrain.set_index('ID').to_dict()
train = pd_mergetrain.set_index('ID').T.to_dict('list')

train_file = 'train.csv'
test_file  = 'test.csv'
soln_file  = 'global_median.csv'



# Load the training data.
train_data = {}
with open(train_file, 'r') as train_fh:
    train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    next(train_csv, None)
    for row in train_csv:
        user   = row[0]
        artist = row[1]
        plays  = row[2]
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = int(plays)

# Compute the global median and per-user median.
plays_array  = []
user_medians = {}
for user, user_data in train_data.iteritems():
    user_plays = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
        user_plays.append(plays)

    user_medians[user] = np.median(np.array(user_plays))
global_median = np.median(np.array(plays_array))


counter = 0
for i, obs in enumerate(pd_mergetrain["Obs"]):
    if len(obs) > 50:
        counter += 1
print counter

test_data = {}
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)
    for row in test_csv:
        id   = row[0]
        user = row[1]
    
        if not user in test_data:
            test_data[user] = {}
        
        test_data[user] = 0
                 



# Write out test solutions. Use clusters of artists when too little info is available from user. (combined with similar users info)
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])
        k = 0
        i = 0
        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            if user in user_medians:
                if len(train[user][1].values()) < 30:
                    i += 1
                    if i%100000 == 0:
                        print i
                    soln_csv.writerow([id, user_medians[user]])
                else:
                    rating = float(pd_artists["rating"][pd_artists['ID'] == artist].values[0])
                    if rating < 0.1: #I am not using 0 because rating is a float, so I am affraid of the conversion.
                        soln_csv.writerow([id, user_medians[user]])
                    else:
                        data_rating = []
                        for artist2, value in pd_mergetrain["Obs"][pd_mergetrain["ID"] == user].values[0].iteritems():
                            if np.abs(float(pd_artists["rating"][pd_artists['ID'] == artist2])-rating) < 1.5:
                                data_rating.append(value)
                        if len(data_rating) > 4:
                            k += 1
                            if k%100 == 0:
                                print k
                            soln_csv.writerow([id, np.median(data_rating)])
                            #print np.median(data_rating)
                            #print " instead of "
                            #print user_medians[user]
                        else:
                            soln_csv.writerow([id, user_medians[user]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])


