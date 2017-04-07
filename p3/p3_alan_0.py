#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 02:38:49 2017

@author: alanlegoallec
"""

import numpy as np
import csv
import pandas as pd

pd_artists = pd.read_pickle("artists.pd")
pd_train = pd.read_pickle("train.pd")

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

#replace the per-user median by the mean per-user median of similar user
# when too little info is avaialbe on this user
pd_train = pd.read_pickle("train.pd")
counter = 0
for i, obs in enumerate(pd_mergetrain["Obs"]):
    if len(obs) < 5:
        counter += 1
        print counter
        gender, age, country = pd_mergetrain["Value"][i]
        if (gender != u'') & (age != u'') & (country != 'u'):
            age = int(age)
            Medians = []
            counter2 = 0
            for values in pd_mergetrain["Value"]:
                if (values[1] != u''):
                    if (values[0] == gender) & (values[2] == country) & (np.abs(int(values[1])-age) < 4):
                        Medians.append(np.median(pd_mergetrain["Obs"][counter2].values()))
                counter2 += 1
            user_medians[pd_mergetrain["ID"][i]] = np.mean(Medians)
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

        for row in test_csv:
            id     = row[0]
            user   = row[1]
            artist = row[2]

            if user in user_medians:
                index = pd_mergetrain["ID"] == user
                if len(pd_mergetrain["Obs"].loc[index]) > 6:
                    soln_csv.writerow([id, user_medians[user]])
                else:
                    gender, age, country = pd_mergetrain["Value"][index]
                    if (gender != u'') & (age != u'') & (country != 'u'):
                        age = int(age)
                        Medians_selected = []
                        Medians_all = []
                        counter = 0
                        rating = pd_artists["rating"][pd_artists['ID'] == artist]
                        for values in pd_mergetrain["Value"]:
                            if (values[1] != u''):
                                if (values[0] == gender) & (values[2] == country) & (np.abs(int(values[1])-age) < 4):
                                    Medians_all.append(np.median(pd_mergetrain["Obs"][counter].values()))
                                    data_individual = []
                                    for artist2, value in pd_mergetrain["Obs"][counter]:
                                        if np.abs(pd_artists["rating"][pd_artists['ID'] == artist2]-rating) < 1.5:
                                            data_individual.append(value)
                                    if len(data_individual) > 0:
                                        Medians_selected.append(np.median(data_individual))
                            counter += 1
                        soln_csv.writerow([id, user_medians[user]*np.mean(Medians_selected)/np.mean(Medians_all)])
                    else:
                        soln_csv.writerow([id, user_medians[user]])
            else:
                print "User", id, "not in training data."
                soln_csv.writerow([id, global_median])



#j = 0
#for values in pd_mergetrain["Value"]:
#    j+= 1
#    if j < 10:
#        print values[0]

#load the dictionary
#with open("train.txt", "r") as f:
#    train_data = json.load(f)
#with open("profiles.txt", "r") as f:
#    profiles_data = json.load(f)
#with open("artists.txt", "r") as f:
#    artists_data = json.load(f)

#musicbrainzngs.set_useragent("Example music app", "0.1", "http://example.com/music")


# cd Desktop/cs181/p3/

#features
# musicbrainzngs.get_area_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_instrument_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_label_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_event_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_place_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_recording_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_release_group_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_release_by_id(id, includes=[], release_status=[], release_type=[])
# musicbrainzngs.get_series_by_id(id, includes=[])
# musicbrainzngs.get_work_by_id(id, includes=[])

#def add_features(id)

#set(test_data.keys()) - set(train_data.keys())
#subset_keys = train_data.keys()[:1000]
#subset_train_data = dict((k, train_data[k]) for k in subset_keys if k in train_data)



