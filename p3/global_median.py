import numpy as np
import pandas as pd
import util
import csv

# Predict via the median number of plays.

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
        plays  = int(row[2])
    
        if not user in train_data:
            train_data[user] = {}
        
        train_data[user][artist] = plays

# Compute the global median.
sol_dic = {}

for user, user_data in train_data.iteritems():
    plays_array = []
    for artist, plays in user_data.iteritems():
        plays_array.append(plays)
    train_data[user]["median"] = np.median(np.array(plays_array))
    if len(plays_array) >= 5:
        if np.std(np.array(plays_array)) / np.median(np.array(plays_array)) < 1.0:
            train_data[user]["median"] = (np.median(np.array(plays_array)) + np.median(np.array(plays_array[1:])) + np.median(np.array(plays_array[:-2])))/3.0

    #print train_data[user]["median"]
print "done this part"
#global_median = np.median(np.array(plays_array))
#print "global median:", global_median

df = pd.read_pickle("newtrain.pd")
df_all = pd.read_pickle("newtrain_0.pd")
dic_df = df.set_index("ID")["ratio"].to_dict()
dic_df_all = df_all.set_index("ID")["ratio"].to_dict()
# Write out test solutions.
with open(test_file, 'r') as test_fh:
    test_csv = csv.reader(test_fh, delimiter=',', quotechar='"')
    next(test_csv, None)

    with open(soln_file, 'w') as soln_fh:
        soln_csv = csv.writer(soln_fh,
                              delimiter=',',
                              quotechar='"',
                              quoting=csv.QUOTE_MINIMAL)
        soln_csv.writerow(['Id', 'plays'])
        counter = 0
        for row in test_csv:
            counter += 1
            if (counter%1000 == 0):
                util.drawProgressBar(counter/4154805.0)
            
            id     = row[0]
            user   = row[1]
            artist = row[2]
            #print df[df['ID'] == str(id)]["ratio"]
            weight = float(dic_df[user])
            weight_all = float(dic_df_all[user])
            #print weight, weight_all

            scale = weight_all
            if weight == 0:
                weight_all = 1 #go back to median

            soln_csv.writerow([id, train_data[user]["median"]])
