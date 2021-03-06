# Tony Tong; baojia.tong@cern.ch
import os, argparse, sys, math, time, csv, util, json, pickle
#for parallel processing!
import multiprocessing as mp
#import musicbrainzngs
#other necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
#load all things~
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors.kde import KernelDensity
from sklearn.feature_selection import SelectKBest, chi2
# from sklearn.metrics import mean_squared_error as MSE
# from sklearn.neural_network import BernoulliRBM
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification
# from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
# from sklearn.pipeline import make_pipeline

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

def user_number(obs):
    sex = obs[0]
    try:
        age = int(obs[1])
    except ValueError:
        age = 100
    country = obs[2]
    code = ""
    if "m" in sex:
        code += "0"
    elif "f" in sex:
        code += "1"
    else:
        code += "2"

    if int(age) < 18:
        code += "0"
    elif int(age) < 21:
        code += "1"
    elif int(age) < 25:
        code += "2"
    elif int(age) < 30:
        code += "3"
    elif int(age) < 35:
        code += "4"
    elif int(age) < 40:
        code += "5"
    elif int(age) < 45:
        code += "6"
    else :
        code += "7"

    if "States" in country:
        code += "0"
    elif "Kingdom" in country:
        code += "1"
    else :
        code += "2"
    return code

def add_artfeature():
    ## make the artist framework
    pd_artists = pd.read_pickle("artists.pd")
    pd_mergetrain = pd.read_pickle("train.pd")
    #[u'work-list', 'release-group-count', 'work-count', u'alias-list', 'id', 'area', 'life-span', 'alias-count', 
    #u'release-group-list', u'begin-area', u'release-list', 'recording-count', 'tag-list', u'artist-relation-list', u'sort-name', 
    #u'work-relation-list', u'url-relation-list', u'event-relation-list', 'release-count', u'recording-list', u'isni-list', 
    #u'recording-relation-list', u'name', 'type', 'country', 'rating', u'release-relation-list', u'disambiguation', u'end-area', 
    #u'release_group-relation-list', 'gender', u'label-relation-list', u'ipi', u'ipi-list', u'annotation', u'series-relation-list', u'place-relation-list']

    ##this is for adding features to artists
    # temp_lst = []##check content
    # add_column = []
    # add_column_name = "gender"
    # for key, value in pd_artists["Value"].iteritems():
    #     try:
    #         #add_column.append(value["artist"][add_column_name])
    #         tag_temp = ""
    #         for i in value["artist"][add_column_name]:
    #             # print 
    #             # if i["name"] not in temp_lst:
    #             tag_temp += str(value["artist"][add_column_name]) + ";"
    #             #tag_temp += (i["name"] + "," +  str(i["count"]) + ";")
    #         add_column.append(tag_temp)
    #     except KeyError:
    #         add_column.append(0)
    #         #pass



    new_dic = {}
    for i, artist in enumerate(pd_artists["ID"]):
        new_dic[artist] = {}

    for i, obs in enumerate(pd_mergetrain["Obs"]):
        if (i%100 == 0):
            util.drawProgressBar(i/233286.0)

        code = user_number(pd_mergetrain["Value"][i])

        #instead of the pure count, just save the average rating...normalized 
        n_total_listen = []
        for artist_id, n_listen in obs.items():
            n_total_listen.append(n_listen)

        n_max = float(max(n_total_listen))
        n_min = float(min(n_total_listen))
        
        def score(n_listen):
            try:
                return (5.0 * (n_listen - n_min)/(n_max - n_min) + 1)
            except ZeroDivisionError:
                return 3.0


        for artist_id, n_listen in obs.items():
            #save each one
            try:
                new_dic[artist_id][code].append(score(n_listen))
            except KeyError:
                new_dic[artist_id][code] = [score(n_listen)]

            #save the all as well
            try:
                new_dic[artist_id]["all"].append(score(n_listen))
            except KeyError:
                new_dic[artist_id]["all"] = [score(n_listen)]

    # for i, artist in enumerate(pd_artists["ID"]):
    #     for key, value in new_dic[artist].items():
    #         new_dic[artist][key] = np.median(np.array(map(float, value)))


    with open("new_artmedian_rating.txt", "w") as f:
        json.dump(new_dic, f)
    #new_art = pd.DataFrame(new_dic.items(), columns=['ID', 'Info']) 
    #print new_art
    #new_art.to_pickle("new_artmedian_rating.pd")

    #print new_dic
    #print temp_lst
    #print add_column[0]
    # print add_column[1]
    # print add_column[-1]
    #pd_artists[add_column_name] = add_column
    #print pd_artists
    #print pd_artists.info()
    #pd_artists.to_pickle("artists.pd")
    return

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

def modeling(X_train, Y_train, X_test, Y_test, model=linear_model.Lasso(alpha=0.2), leg="LR"):
    ##get the median value first

    ##Trim the X_train and Y_train's highest value off
    # max_index = Y_train.index(max(Y_train))
    # del Y_train[max_index]
    # del X_train[max_index]
    # min_index = Y_train.index(min(Y_train))
    # del Y_train[min_index]
    # del X_train[min_index]
    ##this only selects the k variables
    # sel =  SelectKBest(chi2, k=2)
    # sel.fit(X_train, Y_train)
    # X_train = sel.transform(X_train)
    # X_test = sel.transform(X_test)
    #print sel.scores_
    # pca = PCA(n_components=1)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)

    #print kmeans.cluster_centers_ 
    ##modify Y_train

    Y_train = np.array(Y_train)
    # model.fit(X_train, Y_train)
    # model_pred = model.predict(X_test)
    
    ##cluster + fit
    # kmeans = KMeans(n_clusters=2, random_state=0).fit(Y_train.reshape(-1, 1))
    # Y_train_new = kmeans.labels_
    # model.fit(X_train, Y_train_new)
    # model_pred_new = model.predict(X_test)
    # model_pred = [(kmeans.cluster_centers_[i][0]) for i in model_pred_new]

    #model_pred = np.full(len(Y_test), (np.std(Y_train) + np.median(Y_train)))
    #model_pred = np.full(len(Y_test), (kmeans.cluster_centers_[0][0] + kmeans.cluster_centers_[1][0])/2.0)

    #print X_train, X_test
    # X_train_mean_play = np.array([i[-1] for i in X_train])
    # X_test_mean_play = np.array([i[-1] for i in X_test])
    # ratio_play = np.divide(Y_train, X_train_mean_play)
    # #ratio_play.delete(np.amin(ratio_play))
    # ratio = np.median(ratio_play)
    # ratio_2 =  np.median(ratio_play[1:])
    # ratio = (ratio + ratio_2) / 2.0
    # #ratio = math.floor(ratio) ##trick?
    # model_pred = X_test_mean_play * ratio



    ##terrible but whatever
    ratio = 1
    predict = np.median(np.array(Y_train))
    if len(Y_train) >= 5:
        if np.std(np.array(Y_train)) / np.median(np.array(Y_train)) < 1.0:
            predict = (np.median(np.array(Y_train)) + np.median(np.array(Y_train[1:])) + np.median(np.array(Y_train[:-2])))/3.0

    model_pred = np.full(len(Y_test), predict)

    #ratio = 1
    # median = np.median(Y_train)
    # mean = np.mean(Y_train)
    # mode = np.mode(np.around(Y_train, decimals=-(1)))
    # guess = (median + )
    model_median = np.full(len(Y_test), (np.median(np.array(Y_train))))

    #print X_train_mean_play, X_test_mean_play
    #print np.divide(Y_train, X_train_mean_play), ratio
    # print kmeans.cluster_centers_
    #print Y_train, np.median(np.array(Y_train))
    # print Y_train_new, model_pred
    #print Y_test, model_pred
    #print (leg + " MAE: " + str(MAE(model_pred, Y_test)) + " : --- %s seconds ---" % (time.time() - analysis_time))
    #print (leg + " median: " + str(MAE(model_median, Y_test)))
    
    ##bad practice
    # if MAE(model_median, Y_test) < MAE(model_pred, Y_test):
    #     ratio = 0
    #     model_pred = model_median
    
    # RF = RandomForestRegressor()
    # RF.fit(X_train, Y_train)
    # RF_pred = RF.predict(X_test)
    
    #print len(model_pred), len(Y_test)
    # plt.plot(model_pred, Y_test, 'o', label='correlations')
    # plt.xlabel("prediction")
    # plt.ylabel("observation")
    # rms = np.sqrt(MSE(model_pred, Y_test))
    # print leg, ": RMSE is--", rms/1000.0
    # plt.title(leg + ': RMSE:%.3f' % (rms/1000.0))
    # plt.savefig('Plot/' + leg +'_pred.png', bbox_inches='tight')
    # plt.clf()
    return (model_pred, ratio)
    #return [MAE(model_median, Y_test), MAE(model_pred, Y_test), len(Y_test)]

def feature_vector(df, df_user):
    #sex, age, country
    #ID                      2000 non-null object
    # Value                  2000 non-null object
    # rating                 2000 non-null object
    # rating_votes           2000 non-null object
    # work-count             2000 non-null int64
    # release-group-count    2000 non-null int64
    # area                   2000 non-null object
    # life-span              2000 non-null object
    # life-span_end          2000 non-null object
    # alias-count            2000 non-null int64
    # gender                 2000 non-null object
    # country                2000 non-null object
    # type                   2000 non-null object
    # release-count          2000 non-null int64
    # recording-count        2000 non-null int64
    # tag-list               2000 non-null object
    feature = []
    # feature.append(int(float(df["rating"]) * 10))
    # feature.append(int(df["rating_votes"]))
    # feature.append(int(df["work-count"]))
    # feature.append(int(df["release-group-count"]))
    # feature.append(int(df["alias-count"]))
    # feature.append(int(df["release-count"]))
    # feature.append(int(df["recording-count"]))

    #print df["life-span"], type(df["life-span"])
    #feature.append(int(df["life-span"]["life-span"]))
    #try:
    #print int(str(df["life-span"]).split()[1].split("-")[0])
    #print "aha"

    #feature.append((int(str(df["life-span"]).split()[1].split("-")[0]) - 1900)%10)## artist's starting age
    #feature.append(int(bool(str(df_user[2]) in str(df["area"])))) ##if the user and the viewer as the same area
    #feature.append(int(str(df["life-span"]).split()[1].split("-")[0]))## artist's starting age
    #feature.append(int(bool(str(df_user[2]) in str(df["area"])))) ##if the user and the viewer as the same area
    # ##type vector
    # feature.append(1 if "rock"  in str(df["tag-list"]) else 0)
    # feature.append(1 if "pop"   in str(df["tag-list"]) else 0)
    # feature.append(1 if "class" in str(df["tag-list"]) else 0)
    # feature.append(1 if "hop"   in str(df["tag-list"]) else 0)
    # feature.append(1 if "elec"  in str(df["tag-list"]) else 0)
    # feature.append(1 if "jazz"  in str(df["tag-list"]) else 0)
    # feature.append(1 if "metal"  in str(df["tag-list"]) else 0)

    ##this is the play information from the training data
    # for key, value in df["gender"].iteritems():
    #     genders = str(value).split(";")
    #     feature.append(genders.count("Male"))
    #     feature.append(genders.count("Female"))

    # for key, value in df["plays"].iteritems():
    #     plays = np.array(map(int, value))

    # feature.append(np.sum(plays))
    # feature.append(len(plays))
    # feature.append(max(plays))
    # feature.append(min(plays))
    # feature.append(np.std(plays))
    # feature.append(np.mean(plays))
    # feature.append(np.median(plays))

    def reject_outliers(data, m=2):
        return data[abs(data - np.median(data)) < m * np.std(data)]

    code = user_number(df_user)
    
    for key, value in df.iteritems(): ##no longer need info
        #print value
        feature.append(value)

    # for key, value in df["Info"].iteritems():
    #     plays = np.array(map(int, value[code]))
    #     allplays =  np.array(map(int, value["all"]))

    #plays = reject_outliers(plays)
    #save in case
    # if plays.size == 0:
    #     plays = allplays

    # feature.append(np.median(plays))
    return feature



def analysis():
    #add_artfeature()
    # return

    global analysis_time
    analysis_time = time.time()
    NPartdata = int(1000) #takes 1sec to load; 1% of the total size

    #Load the training data.
    info_list = ["recordings", "releases", "release-groups", "works", "various-artists", "discids", "media", "isrcs", "aliases", "annotation", "area-rels", "artist-rels", "label-rels", "place-rels", "event-rels", "recording-rels", "release-rels", "release-group-rels", "series-rels", "url-rels", "work-rels", "instrument-rels", "tags", "user-tags", "ratings", "user-ratings"]

    # train_data = {}
    # with open("artists.csv", 'r') as train_fh:
    #     train_csv = csv.reader(train_fh, delimiter=',', quotechar='"')
    #     next(train_csv, None)
    #     #print row_count
    #     for i, row in enumerate(train_csv):
    #         if (i%20 == 0):
    #             util.drawProgressBar(i/(2000 * 1.0), barLen = 20)
    #         #print row
    #         user   = row[0]
    #         try:
    #             info   = musicbrainzngs.get_artist_by_id(str(user), includes=info_list)
    #         except musicbrainzngs.musicbrainz.ResponseError:
    #             print "cannot find", row
    #             info   = {}
    #         #print user, info
    #         #artist = row[1]
    #         #plays  = int(row[2])
    #         if not user in train_data:
    #             train_data[user] = {}
    #         train_data[user] = info
    # #save as a txt
    # with open("artists.txt", "w") as f:
    #     json.dump(train_data, f)

    #load the dictionary
    # with open("train.txt", "r") as f:
    #     train_data = json.load(f)
    # with open("profiles.txt", "r") as f:
    #     profiles_data = json.load(f)
    # with open("artists.txt", "r") as f:
    #     artists_data = json.load(f)

    # ##merge features into panda dictionay
    # pd_profiles = pd.DataFrame(profiles_data.items(), columns=['ID', 'Value'])
    # pd_train = pd.DataFrame(train_data.items(), columns=['ID', 'Obs'])
    # pd_mergetrain = pd.merge(pd_profiles, pd_train, on='ID')
    # pd_artists = pd.DataFrame(artists_data.items(), columns=['ID', 'Value'])


    ##load the artist framework
    #pd_artists = pd.read_pickle("artists.pd")
    #pd_artists = pd.read_pickle("new_artmedian.pd") ## this is the only median
    #pd_artists = pd.read_pickle("new_art.pd") ## this is the full list
    #pd_artists = pd.read_pickle("new_artmedian_rating.pd") ##this is the current one
    #dic_artists = pd_artists.set_index("ID")["Info"].to_dict()
    pd_mergetrain = pd.read_pickle("train.pd")
    with open("new_artmedian_rating.txt", "r") as f:
        dic_artists = json.load(f)

    ## make the artist framework
    ##add_artfeature()
    for key, value in dic_artists.iteritems():
        for code, code_value in value.iteritems():
            #print code, code_value
            dic_artists[key][code] = np.median(np.array(map(float, code_value)))
            # if code == "all":
            #     print dic_artists[key][code]
    
    ##reverse engineer
    total_MAE = [0, 0, 0]

    test_truth_Y = []
    test_median_Y = []
    test_model_Y = []

    user_ratio = []
    for i, obs in enumerate(pd_mergetrain["Obs"]):
        # this_listen = 0
        # for artist_id, n_listen in obs.items():
        #     this_listen += int(n_listen)
        # total_listen.append(this_listen)

        #total_listen.append(len (obs))
        if i < 233286.0:
            train_Y = []
            test_Y = []
            train_X = []
            test_X = []
            user_Y = []

            if (i%100 == 0):
                util.drawProgressBar(i/233286.0)
            train_size = int(len (obs) * 0.9)
            test_size  = len (obs) - train_size
            #print pd_mergetrain["Value"][i] ##this is the X_vector; sex, age, country
            count_size = 0
            total_listen = 0 ##get the normalization
            for artist_id, n_listen in obs.items():
                total_listen += n_listen

            for artist_id, n_listen in obs.items():
                if (count_size < train_size):
                    train_Y.append(int(n_listen))
                    user_Y.append(int(n_listen))
                    #train_Y.append(int(n_listen/(total_listen * 1.0) * 100))
                    train_X.append(feature_vector(dic_artists[artist_id], pd_mergetrain["Value"][i]))
                else:
                    #train_Y.append(int(n_listen))
                    #user_Y.append(int(n_listen))
                    #train_Y.append(int(n_listen/(total_listen * 1.0) * 100))
                    #train_X.append(feature_vector(pd_artists[pd_artists['ID'] == artist_id], pd_mergetrain["Value"][i]))

                    test_Y.append(int(n_listen))
                    test_truth_Y.append(int(n_listen))
                    #test_Y.append(int(n_listen/(total_listen * 1.0) * 100))
                    test_X.append(feature_vector(dic_artists[artist_id], pd_mergetrain["Value"][i]))
                count_size += 1

            #print train_Y, test_Y
            #print train_X, test_X
            #result =  modeling(train_X, train_Y, test_X, test_Y)
            #result =  modeling(train_X, train_Y, test_X, test_Y, model=LinearRegression(), leg="linear")
            #result  = modeling(train_X, train_Y, test_X, test_Y, model=DecisionTreeRegressor(), leg="DT")
            #result  = modeling(train_X, train_Y, test_X, test_Y, model=KNeighborsRegressor(n_neighbors=3, weights='distance'), leg="KNN")
            result  = modeling(train_X, train_Y, test_X, test_Y, model=KNeighborsClassifier(n_neighbors=2), leg="KNN")
            #
            #result  = modeling(train_X, train_Y, test_X, test_Y, model=RandomForestRegressor(), leg="RFR")
            
            for j in range(test_size):
                test_median_Y.append(np.median(np.array(user_Y)))
            for k in result[0]:
                test_model_Y.append(k)
            
            user_ratio.append(result[1])
            #total_MAE[0] += result[0]
            #total_MAE[1] += result[1]
            #total_MAE[2] += result[2]

                # try:
                #     print pd_artists[pd_artists['ID'] == artist_id]

                # except KeyError:
                #     pass

            if (i%1000 == 0):
                print "median: ", MAE(test_median_Y, test_truth_Y), " model: ", MAE(test_model_Y, test_truth_Y)
        else:
            break



    #result  = modeling(train_X, train_Y, test_X, test_Y, model=DecisionTreeClassifier(max_depth=20), leg="DT")

    print "median: ", MAE(test_median_Y, test_truth_Y), " model: ", MAE(test_model_Y, test_truth_Y)
    #pd_mergetrain["ratio"] = user_ratio
    #pd_mergetrain.to_pickle("newtrain.pd")
    
    #result = modeling(train_X, train_Y, test_X, test_Y, model=linear_model.Lasso(alpha=0.2), leg="DT")
    #result = modeling(train_X, train_Y, test_X, test_Y, model=DecisionTreeRegressor(max_depth=10), leg="DT")
    #result  = modeling(train_X, train_Y, test_X, test_Y, model=KNeighborsRegressor(n_neighbors=10, weights='distance'), leg="KNN")
    
    #print "MAE median is: ", total_MAE[0]/total_MAE[2], " method: ", total_MAE[1]/total_MAE[2]

    #check n listen
    #util.makehst(x=total_listen, label="n music listened", xlabel="N artist", ylabel="count", plotname="Nartist_count")

    # Compute the global median.
    #plays_array = []
    # for artist, artist_data in artists_data.iteritems():
    #     print artist, artist_data

    # print artists_data.keys()[0], artists_data.values()[0].keys()
    # for item in artists_data.values()[0].values():
    #     for key, value in item.iteritems():
    #         print key, value



    # # ##iterate and compute
    # for user, user_data in train_data.iteritems():
    #     for artist, plays in user_data.iteritems():
    #         rate = []
    #         try:
    #             info = artists_data[artist]["artist"]
    #             try:
    #                 #print artists_data[artist]["artist"]["rating"]["rating"]
    #                 rating = float(artists_data[artist]["artist"]["rating"]["rating"])
    #                 #votes = float(artists_data[artist]["artist"]["rating"]["votes-count"])
    #             except KeyError:
    #                 rating = 1
    #             rate.append(rating)
    #         except KeyError:
    #             rate = [1]
    #         plays_array.append(plays * np.mean(np.array(rate)))

    # global_median = np.median(np.array(plays_array))
    # print "global median:", global_median
    

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
    # musicbrainzngs.set_useragent(
    #     "python-musicbrainzngs-example",
    #     "0.1",
    #     "https://github.com/alastair/python-musicbrainzngs/",
    # )
    # musicbrainzngs.auth("tongbaojia", "password")
    main()
