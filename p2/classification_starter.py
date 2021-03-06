## This file provides starter code for extracting features from the xml files and
## for doing some learning.
##
## The basic set-up: 
## ----------------
## main() will run code to extract features, learn, and make predictions.
## 
## extract_feats() is called by main(), and it will iterate through the 
## train/test directories and parse each xml file into an xml.etree.ElementTree, 
## which is a standard python object used to represent an xml file in memory.
## (More information about xml.etree.ElementTree objects can be found here:
## http://docs.python.org/2/library/xml.etree.elementtree.html
## and here: http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/)
## It will then use a series of "feature-functions" that you will write/modify
## in order to extract dictionaries of features from each ElementTree object.
## Finally, it will produce an N x D sparse design matrix containing the union
## of the features contained in the dictionaries produced by your "feature-functions."
## This matrix can then be plugged into your learning algorithm.
##
## The learning and prediction parts of main() are largely left to you, though
## it does contain code that randomly picks class-specific weights and predicts
## the class with the weights that give the highest score. If your prediction
## algorithm involves class-specific weights, you should, of course, learn 
## these class-specific weights in a more intelligent way.
##
## Feature-functions:
## --------------------
## "feature-functions" are functions that take an ElementTree object representing
## an xml file (which contains, among other things, the sequence of system calls a
## piece of potential malware has made), and returns a dictionary mapping feature names to 
## their respective numeric values. 
## For instance, a simple feature-function might map a system call history to the
## dictionary {'first_call-load_image': 1}. This is a boolean feature indicating
## whether the first system call made by the executable was 'load_image'. 
## Real-valued or count-based features can of course also be defined in this way. 
## Because this feature-function will be run over ElementTree objects for each 
## software execution history instance, we will have the (different)
## feature values of this feature for each history, and these values will make up 
## one of the columns in our final design matrix.
## Of course, multiple features can be defined within a single dictionary, and in
## the end all the dictionaries returned by feature functions (for a particular
## training example) will be unioned, so we can collect all the feature values 
## associated with that particular instance.
##
## Two example feature-functions, first_last_system_call_feats() and 
## system_call_count_feats(), are defined below.
## The first of these functions indicates what the first and last system-calls 
## made by an executable are, and the second records the total number of system
## calls made by an executable.
##
## What you need to do:
## --------------------
## 1. Write new feature-functions (or modify the example feature-functions) to
## extract useful features for this prediction task.
## 2. Implement an algorithm to learn from the design matrix produced, and to
## make predictions on unseen data. Naive code for these two steps is provided
## below, and marked by TODOs.
##
## Computational Caveat
## --------------------
## Because the biggest of any of the xml files is only around 35MB, the code below 
## will parse an entire xml file and store it in memory, compute features, and
## then get rid of it before parsing the next one. Storing the biggest of the files 
## in memory should require at most 200MB or so, which should be no problem for
## reasonably modern laptops. If this is too much, however, you can lower the
## memory requirement by using ElementTree.iterparse(), which does parsing in
## a streaming way. See http://eli.thegreenplace.net/2012/03/15/processing-xml-in-python-with-elementtree/
## for an example. 

import os, time, argparse
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt

import util
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, precision_score


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--maketrain",  action="store_true")
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--final",    action="store_true")
    return parser.parse_args()

def split_train(indir="train_full", outdir="train_sort"):
    """split the training dataset into subfolders based on their type"""
    for datafile in os.listdir(indir):
        id_str,clazz = datafile.split('.')[:2]
        util.checkpath(outdir + "/" + clazz)
        os.system("cp " + indir + "/" + datafile + " " + outdir + "/" + clazz)
    print "done!!"

def make_train_test(indir="train_sort", out_train="train", out_test="test"):
    total_train_frac = 0.8 ##this defines the total training datas will be used
    total_test_frac  = 0.25 *  total_train_frac ##this defines the total training datas will be used
    ##Notice the smallest sample has only 21, 
    ##This means if each type is given the same weight, the total size should be 21 * 15 = 320
    ##clear the current train and test samples
    os.system("rm " + out_train + "/*")
    os.system("rm " + out_test + "/*")
    """making the training and testing samples"""
    ##check distributions
    n_clazz = []
    y_nclass = []
    for clazz in os.listdir(indir):
        print clazz, len([f for f in os.listdir(indir + "/" + clazz)
                if os.path.isfile(os.path.join(indir + "/" + clazz, f))])
        n_clazz.append(len([f for f in os.listdir(indir + "/" + clazz)
                if os.path.isfile(os.path.join(indir + "/" + clazz, f))]))
        y_nclass.append(clazz)
    #util.maketypeplot(n_clazz, y_nclass, ylabel="number of samples", title="Nsample", plotname="Nsample")
    print n_clazz#invert the weight
    invert_weight = [1/(i/(sum(n_clazz) * 1.0)) for i in n_clazz] 
    print invert_weight
    weight_dic = {}
    for j, clazz in enumerate(util.malware_classes):
        weight_dic[j] = invert_weight[y_nclass.index(clazz)]
    print weight_dic

    Min_train = int( min(n_clazz) * total_train_frac )
    Min_test  = int( min(n_clazz) * total_test_frac )

    for clazz in os.listdir(indir):
        ##move the files to train and test directory!
        train_count = 0
        test_count = 0
        Min_train = int( len([f for f in os.listdir(indir + "/" + clazz)
                if os.path.isfile(os.path.join(indir + "/" + clazz, f))]) / (min(n_clazz) * 0.1) * total_train_frac)
        Min_test = int( len([f for f in os.listdir(indir + "/" + clazz)
                if os.path.isfile(os.path.join(indir + "/" + clazz, f))]) / (min(n_clazz) * 0.1) * total_test_frac)
        print clazz, Min_train, Min_test
        for f in os.listdir(indir + "/" + clazz):
            if train_count <= Min_train:
                os.system("cp " + indir + "/" + clazz + "/" + f + " " + out_train + "/.")
                train_count += 1
            elif test_count <= Min_test:
                os.system("cp " + indir + "/" + clazz + "/" + f + " " + out_test + "/.")
                test_count += 1
            else:
                pass
    print "done making training and testing samples!!"


def extract_feats(ffs, direc="train", global_feat_dict=None):
    """
    arguments:
      ffs are a list of feature-functions.
      direc is a directory containing xml files (expected to be train or test).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.

    returns: 
      a sparse design matrix, a dict mapping features to column-numbers,
      a vector of target classes, and a list of system-call-history ids in order 
      of their rows in the design matrix.
      
      Note: the vector of target classes returned will contain the true indices of the
      target classes on the training data, but will contain only -1's on the test
      data
    """
    fds = [] # list of feature dicts
    classes = []
    ids = [] 
    N_total = len([f for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))])
    N_current = 0
    for datafile in os.listdir(direc):
        N_current += 1
        if (N_current%100 == 0):
            util.drawProgressBar(N_current/(N_total*1.0))
        # extract id and true class (if available) from filename
        id_str,clazz = datafile.split('.')[:2]
        ids.append(id_str)
        # add target class if this is training data
        try:
            classes.append(util.malware_classes.index(clazz))
        except ValueError:
            # we should only fail to find the label in our list of malware classes
            # if this is test data, which always has an "X" label
            assert clazz == "X"
            classes.append(-1)
        rowfd = {}
        # parse file as an xml document
        tree = ET.parse(os.path.join(direc,datafile))
        # accumulate features
        [rowfd.update(ff(tree)) for ff in ffs]
        fds.append(rowfd)
        
    X,feat_dict = make_design_mat(fds,global_feat_dict)
    return X, feat_dict, np.array(classes), ids


def make_design_mat(fds, global_feat_dict=None):
    """
    arguments:
      fds is a list of feature dicts (one for each row).
      global_feat_dict is a dictionary mapping feature_names to column-numbers; it
      should only be provided when extracting features from test data, so that 
      the columns of the test matrix align correctly.
       
    returns: 
        a sparse NxD design matrix, where N == len(fds) and D is the number of
        the union of features defined in any of the fds 
    """
    if global_feat_dict is None:
        all_feats = set()
        [all_feats.update(fd.keys()) for fd in fds]
        feat_dict = dict([(feat, i) for i, feat in enumerate(sorted(all_feats))])
    else:
        feat_dict = global_feat_dict
        
    cols = []
    rows = []
    data = []        
    for i in xrange(len(fds)):
        temp_cols = []
        temp_data = []
        for feat,val in fds[i].iteritems():
            try:
                # update temp_cols iff update temp_data
                temp_cols.append(feat_dict[feat])
                temp_data.append(val)
            except KeyError as ex:
                if global_feat_dict is not None:
                    pass  # new feature in test data; nbd
                else:
                    raise ex

        # all fd's features in the same row
        k = len(temp_cols)
        cols.extend(temp_cols)
        data.extend(temp_data)
        rows.extend([i]*k)

    assert len(cols) == len(rows) and len(rows) == len(data)
   
    print np.array(data), (np.array(rows), np.array(cols)), (len(fds), len(feat_dict))
    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
    #unzip the matrix
    #X = sparse.csr_matrix.todense(X)
    return X, feat_dict
    

## Here are two example feature-functions. They each take an xml.etree.ElementTree object, 
# (i.e., the result of parsing an xml file) and returns a dictionary mapping 
# feature-names to numeric values.
## TODO: modify these functions, and/or add new ones.
def first_last_system_call_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'first_call-x' to 1 if x was the first system call
      made, and 'last_call-y' to 1 if y was the last system call made. 
      (in other words, it returns a dictionary indicating what the first and 
      last system calls made by an executable were.)
    """
    c = Counter()
    in_all_section = False
    first = True # is this the first system call
    last_call = None # keep track of last call we've seen
    for el in tree.iter():
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            if first:
                c["first_call-"+el.tag] = 1
                first = False
            last_call = el.tag  # update last call seen
            
    # finally, mark last call seen
    c["last_call-"+last_call] = 1
    return c

def system_call_count_feats(tree):
    """
    arguments:
      tree is an xml.etree.ElementTree object
    returns:
      a dictionary mapping 'num_system_calls' to the number of system_calls
      made by an executable (summed over all processes)
    """
    c = Counter()
    c_all = Counter()
    n_el = 0
    in_all_section = False
    for el in tree.iter():
        #print el
        c_all["num_"+str(el.tag)] += 1
        n_el += 1 
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    #calculate proportions for each tag, and merge everything in a dictionary; from Alan      
    for key, val in c_all.items():
        c[key] = val
        c["ratio-"+key] = float(val)/n_el
    c["n_el"] = n_el
    if c['num_processes'] != 0:
        c["ratio-threads-processes"] = float(c['num_threads'])/c['num_processes']        
    if c['num_sections'] != 0:
        c["ratio-system_calls-sections"] = float(c['num_system_calls'])/c['num_sections']
    if c['num_processes'] != 0:
        c["ratio-system_calls-processes"] = float(c['num_system_calls'])/c['num_processes']
    ##finish
    return c

def process_type(tree):
    c = Counter()
    c_ch = Counter()
    ##try root
    root = tree.getroot()
    n_ch = 0
    for child in root:
        #print child.tag, child.attrib
        #print child.attrib, type(child.attrib)
        for key, val in child.attrib.items():
            c_ch["proc_" + str(key)] += 1
        n_ch += 1 
    #calculate proportions for each tag, and merge everything in a dictionary; from Alan      
    for key, val in c_ch.items():
        #print key, val
        c[key] = val
        c["proc_"+key] = float(val)/n_ch
    # c["n_el"] = n_el
    # if c['num_processes'] != 0:
    #     c["ratio-threads-processes"] = float(c['num_threads'])/c['num_processes']        
    # if c['num_sections'] != 0:
    #     c["ratio-system_calls-sections"] = float(c['num_system_calls'])/c['num_sections']
    # if c['num_processes'] != 0:
    #     c["ratio-system_calls-processes"] = float(c['num_system_calls'])/c['num_processes']
    ##finish
    return c


## The following function does the feature extraction, learning, and prediction
def main():
    global ops
    ops = options()
    start_time = time.time()
    train_dir = "train"
    test_dir = "test"
    outputfile = "mypredictions.csv"  # feel free to change this or take it as an argument
    ##only need to call once
    ##split_train()
    ##for each time, make the training set
    if (ops.maketrain):
        #split_train()
        make_train_test()
        return 0
    if (ops.train):

        # TODO put the names of the feature functions you've defined above in this list
        ffs = [first_last_system_call_feats, system_call_count_feats]
        
        # extract features
        print "extracting training features..."
        X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
        print "done extracting training features"
        #print "X_train", X_train
        print "global_feat_dict", global_feat_dict
        ##print "t_train", t_train ##these are the Y vectors
        ##print "train_ids", train_ids ## these are the Y ids
        print "extracting test features..."
        X_test,test_feat_dict,t_test,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
        print "done extracting test features"
        
        # TODO train here, and learn your classification parameters
        print "learning..."
        learned_W = np.random.random((len(global_feat_dict),len(util.malware_classes)))
        print "done learning"
        print

        ##need to add in weights in the classifier
        # class_weight = {}
        # for i in range(15):
        #     class_weight[i] = 1
        class_weight = {0: 27.07017543859649, 1: 61.720000000000006, 2: 83.4054054054054, 3: 96.4375, 4: 75.26829268292683, 5: 79.12820512820512, 6: 58.22641509433962, 7: 75.26829268292683, 8: 1.917961466749534, 9: 146.95238095238096, 10: 5.693726937269372, 11: 96.4375, 12: 8.207446808510637, 13: 52.30508474576271, 14: 77.15}

        ##list of classifiers and their names
        para = [5, 10, 20, 30, 50, 70, 100, 120, 150, 200]
        score = []
        # classifiers = {
        #     #"Nearest_Neighbors": KNeighborsClassifier(3),
        #     #"Linear_SVM": SVC(kernel="linear", C=0.025),
        #     #"RBF_SVM": SVC(gamma=2, C=1),
        #     #"Gaussian_Process": GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
        #     #"Decision_Tree_40" : DecisionTreeClassifier(max_depth=40),
        #     "Decision_Tree_5" : DecisionTreeClassifier(max_depth=5),
        #     "Decision_Tree_7" : DecisionTreeClassifier(max_depth=7),
        #     "Decision_Tree_10" : DecisionTreeClassifier(max_depth=10),
        #     "Decision_Tree_20" : DecisionTreeClassifier(max_depth=20),##optimal
        #     "Decision_Tree_30" : DecisionTreeClassifier(max_depth=30),
        #     "Decision_Tree_40" : DecisionTreeClassifier(max_depth=40),
        #     "Decision_Tree_50" : DecisionTreeClassifier(max_depth=50),##optimal
        #     "Decision_Tree_60" : DecisionTreeClassifier(max_depth=60),
        #     #"Decision_Tree_50_w" : DecisionTreeClassifier(max_depth=50),##optimal
        #     #"Decision_Tree_60" : DecisionTreeClassifier(max_depth=60),
        #     #"Random_Forest"   : RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1, class_weight=class_weight),
        #     #"Random_Forest_2" : RandomForestClassifier(max_depth=20, n_estimators=500, max_features=2),
        #     #"Random_Forest_3" : RandomForestClassifier(max_depth=30, n_estimators=1000, max_features=3),
        #     #"Random_Forest_4" : RandomForestClassifier(max_depth=40, n_estimators=1000, max_features=4),
        #     #"Neural_Net" : MLPClassifier(alpha=1),
        #     #"AdaBoost":AdaBoostClassifier(),
        #     #"Naive_Bayes": GaussianNB(),
        #     #"QDA": QuadraticDiscriminantAnalysis(),
        #     }
        classifiers = {}
        for i in para:
            classifiers["DecisionTree_" + str(i)] = DecisionTreeClassifier(max_depth=10, max_features=5, max_leaf_nodes=i)


        print "truth: ", t_test
        for i in para:
            name = "DecisionTree_" + str(i)
            clf =  classifiers[name]
            clf.fit(X_train, t_train)
            t_test_prediction = clf.predict(X_test)
            acc_score = accuracy_score(t_test_prediction, t_test)
            #pre_score = precision_score(Y_test, t_test)
            print name, " prediction: ", t_test_prediction
            print name, " accuracy_score: ", acc_score #" precision_score: ", pre_score
            score.append(acc_score)
        
        # get rid of training data and load test data
        del X_train
        del t_train
        del train_ids
        del X_test
        del t_test
        del test_ids
    

    util.makeplot(para, score, label="DT, feature", xlabel="Decision Tree Leaf Nodes", ylabel="Acc", plotname="DT_leafnodes")

    # TODO make predictions on text data and write them out; dumb value; comment out
    #print "making predictions..."
    #preds = np.argmax(X_test.dot(learned_W),axis=1)
    #print "done making predictions"
    #print preds
    ##for testing
    #print "Accuracy is: ", np.sum(np.equal(preds, t_test))/(len(preds) * 1.0)

    
    ##for finalizing
    if ops.final:
        X_test,test_feat_dict,t_test,test_ids = extract_feats(ffs, "test_full", global_feat_dict=global_feat_dict)
        preds = classifiers["Random_Forest_4"].predict(X_test)
        print preds
        print "writing predictions..."
        util.write_predictions(preds, test_ids, outputfile)
    ##end of finalizing

    print "done!"
    print("--- %s seconds ---" % (time.time() - start_time))


if __name__ == "__main__":
    main()
    