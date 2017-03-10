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

import os
from collections import Counter
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import numpy as np
from scipy import sparse

import util


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
    for datafile in os.listdir(direc):
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
   

    X = sparse.csr_matrix((np.array(data),
                   (np.array(rows), np.array(cols))),
                   shape=(len(fds), len(feat_dict)))
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
    #initiate counters
    c = Counter()
    c_all = Counter()
    n_el = 0
    in_all_section = False
    for el in tree.iter():
        #keep track of all the kind of tags, and the total number of tags
        c_all["num_"+str(el.tag)] += 1
        n_el += 1        
        # ignore everything outside the "all_section" element
        if el.tag == "all_section" and not in_all_section:
            in_all_section = True
        elif el.tag == "all_section" and in_all_section:
            in_all_section = False
        elif in_all_section:
            c['num_system_calls'] += 1
    #calculate proportions for each tag, and merge everything in a dictionary      
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
        
    return c

## The following function does the feature extraction, learning, and prediction
def main():
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler  
    import matplotlib.pyplot as plt

    from sklearn import svm
    from sklearn.model_selection import train_test_split

    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier

    train_dir = "train"
    test_dir = "test"

    # TODO put the names of the feature functions you've defined above in this list
    ffs = [first_last_system_call_feats, system_call_count_feats]

    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    print "done extracting training features"
    print


    print "extracting test features..."
    X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
   
    #Try SVM--------------------------------------------------------------------
    X_subset_train, X_subset_test, t_subset_train, t_subset_test = train_test_split(X_train, t_train, test_size=0.2, random_state=1)

    kernel = 'rbf'
    tol = 1e-3
    #class_weight = 'balanced'

    scaler = StandardScaler(with_mean = False)
    scaler.fit(X_subset_train)
    X_train_scaled = scaler.transform(X_subset_train)
    clf = svm.SVC(verbose = False, kernel = kernel, tol = tol)
    clf.fit(X_train_scaled, t_subset_train)

    X_test_scaled = scaler.transform(X_subset_test)
    preds = clf.predict(X_test_scaled)

    #test predictions
    num_correct = np.sum(t_subset_test == preds);
    accuracy = num_correct/float(t_subset_test.size)
    print('SVM accuracy = '+str(accuracy))
    
    #Predict using SVM----------------------------------------------------------
    outputfile = 'SVM_preds.csv'
    scaler = StandardScaler(with_mean = False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    clf = svm.SVC(verbose = False, kernel = kernel, tol = tol)
    clf.fit(X_train_scaled, t_train)

    X_test_scaled = scaler.transform(X_test)
    preds = clf.predict(X_test_scaled)
    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

    #Try random forest
    X_subset_train, X_subset_test, t_subset_train, t_subset_test = train_test_split(X_train, t_train, test_size=0.2, random_state=1)
    scaler = StandardScaler(with_mean = False)
    scaler.fit(X_subset_train)
    X_train_scaled = scaler.transform(X_subset_train)
    clf = RandomForestClassifier(n_jobs=-1, n_estimators = 10000)# class_weight='balanced')
    #y, _ = pd.factorize(train['species'])
    clf.fit(X_train_scaled, t_subset_train)

    X_test_scaled = scaler.transform(X_subset_test)
    preds = clf.predict(X_test_scaled)
    accuracy = np.sum(t_subset_test == preds)/float(t_subset_test.size)
    print('RF accuracy = '+str(accuracy))
    
    #predict using MLP----------------------------------------------------------
    outputfile = "MLP_preds.csv" 

    solver = 'adam'
    #momentum = 0.9 #only is sgd
    #nesterovs_momentum = True # only if sgd

    max_iter =100
    batch_size = 200
    solver = 'adam'
    max_iter = 500


    alpha = 1e-5
    tol = 1e-12
    rate_init = 0.01
    learning_rate = 'adaptive' #only matters for sgd
    activation = 'relu'
    early_stopping = True
    #validation_fraction = 0.1
    #warm_start = False

    N_nodes = 25
    N_layers = 3

    layers = np.multiply(N_nodes, np.ones(N_layers)) 
    scaler = StandardScaler(with_mean = False)
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter, alpha=alpha,
                            solver=solver, verbose=10, tol=tol, random_state=1,
                            learning_rate_init=rate_init, activation = activation,
                       learning_rate = learning_rate, early_stopping=early_stopping,
                       #validation_fraction = validation_fraction, warm_start = warm_start, batch_size = batch_size, 
                       #momentum = momentum, nesterovs_momentum=nesterovs_momentum
                       )
    mlp.fit(X_train_scaled, t_train)

    X_test_scaled = scaler.transform(X_test)
    preds = mlp.predict(X_test_scaled)


    print "writing predictions..."
    util.write_predictions(preds, test_ids, outputfile)
    print "done!"

    
    #Try MLP------------------------------------------------------------------------
    solver = 'adam'
    max_iter = 100


    alpha = 1e-6
    tol = 1e-12
    rate_init = 0.01
    learning_rate = 'adaptive'
    early_stopping = True

    accuracy = []
    train_dir = "train_small"
    test_dir = "test_small"
    print "extracting training features..."
    X_train,global_feat_dict,t_train,train_ids = extract_feats(ffs, train_dir)
    print "done extracting training features"
    print


    print "extracting test features..."
    X_test,_,t_ignore,test_ids = extract_feats(ffs, test_dir, global_feat_dict=global_feat_dict)
    print "done extracting test features"
    print
    

    #Vary number of Nodes ----------------------------------------------------------
    N_nodes = np.linspace(1, 201, 41)
    N_layers = 3

    for nodes in N_nodes:
        layers = np.multiply(nodes, np.ones(N_layers)) 
        scaler = StandardScaler(with_mean = False)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter, alpha=alpha,
                            solver=solver, verbose=False, tol=tol, random_state=1,
                            learning_rate_init=rate_init, learning_rate = learning_rate,
                           early_stopping = early_stopping)
        mlp.fit(X_train_scaled, t_train)
        print "done learning"
        print

        X_test_scaled = scaler.transform(X_test)
        preds = mlp.predict(X_test_scaled)
        #test predictions
        num_correct = np.sum(t_ignore == preds);
        accuracy = np.append(accuracy,num_correct/float(t_ignore.size))
        #print('Accuracy  = '+str(accuracy))
    plt.clf()
    plt.plot(N_nodes, accuracy, linestyle='--', marker='o', color='b')
    #plt.title('N_nodes = '+ str(N_nodes)+'_N_layers = ' + str(N_layers))
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy')
    #plt.xlim((n_vals[0]-1, n_vals[-1]+1))
    #plt.ylim((min(RMSE), max(RMSE)))
    #plt.show()
    plt.savefig('Plots/'+'_N_nodes_'+ str(N_nodes[[1]])+'-'+str(N_nodes[[-1]])+'_N_layers' + str(N_layers)+'.pdf')

    #Vary number of layers in MLP-------------------------------------------------
    N_nodes = 50
    N_layers = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 18, 22, 26, 30])
    accuracy = []

    for nlayers in N_layers:

        layers = np.multiply(N_nodes, np.ones(nlayers)) 
        scaler = StandardScaler(with_mean = False)
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=max_iter, alpha=alpha,
                            solver=solver, verbose=False, tol=tol, random_state=1,
                            learning_rate_init=rate_init)
        mlp.fit(X_train_scaled, t_train)

        X_test_scaled = scaler.transform(X_test)
        preds = mlp.predict(X_test_scaled)

        #test predictions
        num_correct = np.sum(t_ignore == preds);
        accuracy = np.append(accuracy,num_correct/float(t_ignore.size))

    plt.clf()
    plt.plot(N_layers, accuracy, linestyle='--', marker='o', color='b')
    #plt.title('N_nodes = '+ str(N_nodes)+'_N_layers = ' + str(N_layers))
    plt.xlabel('Number of layers')
    plt.ylabel('Accuracy')
    plt.savefig('Plots/'+'_N_nodes_'+ str(N_nodes)+'_N_layers' + str(N_layers)+'.pdf')
               
                
if __name__ == "__main__":
    main()
    