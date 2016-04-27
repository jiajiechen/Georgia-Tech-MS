import os
import csv
import numpy as np
import scipy.stats as st
import random
import math
from time import time

from CSE6242HW4Tester import generateSubmissionFile


"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

myname = "Jiajie-Chen"
os.chdir("/Users/jiajiechen/Desktop/CSE6242-HW4/hw4-skeleton/Task1")


class RandomForest(object):

    class __DecisionTree(object):
        class decisionnode:
            def __init__(self,col=-1,value=None,results=None,tb=None,fb=None):
                self.col=col
                self.value=value
                self.results=results
                self.tb=tb
                self.fb=fb

        def __init__(self,tree=None,depth=0,max_depth=None,
                     splitSearchPct=0.5,n_random_feature=None,
                     multi_class=False,threshold=0.5):
            self.tree = tree
            self.depth = depth
            self.max_depth = max_depth
            self.splitSearchPct = splitSearchPct
            self.n_random_feature = n_random_feature
            self.threshold = threshold
            self.multi_class = multi_class


        def entropy(self, attribute_data):
            if self.multi_class:
                _, val_freqs = np.unique(attribute_data, return_counts=True)
                val_probs = 1.0*val_freqs / len(attribute_data)
                return -val_probs.dot(np.log(val_probs,2))
            else: #this runs faster for binary
                p = float(sum(attribute_data))/len(attribute_data)
                if p == 0: return 0.0
                elif p == 1: return 0.0
                else: return -p*math.log(p,2)-(1-p)*math.log(1-p,2)

        def get_count_dict(self, attribute_data):
            attribute_data_values, attribute_data_freqs \
              = np.unique(attribute_data, return_counts=True)
            return dict(zip(attribute_data_values, attribute_data_freqs))

        def divideset(self, data, feature, splitPoint):
            if isinstance(splitPoint,int) or isinstance(splitPoint,float):
                set1=data[data[:,feature]>splitPoint]
                set2=data[data[:,feature]<=splitPoint]
            else:
                set1=data[data[:,feature]==splitPoint]
                set2=data[data[:,feature]==splitPoint]
            return (set1,set2)

        def learn(self, X, y):
            data = np.column_stack((X,y))
            n_row, n_col = data.shape
            #set num of feature to sample
            self.n_random_feature=int(math.log(n_col,2))+1

            label = data[:,-1]
            if n_row == 0: return self.decisionnode()
            current_score = self.entropy(label)

            best_gain=0.0
            best_criteria=None
            best_sets=None

            #It's -1 because the last one is the target attribute and it does not count.
            column_count=n_col-1

            for col in sorted(random.sample(xrange(column_count), 
                                            self.n_random_feature)):
                column_values = self.get_count_dict(data[:,col])
                keys = sorted(column_values.keys())
                # if too manys keys to iter, sample%
                if len(keys) > 50:
                    n_smpl_keys = int(self.splitSearchPct*len(keys))
                    #get sorted sample
                    smpl_keys = [keys[i] for i in sorted(random.sample(xrange(len(keys)), n_smpl_keys))]
                else: smpl_keys = keys
                for value in smpl_keys:
                    (set1, set2)=self.divideset(data, col, value)
                    if set1.shape[0]>0 and set2.shape[0]>0:
                        p = float(set1.shape[0]) / n_row
                        gain = current_score-p*self.entropy(set1[:,-1])-\
                          (1-p)*self.entropy(set2[:,-1])
                        if gain>best_gain:
                            best_gain = gain
                            best_criteria = (col,value)
                            best_sets = (set1,set2)

            if best_gain>0:
                self.depth += 1
                LeftBranch=self.learn(best_sets[0][:,:-1], best_sets[0][:,-1])
                RightBranch=self.learn(best_sets[1][:,:-1], best_sets[1][:,-1])
                return self.decisionnode(col=best_criteria[0],
                                         value=best_criteria[1],
                                         tb=LeftBranch,
                                         fb=RightBranch)
            else:
                if self.multi_class: 
                    result = self.get_count_dict(data[:,-1])
                    return self.decisionnode(results=max(result, 
                                             key=lambda key: result[key]))
                else:
                    return self.decisionnode(results=int(np.mean(data[:,-1])>self.threshold))

        def classify(self,test_instance,tree):
            if tree.results!=None:
                return tree.results
            else:
                v=test_instance[tree.col]
                branch=None
                if isinstance(v,int) or isinstance(v,float):
                    if v>=tree.value: branch=tree.tb
                    else: branch=tree.fb
                else:
                    if v==tree.value: branch=tree.tb
                    else: branch=tree.fb
                return self.classify(test_instance,branch)

        def tree_predict(self, X_test, tree):
            predicted_y = []

            for i in xrange(X_test.shape[0]):
                predicted_y.append(self.classify(X_test[i,:], tree))

            return predicted_y

        def tree_accuracy(self, predicted_y, true_y):
            return float(sum(np.array(predicted_y) == true_y)) / len(predicted_y)

###############################################################################

    decision_trees = []

    def __init__(self, num_trees, verbose=False):
        self.num_trees = num_trees
        self.decision_trees = []
        self.verbose = verbose
    
    def bootstrap(self, data):
        n_sample = data.shape[0]
        a = np.array(range(0, n_sample))
        flag = np.random.choice(a, size=n_sample, replace=True, p=None)
        return data[flag,:]

    def fit(self, X, y):
        data = np.column_stack((X,y))
        for i in xrange(self.num_trees):
            random.seed(time())
            sample = self.bootstrap(data)
            t0 = time()
            clf = self.__DecisionTree()
            clf.tree = clf.learn(sample[:,:-1], sample[:,-1])
            self.decision_trees.append(clf)
            t1 = time()
            if self.verbose: 
                print "Fitting Tree", i
                print "Time elapsed %.4f s" % (t1-t0)


    def predict(self, X):
        y = np.array([], dtype = int)
        for instance in X:
            votes = np.array([int(decision_tree.classify(instance,\
                decision_tree.tree)) for decision_tree in self.decision_trees])
            counts = np.bincount(votes)
            y = np.append(y, np.argmax(counts))
        return y


def main(cv=False,kaggle=True, num_Trees=10):
    X = []
    y = []

    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)

        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])

    X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)

    # Split training/test sets
    # You need to modify the following code for cross validation
    if cv == True:
        K = 10
        cv_accuracy =[]
        for ii in xrange(K):
            X_train = np.array([x for i, x in enumerate(X) if i % K != ii],
                                dtype = float)
            y_train = np.array([z for i, z in enumerate(y) if i % K != ii],
                                dtype = int)
            X_test  = np.array([x for i, x in enumerate(X) if i % K == ii],
                                dtype = float)
            y_test  = np.array([z for i, z in enumerate(y) if i % K == ii],
                                dtype = int)

            randomForest = RandomForest(num_Trees)
            t0 = time()
            randomForest.fit(X_train, y_train)
            t1 = time()
            print "time elapses = %.3f s" % (t1-t0)

            y_predicted = randomForest.predict(X_test)

            results = [prediction == truth for prediction,
                       truth in zip(y_predicted, y_test)]

            # Accuracy
            accuracy = float(results.count(True)) / float(len(results))
            print "test accuracy: %.4f" % accuracy
            cv_accuracy.append(accuracy)
        print "average cv accuracy: %.4f" % np.mean(cv_accuracy)

    elif kaggle==True:
        K = 10
        ii = 3
        X_train = np.array([x for i, x in enumerate(X) if i % K != ii],
                            dtype = float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != ii],
                            dtype = int)
        X_test  = np.array([x for i, x in enumerate(X) if i % K == ii],
                            dtype = float)
        y_test  = np.array([z for i, z in enumerate(y) if i % K == ii],
                            dtype = int)
        
        randomForest = RandomForest(num_Trees, verbose=True)

        t0 = time()
        randomForest.fit(X_train,y_train)
        t1 = time()
        print "time elapses = %.3f s" % (t1-t0)

        y_predicted = randomForest.predict(X_test)
        results = [prediction == truth for prediction,truth in zip(y_predicted,
                                                                   y_test)]
        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        print "test accuracy: %.4f" % accuracy
        generateSubmissionFile(myname, randomForest)

    else:
        K = 10
        ii = 2
        X_train = np.array([x for i, x in enumerate(X) if i % K != ii],
                            dtype = float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != ii],
                            dtype = int)
        X_test  = np.array([x for i, x in enumerate(X) if i % K == ii],
                            dtype = float)
        y_test  = np.array([z for i, z in enumerate(y) if i % K == ii],
                            dtype = int)
        randomForest = RandomForest(num_Trees, verbose=True)
        t0 = time()
        randomForest.fit(X_train,y_train)

        t1 = time()
        print "time elapses = %.3f s" % (t1-t0)

        y_predicted = randomForest.predict(X_test)
        results = [prediction == truth for prediction,truth in zip(y_predicted,
                                                                   y_test)]
        # Accuracy
        accuracy = float(results.count(True)) / float(len(results))
        print "test accuracy: %.4f" % accuracy


main(cv=True, kaggle=False, num_Trees=10)
"""
    @num_Trees:
        number of random decision trees
    @cv: 
        cross validation mode, will run 10-fold cross validation,
        and print average cross validation accuracy
    @kaggle:
        kaggle competition mode will generate submission file
    default: 
"""
