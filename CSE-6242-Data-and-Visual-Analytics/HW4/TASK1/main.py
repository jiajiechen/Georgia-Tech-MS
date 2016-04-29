import os
import csv
import numpy as np
import scipy.stats as st
import random
import math
from time import time

from CSE6242HW4Tester import generateSubmissionFile
from RandomForest import RandomForest

"""
Here, X is assumed to be a matrix with n rows and d columns
where n is the number of samples
and d is the number of features of each sample

Also, y is assumed to be a vector of n labels
"""

myname = "Jiajie-Chen"
os.chdir("/Users/jiajiechen/Desktop/GATECH_MSA/CSE 6242/CSE6242-Assignment/HW4/submission/Task1")

def main(cv=False,kaggle=True, num_Trees=10, verbose=False):
    X = []
    y = []
    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])
    #end

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

            randomForest = RandomForest(num_trees=num_Trees, verbose=verbose)
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
    else:
        ii = 3
        K = 10
        X_train = np.array([x for i, x in enumerate(X) if i % K != ii],
                           dtype = float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != ii],
                           dtype = int)
        X_test  = np.array([x for i, x in enumerate(X) if i % K == ii],
                           dtype = float)
        y_test  = np.array([z for i, z in enumerate(y) if i % K == ii],
                           dtype = int)
        if kaggle==True:
            randomForest = RandomForest(num_trees=num_Trees, verbose=verbose)
            t0 = time()
            # randomForest.fit(X_train,y_train)
            randomForest.fit(X,y) #use the full data
            t1 = time()
            print "time elapses = %.3f s" % (t1-t0)
            # y_predicted = randomForest.predict(X_test)
            # results = [prediction == truth 
            #            for prediction,truth in zip(y_predicted,y_test)]
            # # Accuracy
            # accuracy = float(results.count(True)) / float(len(results))
            # print "test accuracy: %.4f" % accuracy
            generateSubmissionFile(myname, randomForest)
        else:
            randomForest = RandomForest(num_trees=num_Trees, verbose=verbose)
            t0 = time()
            randomForest.fit(X_train,y_train)
            t1 = time()
            print "time elapses = %.3f s" % (t1-t0)
            y_predicted = randomForest.predict(X_test)
            results = [prediction == truth 
                       for prediction,truth in zip(y_predicted,y_test)]
            accuracy = float(results.count(True)) / float(len(results))
            print "test accuracy: %.4f" % accuracy

main(cv=False, kaggle=True, num_Trees=1000, verbose=True)


""" Usage:
    @num_Trees: number of random decision trees
    @cv: cross validation mode, will run 10-fold cross validation,
         and print average cross validation accuracy
    @kaggle: kaggle competition mode will generate submission file
"""







