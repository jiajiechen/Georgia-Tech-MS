from __future__ import print_function

import pandas as pd
import numpy as np
import os
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from time import time

# import matplotlib.pyplot as plt
# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 5, 10

def read_data(path):
    os.chdir(path) #path=data_path
    return pd.read_csv("TFIDF_model.csv").values[:,1:],\
           pd.read_csv("TFIDF_model.csv").values[:,0]

matrix, users=read_data(path="/Users/jiajiechen/Desktop/project-CSE6242/NLP_data")
print(matrix.shape, users.shape)

class Find(cluster_num):
    t0 = time()
    nmf = NMF(n_components=2, init='random',
              random_state=1, verbose=0, tol = 1e-10)

    W = nmf.fit_transform(matrix)
    H = nmf.components_
    print("done in %0.3fs." % (time() - t0))
    print(W.shape, H.shape, nmf.reconstruction_err_)

    userClusters = pd.DataFrame(np.column_stack((users, W)),
                                columns=['userID','t1','t2','t3','t4',
                                         't5','t6','t7','t8','t9','t10'])

    # ['userID','t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']
    Cluster = userClusters.values[:,1:]
    Cluster = normalize(Cluster, norm = 'max',axis=0)

    nor_userCluster = pd.DataFrame(np.column_stack((userClusters.values[:,0], Cluster)),columns=['userID','t1','t2','t3','t4',
                           't5','t6','t7','t8','t9','t10'])

    print(nor_userCluster.head())

    words = np.array(pd.read_csv("./NLP_data/TFIDF_model.csv").columns[1:])
    table1 = pd.DataFrame(H, columns = words).transpose()
    table1.columns = ['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']
    #['t1','t2','t3','t4','t5','t6','t7','t8','t9','t10']

def store_result():
    table1.to_csv("clusterKeywords.csv",index_label=True)
    table1.sort_values(by='t1',axis=0,ascending=False)[['t1']].to_csv("clusterKeywords_1.csv",index_label=True)
    table1.sort_values(by='t2',axis=0,ascending=False)[['t2']].to_csv("clusterKeywords_2.csv",index_label=True)
    table1.sort_values(by='t3',axis=0,ascending=False)[['t3']].to_csv("clusterKeywords_3.csv",index_label=True)
    table1.sort_values(by='t4',axis=0,ascending=False)[['t4']].to_csv("clusterKeywords_4.csv",index_label=True)
    table1.sort_values(by='t5',axis=0,ascending=False)[['t5']].to_csv("clusterKeywords_5.csv",index_label=True)
    table1.sort_values(by='t6',axis=0,ascending=False)[['t6']].to_csv("clusterKeywords_6.csv",index_label=True)
    table1.sort_values(by='t7',axis=0,ascending=False)[['t7']].to_csv("clusterKeywords_7.csv",index_label=True)
    table1.sort_values(by='t8',axis=0,ascending=False)[['t8']].to_csv("clusterKeywords_8.csv",index_label=True)
    table1.sort_values(by='t9',axis=0,ascending=False)[['t9']].to_csv("clusterKeywords_9.csv",index_label=True)
    table1.sort_values(by='t10',axis=0,ascending=False)[['t10']].to_csv("clusterKeywords_10.csv",index_label=True)
#end



