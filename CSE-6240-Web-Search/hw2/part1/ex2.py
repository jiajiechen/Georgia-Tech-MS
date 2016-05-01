from __future__ import print_function
"""
    Author: Jiajie Chen
    Date: Apr 24, 2016
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression import logisticRegression

pathname = os.path.dirname(sys.argv[0])
os.chdir(pathname)

data = pd.read_csv("../ex2data1.txt", delimiter=",", header=None)
X = data.values[:,:2]
y = data.values[:,2]
del(data)

clf = logisticRegression(lambd=0,tol=1e-6,verbose=True)
clf.fit(X,y)

y_pred = clf.predict(X)
print("train accuracy =", clf.accuracy(y_pred, y))

# add decision boundary
xx = np.linspace(X[:,0].min(), X[:,0].max())
yy = -clf.theta[1]/clf.theta[2]*xx - clf.theta[0]/clf.theta[2]

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,s=40,alpha=0.8,cmap=plt.cm.Paired)

plt.figure()
plt.scatter(X[:,0],X[:,1],c=y,s=40,alpha=0.8,cmap=plt.cm.Paired)
plt.plot(xx, yy, 'k-')
plt.show()
