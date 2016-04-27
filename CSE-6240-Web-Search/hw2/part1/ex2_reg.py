from __future__ import print_function
"""
    Author: Jiajie Chen
    Date: Apr 24, 2016
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from logisticRegression import logisticRegression
from sklearn.preprocessing import PolynomialFeatures

os.chdir("/Users/jiajiechen/Desktop/CSE6240-HW2")

data = pd.read_csv("./submit/ex2data2.txt", delimiter=",", header=None)
X = data.values[:,:2]
y = data.values[:,2]
del(data)

poly = PolynomialFeatures(6,include_bias=False)
X_new = poly.fit_transform(X)
print("After polynomial feature transformation, now shape=", X_new.shape)

clf = logisticRegression(lambd=1, verbose=False, tol=1e-10)
clf.fit(X_new,y)
y_pred = clf.predict(X_new)
print("train accuracy = ", clf.accuracy(y_pred, y))

h = .02  # step size in the mesh

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = clf.predict(poly.fit_transform(np.c_[xx.ravel(), yy.ravel()]))

# Put the result into a color plot
Z = Z.reshape(xx.shape)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, s=40,alpha=0.8,cmap=plt.cm.Paired)
plt.title("ex2data2")

# Plot also the training points
plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40,alpha=0.8,cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())
plt.title("ex2data2")
plt.show()




