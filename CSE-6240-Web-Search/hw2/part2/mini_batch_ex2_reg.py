from __future__ import print_function

"""
    Author: Jiajie Chen
    Date: Apr 27, 2016
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from mini_batch_logisticRegression import logisticRegression_sgd
from sklearn.preprocessing import PolynomialFeatures

pathname = os.path.dirname(sys.argv[0])
os.chdir(pathname)

data = pd.read_csv("../ex2data2.txt", delimiter=",", header=None)
X = data.values[:,:2]
y = data.values[:,2]
del(data)

poly = PolynomialFeatures(6,include_bias=False)
X_new = poly.fit_transform(X)
print("After polynomial feature transformation, now shape=", X_new.shape)


"""mini-batch: rnd batch
"""
# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.5,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr1=test.COSTS

# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.3,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr2=test.COSTS

# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.1,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr3=test.COSTS


# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.05,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr4=test.COSTS

# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.03,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr5=test.COSTS

# random batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.01,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = True)
test.fit(X_new,y)
lr6=test.COSTS

pylab.rcParams['figure.figsize']=14,10 # that's default image size for this interactive session

lr1.plot(label = "lr=0.5")
lr2.plot(label = "lr=0.3")
lr3.plot(label = "lr=0.1")
lr4.plot(label = "lr=0.05")
lr5.plot(label = "lr=0.03")
lr6.plot(label = "lr=0.01")

plt.legend()
plt.xlabel("Iteration epoch")
plt.ylabel("Cost")
plt.title("(Stochastic) Mini-batch gradient descent")
plt.show()


"""mini-batch: sequential batch
"""
# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.5,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr1=test.COSTS

# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.3,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr2=test.COSTS

# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.1,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr3=test.COSTS


# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.05,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr4=test.COSTS

# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.03,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr5=test.COSTS

# Sequential batch
test = logisticRegression_sgd(viz = False,
                              learning_rate=0.01,
                              max_iters = 500,
                              tol=1e-9,
                              full_batch=False,
                              mini_batch_size=10,
                              rndbatch = False)
test.fit(X_new,y)
lr6=test.COSTS

pylab.rcParams['figure.figsize']=14,10 # that's default image size for this interactive session

lr1.plot(label = "lr=0.5")
lr2.plot(label = "lr=0.3")
lr3.plot(label = "lr=0.1")
lr4.plot(label = "lr=0.05")
lr5.plot(label = "lr=0.03")
lr6.plot(label = "lr=0.01")

plt.legend()
plt.xlabel("Iteration epoch")
plt.ylabel("Cost")
plt.title("(Sequential) Mini-batch gradient descent")
plt.show()



clf = logisticRegression_sgd(verbose=False,
                             learning_rate=0.5,
                             tol=1e-10,
                             full_batch=False,
                             max_iters = 1000)
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
plt.show()

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




