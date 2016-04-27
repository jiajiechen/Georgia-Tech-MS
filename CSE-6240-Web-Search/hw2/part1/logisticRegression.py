"""
    Author: Jiajie Chen
    Date: Apr 24, 2016
"""

import numpy as np
from scipy.optimize import fmin_ncg

class logisticRegression(object):
    def __init__(self, theta=None, lambd=0.0, 
                 cost=0.0, max_iter=100, tol=1e-5, solver='newton-cg',
                 verbose=False):
        self.theta = theta
        self.lambd = lambd
        self.cost = cost
        self.max_iter = max_iter
        self.tol = tol
        self.solver=solver
        self.verbose=verbose

    def fit(self, X, y):
        X_c = np.column_stack((np.ones(X.shape[0]),X)) 
        self.theta = np.zeros(X.shape[1]+1)
        if self.solver == 'newton-cg':
            self.theta = fmin_ncg(self.costFunction, self.theta, 
                                  fprime=self.gradient, args=(X_c, y),
                                  maxiter=self.max_iter, avextol=self.tol, 
                                  disp=self.verbose)
        else:
            pass
        self.cost = self.costFunction(self.theta, X_c, y)

    def get_prob(self, X):
        X_c = np.column_stack((np.ones(X.shape[0]),X))
        return self.sigmoid(X_c.dot(self.theta))

    def predict(self, X):
        return (self.get_prob(X)>0.5).astype(int)
    
    def accuracy(self, y_pred, y_true):
        if isinstance(y_pred, list): y_pred=np.array(y_pred)
        if isinstance(y_true, list): y_pred=np.array(y_true)
        return np.mean(y_pred == y_true)

    def sigmoid(self, z):
        """z can be a matrix, vector or scalar"""
        if isinstance(z, list): z=np.array(z)
        return 1/(1+np.exp(-z))
    
    def costFunction(self, theta, X_c, y):
        n_row = X_c.shape[0]
        J = (1.0/n_row)*(-np.log(self.sigmoid(X_c.dot(theta))).dot(y)
                         -np.log(1-self.sigmoid(X_c.dot(theta))).dot(1-y))\
             + ((1.0*self.lambd)/(2*n_row))*np.sum(theta[1:]**2)
        return J

    def gradient(self, theta, X_c, y):
        n_row, n_col = X_c.shape
        grad = np.empty(theta.shape)
        grad[0] = (1.0/n_row)*X_c[:,0].T.dot(self.sigmoid(X_c.dot(theta))-y)
        grad[1:] = (1.0/n_row)*X_c[:,1:].T.dot(self.sigmoid(X_c.dot(theta))-y)+(float(self.lambd)/n_row)*theta[1:]
        return grad
# end

