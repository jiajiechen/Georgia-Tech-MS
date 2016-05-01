
import numpy as np
from scipy.optimize import fmin_ncg
import pandas as pd

class logisticRegression(object):
    """Logistic Regression
       @Author: Jiajie Chen
       @Date: Apr 25, 2016
    """
    def __init__(self,
                 theta=None,
                 lambd=0.0, 
                 cost=0.0,
                 max_iter=100,
                 tol=1e-5,
                 solver='newton-cg',
                 verbose = False):
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
            from scipy.optimize import fmin_ncg
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
        X_c = np.column_stack((np.ones(X.shape[0]),X))
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
#end



class logisticRegression_sgd(logisticRegression): 
    """inherited logisticRegression class
       @Author: Jiajie Chen
       @Date: Apr 27, 2016
    """
    def __init__(self,
                 lambd=0.0,
                 mini_batch_size=20, 
                 tol=1e-5,
                 theta=None,
                 COSTS=None,
                 viz=False,
                 learning_rate=0.1,
                 max_iters=4000,
                 verbose=False,
                 full_batch=False,
                 rndbatch=False,
                 THETAS=None):
        self.lambd=lambd
        self.tol=tol
        self.theta=theta
        self.COSTS=COSTS
        self.mini_batch_size=mini_batch_size
        self.viz=viz
        self.learning_rate=learning_rate
        self.max_iters=max_iters
        self.verbose=verbose
        self.full_batch=full_batch
        self.rndbatch=rndbatch
        self.THETAS=THETAS
    
    def fit(self, X, y):
        X_c = np.column_stack((np.ones(X.shape[0]),X)) 
        self.theta = np.zeros(X.shape[1]+1)
        self.theta,_ = self.mini_batch_sgd(X_c,y,
                                           self.theta,
                                           self.learning_rate,
                                           self.max_iters,
                                           self.tol)

    def mini_batch_sgd(self,X,y,theta,learning_rate,max_iters,tol):
        J_old = 0
        J_container = []
        
        if self.full_batch:
            batch = X
            batch_y = y

        remain = X.shape[0]
        idx_old = 0
        idx = self.mini_batch_size
        
        for ii in xrange(max_iters):
            if self.full_batch:
                pass
            else:
                if self.rndbatch:
                    sampleidx = np.random.choice(X.shape[0],self.mini_batch_size)
                    batch = X[sampleidx,:]
                    batch_y = y[sampleidx]
                else:
                    sampleidx = np.array(range(idx_old,idx))
                    batch = X[sampleidx,:]
                    batch_y = y[sampleidx]
                    remain = remain - self.mini_batch_size
                    idx_old = idx
                    if idx+self.mini_batch_size<X.shape[0]:
                        idx=idx+self.mini_batch_size
                    else:
                        idx=X.shape[0]
                    if remain < 0:
                        remain = X.shape[0]
                        idx_old = 0
                        idx = self.mini_batch_size

            gd = self.gradient(theta, batch, batch_y)
            #gd = gd / np.linalg.norm(gd, ord=2)
            theta = theta - learning_rate*gd
            J = self.costFunction(theta, X, y)
            if self.verbose: print J

            if np.abs(J-J_old) < tol:
                print "converaged at ",self.tol
                self.COSTS = pd.Series(J_container)
                if self.viz: self.COSTS.plot()
                return theta, J
            else: J_old = J
            J_container.append(J) #for visualization cost versus iter
        
        self.COSTS = pd.Series(J_container)
        if self.viz: self.COSTS.plot()
        print "failed to converage at ",self.tol
        return theta, J
#end