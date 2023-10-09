# A script for practical class on depth-based classification for MDI341
# Author: Pavlo Mozharovskyi

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import neighbors

def rmvt(mu, sigma, df, n):
    """
    Routine generates Student-t distributed vectors
    Args:
    n - number of observations
    mu - center vector
    sigma - scatter matrix
    df - number of degrees of freedom
    Returns:
    A matrix with n Student-t distributed vectors
    """
    d = len(sigma)
    W = np.tile(np.random.gamma(df/2., 2./df, n), (d, 1)).T
    Z = np.random.multivariate_normal(np.zeros(d), sigma, n)
    return mu + Z / np.sqrt(W)

def depthMah(X, data):
    """
    Routine calculates Mahalanobis depth of X w.r.t. data
    Args:
    X - observations to calculate the depth for
    data - the sample defining the depth fucntion
    Returns:
    An array of Mahalanobis depths
    """
    mu = np.mean(data, axis=0)
    sigmaInv = np.linalg.inv(np.cov(data.T))
    distsSq = np.sum(((X - mu) @ sigmaInv) * (X - mu), axis = 1)
    return 1 / (1 + distsSq)

def depthTuk(X, data, ndirs=100):
    """
    Routine approximates Tukey depth of X w.r.t. data
    Args:
    X - observations to calculate the depth for
    data - the sample defining the depth fucntion
    ndirs - number of directions
    Returns:
    An array of Tukey depths
    """
    # Generate directions
    d = len(data[0])
    dirsTmp = np.random.multivariate_normal(np.zeros(d), np.diag(np.ones(d)), ndirs)
    dirs = (dirsTmp.T / np.sqrt(np.sum(dirsTmp**2, axis = 1))).T
    # Points' depths
    depths = np.ones(len(X))
    # Go throught points
    for i in range(len(X)):
        for j in range(ndirs):
            prjCurX = X[i] @ dirs[j]
            prjData = data @ dirs[j]
            depths[i] = min([depths[i], np.mean(prjData >= prjCurX), np.mean(prjData <= prjCurX)])
    return depths

class MaxDepthClassifier(BaseEstimator, ClassifierMixin):
    """
    Maximum depth classifier class
    """
    def __init__(self, depthName="Mahalanobis", ndirs=100):
        """
        Initialization of the maximum depth classifier
        Args:
        depthName - depth notion to use
        ndirs - number of directions to approximate the depth
        """
        self.depthName_ = depthName
        self.ndirs_ = ndirs
    
    def fit(self, X, y):
        """
        Fit the maximum depth classifier (basically do nothing)
        Args:
        X - input
        y - output
        """
        # Save the data
        self.X_ = X
        self.y_ = np.array(y)
        self.labs_ = np.unique(y)
        # Fit the 1NN classifier for outsiders
        self.oKnn_ = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.oKnn_.fit(self.X_, self.y_)
        return self
    
    def predict(self, X):
        """
        Predict the output for given inputs
        Args:
        X - input
        """
        yPred = np.zeros(len(X))
        # Go throught points
        for i in range(len(X)):
            maxDepth = 0
            maxLab = None
            # Go through classes
            for curLab in self.labs_:
                # Calculate depth
                curClass = self.X_[self.y_ == curLab]
                if self.depthName_ == "Mahalanobis":
                    curDepth = depthMah(X[i].reshape(1, -1), curClass)
                if self.depthName_ == "Tukey":
                    curDepth = depthTuk(X[i].reshape(1, -1), curClass, self.ndirs_)
                # Obtain depth label
                if curDepth > maxDepth:
                    maxDepth = curDepth
                    maxLab = curLab
            if maxDepth > 0: # classified with data depth
                yPred[i] = maxLab
            else: # an outsider -> do 1NN
                yPred[i] = self.oKnn_.predict(X[i].reshape(1, -1))
        return yPred

class DDkNNClassifier(BaseEstimator, ClassifierMixin):
    """
    kNN DD-classifier class
    """
    def __init__(self, depthName="Mahalanobis", ndirs=100, k=11):
        """
        Initialization of the kNN DD-classifier
        Args:
        depthName - depth notion to use
        ndirs - number of directions to approximate the depth
        k - number of neighbors to consider in the DD-plot
        """
        self.depthName_ = depthName
        self.ndirs_ = ndirs
        self.k_ = k
    
    def fit(self, X, y):
        """
        Fit the kNN DD-classifier: calculate the DD-plot
        Args:
        X - input
        y - output
        """
        # Save the data
        self.X_ = X
        self.y_ = np.array(y)
        self.labs_ = np.unique(y)
        # Calculate the DD-plot
        self.ddplot_ = np.empty([len(X), len(self.labs_)])
        for i in range(len(self.labs_)):
            curClass = self.X_[self.y_ == self.labs_[i]]
            if self.depthName_ == "Mahalanobis":
                self.ddplot_[:,i] = depthMah(self.X_, curClass)
            if self.depthName_ == "Tukey":
                self.ddplot_[:,i] = depthTuk(self.X_, curClass, self.ndirs_)
        # Fit the kNN classifier for the DD-plot
        self.ddKnn_ = neighbors.KNeighborsClassifier(n_neighbors=self.k_)
        self.ddKnn_.fit(self.ddplot_, self.y_)
        # Fit the 1NN classifier for outsiders
        self.oKnn_ = neighbors.KNeighborsClassifier(n_neighbors=1)
        self.oKnn_.fit(self.X_, self.y_)
        return self
    
    def predict(self, X):
        """
        Predict the output for given inputs
        Args:
        X - input
        """
        yPred = np.zeros(len(X))
        curDepths = np.zeros(len(self.labs_))
        # Go throught points
        for i in range(len(X)):
            # Go through classes
            for j in range(len(self.labs_)):
                # Calculate depth
                curClass = self.X_[self.y_ == self.labs_[j]]
                if self.depthName_ == "Mahalanobis":
                    curDepths[j] = depthMah(X[i].reshape(1, -1), curClass)
                if self.depthName_ == "Tukey":
                    curDepths[j] = depthTuk(X[i].reshape(1, -1), curClass, self.ndirs_)
            # Obtain depth label
            if max(curDepths) > 0: # classified with data depth
                yPred[i] = self.ddKnn_.predict(curDepths.reshape(1, -1))
            else: # an outsider -> do 1NN
                yPred[i] = self.oKnn_.predict(X[i].reshape(1, -1))
        return yPred
