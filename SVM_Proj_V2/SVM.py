import cvxopt
import cvxopt.solvers
import numpy as np
from numpy import linalg


class SVM:
    def __init__(self, kernel=None, C=None, degree=3, gamma = 1):

        if C is not None:
            C = 0
        if kernel == None:
            kernal = 'linear'
        if gamma is None:
            gamma = 1

        C = float(C)
        gamma = float(gamma)
        degree = int(degree)
        

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel

    def kernal_linear(X1, X2):
            # X1 ; (n_samples1, n_features)
            # X2 ; (n_samples2, n_features)
            return X1 @ X2.T
        
    def kernal_poly(X1, X2, C, degree):
            return (C + X1 @ X2.T) ** degree
        
    def kernal_rbf(X1, X2, gamma):
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            dist = X1_norm + X2_norm - 2 * X1 @ X2.T
            return np.exp(-gamma * dist)
        
    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.kernel == 'linear':
           K = self.kernal_linear(X, X)
        
        elif self.kernel == 'poly':
            K = self.kernal_poly(X, X, self.C, self.degree)

        elif self.kernel == 'rbf':
            K = self.kernal_rbf(X, X, self.gamma)
            self.C = None

        else:
            raise ValueError("Unsupported kernel type. Choose 'linear', 'poly', or 'rbf'.")

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones(n_samples))
        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(-np.eye(n_samples))
            h = cvxopt.matrix(np.zeros(n_samples))
        else:
            G = cvxopt.matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
            h = cvxopt.matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))

        cvxopt.solvers.options['show_progress'] = True
        cvxopt.solvers.options['abstol'] = 1e-10
        cvxopt.solvers.options['reltol'] = 1e-10
        cvxopt.solvers.options['feastol'] = 1e-10

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        alphas = np.ravel(solution['x'])

        sv = alphas > 1e-5
        ind = np.arange(len(alphas))[sv]
        self.alphas = alphas[sv]
        self.sv_X = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.alphas)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alphas * self.sv_y * K[ind[n], sv])
        self.b /= len(self.alphas)
        
        if self.kernel == 'linear':
            self.w = np.sum((self.alphas * self.sv_y)[:, np.newaxis] * self.sv_X, axis=0)
        else:
            self.w = None


