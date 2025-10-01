import cvxopt
import cvxopt.solvers
import numpy as np

class SVM:
    def __init__(self, kernel=None, C=None, degree=3, gamma = 1):

        if C is None:
            C = 0
        if kernel == None:
            kernel = 'linear'
        if gamma is None:
            gamma = 1

        C = float(C)
        gamma = float(gamma)
        degree = int(degree)
        

        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.kernel = kernel

    def kernel_linear(self, X1, X2):
        X1 = X1.reshape(1, -1) if X1.ndim == 1 else X1
        X2 = X2.reshape(1, -1) if X2.ndim == 1 else X2
        result = X1 @ X2.T
        return result[0,0] if result.size == 1 else result
        
    def kernel_poly(self, X1, X2):
        X1 = X1.reshape(1, -1) if X1.ndim == 1 else X1
        X2 = X2.reshape(1, -1) if X2.ndim == 1 else X2
        result = (X1 @ X2.T + self.C) ** self.degree
        return result[0,0] if result.size == 1 else result
        
    def kernel_rbf(self, X1, X2):
        X1 = X1.reshape(1, -1) if X1.ndim == 1 else X1
        X2 = X2.reshape(1, -1) if X2.ndim == 1 else X2
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist = X1_norm + X2_norm - 2 * X1 @ X2.T
        result = np.exp(-self.gamma * dist)
        return result[0,0] if result.size == 1 else result
        
    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.kernel == 'linear':
           K = self.kernel_linear(X, X)
        
        elif self.kernel == 'poly':
            K = self.kernel_poly(X, X)

        elif self.kernel == 'rbf':
            K = self.kernel_rbf(X, X)
            self.C = None

        else:
            raise ValueError("Unsupported kernel type. Choose 'linear', 'poly', or 'rbf'.")

        y = y.astype(float)
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

    def project(self, X):
        if self.w is not None:  # linear case
            return X @ self.w + self.b
        else:
            # Vectorized computation for nonlinear kernels
            if self.kernel == 'linear':
                K = X @ self.sv_X.T
            elif self.kernel == 'poly':
                K = (self.C + X @ self.sv_X.T) ** self.degree
            elif self.kernel == 'rbf':
                X_norm = np.sum(X**2, axis=1).reshape(-1,1)        # (n_samples,1)
                sv_norm = np.sum(self.sv_X**2, axis=1).reshape(1,-1)  # (1,n_sv)
                K = np.exp(-self.gamma * (X_norm + sv_norm - 2*X @ self.sv_X.T))
            y_predict = (K * (self.alphas * self.sv_y)).sum(axis=1) + self.b
            return y_predict
    
    def predict(self, X):
        return np.sign(self.project(X))
    
class MultiClassSVM:
    def __init__(self, kernel='linear', C=None, degree=3, gamma=1):
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.models = {}  # store one SVM per class

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            y_binary = np.where(y == cls, 1, -1)  # one-vs-rest labels
            model = SVM(kernel=self.kernel, C=self.C, degree=self.degree, gamma=self.gamma)
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        # Get decision function from all SVMs
        decisions = np.array([self.models[cls].project(X) for cls in self.classes])
        # Choose the class with the highest score
        return self.classes[np.argmax(decisions, axis=0)]
