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
        if self.w is not None:
            return X @ self.w + self.b
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                s = 0
                for alpha, sv_y, sv_x in zip(self.alphas, self.sv_y, self.sv_X):
                    if self.kernel == 'linear':
                        s += alpha * sv_y *self.kernel_linear(X[i].reshape(1,-1), sv_x.reshape(1,-1))
                    elif self.kernel == 'poly':
                        s += alpha * sv_y * self.kernel_poly(X[i].reshape(1,-1), sv_x.reshape(1,-1))
                    elif self.kernel == 'rbf':
                        s += alpha * sv_y * self.kernel_rbf(X[i].reshape(1,-1), sv_x.reshape(1,-1))
                        self.C = None
                y_predict[i] = s
            return y_predict + self.b
    
    def predict(self, X):
        return np.sign(self.project(X))
    

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------
# Generate toy data
# -----------------------------
X, y = make_blobs(n_samples=200, centers=2, random_state=42, cluster_std=1.5)
y = np.where(y == 0, -1, 1)  # Convert labels to -1, 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------
# Train custom SVM
# -----------------------------
svm = SVM(kernel='rbf', C=1.0, gamma=0.5)
svm.fit(X_train, y_train)

# -----------------------------
# Predictions and accuracy
# -----------------------------
y_pred = svm.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc*100:.2f}%")

# -----------------------------
# Plot decision boundary
# -----------------------------
def plot_decision_boundary(model, X, y, resolution=0.02):
    # Create grid
    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)
    
    # Plot contour
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap=plt.cm.Paired, edgecolors='k')
    plt.title("SVM Decision Boundary")
    plt.show()

plot_decision_boundary(svm, X_test, y_test)
