import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import sys
import os

# make sure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from svm import SVM

# 1. Generate synthetic dataset
X, y = make_blobs(n_samples=50, centers=2, random_state=42)
# Convert labels {0,1} to {-1,+1}
y = np.where(y == 0, -1, 1)

# 2. Train our custom SVM
svm = SVM(C=1, lr=0.001, max_iter=1000)
svm.fit(X, y)

# 3. Predictions
y_pred = svm.predict(X)
print("Predicted labels:", y_pred[:10])
print("True labels     :", y[:10])

# 4. Decision function values
scores = svm.decision_function(X)
print("Decision scores (first 5):", scores[:5])

# 5. Visualization
def plot_svm(X, y, model):
    plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolors='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # grid of points
    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], 50),
        np.linspace(ylim[0], ylim[1], 50)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot decision boundary and margins
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1],
               alpha=0.7, linestyles=['--', '-', '--'])

    plt.title("Custom Linear SVM")
    plt.show()

plot_svm(X, y, svm)
