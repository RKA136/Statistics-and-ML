import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from SVM import SVM

# -----------------------------
# Generate toy data
# -----------------------------
X, y = make_blobs(n_samples=200, centers=2, random_state=55, cluster_std=1.5)
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
