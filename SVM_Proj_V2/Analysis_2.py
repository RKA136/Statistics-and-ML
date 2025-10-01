import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from SVM import SVM, MultiClassSVM
# Generate a sample 3-class dataset
X, y = make_classification(
    n_samples=150, n_features=2, n_informative=2, n_redundant=0,
    n_classes=3, n_clusters_per_class=1, random_state=42
)

# Train the multi-class SVM
clf = MultiClassSVM(kernel='rbf', C=1, gamma=0.5)
clf.fit(X, y)

# Predict
y_pred = clf.predict(X)
print("Predicted classes:", y_pred)
accuracy = np.mean(y_pred == y)
print(f"Training Accuracy: {accuracy*100:.2f}%")

# Plot the decision boundaries
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title("Multi-class SVM Decision Boundaries")
plt.show()