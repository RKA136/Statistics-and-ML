import numpy as np

class SVM:
    """Support Vector Machine (SVM) classifier using a linear kernel."""

    def __init__(self, C = 1.0, lr = 1e-3, max_iter = 1000):
        """
        Initialize the SVM classifier.
        Parameters:
        C : float
            Regularization parameter. Higher value mean that the model tries to fit all training data correctly (less margin). Lower value allows a larger margin with some missclassifications.
        lr : float
            Learning rate for the gradient descent update.
        max_iter : int
            Maximum number of iterations.
        """

        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = None

    def fit(self , X, y):
        """
        Train the SVM classifier using Stochastic Gradient Descent (SGD).

        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Training data.
        y : numpy array of shape (n_samples,)
            Training labels. Should be -1 or 1.
        """
        n_samples, n_features = X.shape 

        #Check if y contains only -1 and 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        #gradient descent training
        for _ in range(self.max_iter):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Only regularization term
                    self.w -= self.lr * (2 * self.w)
                else:
                    # missclassified -> update with hinge loss gradient
                    self.w -= self.lr*(2*self.w - self.C * y_[idx] * x_i)
                    self.b += self.lr * self.C * y_[idx]
    
    def decision_function(self,X):
        """
        Compute the distance of the samples from the hyperplane.
        
        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Input data.
        Returns:
        numpy array of shape (n_samples,)
            Distance of each sample from the hyperplane.
            Positive value indicates class 1, negative value indicates class -1.
        """

        return np.dot(X, self.w) + self.b
    
    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X : numpy array of shape (n_samples, n_features)
            Input data.
        Returns:
        numpy array of shape (n_samples,)
            Predicted class labels (-1 or 1).
        """
        score = self.decision_function(X)
        return np.sign(score)