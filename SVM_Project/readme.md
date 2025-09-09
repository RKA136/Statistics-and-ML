# Custom SVM Project

This project is a step-by-step reimplementation of a Support Vector Machine (SVM) classifier, similar to `sklearn.svm.SVC`.

## Project Structure

```text
SVM_Project/
|-- src/
|    |--svm.py                          #implementation
|-- readme.md                           #documentation
|-- examples/
        |--demo_linear_classifier.py    #test script
```

## Class: `SVM`
A Support Vector Machine implementation with a linear decision boundary, trained using **Stochastic Gradient Descent (SGD)**

### Initialization
```python
class SVM(C = 1.0, lr = 1e-3, max_iter = 1000):
```
---
#### Parameters 
- **C** (float, default = 1.0)
    - Regularization parameter. Controls the tradeoff between maximizing margin and minimizing error.
    - **Large C** $\rightarrow$ less tolerance for missclassification
    - **Small C** $\rightarrow$ more toleranc , wider margin.

- **lr** (float, default = 1e-3)
    - learning rate for SGD updates.

- **max_iter** (int, default = 1000)
    - Number of iteration over the dataset.

#### Attributes
- **w**: nd_array of shape (n_features)
    - weight factor that defines the hyperplane.
- **b**: float
    - Bias term of the hyperplane.
---
### Method: `fit(X, y)`

```python
def fit(self , X, y):
```

Train the SVM on the dataset.

- **Parameters**
    - **X**: ndarray of shape (n_samples, n_features)
        - Training feature matrix
    - **y**: ndarray of shape (n_samples,)

- **Description**
    - Initialize weights `w` and bias `b`.
    - Run Stochastic Gradient Descent for `max_iter` passes over dataset.
    - Updates parameters using the hinge loss gradient:
    $$L(w,b) = \frac{1}{2} ||w||^2 + c\sum_i \max{(0,1-y_i(w\cdot x_i +b))} $$
    - Ensures that corectly classsified points (margin $\geq$ 1) only contribute regularization updates,
    - Misclassified points update both `w` and `b` to enforce margin constraints.
---
### Method: `decision_function(X)`

```python
def decision_function(X):
```

Compute the distance of the point from the hyperplane for each samples.

- **Parameters**
    - **X**: ndarray of shape (n_samples,n_features)
        - Input feature matrix.
- **Returns**
    - ndarray of shape (n_samples,) containing the distance to the hyperplane.

### Method: `predict(X)`

```python
def predict(X):
```

Predict class labels for given samples.

- **Parameters**
    - **X**: ndarray of shape (n_samples,n_features)
        - Input feature matrix.
- **Returns**
    - ndarray of shape (n_samples,).
    - Predicted labels `+1` or `-1`.
- **Description**
    - Computes distance to the hyperplane using the `decision_function()`.
    - Returns `+1` if distance $\geq$ 0, else `-1`.