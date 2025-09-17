# Support Vector Machine

## Idea:
 Use a linear model and try to find a linear decision boundary (hyperplane) that best separates the data. The best hyperplane is the one that yeilds the largest separation/margin between both classes. So we choose the hyperplane so that the distance from it to the nearest data point on each side is maximized.

- Decision
$$ w\cdot x_i - b \geq 1 \quad \text{if } y_i = 1 $$
$$ w\cdot x_i - b \leq 1 \quad \text{if } y_i = - 1 $$
i.e.,
$$ \gamma_i = y_i(w\cdot x_i - b) \geq 1 $$
with $y \in \{-1.1\}$
- This implies that
    - $\gamma_i\geq 0 $ → correct classification.
    - $\gamma_i\leq 0 $ → misclassification.
    - The larger the $\gamma_i$ is the more confident the classifier is.

## Loss Function: Hinge Loss
**Definition**
- The hinge loss for a single example is defined as:
$$L(y,f(x)) = max(0,1-yf(x))$$
- That is 
    - If yf(x) ≥ 1 → loss = 0 (correct classification with margin ≥ 1)
    - If yf(x) < 1 → loss = 1 - yf(x) (penalty grows linearly as margin decreases).

## Regularization

$$J = λ||w||^2 + \frac{1}{n}\sum_{i=1}^n max(0,1-y_i(w⋅x_i -b))$$

For such regularization term we have 

$$\begin{align*}
\text{if } y_if(x_i) ≥ 1 \text{:}&\\
J_i &= λ ||w||^2\\
\text{else}&\\
J_i &= λ ||w||^2 + 1 - y_i(w⋅x_i-b)
\end{align*}$$

## Gradients
$$
\begin{align*}
\text{if } &y_if(x_i) ≥ 1 \text{:}\\
\frac{dJ_i}{dw_k} &= 2λw_k \quad \frac{dJ_i}{db} = 0\\
\text{else}&\\
\frac{dJ_i}{dw_k} &= 2\lambda w_k - y_i.x_{ik} \quad \frac{dJ_i}{db} = y_i
\end{align*}
$$