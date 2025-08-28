# Support Vector Machines


## Problem Setup
- We are given a dataset 
$$\mathcal{D} = [(x_i, y_i)]_{i=1}^N,\quad x_i\epsilon \mathbb{R}^d,\quad y_i \epsilon [-1,+1]$$

- Goal: Learn a classifier of the form
$$f(x) = \text{sign}(w⋅x+b)$$
- where
    - $w$ ϵ $\mathbb{R}^d$ is the weight vector,
    - $b$ ϵ $\mathbb{R}^d$ is the bias,
    - the decision boundary is the hyperplane {x|w⋅x+b=0}


## Geometric Margin
- For any point $(x_i,y_i)$:
    - Functional Margin:
    $$\hat{\gamma}_i = y_i(w⋅x_i+b)$$
    - Geometric Margin (Distance from Hyperplane):
    $$\gamma_i = \frac{\hat{\gamma}_i}{||w||}

- SVM maximizes the minimum margin over all points:
    $$γ = \min_i \gamma_i$$

## Hard Margin SVM (Linearly Separable Case)
- Optimization Problem:
$$\max_{w,b} γ \Leftrightarrow \min_{w,b} \frac{1}{2}||w||^2$$
- Subject to:
$$y_i(w⋅x_i+b)≥ 1, ∀ i$$

## Soft Margin SVM (Non-Separable Data)
- Introduce slcak variable ξᵢ ≥ 0:
$$\min_{w,b,ξ}\frac{1}{2}||w||^2 + C \sum_{i=1}^N\xi_i$$
- Subject to:
$$y_i(w⋅x_i+b)≥ 1 - \xi_i, \quad \xi_i≥ 0$$

- Here C > 0 is a regularization parameter controlling margin-misclassification trade-off.

## Lagrangian Dual Formulation
- Introduce multipliers αᵢ ≥ 0:
$$\mathcal{L}(w,b,ξ,α,μ) = \frac{1}{2}||w||^2 + C\sum_i\xi_i - \sum_i\alpha_i(y_i(w⋅x_i+b)-1+\xi_i) - \sum_i\mu_i\xi_i$$

- KKT (Karush-Kuhn-Tucker) condition give:
$$w = \sum_{i=1}^N \alpha_i y_i x_i, \quad \sum_{i=1}^N \alpha_i y_i = 0, \quad 0≤\alpha_i≤ C$$

- Dual optimization problem:
$$\max_{\alpha}\sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N \alpha_i\alpha_jy_iy_j(x_i⋅x_j)$$
- Subject to:
$$ \sum_{i=1}^N \alpha_iy_i = 0, \quad 0≤\alpha_i≤ C$$

## Support Vectors

- The solution depends only on training smaples with αᵢ > 0.
- These are the **support vectors**
- Decision function:
$$ f(x) = \text{sign}\left( \sum_{iϵSV}\alpha_iy_i(x_i⋅x)+b\right)$$

## Kernels (Nonlinear SVM)

- Replace dot-product with kernel function:
$$ K(x_i,x_j) = φ(x_i)⋅φ(x_j)$$

- Dual becomes:
$$ \max_{\alpha} \sum_{i=1}^N\alpha_i - \frac{1}{2}\sum_{i,j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)$$

- Decision function:
$$f(x) = \text{sign}\left( \sum_{iϵSV}\alpha_iy_iK(x_i,x)+b\right)$$

- Common Kernels
    - **Linear**: K(x,z) = x ⋅ z
    - **Polynomial**: K(x,z) = (x ⋅ z + c)ᵈ 
    - **RBF(Gaussian)**: K(x,z) = exp(-||x-z||²/(2σ²))
    - **Sigmoid**: K(x,z) = tanh(κx ⋅ z + θ)

## Hinge Loss (Link to Deep Learning)
- SVM classification loss can be expressed as:
$$ L_{hinge}(w;x_i,y_i) = max(0,1-y_i(w⋅x_i+b))$$
- Full objective:
$$J(w) = \frac{1}{2}||w||^2 + C\sum_{i=1}^NL_{hinge}(w;x_i,y_i)$$

- This is directly analogous to loss function in deep learning (e.g. cross-entropy).
- In deep nets: hinge loss is used for SVM-style calssifier at the output layer.