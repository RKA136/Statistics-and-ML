# Convex Optimization

In my code in order to implement the optimization in a efficient format I am using **Convex Optimization**. This report gives a overview of the library and the functions that I have used.

## What is CVXOPT (Convex Optimization)
- CVXOPT is a Python package for solving *convex optimization problems*.
- It is especially useful for:
    - Quadratic Programming (QP)
    - Linear Programming (LP)
    - Second-Order Cone Programming (SOCP)
    - Semidefinite Programming (SDP)
## General Problem Formats
1. **Linear Programming**

    **Standard Form**

    minimize: $c^Tx$

    Subject to $Gx ≤ h$     $Ax = b$

    Where:
    - x ∈ $\mathbb{R}^n$ (decision variable)
    - G and A are matrices, h and b are vectors.

2. **Quadratic Programming**

    **Standard Form**

    minimize: $\frac{1}{2}x^TPx + q^Tx$

    Subject to $Gx ≤ h$     $Ax = b$

    Where:
    - P: symmetric, positive semidefinite matrix (quadratic term).
    - q: Linear coefficient vector.
    - G, h: inquuality constraints.
    - A, b; equality constraints.

    This is the form used in SVM dual optimization.

### SVM Dual Problem with CVXOPT

**SVM Primary Problem**:

(For Soft Margin) We want the widest margin hyperplane with slack of misclassification.

**Optimization Problem**    

minimize over (w, b, ξ):
$\frac{1}{2} ||w||^2 + C∑ξ_j$

Subject to: $y_i(w ⋅ x_i + b)≥ 1 - ξ_i$, ∀ i

$ξ_j ≥ 0$

**Construct the Lagrangian**

Introduce Lagrangian Multipliers:
    - α_i ≥ 0 for constraint $y_i(w ⋅ x_i + b)≥ 1 - ξ_i$
    - μ_i ≥ 0 for constraint $ξ_j ≥ 0$

Lagrangian:
    $$L(w, b, ξ; α, μ) = \frac{1}{2} ||w||^2 + C∑ξ_i - ∑ α_i[y_i(w ⋅ x_i + b) - 1 + ξ_i] - ∑ μ_i ξ_i$$

**Stationary Conditions**

To form the *dual*, we minimize L over primal variables (w,b,ξ).

(a). Derivative w.r.t w:
    $$\frac{∂L}{∂w} = w - ∑α_i y_i x_i = 0$$
    $$⟹ w = ∑α_i y_i x_i$$

(b). Derivative w.r.t b:
    $$\frac{∂L}{∂b} = - ∑ α_i y_i = 0$$
    $$⟹ ∑ α_i y_i = 0$$

(c). Derivative w.r.t $ξ_i$:
    $$\frac{∂L}{∂ξ_i} = C - α_i - μ_i = 0$$
    $$⟹ α_i ≤ C$$ (since $μ_i≥0$)

so $0 ≤ α_i ≤ C$

**Substitute back into the Lagrangian**

Now substitute $w = ∑α_i y_i x_i$ in L:
$$L = \frac{1}{2} ||w||^2 - ∑ α_i y_i (w ⋅ x_i) + ∑ α_i y_i b (= 0) + ∑ α_i + (C - α_i - μ_i)(=0)ξ_i$$
$$L = \frac{1}{2} (∑α_i y_i x_i) ⋅ (∑α_j y_j x_j) - ∑ α_i y_i ((∑α_i y_i x_i) ⋅ x_i) + ∑ α_i$$
Simplifying
$$L = ∑ α_i - \frac{1}{2} (∑∑α_i α_j y_i y_j )(x_i ⋅ x_j)$$

**SVM Dual Problem**:

maximize: $∑ α_i - \frac{1}{2}∑∑α_iα_jy_iy_jK(x_i,x_j)$

subject to: $∑α_iy_i = 0$ and $0 ≤ α ≤ C$

**Mapping to CVXOPT QP**

cvxopt expects the problem:

minimize $\frac{1}{2}x^TPx+q^Tx$

Subject to: $Gx ≤ h$     $Ax = b$

with variable x = α 

- Mapping Conditions → cvxopt Matrices
    - a. $P = (yy^T) ∘ K$
        - In the dual the quadratic term is : $\frac{1}{2}∑∑α_iα_jy_iy_jK(x_i,x_j)$
        - That is exactly represented by the matrix multiplication $α^T P α $
            - where $P[i,j] = y_iy_jK(x_i,x_j)$
        - So **P** encodes the interaction between support vectors via kernel + labels.
        
    - b. $q = -1$ (vectors of -1's)
        - In the dual, the linear term is $∑ α_i$
        - cvxopt minimizes so we flip the sign: $q^Tα = - ∑ α $
        - So **q** ensures maximization of $∑ α_i$
    
    - c. $A = y^T, b = 0$
        - Constraints in dual: $∑ α_iy_i = 0$
        - In cvxopt form: $Aα = b$
        - So $A$ is a row vector of labels $(y^T)$ and $b=0$
    
    - d. $G, h$
        - Hard Margin SVM (C = ∞)
            - Constraint: $α_i ≥ 0$
            - Rewritten as $-α_i ≤  0$
            - So $G=-I,h=0$
            -So keeps $\alpha_i$, non-negative
        - Soft Margin SVM (finite C)
            - Constriant: $0 ≤ α_i ≤ C$
            - Already have $\alpha_i ≥ 0$ from above.
            - Add $ α_i ≤ C$.
            - In cvxopt: $G = [-I; I], h= [0;C]$.
            - So keeps $α_i$ bounded between 0 and C (regularization)
            



- $P = (y y^T) ∘ K$ (Element-wise multiply)
- $q = -1$ (Vector of -1's)
- $A = y^T, b = 0$
- G, h:
    - For Hard-margin (C = ∞): G = -I, h = 0 (so α ≥ 0)
    - For Soft-margin: add upper-bound constraints α ≤ C.

