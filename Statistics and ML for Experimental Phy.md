# Statistics and ML for Experimental Physics
- Grader - **Avik(DHEP)**
- 3-4 assignments + term paper + project (preferably from own lab)

## Supervised Learning
- Classification
    - SVM
    - BOT
    - DNN
    - GNN
    - self-attention
    - Opticla Nural Network

- Regression
    - Adversarial Network

- Generative Models (GAN)

- Time series analysis 

- Diffusion Models and Normalization Flows

## Unsupervised Learning
- Clustering 
- anomaly detection

## Elements of Supervised Learning

- Data (̄xᵢ, yᵢ)
```text
̄xᵢ  ----->      [θⱼ]    -----> yᵢ 
Input           Model       Prediction 
Features        Training    Target
```
- yᵢ is true value from the dataset.
- ̂yᵢ is the predicted value of the model.

- Classification
    - multi-classification (true/false)(can be a vector but based on discrete values)
    - regression (can be any value or combination need not be a discrete values)

### Types of Model
- Linear Model
- ̂yᵢ = Σⱼθⱼxᵢⱼ   
- Objective 
    - How do I determine the θs ----> Learning Objective
    - Objective (θ) = L(θ) + Ω(θ)
        - L(θ) -> Loss term
        - Ω(θ) -> Regularization term -> what pinalizaes complexity of the model
            - Without Regularization the problem may be overfitting.
        - "Bias Variance" trade off.
- In classification one of the loss function is the 
    - Logistic Loss: L(θ) = Σᵢ(yᵢ ln(1+e^(-̂yᵢ)) + (1-yᵢ)ln(1+e^(̂yᵢ)))
    - Binary Cross-Entropy.

- In Classification
    - Train-Test_Validation
    - Receiver Operator Characteristics (ROC) curve.
        - ```text
                      fpr|
                1-bkg eff|
                         |_________
                            tpr/signal eff
          ```
        - Being close to the 45° is the worst case scenario
        - Area under the curve (with the y axis) should be as clode to 1 as possible  

- In Regression one of the most popular loss function is 
    - Mean Square Error (MSE): L(θ) = Σᵢ(yᵢ-̂yᵢ)² 
    -                   (MAE): L(θ) = Σᵢ(yᵢ-̂yᵢ)


## Support Vector Mechine

- **Hard Margin Support Vector Machine** 

$$ d= \frac{1}{|̄\vec{W}|}$$

$$ y_i(\bar{X_i}-\bar{W}+b)-1 =0$$

$$L = 0.5 |\bar{W}|^2-∑ λ_i [ y_i(\bar{X_i}-\bar{W}+b)-1 ]
$$

$$ \bar{W} = ∑ λ_iy_i\bar{X}_i$$

$$L = ∑ \lambda_i -\frac{1}{2} \sum_{ij}\lambda_i\lambda_jy_iy_j(\bar{X}_i\bar{X}_j)$$

- Decision Rule 
$$sgn(∑\lambda_iy_i(\bar{X}_i\cdot\bar{X})+b)$$


- If there are dataset which cannot be separated by a single straght hyperplane we need a transformation that transfroms the ̄Xᵢ to ̄Xᵢ' where we can have a hyperplane

- φ(̄X) 
- K(̄Xᵢ,̄Xᵢ) = φ(̄Xᵢ)⋅φ(̄Xⱼ)
    1) (̄Xᵢ̄Xⱼ₊₁)ⁿ 
    2) e^{} 

- **Soft Margin Support Vector Machine**
- Variable - `Margin of error`
- condition -  $$y_i(\bar{X_i}-\bar{W}+b)≥ 1 - ϵ$$ where ϵ ≥ 0
    - Total error - $\sum_i\epsilon_i$

    - $$\frac{1}{2} |\bar{W}|^2+ c\sum\epsilon_i$$
    - c - large → Hard Margin
    - c - small → Soft Margin