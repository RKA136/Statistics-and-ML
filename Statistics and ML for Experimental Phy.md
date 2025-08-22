# Statistics and ML for Experimental Physics
- Grader - **Avik(DHEP)**
- 3-4 assignments + term paper + project (preferably from own lab)

## Supervised Learning

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