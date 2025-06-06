# Logistic Regression Statsmodel Summary Table {#logistic-regression-statsmodel-summary-table}

Statsmodel has this summary table unlike [Sklearn](#sklearn)

[Explanation of summary](https://youtu.be/JwUj5M8QY4U?t=658)

The dependent variable is 'duration'. The model used is a Logit regression (logistic in common lingo), while the method 
- Maximum Likelihood Estimation ([MLE](#mle)). It has clearly converged after classifying 518 observations.
- The Pseudo R-squared is 0.21 which is within the 'acceptable region'.
- The duration variable is significant and its coefficient is 0.0051.
- The constant is also significant and equals: -1.70 (p value close to 0)
- High p value, suggests to remove from model, drop one by one, ie [Feature Selection](#feature-selection).

Specifically a graph such as,
![Pasted image 20240124095916.png](./content/images/Pasted%20image%2020240124095916.png)



$$\mathbb{N}$$