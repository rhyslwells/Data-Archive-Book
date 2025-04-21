Statsmodel has this summary table unlike [Sklearn](./Sklearn.md)

[Explanation of summary](https://youtu.be/JwUj5M8QY4U?t=658)

The dependent variable is 'duration'. The model used is a Logit regression (logistic in common lingo), while the method 
- Maximum Likelihood Estimation ([MLE](./MLE.md)). It has clearly converged after classifying 518 observations.
- The Pseudo R-squared is 0.21 which is within the 'acceptable region'.
- The duration variable is significant and its coefficient is 0.0051.
- The constant is also significant and equals: -1.70 (p value close to 0)
- High p value, suggests to remove from model, drop one by one, ie [Feature Selection](./Feature%20Selection.md).

Specifically a graph such as,
![Pasted image 20240124095916.png]

2.
![Pasted image 20240124095916.png](../images/Pasted%20image%2020240124095916.png)


3.
![Pasted image 20240124095916.png](../images/Pasted%20image%2020240124095916.png)


$$\mathbb{N}$$

[Logistic Regression](./Logistic%20Regression.html)

[Logistic Regression](./Logistic Regression.md)

[[Logistic Regression]]