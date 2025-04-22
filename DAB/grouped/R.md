

# R Squared {#r-squared}


R², or the coefficient of determination, ==measures the proportion of variance in the dependent variable that is explained by the independent variables== in a [regression](#regression) model.

**Interpretation**:  
- R² values range from 0 to 1.
- A value of 1 indicates perfect predictions, meaning the model explains all the variability of the response data around its mean.
- Higher R² values signify a better fit of the model to the data. However, it can be misleading when adding more predictors, as R² will never decrease when more variables are added to a model. See [Adjusted R squared](#adjusted-r-squared).

**Formula**:  
$$R^2 = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}$$

Where:
- $y_i$ = actual values
- $\hat{y}_i$ = predicted values
- $\bar{y}$ = mean of the actual values

**Example**:  
An R² of 0.60 indicates that 60% of the variability observed in the target variable is explained by the regression model.

#### Follow up Questions
- [R-squared metric not always a good indicator of model performance in regression](#r-squared-metric-not-always-a-good-indicator-of-model-performance-in-regression)

# R Squared Metric Not Always A Good Indicator Of Model Performance In Regression {#r-squared-metric-not-always-a-good-indicator-of-model-performance-in-regression}

R-squared (R²) is a commonly used metric for assessing the performance of regression models, but it is not always a reliable indicator of model quality. It should not be the sole criterion for evaluating model performance. It is essential to consider other metrics, such as [Adjusted R squared](#adjusted-r-squared), [Cross Validation](#cross-validation) results, and the overall context of the analysis.

1. **Increased Complexity**: R² will never decrease when more predictors are added to a model, even if those predictors do not have a meaningful relationship with the dependent variable. This can lead to overfitting, where the model captures noise rather than the underlying data pattern.

2. **Lack of Context**: A high R² value does not necessarily imply that the model is appropriate for prediction. ==It only indicates the proportion of variance explained==. A model with a high R² might still have poor predictive performance if it does not generalize well to new data.

3. **Non-linearity**: R² assumes a linear relationship between the independent and dependent variables. If the true relationship is non-linear, R² may provide a false sense of model adequacy.

4. **Ignoring Model Assumptions**: R² does not account for whether the assumptions of the regression model (such as homoscedasticity, independence, and normality of residuals) are met. A model may have a high R² but still violate these assumptions, leading to unreliable results.

5. **Adjusted R²**: To address some of these issues, Adjusted R² is often used, which adjusts the R² value based on the number of predictors in the model. It provides a more accurate measure of model performance when comparing models with different numbers of predictors.



# R {#r}



# Rag {#rag}


Rag is a framework the help [LLM](#llm) be more up to date.

RAG grounds the Gen AI in external data.

>[!Summary]
> Given a question sometimes the answer given is wrong, issue with [LLM](#llm) is no source of data and is out of date.  **RAG** is a specific architecture used in natural language processing ([NLP](#nlp)), where a **retrieval mechanism** is combined with a **generative model** ([Generative](#generative)) (often a [Transformer](#transformer) like GPT). RAG systems are designed to ==enhance the ability of a generative model to answer questions or generate content by incorporating factual knowledge retrieved from external data sources== (such as documents, databases, or knowledge repositories). RAG is the connection of [LLM](#llm)'s with external databases. 

>[!Example]
> **Example of a RAG System**:
> - A user asks: *"What is the capital of France?"*
> - The **retrieval module** fetches a relevant document (e.g., from Wikipedia) that contains the information about France’s capital.
> - The **generation module** synthesizes the response: "The capital of France is Paris."

### [LLM](#llm) Challenges
- Responses are sometimes no sources and out of date.
- LLM's are trained on some store of data (static).  We want this store to be updated.
### Key characteristics of RAG:

![Pasted image 20240928194559.png|500](./images/Pasted%20image%2020240928194559.png|500)

Based on a [Prompting](#prompting).

1. **Retrieval Component**:
   - This module fetches relevant documents (and up to date) or information from an external corpus based on the query or input. It may use traditional search methods like dense vector retrieval (e.g., using embeddings) or keyword-based retrieval.
   - Retriever should be good enough to give the most truthful information based on the store
   
2. **Generative Component**:
   - After retrieving relevant documents, the [Generative](#generative) model (such as GPT or [BERT](#bert)-based models) synthesizes the final response, integrating both the input query and the retrieved information to generate more accurate and contextually informed outputs.
   
3. **Augmentation with External Knowledge**:
   - Instead of solely relying on pre-trained internal knowledge (as in traditional language models), RAG setups use the external knowledge source for augmenting generation, thus improving factual accuracy and reducing the risk of hallucinations (incorrect or fabricated responses).

Model should be able to saay "I dont know" instead of [hallucinating](#hallucinating)

### Resources



[RAG](#rag)
Problems
1. LLMs struggle with memorization > "LLMs may struggle with
tasks that require domain-specific expertise or up-to-date
information.
2. LLMs struggle with generating factually inaccurate content
(hallucinations)
Solution
· A lightweight retriever (SM) to extract relevant document
fragments from external knowledge bases, document collections,
or other tools

 Differnet RaG techinwuqs:
 Spare retrievers, BM25, dense retrtivers. Use Bert for similarity matching:

![Pasted image 20241017165540.png](./images/Pasted%20image%2020241017165540.png)

#### REST API
- REST stands for Representational State Transfer.
- It is a ==standardized== software architecture style used for API communication between a client and a server.

**Benefits of REST APIs:**
1. **Simplicity and Standardization:**
   - Data formatting and request structuring are standardized and widely adopted.
2. **Scalability and Statelessness:**
   - Easily modifiable as service complexity grows without tracking data states across client and server.
3. **High Performance:**
   - Supports ==caching==, maintaining high performance even as complexity increases.

**Main Building Blocks:**
1. **==Request==:**
   - Actions (==[CRUD](#crud)==): Create (POST), Read (GET), Update (PUT), Delete (DELETE).
   - Components: Operation (==HTTP method==), Endpoint, Parameters/Body, Headers.
2. **Response:**
   - Typically in [JSON](#json) format.

**REST API Example:** 
ice cream shop inteacting with cloud database.
- Endpoint example: "icecream.com/api/flavors"
  - "api" indicates the API portion.
  - "flavors" refers to the ==resource== being accessed or modified.

**Real-world Examples:**
1. **Get Flavors:**
   - Operation: ==GET==
   - Endpoint: "/api/flavors"
   - Response: Array of flavor resources.
2. **Update Flavor:**
   - Operation: ==PUT==
   - Endpoint: "/api/flavors/1"
   - Body: New flavor data.
   - Response: Confirmation of update.
3. **Create New Flavor:**
   - Operation: ==POST==
   - Endpoint: "/api/flavors"
   - Body: New flavor data.
   - Response: Confirmation of creation.

# Roc (Receiver Operating Characteristic) {#roc-receiver-operating-characteristic}


**ROC (Receiver Operating Characteristic)** is a graphical representation of a classifier's performance across different thresholds, showing the trade-off between sensitivity (true positive rate) and specificity (1 - false positive rate).

 A graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

It plots the true positive rate (TPR) against the false positive rate (FPR) at different threshold settings.

In [ML_Tools](#ml_tools) see: [ROC_Curve.py](#roc_curvepy)
#### Why Use Predicted Probabilities?

In ROC analysis, predicted probabilities (`y_probs`) are used instead of predicted classes (`y_pred`) because the ROC curve evaluates the model's performance across different threshold levels. Probabilities allow you to adjust the threshold to see how it affects sensitivity and specificity.

#### Threshold Level

The threshold level is the probability value above which an instance is classified as the positive class. Adjusting the threshold affects [Recall](#recall) and [Specificity](#specificity)

- Lower Threshold: Increases sensitivity but may decrease specificity.
- Higher Threshold: Increases specificity but may decrease sensitivity.

#### Example Code

```python
from sklearn.metrics import roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt

# Actual and predicted values
y_act = [1, 0, 1, 1, 0]
y_pred = [1, 1, 0, 1, 0]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_act, y_pred)

# Display ROC curve
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()
```

#### [Logistic Regression](#logistic-regression) Example

```python
from sklearn.linear_model import LogisticRegression

# Train a logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Predict probabilities for the positive class
y_probs = logreg.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', lw=2, label='Random Guessing')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend()
plt.show()
```

# Roc_Curve.Py {#roc_curvepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Selection/ROC_Curve.py
## **Overview**

This script demonstrates how to compute and interpret Receiver Operating Characteristic (ROC) curves and Area Under the ROC Curve (AUROC) scores using Random Forest and Naive Bayes classifiers. Below is the step-by-step breakdown:
## **Script Flow**

- **Generate Synthetic Dataset**  
  - Creates a binary classification dataset with 2,000 samples and 10 features.  
  - Simulates a realistic classification problem.

- **Add Noisy Features**  
  - Appends random, irrelevant features to increase the dataset's complexity.  
  - Mimics challenging real-world scenarios where not all features are informative.

- **Split the Data**  
  - Divides the dataset into training (80%) and testing (20%) subsets.  
  - Ensures unbiased model evaluation on unseen data.

- **Train Classification Models**  
  - Builds two models:  
    - **Random Forest**: A robust ensemble-based classifier.  
    - **Naive Bayes**: A simple probabilistic model based on Bayes' theorem.

- **Generate Prediction Probabilities**  
  - Computes predicted probabilities for each class.  
  - Retains probabilities for the positive class to construct the ROC curve.

- **Compute AUROC and ROC Curve Values**  
  - Calculates:  
    - **AUROC**: Measures model performance (higher is better).  
    - **ROC Values**: False Positive Rate (FPR) and True Positive Rate (TPR) across thresholds.

- **Visualize ROC Curve**  
  - Plots FPR (x-axis) against TPR (y-axis) for each model.  
  - Includes AUROC scores in the legend for comparison.
## **Key Outputs**
- **AUROC Scores**  
  - Evaluates the overall discriminative power of the classifiers.  

- **ROC Plot**  
  - Visualizes how well each model distinguishes between positive and negative classes across thresholds.  
  - A random prediction baseline is included for reference.
## **Conclusion**
This script illustrates the process of building, evaluating, and visualizing classification models using ROC curves. It highlights the strengths and weaknesses of different models in distinguishing classes.

# Output

### Interpretation of the Script Output

- **Random (Chance) Prediction: AUROC = 0.500**
    - An AUROC score of **0.500** represents a random guessing model with no predictive power.
    - The model's True Positive Rate (TPR) is equal to its False Positive Rate (FPR) across all thresholds, resulting in a diagonal line on the ROC curve.
    
- **Random Forest: AUROC = 0.922**
    - An AUROC score of **0.922** indicates excellent model performance.
    - The Random Forest classifier has a high ability to distinguish between positive and negative classes, with a much higher TPR than FPR across thresholds.
- 
- **Naive Bayes: AUROC = 0.993**
    - An AUROC score of **0.993** suggests near-perfect model performance.
    - The Naive Bayes classifier has an extremely high discriminative power, with TPR approaching 1 and FPR close to 0 for most thresholds.

### Summary

- The **Naive Bayes classifier** outperforms the **Random Forest classifier** in this specific setup.
- Both models significantly outperform random guessing (baseline AUROC = 0.500), indicating their utility for this classification task.
- However, such high performance (especially for Naive Bayes) may suggest that the dataset or features are particularly well-suited to the model, or there may be minimal noise in the classification task. Further evaluation (e.g., on new datasets) is recommended to confirm robustness.

# Race Conditions {#race-conditions}



# Random Forest Regression {#random-forest-regression}

Random Forest Regression
: Like random forests for classification, random forest regression combines multiple regression trees to improve prediction accuracy.

# Random Forests {#random-forests}



A random forest is an [Model Ensemble](#model-ensemble) of [Decision Tree](#decision-tree)s. Take many decision trees decisions to get better result.

What is the Random Forest method;; an ensemble learning method based on constructing multiple decision trees during training and combining their predictions through averaging. Random Forests are flexibility, robustness, and ability to handle high-dimensional data, as well as their resistance to overfitting.

What is an issue with [Random Forests](#random-forests);; susceptible to overfitting, especially when dealing with noisy or high-dimensional data. Proper tuning of hyperparameters like the number of trees and maximum depth is crucial to mitigate this.

Random forests combine multiple decision trees to improve accuracy and generalisation.

**What is Random Forest, and how does it work?**;; Random Forest is an method that can perform regression, classification, dimensionality reduction, and handle missing values. It builds multiple decision trees and combines their outputs. Each tree is grown using a subset of the data and features, and the final output is determined by aggregating the predictions of individual trees.

Remember that for a Random Forest, we randomly choose a subset of the features AND randomly choose a subset of the training examples to train each individual tree.

if $n$ is the number of features, we will randomly select $\sqrt{n}$ of these features to train each individual tree. 
- Note that you can modify this by setting the `max_features` parameter.

You can also speed up your training jobs with another parameter, `n_jobs`. 
- Since the fitting of each tree is independent of each other, it is possible fit more than one tree in parallel. 
- So setting `n_jobs` higher will increase how many CPU cores it will use. Note that the numbers very close to the maximum cores of your CPU may impact on the overall performance of your PC and even lead to freezes. 
- Changing this parameter does not impact on the final result but can reduce the training time.

![Pasted image 20240128194716.png|500](./images/Pasted%20image%2020240128194716.png|500)
![Pasted image 20240118145117.png|500](./images/Pasted%20image%2020240118145117.png|500)





What is an issue with [Random Forests](#random-forests);; susceptible to overfitting, especially when dealing with noisy or high-dimensional data. Proper tuning of hyperparameters like the number of trees and maximum depth is crucial to mitigate this.

[Decision Tree](#decision-tree) are not the best - need to make flexible for new data. THey work well with the data set they are defined on.

How to proceed with random forest: (build tree's randomly) i.e solve the issue with decision trees. Processis called [Bagging](#bagging)
1) randomly select a dataset (bootstrap)
2) randomly select two (or multiple) features for each branch and proceed like in decision tree.

variety makes trees better.

To make a prediction , run data through trees in forest, and get prediction, conclude with majority prediciton.

How to know if random forest is good ?

Use data that was not in boot strap data set - measure the accuracy based on these classiifcations.

Refine the random forest by qweaking the [Hyperparameter](#hyperparameter) of number of features used per step.


# React {#react}


React is a [JavaScript](#javascript) library developed by Meta for building user interfaces, particularly in web development.

Related to:
- [Dashboarding](#dashboarding)

### Core Concepts

React's component-based architecture allows for reusable UI elements, enhancing maintenance and testing. It uses a Virtual DOM for efficient updates, minimizing direct DOM manipulation. Data flows unidirectionally from parent to child components.

### Main Use Cases

React is ideal for Single Page Applications (SPAs) that load once and update dynamically, as well as for complex, interactive user interfaces and real-time applications like dashboards.
### Common Tools

Popular UI libraries include Tailwind CSS and shadcn/ui.


# Reasoning Tokens {#reasoning-tokens}


Transformers rely on pattern recognition and language-based reasoning.

Thus, **reasoning tokens serve as a mechanism for token-based logical progression**, allowing models like ChatGPT to simulate math insights by leveraging pattern recognition, token relationships, and sequential reasoning, even without explicit symbolic or mathematical processing built into the model itself.

[Mathematical Reasoning in Transformers](#mathematical-reasoning-in-transformers)

In the context of models like ChatGPT, **reasoning tokens** refer to the individual pieces of language that contribute to the step-by-step logical process used by the model to solve problems, including mathematical ones.

### 5. **Logical Continuity and Error Correction**

- Reasoning tokens enable the model to maintain **logical continuity**, allowing it to backtrack or adjust outputs based on the sequence of previously generated tokens. For example, if the model makes a mistake in an earlier step (like a miscalculation), it can revise its response as it generates subsequent tokens that recognize the inconsistency.

# Recall {#recall}


**Recall Score** is a [Evaluation Metrics](#evaluation-metrics) used to evaluate the performance of a [Classification](#classification) model, focusing on the model's ability to **identify all relevant instances of the positive class**.

It answers the question: ==**How many relevant items are retrieved?**==

High recall means that the model is effective at identifying most of the actual spam emails. This is useful in environments where missing a spam email could lead to security risks such as in corporate email systems.

The formula for recall is:

$$\text{Recall} = \frac{TP}{TP + FN}$$

Importance
- Recall is crucial in scenarios where ==**missing a positive instance is costly**==, such as in disease screening or fraud detection.
- It helps in understanding how well the model captures all the actual positive instances.

Related Concepts
- **Sensitivity** (also known as recall or the true positive rate) measures the proportion of actual positives that are correctly identified by the model. It indicates how well the model is at identifying positive instances.

![Pasted image 20241222091831.png](./images/Pasted%20image%2020241222091831.png)

# Recommender Systems {#recommender-systems}


Crab on Python

A recommender system, or recommendation system, is a type of information filtering system that aims to predict the preferences or interests of users by analyzing their behavior and the behavior of similar users or items. These systems are widely used in various applications, such as e-commerce, streaming services, social media, and content platforms, to provide personalized recommendations to users.

### Key Components of Recommender Systems:

1. **User Data**: Information about users, such as their preferences, ratings, purchase history, and interactions with items.

2. **Item Data**: Information about the items being recommended, which can include attributes, descriptions, and metadata.

3. **Recommendation Algorithms**: The methods used to generate recommendations. Common approaches include:
   - **Collaborative Filtering**: This technique relies on the behavior and preferences of similar users. It can be user-based (finding similar users) or item-based (finding similar items).
   - **Content-Based Filtering**: This approach recommends items based on the features of the items and the preferences of the user. For example, if a user likes action movies, the system will recommend other action movies based on their attributes.
   - **Hybrid Methods**: Combining collaborative and content-based filtering to improve recommendation accuracy and overcome limitations of each method.

4. **Evaluation Metrics**: Metrics used to assess the performance of the recommender system, such as precision, recall, F1 score, and mean average precision.

### Applications of Recommender Systems:

- **E-commerce**: 
- **Streaming Services**: 
- **Social Media**:
- **News and Content Platforms**:


# Recurrent Neural Networks {#recurrent-neural-networks}


Recurrent Neural Networks (RNNs) are a type of [neural network](#neural-network) designed to process sequential data by maintaining a memory of previous inputs through hidden states. This makes them suitable for tasks where the order of data is needed, such as:

- [Time Series](#time-series) prediction, 
- speech recognition, 
- and [NLP|natural language processing](#nlpnatural-language-processing) (NLP). 

RNNs have loops in their architecture, ==allowing information to persist across sequence steps.== However, they face challenges with long sequences due to [vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem). To address these issues, variants like Long Short-Term Memory ([LSTM](#lstm)) and [Gated Recurrent Unit](#gated-recurrent-unit) (GRU) have been developed.
### Resources:
[Video link](https://www.youtube.com/watch?v=TLLqsEyt8NI&list=PLcWfeUsAys2nPgh-gYRlexc6xvscdvHqX&index=9)
https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks

### Key Concepts of RNNs

#### Sequential Data Handling
- RNNs maintain a hidden state that acts as memory, enabling them to model temporal dependencies. This is essential for tasks where the current output depends on both current and previous inputs.
- At each time step, RNNs process an input, combine it with the previous hidden state, and produce an output along with an updated hidden state.
- The hidden state carries forward information influenced by all previous inputs, theoretically allowing RNNs to remember long-term dependencies.


![Pasted image 20241219073017.png](./images/Pasted%20image%2020241219073017.png)


#### [Backpropagation](#backpropagation) Through Time (BPTT)
- RNNs are trained using BPTT, a variant of backpropagation. The network unrolls over time, treating each time step as a layer in a deep network.
- Gradients are computed for each time step, and weights are updated based on cumulative error across the sequence. This allows learning of long-term dependencies but can lead to vanishing and exploding gradients in long sequences.
#### Variants of RNNs
- **LSTM**: Introduces gates (input, forget, output) to control information flow, addressing vanishing gradients and improving long-sequence handling.
- **GRU**: A simpler variant of LSTM with fewer parameters, offering efficiency and ease of training while maintaining performance on sequence tasks.
### Example Code (RNN in Python with [PyTorch](#pytorch))

See RNN_Pytorch.py

### Problem of Long Term Dependencies

The more time steps we include the less data we keep from the past.

Solutions: [LSTM](#lstm) and [GRU](#gru): use gates: but are costly in computation.

![Pasted image 20241219073440.png](./images/Pasted%20image%2020241219073440.png)

### Other areas

[Use of RNNs in energy sector](#use-of-rnns-in-energy-sector)

### RNNS and [Transformer|Transformers](#transformertransformers)

Why have [Transformer|Transformer](#transformertransformer)'s have replaced traditional RNN.

[Inductive Reasoning, Memories and Attention.](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

How to address the limitations of vanilla recurrent networks. 

Issues:
- RNNs are not inductive: They memorize sequences extremely well, but don't generalise well.
- They couple their representation size to the amount of computation per step. 


# Recursive Algorithm {#recursive-algorithm}



# Regression Metrics {#regression-metrics}


These metrics provide various ways to evaluate the performance of regression models.

### Evaluating Regression Models

These metric provide:

1. **Comprehensive Evaluation**: Each metric provides a different perspective on model performance. For example, while MSE and RMSE give insights into the average error magnitude, MAE provides a straightforward average error measure, and R-squared indicates how well the model explains the variance in the data.

2. **Sensitivity to [standardised/Outliers](#standardisedoutliers)**: Metrics like MSE and RMSE are sensitive to outliers due to the squaring of errors, which can be useful if you want to emphasize larger errors. In contrast, MAE and Median Absolute Error are more robust to outliers.

3. **[Interpretability](#interpretability)**: RMSE is in the same units as the target variable, making it easier to interpret in the context of the data. This can be particularly useful for stakeholders who need to understand the model's performance in practical terms.

4. **Model Comparison**: These metrics allow you to compare different models or configurations to determine which one performs best on your data.

5. **Variance Explanation**: R-squared and Explained Variance Score provide insights into how much of the variability in the target variable is captured by the model, which is crucial for understanding the model's effectiveness.
### Common Regression Metrics

#### Mean Absolute Error (MAE):
   - Definition: MAE measures the average absolute differences between predicted and actual values.
   - Interpretation: Lower values indicate better model performance, as it reflects fewer errors in predictions.
   - Formula: 
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
   - Where:
     - $n$ = number of observations
     - $y_i$ = actual value
     - $\hat{y}_i$ = predicted value

#### Mean Squared Error (MSE):
   - Definition: MSE calculates the average of the squares of the errors (the differences between predicted and actual values).
   - Interpretation: Like MAE, lower values are better. However, MSE is more sensitive to outliers due to the squaring of errors, which can disproportionately affect the metric. Greater error values are exaggerated.
   - Formula: 
   $$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$
   - Where:
     - $n$ = number of observations
     - $y_i$ = actual value
     - $\hat{y}_i$ = predicted value

#### Root Mean Squared Error (RMSE):
   - Definition: RMSE is the square root of MSE, providing an error metric in the same units as the target variable.
   - Interpretation: Lower RMSE values indicate better model performance, and it also emphasizes larger errors due to the squaring process. Easier to interpreted ([interpretability](#interpretability)), back to the same scale as the input.
   - Formula: 
   $$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

#### [R squared](#r-squared)

#### [Adjusted R squared](#adjusted-r-squared)

#### Median Absolute Error:
   - Definition: This metric measures the median of the absolute errors between predicted and actual values.
   - Interpretation: It provides a robust measure of prediction accuracy, especially in the presence of [standardised/Outliers](#standardisedoutliers).
   - Formula: 
   $$ \text{MedAE} = \text{median}(|y_i - \hat{y}_i|) $$

#### Explained Variance Score:
   - Definition: This metric measures the proportion of variance in the target variable that is predictable from the features.
   - Interpretation: Higher values indicate that the model explains a greater proportion of the [Variance](#variance) in the target variable.
   - Formula: 
  $$\text{Explained Variance} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$
   - Where:
     - $\text{Var}(y)$ = variance of the actual values
     - $\text{Var}(y - \hat{y})$ = variance of the prediction errors

### Implementation

See Regression_Metrics.py



# Regression {#regression}



>[!Summary]  
> [Regression](#regression) analysis is a statistical method used to ==predict== a continuous variable based on one or more predictor variables. The most common form, [Linear Regression](#linear-regression), assumes a linear relationship between the dependent variable $y$ and independent variables $x_1, x_2, \dots, x_n$. The goal is to minimize the residual sum of squares (RSS) between observed and predicted values. Other forms, such as [Logistic Regression](#logistic-regression), handle classification problems.
> 
> Regression models can incorporate techniques like regularization ($L_1$, $L_2$) to improve performance and prevent overfitting, especially with high-dimensional data. Advanced methods like [Polynomial Regression](#polynomial-regression) address non-linearity, while generalized linear models (GLMs) extend regression to non-normal response variables.  
> 
> **Regressor**: This is a type of model used for regression tasks, where the goal is to predict continuous values. For example, a regressor might be used to predict the price of a house based on its features, or to forecast future sales figures.


>[!Breakdown]  
> Key Components:  
> - [Linear Regression](#linear-regression): Predicts $y = \beta_0 + \beta_1x_1 + \dots + \beta_nx_n + \epsilon$, where $\epsilon$ is the error term.  
> - Regularization: Adds $L_1$ ([Lasso](#lasso)) or $L_2$ ([Ridge](#ridge)) penalty to prevent overfitting in high-dimensional data.  
> - Feature transformation: [Polynomial Regression](#polynomial-regression) and logarithmic transformations adjust for non-linearity in data.  
> - Regression is a type of [Supervised Learning](#supervised-learning).

>[!important]  
> - $R^2$ is a key metric, showing how much of the variance in $y$ is explained by $x$.  
> - [Multicollinearity](#multicollinearity) can inflate variances of coefficient estimates, harming model reliability.  

>[!attention]  
> - Regression assumes [linearity](#linearity), so improper application to non-linear data can lead to biased predictions.  
> - Overfitting can occur with too many predictors, especially in small datasets.  

>[!Example]  
> In predicting insurance claims, a linear regression model could take input variables like age and driving history to estimate the expected claim amount. A transformation, such as logarithmic scaling, could address any non-linear patterns between variables.  

>[!Follow up questions]  
> -  How can [Polynomial Regression](#polynomial-regression) improve predictions in non-linear datasets?  
> -  What are the benefits of combining [Linear Regression](#linear-regression) with [Feature Engineering](#feature-engineering) for complex datasets?  


# Regression_Logistic_Metrics.Ipynb {#regression_logistic_metricsipynb}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Selection/Model_Evaluation/Classification/Regression_Logistic_Metrics.ipynb

# Regularisation Of Tree Based Models {#regularisation-of-tree-based-models}


Tree models, such as Random Forests and Gradient Boosting, can also be regularized, although they don’t use L1 or L2 regularization directly. Instead, they are regularized through hyperparameters like max depth, min samples split, and learning rate to control the complexity of the trees.

In tree-based models, regularization is not applied in the same way as it is in linear models (i.e., using L1 or L2 penalties). 

In tree models, [Regularisation](#regularisation) is done by controlling the growth of the trees using [hyperparameters](#hyperparameters) like 

- `max_depth`,
- `min_samples_split`,
- `min_samples_leaf`.

These hyperparameters restrict the growth of the tree, preventing it from becoming too complex.

For [Model Ensemble](#model-ensemble) methods like Random Forests and Gradient Boosting, additional regularization techniques like 

- subsampling, 
- bootstrap sampling, 
- and learning rate control

to help prevent overfitting. These techniques effectively restrict the model complexity, leading to better generalization .

Below are the common regularization techniques used in tree models such as [Decision Tree](#decision-tree), [Random Forests](#random-forests).

### Regularization in Different Tree Models

- Decision Trees: Prone to overfitting when not regularized, since they tend to grow large and complex trees. Regularization through pruning, limiting tree depth, and controlling minimum samples per split or leaf is critical.

- Random Forests: Regularization is mainly achieved through the use of multiple decision trees, random feature selection (`max_features`), and bootstrapping (`bootstrap`). Each tree learns a different part of the data, which reduces overfitting.

- Gradient Boosting Models (GBMs): Regularized by tuning the `learning_rate`, `subsample`, and controlling the tree depth and other tree-based hyperparameters like `min_samples_split`. The slower learning process with a smaller learning rate combined with these hyperparameters helps prevent overfitting.
### Regularization Techniques for Tree Models

1. Limiting Tree Depth:
    - Max Depth (`max_depth`): This parameter restricts the maximum depth of the tree. Trees that are too deep can model complex patterns, but they are prone to overfitting. By limiting the depth, you constrain the tree's ability to learn highly specific patterns in the training data.
    - Example: In scikit-learn, setting `max_depth` for a Decision Tree or Random Forest.

    ```python
    from sklearn.tree import DecisionTreeClassifier
    
    # Limit tree depth to regularize the model
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    ```

2. Minimum Samples for Splitting:
    - Min Samples Split (`min_samples_split`): This parameter specifies the minimum number of samples required to split an internal node. Increasing this value makes the tree more conservative and prevents it from splitting when there are too few samples, thus controlling its complexity.
    - This helps avoid creating splits based on noise, which could lead to overfitting.

    ```python
    model = DecisionTreeClassifier(min_samples_split=10)
    model.fit(X_train, y_train)
    ```

3. Minimum Samples per Leaf:
    - Min Samples Leaf (`min_samples_leaf`): This parameter sets the minimum number of samples a node must have after a split to be a leaf. Higher values result in fewer splits, producing simpler trees that are less likely to overfit.
    - This also encourages broader splits that require more data, reducing the sensitivity to outliers.

    ```python
    model = DecisionTreeClassifier(min_samples_leaf=4)
    model.fit(X_train, y_train)
    ```

4. Max Number of Features:
    - Max Features (`max_features`): This controls the number of features to consider when looking for the best split. Reducing the number of features makes the model less likely to overfit, as it limits the search space for splits. For Random Forests, this also introduces randomness that can improve generalization.

    ```python
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(max_features='sqrt')  # Uses sqrt of total features
    model.fit(X_train, y_train)
    ```

5. Max Leaf Nodes:
    - Max Leaf Nodes (`max_leaf_nodes`): This parameter limits the total number of leaf nodes the tree can have. Fewer leaf nodes result in simpler trees that are less likely to overfit the training data.

    ```python
    model = DecisionTreeClassifier(max_leaf_nodes=10)
    model.fit(X_train, y_train)
    ```

6. Subsampling (for Ensemble Methods like Random Forests and Gradient Boosting):
    - Bootstrap Sampling (`bootstrap`): For Random Forests, regularization is achieved through bootstrapping (random sampling with replacement) during training. This introduces variability and helps prevent overfitting.
    - Subsample (`subsample`): For Gradient Boosting, the `subsample` parameter controls the fraction of the training data used for fitting each individual tree. A value less than 1 introduces randomness and reduces the chance of overfitting, similar to how dropout works in neural networks.

    ```python
    from sklearn.ensemble import GradientBoostingClassifier
    
    model = GradientBoostingClassifier(subsample=0.8)  # Use 80% of data for each tree
    model.fit(X_train, y_train)
    ```

7. Learning Rate (for Gradient Boosting Models):

    - Learning Rate (`learning_rate`): This parameter controls how much each tree contributes to the overall model in Gradient Boosting. A lower learning rate slows down the learning process, requiring more trees but helping to avoid overfitting by making small adjustments at each step.

    ```python
    model = GradientBoostingClassifier(learning_rate=0.1)
    model.fit(X_train, y_train)
    ```

8. Pruning:
    - For Decision Trees, pruning is a post-processing regularization technique where branches that contribute little to the overall performance of the model are removed. This prevents the tree from learning noise in the data.
    - In scikit-learn, Cost Complexity Pruning (`ccp_alpha`) is used for pruning. A larger value of `ccp_alpha` leads to more aggressive pruning, simplifying the tree.

    ```python
    model = DecisionTreeClassifier(ccp_alpha=0.01)
    model.fit(X_train, y_train)
    ```



# Regularisation {#regularisation}


Regularization is a technique in machine learning that reduces the risk of overfitting by adding a penalty to the [Loss function](#loss-function) during model training. This penalty term restricts the magnitude of the model's parameters, thereby controlling the complexity of the model. It is especially useful in linear models but can also be applied to more complex models like neural networks.

### Key Concepts

- **$L_1$ Regularization ([Lasso](#lasso)):** Adds the absolute value of the coefficients to the loss function, encouraging sparsity by driving some coefficients to zero, effectively selecting a subset of features.
  
- **$L_2$ Regularization ([Ridge](#ridge)):** Adds the square of the coefficients to the loss function, shrinking them toward zero. It encourages smaller coefficients but does not push them exactly to zero, helping reduce overfitting by penalizing large weights.

- **Elastic Net:** Combines both Lasso and Ridge regularization.

### Benefits

- **Prevents Overfitting:** Regularization adds a penalty term to the loss function to avoid overfitting.
- **Feature Sparsity:** $L_1$ encourages feature sparsity, while $L_2$ reduces coefficient magnitudes.
- **Enhanced Generalization:** Dropout enhances generalization by preventing unit co-adaptation in neural networks.

### Considerations

- **Underfitting Risk:** Over-penalizing parameters can lead to underfitting, where the model becomes too simplistic.
- **Tuning $\lambda$:** Choosing the right penalty term (i.e., $\lambda$) is crucial for balancing bias and variance.

[When and why not to us regularisation](#when-and-why-not-to-us-regularisation)
### Questions
- How does the balance between $L_1$ and $L_2$ regularization impact model performance in large feature spaces?
- What are the best practices for tuning the $\lambda$ parameter in regularization? [Model Parameters Tuning](#model-parameters-tuning).

### Example

Consider a linear regression model with $L_2$ regularization (Ridge). The [loss function](#loss-function) would be:

$$ J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 $$

Here, $\lambda$ controls the strength of the regularization. Higher $\lambda$ values shrink the coefficients more.
### Related Topics

- [Feature Selection](#feature-selection): L1 regularization can zero out irrelevant features, improving model [interpretability](#interpretability) and reducing computational costs.
- [Model Selection](#model-selection) techniques for high-dimensional data.

### Applications

Regularization is widely used in linear models but is also applied in other machine learning models, particularly those prone to overfitting:

- [Neural network](#neural-network)
- [Regularisation of Tree based models](#regularisation-of-tree-based-models)

### Implementation

In [ML_Tools](#ml_tools) see: [Regularisation.py](#regularisationpy)


# Regularisation.Py {#regularisationpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Optimisation/Regularisation/Regularisation.py

# Reinforcement Learning {#reinforcement-learning}


Reinforcement Learning ( [Reinforcement learning|RL](#reinforcement-learningrl)) is a branch of machine learning where an agent learns to make decisions by interacting with an environment. The agent receives rewards or penalties based on its actions, and its goal is to ==maximize cumulative reward.==
#### Current Research Problems

1. Sample Efficiency: Many RL algorithms require a large number of interactions with the environment to learn effectively. Research is focused on developing methods to improve sample efficiency, such as model-based approaches and transfer learning.
    
2. [Exploration vs. Exploitation](#exploration-vs-exploitation): Balancing exploration (trying new actions) and exploitation (choosing known rewarding actions) remains a challenge. New strategies, such as curiosity-driven learning and bandit algorithms, are being investigated.
    
3. [Multi-Agent Reinforcement Learning](#multi-agent-reinforcement-learning): Extending RL to environments with multiple interacting agents introduces complexity in learning optimal strategies. Research includes coordination, competition, and communication between agents.
    
4. Robustness and Stability: Ensuring that RL agents perform reliably in changing or adversarial environments is a key area of study. Techniques for robust control and stability analysis are being explored.

### Algorithms in Reinforcement Learning

[Q-Learning](#q-learning)
[Deep Q-Learning](#deep-q-learning)
[Sarsa](#sarsa)
#### Components

- Agent: An entity that interacts with the environment and learns to optimize its actions based on rewards.
- State ($s$): The current situation in the environment, often defined by the positions and attributes relevant to the agent's decision-making.
- Action ($a$): The available moves or decisions an agent can take.
- Reward ($r$): A scalar value received after taking an action, representing feedback from the environment.
- [Policy](#policy) ($\pi$): A strategy that the agent follows, mapping states to actions.
- Q-Value ($Q(s, a)$): The expected cumulative reward for taking a particular action in a given state and following the policy thereafter. The Q-values guide the agent in making decisions that maximize long-term rewards. [Q-Learning](#q-learning)
#### Mathematical Foundations

- [Markov Decision Processes](#markov-decision-processes)
- Dynamic Programming: Techniques such as [Bellman Equations](#bellman-equations) equations are central to RL, as they provide a way to break down complex decision-making problems into simpler subproblems.
- Optimization Techniques: Finding optimal [Policy|polices](#policypolices) often involves advanced optimization methods, including gradient ascent and policy iteration.

### Reinforcement Learning Implementation

Use import gym

Action Space: What actions are available to the agent?
Observation Space: What information is available to the agent?
Reward Envioronment: What rewards can be given to the agent?

Loads the enviroemnt for examples
env = gym.make('LunarLander-v2')

Can step though the environment dynamics
next_state, reward, done, info = env.step(action)

```python
# Select an action
action = 0

# Run a single time step of the environment's dynamics with the given action.
next_state, reward, done, info = env.step(action)

with np.printoptions(formatter={'float': '{:.3f}'.format}):
    print("Initial State:", initial_state)
    print("Action:", action)
    print("Next State:", next_state)
    print("Reward Received:", reward)
    print("Episode Terminated:", done)
    print("Info:", info)
```

# Relating Tables Together {#relating-tables-together}


Implementing these concepts, database tables can be effectively related, ensuring [Data Integrity](#data-integrity), efficient retrieval, and easy maintenance.

Resources:
- [LINK](https://cs50.harvard.edu/sql/2024/weeks/1/)
### Notes on Relating Database Tables

[Primary Key](#primary-key)

[Foreign Key](#foreign-key)

One-to-One Relationships:
   - Each record in Table A relates to one record in Table B and vice versa.
   - Example: Each employee has one unique profile.

One-to-Many Relationships:
   - A single record in Table A can relate to multiple records in Table B.
   - Example: One department has many employees.

[Many-to-Many Relationships](#many-to-many-relationships)

[Junction Tables](#junction-tables)
   - Used to manage many-to-many relationships.
   - Contains foreign keys from both tables it connects.
   - Example: Enrollments table with StudentID and CourseID as foreign keys.

Referential Integrity
   - Ensures that relationships between tables remain consistent.
   - Example: If an employee is assigned a department, the DepartmentID must exist in the Departments table.

Cascading Actions:
   - Cascade Update: Updates related records automatically when a primary key is updated.
   - Cascade Delete: Deletes related records automatically when a primary key is deleted.

[ER Diagrams](#er-diagrams)


# Relational Database {#relational-database}



# Relationships In Memory {#relationships-in-memory}


In managing the memory of a large language model (LLM), several key concepts and techniques play a crucial role in forming and maintaining relationships between data points:

**[RAG](#rag) (Retrieval-Augmented Generation)**:

This technique enhances LLMs by integrating external data retrieval with generative capabilities. By employing chunking and reranking, RAG refines outputs, ensuring that the model can access and utilize relevant information efficiently. This process strengthens the model's ability to form and maintain relationships between different pieces of information, improving its memory and response accuracy.

**Ontology and Correlating Data Points**:

Ontologies establish [semantic relationships](#semantic-relationships) between data points by defining a structured framework of concepts and their interrelations. This structured understanding aids in memory management by providing a clear map of how different pieces of information are related. Similarly, correlating data points involves understanding and forming connections, which is essential for enhancing memory management. Together, these approaches help LLMs organize and interpret information more effectively.

**Vector Store and Metadata Management**:

Utilizing [Vector Database](#vector-database) vector databases allows for efficient storage and retrieval of memory representations. These databases preserve the relationships between data points, enabling LLMs to access and utilize memory more effectively. Alongside this, managing metadata is crucial for organizing, retrieving, and correlating data points. Effective metadata management helps LLMs understand the context and relationships between different pieces of information, enhancing their memory capabilities.

**Structure Memory Graph**: [GraphRAG](#graphrag)

Organizing memory in graph structures allows LLMs to improve relational understanding and connection-making. Graphs provide a visual and structural representation of how information is interconnected, aiding the model's ability to form and maintain complex relationships in memory.

**Cognitive Sciences**:

Insights from cognitive science inform memory design and improve human-AI interaction. By integrating these insights, LLMs can mimic human-like memory processes, enhancing their ability to form, maintain, and retrieve relationships in memory, leading to more natural and effective interactions.

# Reward Function {#reward-function}

[Recurrent Neural Networks|RNN](#recurrent-neural-networksrnn)

# Ridge {#ridge}


L2 Regularization, also known as Ridge Regularization, adds a penalty term proportional to the square of the weights to the [loss function](#loss-function). 

This technique enhances the robustness of linear regression models (and [Logistic Regression](#logistic-regression)) by penalizing large coefficients, encouraging smaller weights overall, and ==distributing weight values more evenly across all features.==

### Key Points

- **Overfitting Mitigation**: Ridge helps mitigate [overfitting](#overfitting), especially in high-dimensional datasets, and is effective in managing [Multicollinearity](#multicollinearity) among predictors.
- **Coefficient Shrinkage**: Unlike Lasso regularization (L1), which can eliminate some features entirely by driving their coefficients to zero, Ridge reduces the magnitudes of coefficients but retains all features.
- **Multicollinearity Handling**: Particularly useful when predictors are highly correlated, as it stabilizes estimates by shrinking the coefficients of correlated features.
- **Feature Retention**: Ridge retains all features in the model, unlike Lasso, which can perform [Feature Selection](#feature-selection)by setting some coefficients to zero.

### Understanding Ridge Regularization

#### 1. Purpose of Ridge Regularization

- **Penalty Addition**: Adds a penalty term to the loss function, proportional to the square of the coefficients (weights), discouraging overly complex models by shrinking the coefficients.

#### 2. Mathematical Formulation

- The loss function for Ridge regression can be expressed as:
  $$\text{Loss} = \text{SSE} + \lambda \sum_{j=1}^{p} b_j^2$$
  - Where:
    - SSE (Sum of Squared Errors) is the original loss function for [linear regression](#linear-regression)
    - $\lambda$ is the regularization parameter (penalty term) that controls the strength of the penalty.
    - $b_j$ are the coefficients of the model.
    - $p$ is the number of predictors.

#### 3. Effect of the Regularization Parameter ($\lambda$)

- **Range**: $\lambda$ can take values from 0 to infinity.
- **Impact**:
  - A small $\lambda$ (close to 0) means the model behaves similarly to ordinary least squares (OLS) regression, with minimal regularization.
  - A large $\lambda$ increases the penalty, leading to smaller coefficients and a simpler model.

#### 4. Finding the Best $\lambda$

- Use techniques like cross-validation to determine the optimal value of $\lambda$. By testing various values and evaluating model performance, select the one that minimizes prediction error (or variance).

### Example Code

```python
from sklearn.linear_model import Ridge

# Initialize a Ridge model
model = Ridge(alpha=0.1)  # alpha is the regularization strength
model.fit(X_train, y_train)
```

### Resources

- [Understanding Ridge Regularization](https://www.youtube.com/watch?v=Q81RR3yKn30)

---

### Understanding the Content

- **L2 Regularization (Ridge)**: This technique is crucial for improving model generalization by penalizing large coefficients, which helps in reducing overfitting and handling multicollinearity. The regularization parameter $\lambda$ controls the trade-off between fitting the training data well and keeping the model coefficients small.

[Ridge](#ridge)
### L2 Regularization (Ridge Regression): for [Neural network](#neural-network)



Adds a penalty term to the loss: \( L_{\text{regularized}} = L + \lambda \cdot ||W||^2 \). This discourages overly complex models by penalizing large weights.

Example:

```python
from tensorflow.keras.regularizers import l2
Dense(25, activation="relu", kernel_regularizer=l2(0.01))
```

# Row Based Storage {#row-based-storage}

Data is stored in consecutive rows, allows [CRUD](#crud)

Row-based storage is well-suited for transactional systems ([OLTP](#oltp)) 

Less efficient than [Columnar Storage](#columnar-storage) in largedatasets.

Row-based Storage Example (Transactional Workloads)**:
For the same table, if the goal is to efficiently handle transactions like inserting or updating an order, **row-based storage** organizes data row by row.

| `order_id` | `customer_id` | `order_date` | `order_amount` |
| ---------- | ------------- | ------------ | -------------- |
| 1          | 101           | 2024-10-01   | $100           |
| 2          | 102           | 2024-10-02   | $150           |
| 3          | 103           | 2024-10-03   | $200           |

In **row-based storage**, entire records (rows) are stored together. For example:
- Row 1: [1, 101, 2024-10-01, $100]

When performing an **insert** or **update**, the entire row can be read and written back quickly, making this method ideal for transactional operations where complete records need to be processed together.

Use case [OLTP](#oltp). For instance, inserting a new order or updating an existing one (like modifying `order_amount` or `customer_id`) is efficient because all the data for a single record is stored together in a row.

# Requirements.Txt {#requirementstxt}



# Reverse Etl {#reverse-etl}


Reverse [ETL](#etl) is the flip side of the [ETL](ETL.md)/[ELT](term/elt.md). **With Reverse ETL, the data warehouse becomes the source rather than the destination**. Data is taken from the warehouse, transformed to match the destination's data formatting requirements, and loaded into an application – for example, a CRM like Salesforce – to enable action.

In a way, the Reverse ETL concept is not new to data engineers, who have been enabling data movement warehouses to business applications for a long time. 

As [Maxime Beauchemin](term/maxime%20beauchemin.md) mentions in [his article](https://preset.io/blog/reshaping-data-engineering/), Reverse ETL “appears to be a modern new means of addressing a subset of what was formerly known as  [Master Data Management (MDM)](master%20data%20management.md).”

Read more about in [Reverse ETL Explained](https://airbyte.com/blog/reverse-etl#so-what-is-a-reverse-etl).

# Rollup {#rollup}


Rollup refers to aggregating data to a higher level of [granularity](#granularity), such as summarizing hourly data into daily totals.

[Database](#database)