# A

## Table of Contents
* [AB testing](#ab-testing)
* [ACID Transaction](#acid-transaction)
* [AI Engineer](#ai-engineer)
* [AI governance](#ai-governance)
* [API Driven Microservices](#api-driven-microservices)
* [API](#api)
* [ARIMA](#arima)
* [AUC](#auc)
* [AWS Lambda](#aws-lambda)
* [Accessing Gen AI generated content](#accessing-gen-ai-generated-content)
* [Accuracy](#accuracy)
* [Activation Function](#activation-function)
* [Activation atlases](#activation-atlases)
* [Active Learning](#active-learning)
* [Ada boosting](#ada-boosting)
* [Adam Optimizer](#adam-optimizer)
* [Adaptive Learning Rates](#adaptive-learning-rates)
* [Adding a database to PostgreSQL](#)
* [Addressing Multicollinearity](#addressing-multicollinearity)
* [Addressing_Multicollinearity.py](#addressing_multicollinearitypy)
* [Adjusted R squared](#adjusted-r-squared)
* [Agent-Based Modelling](#agent-based-modelling)
* [Agentic Solutions](#agentic-solutions)
* [Aggregation](#aggregation)
* [Algorithms](#algorithms)
* [Alternatives to Batch Processing](#alternatives-to-batch-processing)
* [Amazon S3](#amazon-s3)
* [Anomaly Detection in Time Series](#anomaly-detection-in-time-series)
* [Anomaly Detection with Clustering](#anomaly-detection-with-clustering)
* [Anomaly Detection with Statistical Methods](#anomaly-detection-with-statistical-methods)
* [Anomaly Detection](#anomaly-detection)
* [Apache Airflow](#apache-airflow)
* [Apache Kafka](#apache-kafka)
* [Apache Spark](#apache-spark)
* [Asking questions](#asking-questions)
* [Attention Is All You Need](#attention-is-all-you-need)
* [Attention mechanism](#attention-mechanism)
* [Automated Feature Creation](#automated-feature-creation)
* [Azure](#azure)



# Ab Testing {#ab-testing}

A/B testing is a method of performance testing two versions of a product like an app.

# Acid Transaction {#acid-transaction}


An ACID [Transaction](#transaction) ensures that either all changes are successfully committed or rolled back, preventing the database from ending up in an inconsistent state. This guarantees the integrity of the data throughout the transaction process.

### Key Properties of ACID Transactions

1. Atomicity: This property ensures that transactions are treated as a single, indivisible unit. If any part of the transaction fails, the entire transaction is rolled back, and none of the changes are applied. Users do not see intermediate states of the transaction.

2. Consistency: Transactions must leave the database in a valid state, adhering to all defined constraints. If a transaction violates a constraint, it is rolled back to maintain the database's stable state.

3. Isolation: This property ensures that concurrent transactions do not interfere with each other. Each transaction operates independently, and the results of one transaction are not visible to others until it is committed.

4. Durability: Once a transaction has been committed, the changes are permanent, even in the event of a system failure. The data remains intact and recoverable.



# Ai Engineer {#ai-engineer}


They know what
- [LSTM](#lstm) means
- [Attention mechanism](#attention-mechanism)
- [Prompting](#prompting) optimisation
- [Neural network](#neural-network)



# Ai Governance {#ai-governance}

AI Governance

[Data Governance](#data-governance)

Used in regulated sectors.  

Constraints to using ai: 
- legal,
- transparency, 
- security, 
- historical bias  

AI acts and standards: 
- eu and AI acts
- NIST standards framework  
- Security OWASP standards for LLM  

how will beaurcracy keeps up with ai innovation.  

Governance can stifle innovation.

# Api Driven Microservices {#api-driven-microservices}


API-driven microservices refer to a [software architecture](#software-architecture) approach where [microservices](#microservices) communicate with each other and with external systems primarily through well-defined [API](#api) (Application Programming Interfaces). 

This architecture is designed to enhance modularity, scalability, and flexibility by breaking down an application into smaller, independent services that can be developed, deployed, and scaled independently.

API-driven microservices architecture is particularly beneficial for large, complex applications that require frequent updates and scaling. It allows organizations to innovate faster, improve fault isolation, and better align development efforts with business needs. However, it also introduces complexity in terms of service orchestration, data consistency, and network communication, which must be carefully managed.

Key characteristics of API-driven microservices include:

1. **Decoupled Services**: Each microservice is a separate, self-contained unit that performs a specific business function. Services are loosely coupled, meaning changes to one service do not directly impact others.

2. **API Communication**: Microservices interact with each other and with external clients through APIs. These APIs are typically RESTful, but they can also use other protocols like gRPC, GraphQL, or messaging systems like Kafka.

3. **Independent Deployment**: Each microservice can be developed, tested, deployed, and scaled independently of the others. This allows for more agile development and continuous deployment practices.

4. **Technology Agnostic**: Different microservices can be built using different technologies or programming languages, as long as they adhere to the agreed-upon API contracts.

5. **Scalability and Resilience**: Microservices can be scaled independently based on demand. If one service fails, it does not necessarily bring down the entire system, enhancing resilience.

6. **Focused Functionality**: Each microservice is designed to handle a specific business capability, making it easier to understand, develop, and maintain.

7. **API Gateway**: Often, an API gateway is used to manage and route requests to the appropriate microservices. It can also handle cross-cutting concerns like authentication, logging, and rate limiting.



# Api {#api}


An API (Application Programming Interfaces) allows one system (client) to <mark>request specific actions from another system</mark> (server).

Using a predefined set of rules and <mark>protocols</mark>. 

Good API documentation is necessary for developers to integrate and use APIs effectively.
####  Resources:
- [Link](https://www.youtube.com/watch?v=yBZO5Rb4ibo)
- [REST API](#rest-api)
- [FastAPI](#fastapi)
#### API Principles

1. **Controlled Access**: APIs provide access to certain parts of a system while keeping the core functionalities secure.
2. **System Independence**: APIs function independently of changes in the underlying system.
3. **Simplicity**: APIs are designed to be <mark>user-friendly</mark> and come with comprehensive documentation to guide developers.
#### Implementation

In [ML_Tools](#ml_tools) see: [Wikipedia_API.py](#wikipedia_apipy)

#### Example:

For instance, a weather app querying a weather API to fetch the current weather conditions involves sending a structured request and receiving a response.

Types of API Connections:
1. **Web APIs**: These facilitate communication between web clients (browsers or apps) and servers. For example, online shopping apps use APIs to process transactions on remote servers.
2. **Database APIs**: These allow applications to interact with databases, ensuring data is accessed and manipulated efficiently.
3. **Device APIs**: When apps like Instagram or WhatsApp request access to your phone's camera or microphone, they use device APIs.

# Arima {#arima}


**ARIMA** (AutoRegressive Integrated Moving Average) is a popular method for [Time Series Forecasting](#time-series-forecasting) that models the autocorrelations within the data. It is particularly useful for datasets with trends and patterns that are not seasonal. However, it is not perfect; for example, it struggles with predicting stock trading data.

## ARIMA Components

- **AR (AutoRegressive)**: Utilizes the dependency between an observation and a number of lagged observations.
- **I (Integrated)**: Involves differencing the data to achieve stationarity.
- **MA (Moving Average)**: Models the dependency between an observation and a residual error from a moving average model applied to lagged observations.

### ARIMA Explained

**ARIMA** stands for:

- **A**uto**R**egressive (AR): Uses past values to predict the current one.
- **I**ntegrated (I): Differencing to make the series stationary.
- **MA**ving Average (MA): Uses past forecast errors for prediction.

A typical **ARIMA(p,d,q)** model has:

- $p$: Number of **lagged values** (AR terms)
- $d$: Number of **differences** needed to make the series stationary
- $q$: Number of **lagged forecast errors** (MA terms)

#### 🔧 What ARIMA Does:

1. **Checks for stationarity** – if not, applies differencing ($d$ times).
2. **Models relationships** between current values and:
   - Past values (AR part)
   - Past errors (MA part)
3. **Fits** parameters by minimizing a loss (typically log-likelihood).
4. **Forecasts** future values using this learned structure.
### Why ARIMA Isn’t Enough for Seasonal Data

Your quarterly data might show **repeating patterns every 4 quarters** (seasonality), and ARIMA has **no built-in mechanism to model this periodic structure**. This is where **SARIMA** comes in.
### SARIMA: Seasonal ARIMA

SARIMA extends ARIMA by adding **seasonal components**. It is written as:

$$\text{SARIMA}(p, d, q)(P, D, Q)_s$$

Where:

- $(p, d, q)$ are **non-seasonal** ARIMA parameters (as above).
- $(P, D, Q)_s$ are **seasonal** ARIMA parameters:
  - $P$: Seasonal autoregressive terms.
  - $D$: Seasonal differences.
  - $Q$: Seasonal moving average terms.
  - $s$: Seasonality period (e.g., $s=4$ for quarterly data).

#### What SARIMA Adds:

- **Seasonal differencing** ($D$): e.g., subtract the value from 4 quarters ago to remove annual cycles.
- **Seasonal AR/MA** terms: Model relationships from past seasonal lags and errors.

### ✅ Summary

| Model   | Handles Trend | Handles Seasonality | Use When...                                   |
|---------|---------------|---------------------|------------------------------------------------|
| ARIMA   | ✅            | ❌                  | Data has trend, but no regular cycles         |
| SARIMA  | ✅            | ✅                  | Data has both trend and seasonality           |
## Advanced Variants

### SARIMAX (Seasonal ARIMA with Exogenous Variables)

- Incorporates external variables (e.g., interest rates, volume) into the forecasting model.
- Useful for [Datasets](#datasets) where external factors influence the time series.

### Related terms

In [ML_Tools](#ml_tools) see:
- https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries/ARIMA.ipynb
- [Forecasting_AutoArima.py](#forecasting_autoarimapy)
- [pmdarima](#pmdarima)

# Auc {#auc}


**AUC (Area Under the Curve)** is a metric for binary classification problems, representing the area under the [ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic)

#### Key Concepts
Represents the area under the ROC curve.

AUC values range from 0 to 1, where 1 indicates perfect classification and 0.5 suggests no discriminative power (equivalent to random guessing).

#### Roc and Auc Score

The `roc_auc_score` is a function from the `sklearn.metrics` module in Python that computes the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. It is a widely used metric for evaluating the performance of binary classification models.

Key Points about `roc_auc_score`:

- **Purpose**: It quantifies the overall ability of the model to discriminate between the positive and negative classes across all possible classification thresholds.
- **Range**: The score ranges from 0 to 1, where:
    - 1 indicates perfect discrimination (the model perfectly distinguishes between the positive and negative classes).
    - 0.5 suggests no discriminative power (equivalent to random guessing).
    - Values below 0.5 indicate a model that performs worse than random guessing.
- **Input**: The function takes the true binary labels and the predicted probabilities (or decision function scores) as inputs.
- **Output**: It returns a single scalar value representing the AUC.

#### Example Code

```python
from sklearn.metrics import roc_auc_score

# Actual and predicted values
y_act = [1, 0, 1, 1, 0]
y_pred = [1, 1, 0, 1, 0]

# Compute AUC
auc = roc_auc_score(y_act, y_pred)
print(f'AUC: {auc}')
```

# Aws Lambda {#aws-lambda}


AWS Lambda is a serverless computing service provided by Amazon Web Services (AWS) that allows you to run code without provisioning or managing servers. 

AWS Lambda is a powerful tool for building scalable, event-driven applications without the overhead of managing server infrastructure.

With AWS Lambda, you can execute your code in response to various events, such as HTTP requests via Amazon API Gateway, changes to data in an Amazon S3 bucket, updates to a DynamoDB table, or messages arriving in an Amazon SQS queue.

Key features of AWS Lambda include:

1. **[Event Driven Events](#event-driven-events)**: AWS Lambda functions are triggered by events, which can come from a wide range of AWS services or custom applications.

2. **Automatic Scaling**: Lambda automatically scales your application by running code in response to each trigger. Your code runs in parallel and processes each trigger individually, scaling precisely with the size of the workload.

3. **Pay-as-You-Go**: You are charged based on the number of requests for your functions and the time your code executes. This means you only pay for the compute time you consume.

4. **No Server Management**: AWS Lambda abstracts the underlying infrastructure, so you don't need to manage servers, patch operating systems, or worry about scaling.

5. **Supports Multiple Languages**: AWS Lambda supports several programming languages, including Python, Java, Node.js, C#, Ruby, and Go, among others.

6. **Integration with AWS Services**: Lambda integrates seamlessly with other AWS services, allowing you to build complex, scalable applications.

Here's a simple example of how AWS Lambda might be used:

- You have an [S3 bucket](#s3-bucket) where users upload images.
- An AWS Lambda function is triggered whenever a new image is uploaded.
- The Lambda function processes the image, such as generating thumbnails or extracting metadata.
- The processed data is then stored back in S3 or sent to another AWS service for further processing.

# Accessing Gen Ai Generated Content {#accessing-gen-ai-generated-content}


To assess whether the content generated by a [Generative AI](#generative-ai) is truthful and faithful, several methods and frameworks can be employed. Truthfulness refers to whether the generated content <mark>is factually correct</mark>, while faithfulness refers to whether it <mark>accurately</mark> reflects the input data or prompt.

[interpretability](#interpretability)
### 1. Frameworks for Truthfulness and Faithfulness

   - Subject Matter Expert (SME) Reviews: One of the most reliable methods for verifying truthfulness and faithfulness is through SME validation. SMEs can manually check the content to ensure it aligns with domain-specific knowledge and is factually accurate.
   
   - [Knowledge Graph](#knowledge-graph) and External Data: Generative AI models can be linked to external sources of truth, such as knowledge graphs, databases, or other verified resources. This allows the system to cross-check facts and improve the truthfulness of the content.
   
   - Retrieval-Augmented Generation ([RAG](#rag)): This framework involves retrieving relevant information from trusted sources before generating content. It helps ensure that the AI is providing up-to-date, reliable, and contextually accurate responses.
   
   - Evaluation Metrics: Some metrics can be used to evaluate faithfulness:
     - Factual Consistency Metrics: Tools such as [BERTScore](#bertscore) or FactCC can compare generated text with reference text or factual databases to check for consistency.
     - Human Evaluation: In certain contexts, human evaluators rate the content on aspects of truthfulness and faithfulness. This can be part of quality assurance processes.
     

   - Cross-Referencing Data: AI-generated content should be cross-referenced with existing, credible sources to confirm its accuracy. For example, if the AI makes a historical claim or provides statistical data, those facts should be verifiable through known data repositories.

   - Fact-Checking Tools: Using automated fact-checking tools or models trained to detect false information can provide another layer of defence against untruthful content.

# Accuracy {#accuracy}


## Definition

- Accuracy Score is the proportion of correct predictions out of all predictions made. In other words, it is the percentage of correct predictions.
- Accuracy can have issues with [Imbalanced Datasets](#imbalanced-datasets)where there is more of one class than another.

## Formula

- The formula for accuracy is:
  $$\text{Accuracy} = \frac{TN + TP}{\text{Total}}$$
In the context of [Classification](#classification) problems, particularly binary classification, TN and TP are components of the confusion matrix:

- TP (True Positive): The number of instances that are correctly predicted as the positive class. For example, if the model predicts a positive outcome and it is indeed positive, it counts as a true positive.
- TN (True Negative): The number of instances that are correctly predicted as the negative class. For example, if the model predicts a negative outcome and it is indeed negative, it counts as a true negative.

The [Confusion Matrix](#confusion-matrix) also includes:

- FP (False Positive): The number of instances that are incorrectly predicted as the positive class. This is also known as a "Type I error."
- FN (False Negative): The number of instances that are incorrectly predicted as the negative class. This is also known as a "Type II error."

These metrics are used to evaluate the performance of a classification model, providing insights into not just accuracy but also precision, recall, and other performance measures.
## Exploring Accuracy in Python

To explore accuracy in Python, you can use libraries such as `scikit-learn`, which provides the `accuracy_score` function. This function compares the predicted labels with the true labels and calculates the accuracy.

### Example Usage

```python
from sklearn.metrics import accuracy_score
# Assuming pred and y_test are defined
accuracy = accuracy_score(y_test, pred)
print("Prediction accuracy: {:.2f}%".format(accuracy  100.0))
```

- Make sure to replace `pred` and `y_test` with your actual prediction and test data variables.

# Activation Function {#activation-function}


Activation functions play a role in [Neural network](#neural-network) by introducing non-linearity, allowing models to learn from complex patterns and relationships in the data.

[How do we choose the right Activation Function](#how-do-we-choose-the-right-activation-function)
### Key Uses of Activation Functions:

1. Non-linearity: Without activation functions, neural networks would behave as linear models, unable to capture complex, non-linear patterns in the data
2. [Data transformation](#data-transformation): Activation functions modify input signals from one layer to another, helping the model focus on important information while ignoring irrelevant data,
3. [Backpropagation](#backpropagation): They enable gradient-based optimization by making the network differentiable, essential for efficient learning.

### Purpose of Typical Activation Functions

Linear: Outputs a continuous value, suitable for regression.

ReLU (Rectified Linear Unit): 
  - Purpose: ReLU is used to introduce non-linearity by turning neurons "on" or "off." It outputs the input directly if it is positive; otherwise, it outputs zero. This helps in efficiently training deep networks by mitigating the vanishing gradient problem.
  - Function:$f(x) = \max(0, x)$

Sigmoid:
  - Purpose: Sigmoid is used primarily in [Binary Classification](#binary-classification) tasks. It squashes input values to a range between 0 and 1, making it suitable for representing probabilities.
  - Function:$f(x) = \frac{1}{1 + e^{-x}}$

 Tanh:
  - Purpose: Tanh is similar to the sigmoid function but outputs values in the range of -1 to 1. This zero-centered output can be beneficial for optimization in certain scenarios.
  - Function:$f(x) = \tanh(x)$

Softmax:
  - Purpose: Softmax is used in multi-class classification tasks. It converts a vector of raw scores (logits) into a probability distribution, where each value is between 0 and 1, and the sum of all values is 1. This allows the outputs to be interpreted as probabilities, with larger inputs corresponding to larger output probabilities.
  - Application: In both softmax regression and neural networks with softmax outputs, a vector$\mathbf{z}$is generated by a linear function and then passed through the softmax function to produce a probability distribution. This enables the selection of one output as the predicted category.

The softmax function converts a vector of raw scores (logits) into a probability distribution. The formula for the softmax function for a vector $\mathbf{z} = [z_1, z_2, \ldots, z_N]$is given by:

$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}$$

This ensures that the output values are between 0 and 1 and that they sum to 1, making them [interpretability|interpretable](#interpretabilityinterpretable) as probabilities.






# Activation Atlases {#activation-atlases}


is a viewing method for high dimensional space that AI system use for predictions.

Example AlexNet (cofounder of OpenAI)

# Active Learning {#active-learning}


Think captchas for training.
  
To help the [Supervised Learning](#supervised-learning) models when they are less confident.  
  
Reducing labelling time or need for it.


# Ada Boosting {#ada-boosting}


Resources:
[LINK](https://www.youtube.com/watch?v=LsK-xG1cLYA)
# Overview:

Ada Boosting short for <mark>Adaptive Boosting</mark>, is a specific type of [Boosting](#boosting) algorithm that focuses on improving the accuracy of predictions by <mark>combining multiple weak learners</mark> into a strong learner. It is particularly known for its <mark>simplicity</mark> and effectiveness in classification tasks.

### How AdaBoost Works:

1. **Base Learners**: In AdaBoost, the base learners are typically low-depth trees, also known as <mark>stumps</mark>. These are simple models that perform slightly better than random guessing.

2. **Sequential Training**: AdaBoost trains these stumps sequentially. Each stump is trained to correct the errors made by the previous stumps. This sequential approach ensures that each new model focuses on the data points that were misclassified by earlier models.

3. **Weighting**: After each stump is trained, <mark>AdaBoost assigns a weight to it based on its accuracy</mark>. More accurate stumps receive higher weights, giving them more influence in the final prediction.

4. **Error Focus**: The algorithm <mark>increases the weights of the misclassified data points</mark>, making them more prominent in the training of the next stump. This ensures that subsequent models pay more attention to the difficult-to-classify instances.

5. **Final Prediction**: The final prediction is a weighted sum of the predictions from all the stumps. The stumps with higher accuracy have more say in the final classification.

# Further Understanding

### Creating a Forest with AdaBoost:

To create a forest using AdaBoost, you start with a [Decision Tree](#decision-tree) or [Random Forests](#random-forests) approach, but instead of using full-sized trees, you use stumps. 

These stumps are trained sequentially, with each one focusing on the errors of the previous stumps. 

The final prediction is a weighted sum of the predictions from all the stumps, where more accurate stumps have more influence on the final outcome.

### Key Differences from [Random Forests](#random-forests):

- **Tree Depth**: In [Random Forests](#random-forests), full-sized trees are used, and each tree gets an equal say in the final prediction. In contrast, AdaBoost uses low-depth trees (stumps) and assigns different weights to each stump based on its accuracy.

- **Order and Sequence**: In AdaBoost, the order of the stumps is important because errors are passed on in sequence. In [Random Forests](#random-forests), trees are built independently and simultaneously.

### Advantages of AdaBoost:

- **Increased Accuracy**: By focusing on the errors of previous models, AdaBoost can significantly improve the accuracy of predictions.
- **Simplicity**: AdaBoost is relatively simple to implement and understand compared to other ensemble methods.
- **Flexibility**: It can be applied to various types of base models and is not limited to a specific algorithm.

### Challenges of AdaBoost:

- **Sensitivity to Noisy Data**: AdaBoost can be sensitive to noisy data and outliers, as it focuses heavily on correcting errors.
- **Complexity**: While simpler than some other boosting methods, AdaBoost can still be computationally intensive due to its sequential nature.

# Adam Optimizer {#adam-optimizer}


Adam (Adaptive Moment Estimation) is an advanced optimization algorithm that combines the benefits of both [Momentum](#momentum) and adaptive learning rates. It is widely used due to its efficiency and effectiveness in training [deep learning](#deep-learning) models.

Adam is particularly effective for large datasets and complex models, as it provides robust convergence and requires minimal tuning compared to other optimization algorithms. Its ability to <mark>dynamically adjust learning rates</mark> makes it a popular choice in the deep learning community.
#### Key Features of Adam:

**Adaptive Learning Rates:** Adam adjusts the [learning rate](#learning-rate) for each parameter individually, based on the first and second moments of the gradients. This allows for more precise updates and better convergence.

**Momentum and RMSProp Combination:** Adam incorporates the concept of momentum by using moving averages of the gradients (first moment) and the squared gradients <mark>(second moment)</mark>, similar to RMSProp.

**Parameter Update Rule:** The update rule involves computing biased estimates of the first and second moments, which are then corrected to provide unbiased estimates. These are used to update the parameters.

**[Hyperparameter](#hyperparameter):**
  - **Learning Rate (\($\alpha$\)):** Typically set to 0.001, but can be tuned for specific tasks.
  - **Beta1 and Beta2:** Control the decay rates for the moving averages of the first and second moments. Common values are 0.9 and 0.999, respectively.
  - **Epsilon (\($\epsilon$\)):** A small constant added for numerical stability, usually set to \(1 \times 10^{-8}\).

**Implementation Challenges:**
  - **Parameter Tuning:** Careful tuning of learning rate, beta1, and beta2 is essential for optimal performance.
  - **Numerical Stability:** Adjusting epsilon can help prevent division by zero and other numerical issues.  
## Related concepts
- [Gradient Descent](#gradient-descent)
- [Why does the Adam Optimizer converge](#why-does-the-adam-optimizer-converge)
- 

# Adaptive Learning Rates {#adaptive-learning-rates}

 [Adam Optimizer](#adam-optimizer)

Adaptive [learning rate](#learning-rate) adjust the learning rate for each parameter based on the estimates of the first and second moments of the gradients. Adam (short for Adaptive Moment Estimation) combines ideas from [Momentum](#momentum) and adaptive learning rates to help the optimization process.

### How to Add a Database to PostgreSQL  

### Using pgAdmin (GUI)  

1. Open pgAdmin and log in.  
2. In the Object Explorer, right-click Databases → Create → Database.  
3. Enter Database Name (e.g., `mydatabase`).  
4. Choose an Owner (optional).  
5. Click Save.  
### Using Python (`psycopg2`)  

If you're using Python (e.g., in a Jupyter Notebook), install the `psycopg2` package if needed:  

```python
!pip install psycopg2-binary
```

Then, run this script to create a PostgreSQL database:  

```python
import psycopg2

# Connect to the PostgreSQL server (default 'postgres' database)
conn = psycopg2.connect(
    dbname="postgres",  # Default DB to connect before creating a new one
    user="postgres",
    password="your_password",
    host="localhost"
)
conn.autocommit = True  # Required for CREATE DATABASE
cursor = conn.cursor()

# Create a new database
cursor.execute("CREATE DATABASE mydatabase;")

# Close connection
cursor.close()
conn.close()
print("Database 'mydatabase' created successfully!")
```

# Addressing Multicollinearity {#addressing-multicollinearity}

In [ML_Tools](#ml_tools) see: [Addressing_Multicollinearity.py](#addressing_multicollinearitypy)

Multicollinearity can impact the performance and [interpretability](#interpretability) of regression models by causing instability in coefficient estimates and complicating the analysis of variable significance. Techniques like PCA can help by transforming correlated variables into uncorrelated principal components, thereby improving model stability and interpretability.

[Principal Component Analysis](#principal-component-analysis) (PCA) is a [dimensionality reduction](#dimensionality-reduction) technique that can help address [multicollinearity](#multicollinearity) in regression models.

1. **Combining Correlated Variables**: PCA transforms the correlated independent variables into a set of uncorrelated variables called principal components. These components capture the majority of the variance in the data while reducing redundancy.

2. **Reducing Dimensionality**: By selecting a smaller number of principal components that explain most of the variance, PCA can simplify the model. This reduces the complexity and potential overfitting associated with having too many correlated predictors.

3. **Improving Model Stability**: By using principal components instead of the original correlated variables, the regression model can achieve greater stability and reliability in coefficient estimates, as the issues caused by multicollinearity are mitigated.

4. **Enhanced Interpretability**: While the principal components may not have a direct interpretation in terms of the original variables, they can still provide insights into the underlying structure of the data and the relationships among variables.
### Example Code

```python

# edit this to explore how to address multi collinusing pca 
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

variables = data_cleaned['var1', 'var2', 'var3'](#var1-var2-var3)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns

# Drop feature with VIF > 10
data_no_multicollinearity = data_cleaned.drop(['Year'], axis=1)
```



# Addressing_Multicollinearity.Py {#addressing_multicollinearitypy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/Regression/Addressing_Multicollinearity.py

# Adjusted R Squared {#adjusted-r-squared}


Adjusted R-squared is a [Regression Metrics](#regression-metrics)or assessing the quality of a regression model, <mark>especially when multiple predictors</mark> are involved. It helps ensure that the model remains [parsimonious](#parsimonious) while still providing a good fit to the data.

When evaluating a regression model, if you notice a <mark>large difference</mark> between [R squared](#r-squared) and adjusted R², it indicates that the additional predictors may not be improving the model's performance.

In such cases, it may be beneficial to drop those extra variables to simplify the model without sacrificing predictive power.

Key features:

1. **Penalty for Number of Predictors**:
   - Adjusted R² adjusts the R² value by penalizing the addition of <mark>unnecessary predictors</mark>. This means that if you add a variable that does not improve the model significantly, the adjusted R² will decrease, reflecting that the model may be overfitting.

2. **Comparison with R-squared**:
   - Adjusted R² is always less than or equal to R². While R² can artificially inflate with the addition of more predictors (even if they are not useful), adjusted R² provides a <mark>more reliable</mark> assessment of model fit by considering the number of predictors relative to the number of observations.

3. **Interpretation**:
   - Like R², adjusted R² values range from 0 to 1, where values closer to 1 indicate a better fit. However, a significant difference between R² and adjusted R² suggests that the model may be penalized for including extra variables that do not contribute meaningfully to the prediction.

4. **Formula**:
   $$R^2_{adj.} = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$
   - Where:
     - $R^2$ = R-squared value
     - $n$ = number of observations
     - $p$ = number of predictors (independent variables)




# Agent Based Modelling {#agent-based-modelling}


(ABM) is a computational approach that simulates the interactions of individual agents within a defined environment to observe complex phenomena and [emergent behavior](#emergent-behavior) at a system level. 

Agent-based modeling provides a robust framework for understanding and analyzing complex systems, particularly in the energy sector. 

By simulating individual agents and their interactions, researchers and practitioners can gain insights into system dynamics, evaluate [Policy|policies](#policypolicies), and optimize strategies for energy production and consumption.
### Principles of Agent-Based Modelling

1. **Agents**: The primary components of ABM, agents can represent individuals, groups, or entities with defined behaviors and attributes. Each agent operates based on its rules and interactions with other agents and the environment.

2. **Environment**: The space in which agents operate, which can be a physical or abstract setting. The environment influences agent behavior and can include various elements like resources, obstacles, or rules governing interactions.

3. **Interactions**: Agents communicate and interact with each other and their environment. These interactions can be cooperative, competitive, or neutral, leading to complex system dynamics.

4. **Emergence**: ABM focuses on emergent phenomena, where the collective behavior of agents leads to unexpected outcomes not evident from examining individual agents alone. This principle helps understand complex systems' dynamics and behaviors.

### Techniques in Agent-Based Modeling

1. **Model Development**: 
   - **Define Agents**: Specify agent types, behaviors, attributes, and decision-making processes.
   - **Environment Design**: Create a representation of the environment, including spatial aspects and available resources.
   - **Interaction Rules**: Establish rules governing how agents interact with each other and their environment.

2. **Simulation**: 
   - Execute the model over time, allowing agents to make decisions, interact, and adapt based on predefined rules.
   - Collect data on agents' behaviors and system-wide outcomes during the simulation.

3. **Analysis**: 
   - Analyze the results to understand emergent patterns and behaviors. This can involve statistical analysis, visualization of agent interactions, and evaluating how different parameters influence outcomes.

4. **Validation**: 
   - Compare model outputs with real-world data to validate the model's accuracy. Calibration may be necessary to ensure the model reflects observed behaviors accurately.

### Applications of Agent-Based Modeling

1. **Energy Systems**: 
   - **Demand Response**: ABM can simulate consumer behavior in response to dynamic pricing or demand response programs, providing insights into how to encourage energy conservation during peak demand.
   - **Renewable Energy Integration**: It helps model the interactions between different energy producers (e.g., solar, wind) and consumers, examining how they adapt to changes in supply and demand.

2. **Market Dynamics**: 
   - Simulate interactions between different energy providers and consumers to understand competitive behavior, pricing strategies, and market outcomes.
   - Evaluate the impact of regulatory changes on market behavior and investment decisions.

3. **Resource Management**: 
   - Model interactions among agents in managing shared resources, such as water or electricity, to study the effects of cooperation and competition on resource depletion or sustainability.

### Example Frameworks and Tools

Several tools and frameworks are available for building and simulating agent-based models, including:

- **NetLogo**: A user-friendly platform for creating agent-based models, particularly in education and research.
- **AnyLogic**: A powerful commercial tool that supports agent-based, system dynamics, and discrete event modeling.
- **Repast**: An open-source framework for building ABMs, widely used in academic research.
- **MASON**: A fast and flexible discrete-event simulation library for Java, suitable for developing complex ABMs.

---

[Agent-Based Modelling](#agent-based-modelling)
- **Multi-Agent Systems** in [LLM|LLMs](#llmllms)
  - The need for shared context across different agents, ensuring consistency and coherence in interactions.

## **Agent Interactions**

### How do Agents Interact?
- **Horizontal Collaboration:** Agents with local goals coordinate to achieve a shared objective.
- **Hierarchical Collaboration:** Primary agents oversee specialized agents to manage complex tasks effectively.

## **Understanding Agents**

### Core Components:
1. **Tools:** Resources such as APIs, databases, or GitHub.
2. **Strategy:** Techniques like self-criticism, [chain of thought](#chain-of-thought) (CoT), and planning to improve reasoning.
3. **States:** Memory, context tracking, and microservices for modularity.
4. **Goals:** Specific objectives defined for each agent.

## **Agent Planning and Interaction**
1. **Planning:** Agents plan operations, such as managing workflows in a support center.
2. **Agent Collaboration:** Agents align their individual goals with shared objectives to enhance system performance.
3. Multi-step

## **Compounding Systems**

### Multi-Agent Systems
These systems reduce reliance on extensive [prompt engineering](#prompt-engineering) by compartmentalizing tasks across specialized agents.  
**Example:** A writer agent drafts content, while a reviewer agent ensures quality, both operating within defined scopes.

# Agentic Solutions {#agentic-solutions}


Agentic solutions leverage multiple autonomous agents (usually [Small Language Models|SLM](#small-language-modelsslm)) to achieve goals collaboratively. These systems distribute tasks across agents that operate individually or collectively to solve complex problems.

Agents can model specific business functions. Role clarity enhances the effectiveness of these systems.

Related terms:
- [GraphRAG](#graphrag)
- [Agent-Based Modelling](#agent-based-modelling)
## **Types of Agentic Solutions**

**Reactive Solutions (Ask Approach):**  
    Systems like chatbots and retrieval-augmented generation (RAG) tools respond to user queries.
    
**Autonomous Solutions (Do Approach):**  
    Agents perform tasks proactively, e.g., drafting documents or scheduling meetings.
## **Business Process Integration**

### Workflow:
1. Identify a business problem.
2. Define personas (agents) required.
3. Develop an agentic workflow.

### Agentic Architectures:
1. **Vertical:** Hierarchical structures for task delegation.
2. **Horizontal:** Collaborative structures with high feedback loops.
3. **Mixed:** Combines vertical delegation with horizontal collaboration.

**Vertical Example:** A primary agent delegates tasks to lower-level agents for execution.

## **The Orleans Framework**

A framework for building distributed applications in .NET.
- **Grains:** Individual agents performing specific tasks.
- **Silos:** Distributed nodes managing grains.
- **Clusters:** Collections of silos for scalability.

## **Benefits of Using Agents**

1. **Performance Gains:** Task parallelization enhances throughput.
2. **Developer Abstraction:** Modular design simplifies system understanding and debugging.
3. **Workflow Integration:** Aligns AI agents with organizational processes.

## **Example Use Cases**

1. **IT Helpdesk Agent:** Automates troubleshooting and network access requests.
2. **Device Refresh Agent:** Manages hardware upgrades and approvals.
3. **Lead Generation Agent:** Identifies and researches potential leads.
4. **Budget Management Agent:** Reviews financial data and aids in planning.
5. **Customer Support Agent:** Triage support issues for faster resolution.
6. **Project Tracker Agent:** Tracks project milestones and budget compliance.





# Aggregation {#aggregation}

Aggregation

Summarizing data for analysis ([Pandas Pivot Table](#pandas-pivot-table) and [Groupby](#groupby)).

In [DE_Tools](#de_tools) see:
	- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/group_by.ipynb
	- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/pivot_table.ipynb

# Algorithms {#algorithms}


###  [Recursive Algorithm](#recursive-algorithm)

### Backtracking

Backtracking is a method for solving problems incrementally, by trying partial solutions and then abandoning them if they are not valid.

Example: Graph coloring with the 4-color theorem.

### Divide and Conquer

Divide and conquer is a strategy that involves breaking a problem into smaller subproblems of the same type, solving these subproblems recursively, and then combining their solutions to solve the original problem.

Example: Merge sort, where the array is split in half, and each smaller part is sorted.

Note: Subproblems do not generally overlap.

### Dynamic Programming

Dynamic programming is used for optimization problems and involves storing past results to find new results efficiently.

- [Memoization](#memoization): This technique "remembers" past results to avoid redundant calculations.
- It is characterized by overlapping substructure and overlapping subproblems.

Examples: Fibonacci numbers, Towers of Hanoi, etc.

### Greedy Algorithms

A greedy algorithm builds up a solution piece by piece, always choosing the next piece that offers the most immediate benefit without regard for future consequences. The hope is that by choosing a local optimum at each step, a global optimum will be reached.

Examples: Dijkstra's shortest path algorithm, Knapsack problem.

### Branch and Bound

Branch and bound algorithms are typically used for optimization problems and are similar to backtracking.

- As the algorithm progresses, a tree of subproblems is formed, with the original problem as the "root problem."
- A method is used to construct upper and lower bounds for a given problem.
- At each node, bounding methods are applied:
    - If the bounds match, it is deemed a feasible solution for that subproblem.
    - If the bounds do not match, the problem represented by that node is partitioned into two subproblems, which become child nodes.
- The process continues, using the best known feasible solution to trim sections of the tree until all nodes have been solved or trimmed.

Example: Traveling salesman problem.

### Brute Force

Brute force algorithms try all possible solutions until they find an optimized or satisfactory answer. Heuristics can be used to assist in this process.

### Randomized Algorithms

Randomized algorithms use random numbers to influence their behaviour.

Example: [K-means](#k-means) clustering initialization.

# Alternatives To Batch Processing {#alternatives-to-batch-processing}

If you’re working with a streaming dataset ([Data Streaming](#data-streaming)), why might [batch processing](#batch-processing) not be suitable, and what alternatives would you consider?  

**Latency**: Batch processing involves collecting data over a period and processing it in large chunks. This can introduce significant delays, making it unsuitable for applications that require real-time or near-real-time insights.

**Timeliness**: Streaming datasets often require immediate processing to respond to events as they occur. Batch processing cannot meet the demand for timely [data analysis](#data-analysis).

**Data Freshness**: In streaming scenarios, data is continuously generated, and waiting for a batch interval can result in outdated information being analyzed.

### Alternatives to Batch Processing

1. **Stream Processing**: This approach processes data in real-time as it arrives. Tools like [Apache Kafka](#apache-kafka), [Apache Flink](#apache-flink), and [Apache Spark Streaming](#apache-spark-streaming) are designed for handling streaming data efficiently.

2. **Event-Driven Architectures**: Implementing an [event-driven architecture](#event-driven-architecture) allows systems to react to data changes or events immediately, ensuring timely processing and response.

3. **Micro-batching**: This technique processes small batches of data at very short intervals, striking a balance between batch and stream processing. Tools like [Apache Spark Streaming](#apache-spark-streaming) can utilize micro-batching to handle streaming data more effectively.

4. **Complex Event Processing (CEP)**: CEP systems analyze and process streams of events in real-time, allowing for the detection of patterns and trends as they happen.

# Amazon S3 {#amazon-s3}


# Amazon S3
## Amazon S3 buckets (onedrive essentially)

Amazon Simple Storage Service (S3) is a versatile <mark>object storage solution</mark> known for its scalability, data availability, security, and performance.

Stands for Simple Storage Service.

**What is Amazon S3?**

It's an object storage service, allowing you to upload and store various types of objects, including images, text, videos, and more. S3's structure resembles that of a typical file system, with folders, subfolders, and files. Notably, it's extremely cost-effective, with storage costs starting at only 0.023 cents per GB, and it offers high durability with data replicated across three availability zones.

**Why is Amazon S3 Useful?**

1. **Cost-Effective Storage**: S3 provides cheap, reliable storage for objects of any type.
2. **Low Latency, High Throughput**: It offers fast access to your data, making it suitable for hosting static websites.
3. **Integration with AWS Services**: S3 can be integrated with other AWS services like SNS, SQS, and Lambda for powerful event-driven applications.
4. **Lifecycle Management**: S3 offers lifecycle management to automatically transition data to lower-cost storage tiers based on access patterns.

**Step-by-Step Walkthrough in the AWS Console**

1. **Creating a Bucket**: Start by navigating to the AWS Management Console and selecting S3. Create a bucket with a unique name and choose the region closest to your application.
2. **Uploading a File**: Once the bucket is created, upload a file using the console. You can add metadata and set permissions as needed.
3. **Exploring Settings**: Dive into bucket settings to configure options like versioning, logging, encryption, and lifecycle management. Block public access to ensure data security.

**Additional Features**

1. **Transfer Acceleration**: Accelerate data transfer to and from S3 using optimized network paths.
2. **Events**: Configure event notifications to trigger actions in response to S3 events like object creation or deletion.
3. **Requester Pays**: Enable Requester Pays to allow bucket owners to pass on data transfer costs to requesters.

# Anomaly Detection In Time Series {#anomaly-detection-in-time-series}

In [Time Series](#time-series)

In [ML_Tools](#ml_tools) see:
- [TS_Anomaly_Detection.py](#ts_anomaly_detectionpy)

To perform anomaly detection specifically for time series data, you can utilize various techniques that account for the <mark>temporal nature</mark> of the data. Here are some common methods:

1. Statistical Methods:
   - Moving Average: Calculate a moving average and identify points that deviate significantly from this average.
   - Seasonal Decomposition: Decompose the time series into trend, seasonal, and residual components. Anomalies can be identified in the residuals.

2. Time Series Models:
   - AutoRegressive Integrated Moving Average: Fit an ARIMA model to the time series data and analyze the residuals for anomalies.
   - State Space Model (ETS): Similar to ARIMA, this model can be used to forecast and identify anomalies in the residuals.

3. Machine Learning Approaches:
   - [LSTM](#lstm) (Long Short-Term Memory): Use LSTM networks to model the time series and detect anomalies based on prediction errors.
   - [Isolated Forest](#isolated-forest): This algorithm can be adapted for time series data by treating time as an additional feature.

4. Change Point Detection:
   - Identify points in time where the statistical properties of the time series change significantly, which may indicate anomalies.

5. Visual Methods:
   - Time Series Plots: Visual inspection of time series plots can help identify anomalies.
   - Control Charts: Use control charts to monitor the time series and flag points that fall outside control limits.


# Anomaly Detection With Clustering {#anomaly-detection-with-clustering}

[Clustering](#clustering)
- Description: Outliers often form small clusters or are isolated from main clusters.

### **8. [DBSCAN](#dbscan) (Density-Based Spatial Clustering of Applications with Noise)**

- **Purpose:**  
    Finds anomalies based on density rather than explicit statistical assumptions.
- **Steps:**
    - Identify points in low-density regions as anomalies.

[Isolated Forest|iForest](#isolated-forestiforest)

### **2. Local Outlier Factor (LOF)**

LOF is a density-based anomaly detection method that identifies anomalies by comparing the local density of a point with that of its neighbors.

**Steps:**

- For each point, calculate the local density based on the distance to its k-nearest neighbors.
- Compute the LOF score, which measures the degree of isolation of a point relative to its neighbors.
- Points with a LOF score significantly greater than 1 are considered anomalies.

# Anomaly Detection With Statistical Methods {#anomaly-detection-with-statistical-methods}


Basic:
- [Z-Normalisation|Z-Score](#z-normalisationz-score)
- [Interquartile Range (IQR) Detection](#interquartile-range-iqr-detection)
- [Percentile Detection](#percentile-detection)

Advanced:
- [Gaussian Model](#gaussian-model)
- [Isolated Forest](#isolated-forest)

### Grubbs' Test

Context:  
Grubbs' test is a hypothesis test designed to detect a single outlier in a normally distributed dataset. It tests the largest deviation from the mean relative to the standard deviation. This test is iterative and removes one outlier at a time.

Purpose:  
To determine whether the most extreme data point (either smallest or largest) is a statistical outlier.

Steps:
- Compute the test statistic:  
    $G = \frac{\max(|X - \mu|)}{\sigma}$  
    where:
    - $X$: Data points
    - $\mu$: Mean of the dataset
    - $\sigma$: Standard deviation of the dataset.
- Compare $G$ to a critical value:
    - The critical value depends on the sample size $n$ and significance level $\alpha$ (e.g., 0.05).
    - If $G$ exceeds the critical value, the data point is considered an outlier.

Limitations:
- Assumes data follows a normal distribution.
- Inefficient for detecting multiple outliers simultaneously.

### Histogram-Based Outlier Detection (HBOS)

Context:  

HBOS is a non-parametric method that detects anomalies by analyzing the distribution of individual features independently. It relies on histograms, which estimate feature density.

Purpose:  
To identify outliers as data points falling in bins with low frequencies or densities.

Steps:
- Create histograms for each feature:
    - Divide each feature's range into bins.
    - Count the frequency of data points in each bin.
- Calculate scores for each data point:
    - Outliers are points in bins with significantly lower densities compared to others.

Advantages:
- Does not assume a specific data distribution.
- Scales well to large datasets.

Limitations:
- Assumes feature independence (not ideal for [multivariate data](#multivariate-data)).
- Sensitive to bin size selection.

### One-Class SVM

One-Class Support Vector Machine is a variation of the SVM algorithm used for anomaly detection. It learns a decision boundary around the normal data points.

Steps:
- Train the model on the normal data points.
- The model attempts to find a hyperplane that separates the normal data from the origin.
- Points that fall outside this boundary are classified as anomalies.

# Anomaly Detection {#anomaly-detection}

Anomaly detection involves identifying [standardised/Outliers|Outliers](#standardisedoutliersoutliers). Detecting these anomalies is crucial for maintaining [data integrity](#data-integrity) and improving model performance.
## Methods for Detecting Anomalies

In [ML_Tools](#ml_tools) see: [Pycaret_Anomaly.ipynb](#pycaret_anomalyipynb)
## Process of Detection

Data Preparation
   - [Data Cleansing](#data-cleansing): Handle missing values and remove any irrelevant data points.
   - [Normalisation](#normalisation)/[Standardisation](#standardisation): Scale the data if necessary, especially if using methods sensitive to the scale.

Anomaly Detection with a model: Use a chosen method to flag anomalies
- [Anomaly Detection with Clustering](#anomaly-detection-with-clustering)
- [PCA-Based Anomaly Detection](#pca-based-anomaly-detection)
- [Anomaly Detection in Time Series](#anomaly-detection-in-time-series)
- [Anomaly Detection with Statistical Methods](#anomaly-detection-with-statistical-methods)

Validation
   - Validate the detected anomalies by comparing them against known anomalies (if available) or using domain knowledge.
   - Adjust thresholds or methods based on validation results.

Visualization

- Visualize the results using plots (e.g., scatter plots, box plots) to understand the distribution of data and the identified anomalies.
- [Boxplot](#boxplot): Displays the distribution and identifies outliers using the interquartile range (IQR).
- Scatter Plot: Helps visually identify outliers.

# Apache Airflow {#apache-airflow}


Schedular think CROM jobs with python.

Apache Nifi may be better.

[Airflow](https://airflow.apache.org/) is a [data orchestrator](term/data%20orchestrator.md) and the first that made task scheduling popular with [Python](term/python.md). 

Airflow programmatically author, schedule, and monitor workflows. It follows the [imperative](term/imperative.md) paradigm of schedule as *how* a DAG [Directed Acyclic Graph (DAG)](#directed-acyclic-graph-dag) is run has to be defined within the Airflow jobs. Airflow calls its *Workflow as code* with the main characteristics
- **Dynamic**: Airflow pipelines are configured as Python code, allowing for dynamic pipeline generation.
- **Extensible**: The Airflow framework contains operators to connect with numerous technologies. All Airflow components are extensible to easily adjust to your environment.
- **Flexible**: Workflow parameterization is built-in leveraging the [Jinja Templating](term/jinja%20template.md) engine.


# Apache Kafka {#apache-kafka}


Apache Kafka is an open-source distributed event streaming platform used for building real-time data pipelines and data streaming  applications. It is designed to handle high-throughput, fault-tolerant, and scalable messaging. 

### Features

Immutable commit log: Kafka maintains an append-only log of messages, which ensures [Data Integrity](#data-integrity) and replicability.

Kafka allows applications to [Publish and Subscribe](#publish-and-subscribe) to streams of records, similar to a message queue or enterprise messaging system.

Durability and Reliability: Kafka stores streams of records in a fault-tolerant way, ensuring data durability and reliability. It replicates data across multiple nodes to prevent data loss.

[Scalability](#scalability): Kafka is designed to scale horizontally by adding more brokers (servers) to the cluster, which can handle increased load and data volume.

Kafka is optimized for high throughput, making it suitable for processing large volumes of data in real-time.

Data in Kafka is organized into topics, which are further divided into partitions. Each partition is an ordered, immutable sequence of records that is continually appended to.

Producers are applications that publish data to Kafka topics, while consumers are applications that subscribe to topics and process the data.

Kafka is commonly used for log aggregation, real-time analytics, [data integration](#data-integration), stream processing, and building event-driven architectures.

Kafka integrates ([Data Integration](#data-integration)) well with various data processing frameworks like Apache Spark, Apache Flink, and Apache Storm, as well as with databases and other [data storage](#data-storage) systems.


# Apache Spark {#apache-spark}


Apache Spark is an open-source multi-language engine for executing [Data Engineer](Data%20Engineer.md)  and [Machine Learning](Machine%20Learning.md) on single-node machines or clusters. It's optimized for large-scale data processing.

Spark runs well with [Kubernetes](term/kubernetes.md).

Spark is a highly popular framework for large-scale data processing. It allows  [Data Engineer](#data-engineer) to process massive datasets in memory, which makes it faster than traditional disk-based approaches. Spark is versatile, supporting batch processing, real-time data streaming, machine learning, and graph processing.



# Asking Questions {#asking-questions}


### Why Ask Better Questions?

Asking better questions enhances thinking, learning, problem-solving, and communication. Good questions:

- Guide inquiry and exploration.
- Clarify assumptions.
- Elicit insights or novel responses.
- Enable self-reflection and deeper understanding.

### What Makes a Good Question?

A good question:
- **Elicits a novel or thoughtful response** — not just a fact or yes/no.
- **Opens possibilities** rather than closing them.
- **Reveals assumptions** or **forces re-evaluation** of mental models.
- **Matches the context and audience** — good questions for brainstorming differ from those for debugging.
- **Fosters chain-of-thought reasoning**, helping others articulate how they arrive at conclusions.
### Types of Questions

Questions can be classified by their **function**, **depth**, or **structure**.

#### 1. By Function

| Type        | Purpose                                   | Example                             |
|-------------|-------------------------------------------|-------------------------------------|
| Clarifying  | Understand what's being said              | "What do you mean by X?"           |
| Probing     | Dig deeper into reasoning or logic        | "Why do you think that?"           |
| Exploratory | Generate ideas or new perspectives        | "What if we reversed the problem?" |
| Reflective  | Encourage self-awareness                  | "What assumption am I making here?"|
| Critical    | Test or challenge statements              | "What evidence supports that?"     |

#### 2. By Depth
- **Surface-level:** "What is X?"
- **Mid-level:** "How does X relate to Y?"
- **Deep-level:** "Why does X matter?" or "What are the implications of X?"

#### 3. By Structure

- **Open-ended:** Encourage elaboration.  
    → _"How might we design this differently?"_
    
- **Closed:** Seek a specific answer.  
    → _"Is this implementation correct?"_
    
- **Leading:** Suggest an answer.  
    → _"Wouldn't you agree that…?"_
    
- **Falsifiable/Testable:** Can be proven right or wrong.  
    → _"Does increasing X always decrease Y?"_
### Characteristics of Good Questions

| Characteristic | Description                                                       |
|----------------|-------------------------------------------------------------------|
| **Purposeful** | Serves a clear goal in the conversation or inquiry                |
| **Contextual** | Relevant to the topic or the respondent                           |
| **Open/Expansive** | Invites multiple viewpoints or lines of reasoning             |
| **Challenging** | Pushes beyond defaults or surface-level answers                  |
| **Precise**    | Minimizes ambiguity while leaving room for elaboration           |
| **Sequenced**  | Ordered to build thought step-by-step (chain of thought)        |

### Related Concepts
- [Chain of Thought](#chain-of-thought)
- [Design Thinking Questions](#design-thinking-questions)
- [Prompting](#prompting)

### Questions:
- How LLMs generate or refine questions using [Prompting](#prompting) or [Chain of thought](#chain-of-thought) approaches?

# Attention Is All You Need {#attention-is-all-you-need}




# Attention Mechanism {#attention-mechanism}


 Think of attention like human reading behavior: when reading a complex sentence, we don't process all the words equally at every moment. Instead, we might "attend" more to certain words based on the context of what we’ve read so far and what we're trying to understand. This is similar to what the attention mechanism does in neural networks.

The attention mechanism is a key concept in modern [ML](#ml) particularly in natural language processing ([NLP](#nlp)/[LLM](#llm)) and sequence-based models like neural machine translation (NMT). It was introduced to address the limitations of earlier models, like [Recurrent Neural Networks|RNN](#recurrent-neural-networksrnn) and [LSTM](#lstm) network), in handling long sequences and capturing important dependencies within data.

The attention mechanism improves a model's ability to focus on important parts of the input data, helping it manage long-range dependencies, which is especially useful in tasks like machine translation, text generation, and various NLP tasks.

### Core Idea of Attention

The  [Transformer|Transformer](#transformertransformer) architecture, introduced by Vaswani et al. in 2017 ("[Attention Is All You Need](#attention-is-all-you-need)"), is the most popular use of the attention mechanism. The key innovation of Transformers is the <mark>self-attention mechanism</mark>, which <mark>allows each token in a sequence to attend to all other tokens</mark>, making the model more efficient and scalable for parallel processing. <mark>Transformers replaced traditional RNN-based</mark> models and have become the foundation of models like [BERT](#bert), GPT, and T5.

The attention mechanism allows a model to focus on different parts of an input sequence when making predictions, rather than relying on a fixed-size hidden state to encode all information. This selective "focus" can greatly enhance the model's ability to handle long-range dependencies.

For example, in machine translation, while translating a sentence from one language to another, different words in the input sentence might hold varying degrees of relevance to the word currently being translated. Attention helps the model dynamically weigh the importance of different words at each step of the translation.

In its simplest form, attention assigns weights to each input token based on its relevance to the current output token being generated. These weights are typically calculated using a score function and then normalized (usually through a softmax) to produce a distribution over the input sequence. The weighted sum of the input tokens is then passed as context for generating the output token.

### Key Components of Attention and Formula

I see! Here's the text with the math terms wrapped in as requested:

1. Query: Represents the current word or position that requires attention: $Q$
2. Key: Represents each word in the input sequence: $K$
3. Value: Represents the actual content or information in the input sequence:  $V$
4. Attention Scores: The attention mechanism computes the relevance between the query and each key by computing a similarity score (such as dot-product or other scoring methods).
5. Softmax: These scores are then passed through a softmax function to form a probability distribution, which gives us the attention weights.
6. Context Vector: A weighted sum of the values ($V$), using the attention weights, is computed. This context vector is what the model uses to generate the output token.

Given a query matrix, key matrix, and value matrix, attention is calculated as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- $Q$,$K$, and $V$are matrices of query, key, and value vectors.
- $d_k$ is the dimension of the keys.
- The softmax is applied row-wise to produce attention weights.

### Applications of Attention

- Machine Translation: Aligns source and target words in a more dynamic and context-dependent way.
- Text Summarization: Helps to identify the most relevant parts of a document.
- Speech Recognition: Enhances the model’s ability to focus on important features over long audio sequences.
- Vision: Self-attention is now used in computer vision tasks (Vision Transformers or ViTs) to model dependencies between different parts of an image.

### Types of Attention Mechanisms
1. Additive Attention: Introduced in the original attention paper by Bahdanau et al. (2014), this method uses a feed-forward neural network to compute the relevance between the query and each key.
   
2. Dot-Product (Multiplicative) Attention: Introduced in the Transformer paper by Vaswani et al. (2017), this method computes the relevance using a simple dot product between the query and key vectors. It is computationally efficient and widely used.

3. Scaled Dot-Product Attention: A variant of dot-product attention, used in the Transformer architecture, where the dot product is divided by the square root of the dimension of the key vectors to avoid excessively large gradients.

4. Self-Attention: In this mechanism, the model applies attention to itself. This means each word in the input sequence attends to all other words in the sequence, including itself. Self-attention is used in models like Transformers to capture dependencies within a sentence.

### Self attention vs multi-head attention

https://www.youtube.com/shorts/Muvjex0nkes

Take every word pays attention to every other word to capture contetxt by:

1. take input word vectors,
2. break words into Q,K,V vectors,
3. compute attention matrix
4. generate final word vectors.
these vectors

Multi-head attention: perform self attention in parallel.

1. take word vectors,
2. break words into Q,K,V vectors,
	1. Break each Q,K,V vector into the number of heads parts
3. compute attention matrix for each head
4. generate final word vectors for each head
5. Combine back together

These have better understanding of the context.

[Multi-head attention](#multi-head-attention)

- This allows the model to weigh the importance of different words in a sequence when making predictions. It captures relationships between words, even if they are far apart in the sequence.
   - The model computes attention scores for each pair of words to determine how much focus one word should place on another in a sequence.

# Automated Feature Creation {#automated-feature-creation}


**Question:** Can we autodetect meaningful features.

[Feature Engineering](#feature-engineering) is an ad-hoc manual process that depends on domain knowledge, intuition, data exploration, and creativity. However, this process is dataset-dependent, time-consuming, tedious, subjective, and not a scalable solution.

[Automated Feature Creation](#automated-feature-creation) automatically generates features using a framework; these features can be filtered using [Feature Selection](#feature-selection) to avoid feature explosion. 

Below are some popular open-source libraries for automated feature engineering:

- Pycaret – [PyCaret](https://pycaret.org/)
- Featuretools for advanced usage – [Home](https://www.featuretools.com/) | [What is Featuretools? — Featuretools 1.1.0 documentation](https://featuretools.alteryx.com/en/stable/)
- Optuna – [A hyperparameter optimization framework](https://optuna.org/)
- Feature-engine – [A Python library for Feature Engineering for Machine Learning — 1.1.2](https://feature-engine.readthedocs.io/en/1.1.x/)
- ExploreKit – [GitHub – giladkatz/ExploreKit](https://github.com/giladkatz/ExploreKit)
- https://www.turintech.ai/blog/feature-generation-what-it-is-and-how-to-do-it


# Azure {#azure}


Public cloud computing platform from Microsoft offering various services like infrastructure, data storage, and machine learning.