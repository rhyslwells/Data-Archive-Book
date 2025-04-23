

# Olap (Online Analytical Processing) {#olap-online-analytical-processing}


OLAP, or Online Analytical Processing, is a category of database technology.

OLAP systems allow organizations to gain insights by examining data across various dimensions, such as time, product, and region.

[Excel pivot table](#excel-pivot-table)
## Key Features of OLAP & Operations

Query Performance
Aggregation and Summarization across dimensions.

- **Slicing**: Extracting a single layer of data from the cube by selecting a specific dimension (e.g., sales for Q1).
- **Dicing**: Selecting a subcube by specifying values for multiple dimensions.
- **Drill Down**: Moving from a summary level to a more detailed level (e.g., from yearly to monthly sales).
- **Roll Up**: Aggregating data to a higher level (e.g., from daily to monthly sales).
- **Pivoting**: Rotating the data to view it from a different perspective (e.g., switching rows and columns).

### Use Cases of OLAP
- **Business Intelligence (BI)**: OLAP tools are integral to BI solutions, allowing for the analysis of financial data, sales performance, and other key metrics.
- **Data Warehousing**: OLAP is commonly used with data warehouses, where large volumes of historical data are stored for reporting and analysis.
### Visualization Tools
To interact with the OLAP cube, users typically utilize tools such as:
- **Microsoft Power BI**: For creating dashboards and visualizations.
- **Excel with Pivot Tables**: For slicing, dicing, and reporting.
- **Tableau**: For visual analysis.

# Olap {#olap}



# Oltp {#oltp}

In online transaction processing (**OLTP**), information systems typically facilitate and manage **transaction-oriented** applications. It's the opposite of [OLAP (Online Analytical Processing)](standardised/OLAP%20(online%20analytical%20processing).md).

# Proposal: [Project name]

# About this doc

_Metadata about this document. Describe the scope and current status._

This doc is a proposal for [feature or change]. Upon approval, we will look to have this prioritized as a project and do a full Technical Design Document.

|   |   |
|---|---|
|Sign off deadline|_Date_|
|Status|_Draft_|
|Author(s)|_Name 1, Name 2_|

Sign offs

- *Name 1*
    
- *Name 2*
    
- Add your name here to sign off
    

# Problem

_What is the problem being solved? What are the pain points? What is the current solution and why is not good enough?_

# High level goal

_Why should we do this? Answer this in metrics ideally but otherwise a sentence or two is okay._

# What will happen if we don’t solve this?

_Make it clear the downsides of what will happen if we don’t invest the time into this._

# Proposed solution: [Option name]

_State the option you suggest and explain your reasoning. What benefits will we get from this approach? Time, money, risk, convenience, etc._

# Alternatives

_A table or summary of the other options to achieve the goal. Also, consider adding this to an Appendix to keep the doc focused too._

- Option 1: …
    
    - Pros: …
        
    - Cons: …
        
- Option 2: …
    
    - Pros: …
        
    - Cons: …
        
- …
    

# Risks

_What can go wrong with the proposed approach? How are you mitigating that?_

- _Risk 1_
    
- _Risk 2_
    
- _…_
    

# Open Questions [optional]

_Anything still being figured out that you could use some additional eyes or thoughts on._

# One Hot Encoding {#one-hot-encoding}


Related terms:
- Why do we need to drop one of the dummy columns? [Dummy variable trap](#dummy-variable-trap): 

<mark>Dummy variables & One-hot encoding are fundamentally different from [Label encoding](#label-encoding)</mark>

[Why does label encoding give different predictions from one-hot encoding](#why-does-label-encoding-give-different-predictions-from-one-hot-encoding)

One-hot encoding is a technique used to convert categorical data into a numerical format that can be used by machine learning algorithms. It is particularly useful when dealing with categorical variables that have no ordinal relationship. 

In one-hot encoding, each category is transformed into a binary vector. If there are \( n \) unique categories, each category is represented by a vector of length \( n \) where one element is "hot" (set to 1) and the rest are "cold" (set to 0). For example, if you have a categorical variable with three categories: "red," "green," and "blue," one-hot encoding would represent them as:

- "red" -> [1, 0, 0]
- "green" -> [0, 1, 0]
- "blue" -> [0, 0, 1]

One-hot encoding is used because many machine learning algorithms cannot work directly with categorical data. By converting categories into a numerical format, one-hot encoding allows these algorithms to process the data effectively. It is commonly used in preprocessing steps for machine learning models, especially in neural networks and other algorithms that require numerical input.
### Implementation

In [ML_Tools](#ml_tools) see: [One_hot_encoding.py](#one_hot_encodingpy)

```python
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df, prefix = cat_variables, columns = cat_variables)
```

# One_Hot_Encoding.Py {#one_hot_encodingpy}

Explorations\Preprocess\One_hot_encoding\One_hot_encoding.py

This script demonstrates how to preprocess categorical variables and apply linear regression for house price prediction. Key steps include:

1. **Data Loading**: It loads a dataset of house prices.
2. **Dummy Variables**: It creates dummy variables for the 'town' column using `pd.get_dummies()` and merges them with the original dataframe.
3. **Dummy Variable Trap**: It drops one dummy variable to avoid multicollinearity (dummy variable trap).
4. **Feature and Target Split**: It separates the dataset into features (X) and the target variable (price).
5. **Model Training**: A Linear Regression model is trained on the data.
6. **Predictions**: It predicts house prices based on various features and evaluates the model's accuracy.
7. **Label Encoding and One-Hot Encoding**: It applies `LabelEncoder` to convert 'town' names into numbers and uses `OneHotEncoder` to create dummy variables for categorical columns.
8. **Final Predictions**: It predicts prices using the transformed features and evaluates the model's performance.

# Optimisation Function {#optimisation-function}


Optimization functions adjust the [Model Parameters](#model-parameters) to minimize the [Loss function](#loss-function), which measures how well the model performs. This is a fundamental step in training machine learning models.  

General Optimization Process:

The [Optimisation function](#optimisation-function) (e.g., LBFGS, Newton-CG) iteratively updates the [Model Parameters](#model-parameters) by:  
1. Calculating the gradient of the loss function with respect to the parameters.  
2. Updating the parameters in the direction of the negative gradient (as described in [Gradient Descent](#gradient-descent)).  

This process is repeated until:  
- The cost function converges (i.e., the change in the loss function becomes negligible), or  
- The maximum number of iterations is reached.  

See [Optimisation techniques](#optimisation-techniques).

# Optimisation Techniques {#optimisation-techniques}

Optimisation techniques
- [Adam Optimizer](#adam-optimizer)
- RMSprop
- [Stochastic Gradient Descent](#stochastic-gradient-descent)
- [standardised/Optuna](#standardisedoptuna)

[Gradient Descent](#gradient-descent)
- Iteratively updates parameters using the gradient of the [Cost Function](#cost-function) with respect to the parameters.  
- Requires careful tuning of the [Learning Rate](#learning-rate) ($\alpha$), which controls the size of each update.  

Optimization Solvers in [Sklearn](#sklearn) : Scikit-learn solvers improve on Gradient Descent by leveraging advanced techniques:  
- Use second-order information, such as approximations to the Hessian matrix.  
- Achieve faster and more reliable convergence compared to Gradient Descent.  
- Automatically adapt step sizes [Adaptive Learning Rates](#adaptive-learning-rates), eliminating the need for manual tuning.



# Optimising Neural Networks {#optimising-neural-networks}



[Deep Learning](#deep-learning)

Ways to improve in using a [Neural network](#neural-network) 
more data, 
bigger network, 
diverse training set, 
try dropout, 
change network architechure.

<mark>Need strategies that will point towards whats the best methods to try.</mark>



# Optimising a [Logistic Regression](#logistic-regression) Model

In `sklearn`, the logistic regression model uses an optimization algorithm to find the best parameters (intercept and coefficients) that minimize a loss function, typically the logistic loss (cross-entropy loss). Here's an overview of how it works:

Optimization process: `sklearn` optimizes the logistic regression parameters using iterative solvers (like LBFGS, newton-cg, or liblinear) that minimize the logistic loss function.`sklearn` is using one of these optimization algorithms (likely `lbfgs` or `liblinear` by default) to minimize the logistic loss function. 

### Objective Function (Loss Function)

The objective of logistic regression is to minimize the logistic [loss function](#loss-function) (also called cross-entropy loss) given by:

$\text{Cost}(\theta) = -\frac{1}{m} \sum_{i=1}^m \left( y^{(i)} \log(h_{\theta}(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_{\theta}(x^{(i)})) \right)$

Where:
- $h_{\theta}(x) = \frac{1}{1 + \exp(-\theta^T x)}$ is the sigmoid function that predicts probabilities.
- $y^{(i)}$ is the actual label for the $i$-th sample (either 0 or 1).
- $m$ is the number of samples.
- $\theta$ is the vector of parameters (intercept and coefficients).
  
The goal is to minimize this [cost function](#cost-function) by finding the optimal $\theta$ values (intercept and coefficients).

### Optimization Algorithm (Solvers)

[sklearn](#sklearn) provides different [Optimisation function](#optimisation-function) to find the optimal [Model Parameters](#model-parameters) for logistic regression. These solvers use optimization techniques like Gradient Descent or more advanced methods like Newton’s Method or Limited-memory Broyden–Fletcher–Goldfarb–Shanno (LBFGS). Here are the main solvers used in `sklearn`:

- 'liblinear': This solver uses coordinate descent or regularized Newton's method and is a good choice for smaller datasets. It supports both L1 (lasso) and L2 ([Ridge](#ridge)) regularization.
  
- 'lbfgs': This is a quasi-Newton method (Limited-memory BFGS), which approximates the Newton's method and is more efficient for larger datasets. It’s an iterative solver that converges faster and requires fewer iterations than simple gradient descent.
  
- 'newton-cg': This solver is based on Newton’s method, which computes the second-order derivative (Hessian matrix) and updates parameters using the inverse of this matrix. It works well for logistic regression with large datasets and is efficient when the number of features is large.
  
- 'saga': An efficient solver that can handle L1 regularization (for sparse solutions) and large datasets. It’s a variant of Stochastic Gradient Descent (SGD) and supports L1, L2, and ElasticNet regularization.

- 'sgd': This is a solver for stochastic [Gradient Descent](#gradient-descent), which updates parameters iteratively using only a single data point (or a small batch) at a time. It can be slower but is useful for very large datasets.

### 5. Convergence Criteria

`sklearn`’s logistic regression solvers have specific criteria for when to stop the optimization process:
- Convergence tolerance (`tol`): This is the threshold for when the optimization stops. When the improvement in the cost function between iterations becomes smaller than `tol`, the process stops.
- Maximum iterations (`max_iter`): The maximum number of iterations allowed before the solver stops. If convergence is not achieved within the allowed iterations, the algorithm will stop, and a warning will be issued.




# Optuna {#optuna}

Optuna is a [hyperparameter](#hyperparameter) optimization framework used to automatically tune hyperparameters for machine learning models.

Optuna automates the process of tuning hyperparameters by defining an objective function, testing different hyperparameter combinations, training the model, and evaluating its performance. The best set of hyperparameters is chosen based on the performance metric (e.g., test accuracy) returned by the objective function.

[Hyperparameter Tuning](#hyperparameter-tuning)
## Benefits of Using Optuna

1. Efficient Search: Utilizes algorithms like TPE (Tree-structured Parzen Estimator) to search the hyperparameter space more efficiently than grid search.
2. Dynamic Search Space: Can explore continuous, categorical, and discrete spaces.
3. Automatic Pruning: Supports pruning of unpromising trials during training, improving computational efficiency.
4. Visualization: Offers built-in tools for visualizing the optimization process, aiding in understanding the impact of hyperparameters.

## Steps to Use Optuna

1. Define Objective Functions:
   - For each model (e.g., [LightGBM](#lightgbm), [XGBoost](#xgboost), [CatBoost](#catboost)), define an objective function.
   - The objective function takes trial parameters as input and returns a score to optimize.
   - Specify hyperparameters to tune within each function, such as:
     - LightGBM: learning rate, number of leaves
     - XGBoost: eta, max depth
     - CatBoost: learning rate, depth

2. Running Hyperparameter Optimization:
   - Create a study object for each model using `optuna.create_study()`.
   - Run the optimization process using the `.optimize()` method, specifying the objective function and the number of trials.
   - Retrieve the best hyperparameters from each study object using `.best_params`.

3. Comparison and Evaluation:
   - Compare the best hyperparameters obtained for each model.
   - Evaluate the performance of the tuned models on a validation dataset.

## Differences between Models with Optuna

Hyperparameters:
  - The specific hyperparameters to tune may vary between models.
  - Example: LightGBM involves tuning parameters like learning rate and number of leaves, while XGBoost involves parameters like eta and max depth.

Objective Function:
  - Tailor the objective function for each model to its respective API and requirements.
  - Ensure the objective function properly trains and evaluates the model using the specified hyperparameters.

Optimization Strategy:
  - Optuna provides different optimization algorithms (e.g., TPE, CMA-ES) that may behave differently depending on the model and hyperparameter space.
### Implementation

```python
import optuna

# 1. Creating an Optimization Study
# Initialize a study to track and manage the optimization process.
# The 'direction' parameter specifies whether to maximize or minimize the objective function.
study = optuna.create_study(direction='maximize')

# 2. Defining the Objective Function
def objective(trial):
    # 3. Suggesting Hyperparameters
    # Use trial.suggest_* methods to specify the hyperparameters to optimize.
    # Each method defines the type and range of values to explore.
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    dropout = trial.suggest_uniform('dropout', 0.2, 0.5)
    units_1 = trial.suggest_int('units_1', 64, 128)
    
    # 4. Training the Model
    # Train the model using the suggested hyperparameters.
    # Evaluate the model's performance on the validation dataset.
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=config['epochs'], batch_size=batch_size, callbacks=[...])
    
    # Returning the Result
    # Return the model's test accuracy as the result of the trial.
    return test_accuracy

# Optimization
# Run the optimization process, executing the objective function multiple times.
# The 'n_trials' parameter specifies the number of trials to run.
study.optimize(objective, n_trials=15)

# Best Trial
# After optimization, retrieve the best trial with the highest test accuracy.
# The best hyperparameters are stored in 'best_trial.params'.
best_trial = study.best_trial
```

# Ordinary Least Squares {#ordinary-least-squares}

**Derivation of Coefficients**:
    - OLS derives the coefficients by setting the partial derivatives of the SSE with respect to each coefficient to zero. This results in a set of normal equations that can be solved to find the optimal coefficients.
    - In matrix form, the solution can be expressed as:  
$$b=(X^{T}X)^{-1} X^{T}y$$
    - Here, $X$ is the matrix of input features (including a column of ones for the intercept),  is the vector of observed values, and  is the vector of coefficients.

# Orthogonalization {#orthogonalization}

[link](https://www.youtube.com/watch?v=UEtvV1D6B3s&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=2)

When training ML model need Orthogonalization in order to  Determine what to tune, and observe the effect it has.

Each button does one thing.

Example: Car with Accelerator and angle of steering wheel. 

Assumptions (controls for tuning):
- model works well with cost functions
	- Try [Adam Optimizer](#adam-optimizer), bigger network
- work on training set
	- [Regularisation](#regularisation)
	- Try bigger training set
- works on test set of data
	- Try bigger training set.
- works well in real life.
	- Change training set
	- Change cost function.

Avoid early stopping as effects network size, and training set size.

Related links:
- [Optimising Neural Networks](#optimising-neural-networks)
- [DS & ML Portal](#ds--ml-portal)

# Outliers {#outliers}


Outliers are data points that differ significantly from other observations in the dataset. They can skew and mislead the training of machine learning models, especially those sensitive to the scale of data, such as [Linear Regression](#linear-regression). They can sway the generality of the model, skewing predictions and increasing the standard deviation.

Related Concepts:
- Handling outliers in similar to [Handling Missing Data](#handling-missing-data)
- [Methods for Handling Outliers](#methods-for-handling-outliers)
- [Anomaly Detection](#anomaly-detection)
#### Why Removing Outliers May Improve Regression but Harm Classification

Impact on Regression Model:

Regression models, particularly linear regression, are sensitive to outliers because they attempt to minimize the sum of squared errors. By removing outliers, the model can better capture the underlying trend of the data, leading to improved performance metrics such as R-squared and reduced mean squared error.

Impact on Classification Models

- Class Boundary Distortion: Classification models, such as decision trees or support vector machines, rely on the distribution of data points to define class boundaries. <mark>Outliers can provide valuable information about the variability within classes.</mark>

- Loss of Information: Removing outliers may lead to the loss of important data points that could help in distinguishing between classes, potentially resulting in a less accurate model. For example, an outlier might represent a rare but important class that the model needs to learn from.

# Over Parameterised Models {#over-parameterised-models}

[Neural network](#neural-network)

Universal approximation theory

# Overfitting {#overfitting}



>[!Summary]  
> Overfitting in machine learning occurs when a model captures not only the underlying patterns in the training data <mark>but also the noise</mark>, leading to poor performance on unseen data, and is unable to generalise.
> 
>Mathematically, overfitting results in a model with low bias but high variance, meaning it adapts too closely to the training data and fails to generalize well.
>
>Key methods to address overfitting include [Regularisation](#regularisation) (such as $L_1$ and $L_2$ regularization), [Cross Validation](#cross-validation), and simpler models.
>
>In statistical terms, it indicates a model with high complexity and too many parameters relative to the amount of training data, which results in $f(x)$ poorly representing the population distribution.

>[!Breakdown]  
> Key Components:  
> - Regularization (Lasso: $L_1$, Ridge: $L_2$) to penalize model complexity.  
> - [Cross Validation](#cross-validation) to ensure model generalization.  
> - Early stopping in training to avoid learning noise.  
> - Simplification of models to prevent fitting irrelevant patterns.

>[!important]  
> - Overfitting indicates high variance in the model’s performance, which can be <mark>identified by a significant drop in accuracy between training and test datasets.</mark>  
> - Regularization adds penalty terms to the cost function, reducing model complexity and mitigating overfitting.

>[!attention]  
> - Overfitting is more common in models with high-dimensional datasets.  
> - Excessive model tuning (hyperparameter optimization) may inadvertently increase overfitting.

>[!Follow up questions]  
> - How does the choice of regularization type (e.g., $L_1$ vs. $L_2$) affect model generalization in overfitting scenarios?  
> - What role does the size of the training dataset play in mitigating overfitting?

>[!Related Topics]  
> - [Cross Validation](#cross-validation) techniques (e.g., $k$-fold, Leave-One-Out cross-validation)  
> - [Bias and variance](#bias-and-variance)radeoff in machine learning models  

# Oltp (Online Transactional Processing) {#oltp-online-transactional-processing}

