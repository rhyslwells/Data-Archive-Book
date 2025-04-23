# L

## Table of Contents
* [LBFGS](#lbfgs)
* [LLM Evaluation Metrics](#llm-evaluation-metrics)
* [LLM](#llm)
* [LSTM](#lstm)
* [Label encoding](#label-encoding)
* [Labelling data](#labelling-data)
* [Langchain](#langchain)
* [Language Model Output Optimisation](#language-model-output-optimisation)
* [Language Models Large (LLMs) vs Small (SLMs)](#)
* [Language Models](#language-models)
* [Lasso](#lasso)
* [Latency](#latency)
* [Latent Dirichlet Allocation](#latent-dirichlet-allocation)
* [Learning Styles](#learning-styles)
* [LightGBM vs XGBoost vs CatBoost](#lightgbm-vs-xgboost-vs-catboost)
* [LightGBM](#lightgbm)
* [Linear Discriminant Analysis](#linear-discriminant-analysis)
* [Linear Regression](#linear-regression)
* [Linked List](#linked-list)
* [Load Balancing](#load-balancing)
* [Local Interpretable Model-agnostic Explanations](#local-interpretable-model-agnostic-explanations)
* [Logical Model](#logical-model)
* [Logistic Regression Statsmodel Summary table](#logistic-regression-statsmodel-summary-table)
* [Logistic Regression does not predict probabilities](#logistic-regression-does-not-predict-probabilities)
* [Logistic Regression](#logistic-regression)
* [Logistic regression in sklearn & Gradient Descent](#)
* [Looker Studio](#looker-studio)
* [Loss function](#loss-function)
* [Loss versus Cost function](#loss-versus-cost-function)
* [lambda architecture](#lambda-architecture)
* [learning rate](#learning-rate)
* [lemmatization](#lemmatization)



# Lbfgs {#lbfgs}

LBFGS stands for Limited-memory Broyden-Fletcher-Goldfarb-Shanno, which is an [Optimisation function](#optimisation-function)optimization algorithm used to find the minimum of a function. In the context of [logistic regression](#logistic-regression), LBFGS is a method for optimizing the cost function to find the optimal [model parameters](#model-parameters) (such as the intercept and coefficients).

Here's a breakdown of the key features of LBFGS:

1. Quasi-Newton Method: LBFGS is a type of Quasi-Newton method, which approximates the inverse of the Hessian matrix (second-order derivatives of the cost function). Instead of computing the full Hessian matrix, it uses an approximation, which makes it more efficient for large datasets.
    
2. Limited Memory: The "limited-memory" part refers to the fact that LBFGS does not store the entire Hessian matrix, which is computationally expensive and memory-intensive. Instead, it keeps a limited amount of information from previous iterations, making it well-suited for large-scale problems where full memory-based methods might not be feasible.
    
3. Optimization for Smooth, Differentiable Functions: It is designed to optimize smooth, differentiable functions like the [cost function](#cost-function) in logistic regression.
    

In the context of logistic regression with sklearn, LBFGS is used as a solver for optimization. When you set `solver='lbfgs'`, [Sklearn](#sklearn)'s logistic regression uses this algorithm to iteratively adjust the model parameters (the intercept and coefficients) to minimize the logistic loss (the cost function) while possibly incorporating regularization.

LBFGS is often preferred for its efficiency and ability to converge quickly without needing a lot of iterations, especially when the number of features is large.

# Llm Evaluation Metrics {#llm-evaluation-metrics}

[LLM Evaluation Metrics](#llm-evaluation-metrics)
- BLEU, 
- ROUGE, 
- perplexity
which quantify the similarity between generated text and reference outputs.

[LLM](#llm)

# Llm {#llm}


A Large Language Model (LLM) is a type of language model designed for language understanding and generation. They can perform a variety of tasks, including:

- Text generation
- Machine translation
- Summary writing
- Image generation from text
- Machine coding
- Chatbots or Conversational AI
# Questions

- [How do we evaluate of LLM Outputs](#how-do-we-evaluate-of-llm-outputs)
- [Memory|What is LLM memory](#memorywhat-is-llm-memory)
- [Relationships in memory|Managing LLM memory](#relationships-in-memorymanaging-llm-memory)
- [Mixture of Experts](#mixture-of-experts): having multiple experts instead of one big model.
- [Distillation](#distillation)
- Mathematics on the parameter usage [Attention mechanism](#attention-mechanism)
- Use of [Reinforcement learning](#reinforcement-learning) in training [Chain of thought](#chain-of-thought) methods in LLM's (deepseek)

## How do Large Language Models (LLMs) Work?

Large [Language Models](#language-models) (LLMs) are a type of artificial intelligence model that is designed to understand and generate human language. Key aspects of how they work include:

- Word Vectors: LLMs represent words as long lists of numbers, known as word vectors ([standardised/Vector Embedding|word embedding](#standardisedvector-embeddingword-embedding)).
- Neural Network Architecture: They are built on a neural network architecture known as the [Transformer](#transformer). This architecture enables the model to identify relationships between words in a sentence, irrespective of their position in the sequence.
- [Transfer Learning](#transfer-learning): LLMs are trained using a technique known as transfer learning, where a pre-trained model is adapted to a specific task.

## Characteristics of LLMs

- <mark>Non-Deterministic:</mark> LLMs are non-deterministic, meaning the types of problems they can be applied to are of a probabilistic nature (<mark>temperature</mark>).
- Data Dependency: The performance and behaviour of LLMs are heavily influenced by the data they are trained on.




# Lstm {#lstm}


# What is LSTM

LSTM (Long Short-Term Memory) networks are a specialized type of Recurrent Neural Network (RNN) designed to overcome the [vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem) that affects traditional [Recurrent Neural Networks](#recurrent-neural-networks). 

LSTMs address this challenge through their unique architecture.

Used for tasks that require the retention of information over time, and problems involving <mark>sequential data.</mark> 

The key strength of LSTMs is their ability to manage <mark>long-term dependencies</mark> using their <mark>gating mechanisms</mark>.
### Key Components of LSTM Networks:

Resources: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

$x_t$ input, $h_t$ output, cell state $C_t$, conveyer belt

![Pasted image 20241015211424.png|500](../content/images/Pasted%20image%2020241015211424.png|500)

Memory Cell:
   - The core of an LSTM network is the memory cell, which maintains information over long time intervals. This cell helps store, forget, or pass on information from previous time steps.

<mark>Gates</mark>:
   - Input Gate: Controls how much of the input should be allowed into the memory cell.
   - Forget Gate: Determines which information should be discarded from the memory cell.
   - Output Gate: Controls what part of the cell's memory should be output as the hidden state for the current time step.

These gates are regulated by <mark>sigmoid</mark> activation, which output values between 0 and 1, acting like a filter to determine the amount of information that should pass through. This gate mechanism allows the LSTM network to maintain a balance between retaining relevant data and discarding unnecessary information over time.
# Why is LSTM less favourable over using transformers

>[!Summary]  
> Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network ([RNN](#rnn)), are less favorable than [Transformer](#transformer) for many modern tasks, especially in Natural Language Processing ([NLP](#nlp)). LSTMs process sequences of data one step at a time, making them inherently sequential and difficult to parallelize. Transformers, on the other hand, leverage a self-attention mechanism that allows them to process entire sequences simultaneously, leading to faster training and the ability to capture long-range dependencies more effectively. 
> 
> Mathematically, LSTM’s sequential nature leads to slower computations, while the Transformer’s attention mechanism computes relationships between all tokens in a sequence, allowing better scalability and performance for tasks like translation, summarization, and language modeling.

>[!Breakdown]  
> Key Components:  
> - Sequential Processing in LSTM: Each time step relies on the previous one, creating a bottleneck for long sequences.  
> - Self-Attention Mechanism in Transformers: Allows simultaneous processing of all elements in a sequence.  
> - Parallelization: Transformers leverage parallel computing more effectively due to non-sequential data processing.  
> - Positional Encoding: Used by Transformers to retain the order of the sequence, overcoming the need for explicit recurrence.

>[!important]  
> - LSTMs are slower in training due to their sequential nature, as calculations depend on previous states.  
> - Transformers efficiently handle long-range dependencies using self-attention, which calculates the relationships between tokens directly without needing previous time steps.

>[!attention]  
> - LSTMs suffer from vanishing/exploding gradient issues, especially in long sequences, limiting their effectiveness for long-term dependencies.  
> - Transformers require more data and computational power to train, which can be a limitation in resource-constrained environments.

>[!Example]  
> In a language translation task, LSTMs process words sequentially, making them less efficient in handling long sentences. In contrast, a Transformer can analyze the entire sentence at once, using self-attention to determine relationships between all words, leading to faster and more accurate translations.

>[!Follow up questions]  
> - How does the attention mechanism in Transformers improve the model's ability to capture long-range dependencies compared to LSTM’s <mark>cell structure?</mark>  
> - In what cases might LSTM still be a better option over Transformers, despite their limitations?

>[!Related Topics]  
> - [Attention mechanism](#attention-mechanism) in deep learning  
> - [BERT](#bert) (Bidirectional Encoder Representations from Transformers)  
> - [GRU](#gru) (Gated Recurrent Unit) as an alternative to LSTM


### Example Workflow in Python using Keras:

In this example, we define a simple LSTM model in [Keras](#keras) for a time series forecasting task. The model processes sequences with 1000 time steps, and the LSTM layer has 50 units, followed by a fully connected (Dense) layer for the final prediction.

```python
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Sample data: time series with 1000 timesteps and 1 feature
time_steps = 1000
features = 1
X_train = np.random.rand(1000, time_steps, features)
y_train = np.random.rand(1000)

# Define LSTM model
model = Sequential()
model.add(LSTM(50, activation='tanh', return_sequences=False, input_shape=(time_steps, features)))
model.add(Dense(1))  # Output layer for regression tasks

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64)
```

# notes

[LSTM](#lstm)
How to implement [LSTM](#lstm) with [PyTorch](#pytorch)?
https://lightning.ai/lightning-ai/studios/statquest-long-short-term-memory-lstm-with-pytorch-lightning?view=public&section=all
without lightning - there is a script there

[LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

# Label Encoding {#label-encoding}



# Labelling Data {#labelling-data}

Possible missing labelling or bias in the data, or under-represented data. Construction of the data set comes from the group collecting it.

Examples:
- ImageNet


# Langchain {#langchain}

[Python](#python) framework

For building apps with [LLM](#llm) and interaction with them and combining models. 

Its end to end, through composability

Example:
[Pandas Dataframe Agent](#pandas-dataframe-agent)

## Modules
### models

interface

### prompts

### chains

sequences of calls too a LLM

### Memory

### Indexes

### Agents and tools

set up [Agentic Solutions](#agentic-solutions)

# Language Model Output Optimisation {#language-model-output-optimisation}



What techniques from [information theory](#information-theory) can be used to measure and optimize the amount of information conveyed by an language model?

In information theory, several techniques can be applied to measure and optimize the amount of information conveyed by an [Language Models](#language-models).

1. Entropy: Entropy measures the uncertainty or unpredictability of a random variable. In the context of language models, it can be used to quantify the uncertainty in predicting the next word in a sequence. Lower entropy indicates more predictable and informative outputs.

2. [Cross Entropy](#cross-entropy): This measures the difference between two probability distributions. For language models, cross-entropy can be used to evaluate how well the predicted distribution of words matches the actual distribution in the data. Minimizing cross-entropy during training helps optimize the model's predictions.

3. Perplexity: Perplexity is a common metric for evaluating language models. It is the exponentiation of the cross-entropy and represents the model's uncertainty in predicting the next word. Lower perplexity indicates a better-performing model.

4. Mutual Information: This measures the amount of information shared between two variables. In language models, it can be used to assess how much information about the input is retained in the output, helping to optimize the model's ability to convey relevant information.

5. KL Divergence: Kullback-Leibler divergence measures how one probability distribution diverges from a second, expected probability distribution. It can be used to optimize language models by minimizing the divergence between the predicted and true distributions.

6. Information Bottleneck: This technique involves finding a balance between compressing the input data and preserving relevant information for the task. It can be used to optimize models by focusing on the most informative features.

7. Rate-Distortion Theory: This involves finding the trade-off between the fidelity of the information representation and the amount of compression. It can be applied to optimize language models by balancing the complexity of the model with the quality of the information conveyed.

8. [Attention Mechanism](#attention-mechanism): While not strictly an information theory concept, attention mechanisms in neural networks can be seen as a way to dynamically allocate information processing resources, focusing on the most informative parts of the input.



### Overview
Language models can be categorized into **large language models ([LLM](#llm))** and **small language models ([SLM](#slm))**. While LLMs boast extensive general-purpose knowledge and capabilities, SLMs offer distinct advantages in certain scenarios, particularly when it comes to efficiency, resource constraints, and task-specific environments.

### Key Differences

| Aspect             | LLMs                                              | SLMs                                                 |
|--------------------|---------------------------------------------------|------------------------------------------------------|
| **Accuracy**        | Higher accuracy across broad tasks due to large datasets and extensive training. | Comparable performance in domain-specific tasks after fine-tuning. |
| **Efficiency**      | Computationally expensive; requires significant resources for training and inference. | More resource-efficient; suited for edge devices and real-time applications. |
| **Interpretability**| Often a "black box"; difficult to explain decision-making. | More interpretable due to simpler architecture. |
| **Generality**      | General-purpose; capable of handling a wide range of tasks. | Task-specific; excels in specific domains and structured data. |

# Language Models {#language-models}


A language model is a machine learning model that is designed to understand, generate, and predict human language. 

It does this by analyzing large amounts of text data to learn the patterns, structures, and relationships between words and phrases. 

They work by assigning probabilities to sequences of words, allowing them to predict the next word in a sentence or generate coherent text based on a given prompt.

Related to:
[LLM](#llm)
[Small Language Models|SLM](#small-language-modelsslm)

# Lasso {#lasso}



1. **L1 Regularization (Lasso Regression)**:
    - In L1 regularization, a penalty proportional to the absolute value of the coefficients is added to the loss function.
    - The L1 penalty tends to shrink some coefficients to exactly zero, effectively performing **feature selection** by eliminating irrelevant features.
    - **Lasso regression** (Least Absolute Shrinkage and Selection Operator) is an example of a model that uses L1 regularization.

    The L1-regularized loss function is:
    $\text{Loss} = \text{MSE} + \lambda \sum_{i=1}^{n} |w_i|$
    where $\lambda$ is the regularization parameter, $w_i$ are the model weights, and MSE is the mean squared error.


For **Lasso Regression (L1)**:

```python
from sklearn.linear_model import Lasso

# Initialize a Lasso model
model = Lasso(alpha=0.1)  # alpha is the regularization strength
model.fit(X_train, y_train)
```

[LINK](https://www.youtube.com/watch?v=NGf0voTMlcs)

In L1 regularization, a penalty term proportional to the absolute value of the weights is added to the loss function. This encourages sparsity in the model by driving some weights to exactly zero, effectively performing feature selection.

- Adds a penalty term proportional to the <mark>absolute value of the model's coefficients.</mark>
- Encourages sparsity by <mark>driving some coefficients to exactly zero.</mark>
- Useful for [Feature Selection](#feature-selection) , as it tends to eliminate less important features by setting their corresponding coefficients to zero.
- Can result in a sparse model with fewer features retained.
-  Lasso regression adds a penalty term to the loss function proportional to the absolute value of the coefficients of the features. This encourages sparsity in the coefficient vector, effectively setting some coefficients to zero and performing feature selection.

# Latency {#latency}



# Latent Dirichlet Allocation {#latent-dirichlet-allocation}


Related terms:
- [topic modeling](#topic-modeling)
- [Semantic Relationships](#semantic-relationships)
- [NLP](#nlp)

**Latent Dirichlet Allocation (LDA)** is a generative probabilistic model used in Natural Language Processing (NLP) and machine learning for topic modeling. It assumes that documents are mixtures of topics, and topics are mixtures of words. The goal of LDA is to uncover the latent topics in a collection of text documents by identifying groups of words that frequently appear together in the same documents.

Libraries like gensim in Python are highlighted as being particularly suitable for topic modeling

### Key Concepts:
- **Topic**: A distribution over a fixed vocabulary of words. A topic might represent a certain theme, like "sports" or "politics."
- **Document**: A mixture of topics. A document is modeled as a combination of topics with different proportions.
- **Words**: Each word in the document is associated with a topic based on the topic proportions of that document.

### How LDA Works:
1. **Assume each document has a mixture of topics**: Each document is made up of a certain proportion of topics. For example, a document could be 70% about "sports" and 30% about "politics."
2. **Assume each topic is a distribution of words**: Each topic has a specific distribution over words. For example, the "sports" topic might have a high probability for words like "football," "game," and "score."
3. **Infer the topics from the documents**: LDA tries to discover these hidden topics from the words in the documents, without knowing the topics in advance.

### Example:

Imagine you have a set of three simple documents:

1. **Document 1**: "Football is a great sport"
2. **Document 2**: "Basketball is also fun to play"
3. **Document 3**: "Soccer and football are similar sports"

LDA might identify two topics:
- **Topic 1**: Words related to "sports" (e.g., "football," "basketball," "soccer").
- **Topic 2**: Words related to "competition" or "play" (e.g., "game," "score," "fun").

The algorithm would assign a mixture of these topics to each document based on the words in the documents.

### Example Code:

Here’s a simple implementation of LDA using `sklearn` in Python:

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
docs = [
    "Football is a great sport",
    "Basketball is also fun to play",
    "Soccer and football are similar sports"
]

# Step 1: Convert the documents into a document-term matrix (DTM)
vectorizer = CountVectorizer(stop_words='english')
dtm = vectorizer.fit_transform(docs)

# Step 2: Apply LDA to discover topics
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(dtm)

# Step 3: Display topics
terms = vectorizer.get_feature_names_out()

for index, topic in enumerate(lda.components_):
    print(f"Topic #{index + 1}:")
    print([terms[i] for i in topic.argsort()[-5:]])  # Print the top 5 words in the topic
    print()

# Example Output:
# Topic #1:
# ['football', 'sports', 'game', 'score', 'play']
# Topic #2:
# ['basketball', 'soccer', 'sport', 'fun', 'great']
```

### Output Explanation:
- **Topic #1** might have words like **football**, **sports**, **game**, because these terms appear frequently in the documents related to sports.
- **Topic #2** could have terms like **basketball**, **soccer**, **fun**, because these are associated with the activities discussed in the documents.

### How to Interpret:
- **Document 1 ("Football is a great sport")**: LDA might classify this document as being 60% about Topic #1 (sports-related) and 40% about Topic #2 (competitive play).
- **Document 2 ("Basketball is also fun to play")**: LDA might classify this document as being 80% about Topic #2 (competitive play) and 20% about Topic #1 (sports-related).

### Why Use LDA?
- **Topic Discovery**: It helps discover hidden themes in a large corpus of text data.
- **Dimensionality Reduction**: LDA reduces the complexity of the text data by modeling it with a smaller number of topics instead of many individual words.


# Learning Styles {#learning-styles}


What does the data look like [continuous](#continuous) or [categorical](#categorical)? 

![ Pasted image 20240112101344.png|500](../content/images/%20Pasted%20image%2020240112101344.png|500)

[Unsupervised Learning](#unsupervised-learning)
	 [Regression](#regression) 
	 [Classification](#classification)

[Unsupervised Learning](#unsupervised-learning)
	[Dimensionality Reduction](#dimensionality-reduction)
	[Clustering](#clustering)





# Lightgbm Vs Xgboost Vs Catboost {#lightgbm-vs-xgboost-vs-catboost}


This table summarizes the key differences and strengths of each [Gradient Boosting](#gradient-boosting) framework.

| Feature/Aspect                    | [LightGBM](#lightgbm) (LGBM)                                                        | [XGBoost](#xgboost)                                                                     | [CatBoost](#catboost)                                                           |
| --------------------------------- | -------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Tree Growth Strategy          | Leaf-wise growth, leading to deeper trees and potentially better accuracy. | Level-wise growth, resulting in more balanced trees.                            | Ordered boosting, reducing overfitting and improving generalization.   |
| Speed and Memory              | High speed and low memory usage, especially with large datasets.           | Balanced speed and accuracy with robust regularization options.                 | Competitive performance with minimal hyperparameter tuning.            |
| Handling Categorical Features | Requires preprocessing (e.g., label encoding).                             | Requires preprocessing of categorical features.                                 | Natively handles categorical features without preprocessing.           |
| [Regularisation](#regularisation)            | Supports regularization but not as robust as XGBoost.                      | Strong regularization options (L1 and L2) to prevent overfitting.               | Utilizes techniques like ordered boosting to mitigate overfitting.     |
| Use Cases                     | Ideal for large datasets and when computational efficiency is a priority.  | Suitable for structured data and tabular datasets; widely used in competitions. | Useful for datasets with many categorical features and missing values. |
| Performance                   | Fast training and efficient on large datasets.                             | Accurate and flexible, often used in competitions.                              | Provides competitive performance, especially with categorical data.    |



# Lightgbm {#lightgbm}


LightGBM is a gradient boosting framework that is designed for speed and efficiency. It is particularly well-suited for handling large datasets and high-dimensional data.

- **Tree Growth**: Splits the tree leaf-wise, which can lead to faster convergence compared to level-wise growth.
- **Learning Rate**: Similar to [Gradient Descent](#gradient-descent), LightGBM uses a learning rate to control the contribution of each tree.
- **DART**: A variant of LightGBM known for its performance.
- **Parameter Definition**: Requires parameters to be defined in a dictionary for model configuration.

[Watch Video Explanation](https://www.youtube.com/watch?v=n_ZMQj09S6w)

### Key Parameters

- **Learning Rate**: Controls the step size at each iteration while moving toward a minimum of the loss function.
- **Number of Leaves**: Determines the complexity of the tree model.

### Advantages

- **Speed**: Renowned for its speed, often outperforming other gradient boosting implementations.
- **Memory Usage**: Optimizes memory usage, enabling efficient handling of large datasets.
- **Leaf-Wise Growth**: Grows trees leaf-wise, leading to faster convergence.
- **Parallel and GPU Learning**: Supports parallel and GPU learning for further speedup.

### Use Cases

- **Large Datasets**: Ideal for applications where speed is crucial.
- **High-Dimensional Data**: Efficient when dealing with high-dimensional data and categorical features.

# Linear Discriminant Analysis {#linear-discriminant-analysis}



# Linear Regression {#linear-regression}


# Description

Linear regression assumes [linearity](#linearity) between the input features and the target variable. Assumes that the relationship between the independent variable(s) and the dependent variable is linear.

During the training phase, the algorithm adjusts the slope (m) and the intercept (b) of the line to minimize the [Loss function](#loss-function).

The linear regression model is represented as:

$y = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n$

- $y$ is the dependent variable (the variable we want to predict).
- $x_1, x_2, \ldots, x_n$ are the independent variables (features or predictors).
- $b_0, b_1, b_2, \ldots, b_n$ are the coefficients (weights) associated with each independent variable.
- $b_0$ is the intercept term.

You <mark>evaluate</mark> the performance of your model by comparing its predictions to the actual values in a separate test dataset. Common metrics for evaluating regression models  are:

-  Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
-  [R squared](#r-squared).

The goal of linear regression is to find the values of coefficients $$b_0, b_1, b_2, \ldots, b_n$$ that <mark>minimize the sum of squared errors (SSE),</mark> also known as the residual sum of squares (RSS) or (MSE - mean square error).

Mathematically, SSE is a [Loss function](#loss-function) given by:

$$SSE = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

where $N$ is the number of observations, $y_i$ is the actual value of the dependent variable for observation $i$, and $\hat{y}_i$ is the predicted value based on the linear regression model.

The formula for the Regression Sum of Squares (SSR) in the context of linear regression is:

$$SSR = \sum_{i=1}^{N} (\hat{y}_i - \bar{y})^2 $$

Where:
- $\hat{y}_i$ is the predicted value of the dependent variable for observation $i$ based on the linear regression model.
- $\bar{y}$ is the mean of the observed values of the dependent variable.
- $N$ is the total number of observations.

SSR measures the amount of variability in the dependent variable that is explained by the independent variables in the model. It <mark>reflects how well the regression model captures the relationship between the independent and dependent variables.</mark>

Total Sum of Squares (SST) represents the total variability in the dependent variable $y$. The relationship between SST, SSR, and SSE is given by:

 $SST = SSR + SSE$ 

This equation reflects the decomposition of total variability into explained variability (SSR) and unexplained variability (SSE) due to errors.
### [Ordinary Least Squares](#ordinary-least-squares)

The Ordinary Least Squares method is used to minimize SSE. It achieves this by finding the values of  that minimize the sum of squared differences between the observed and predicted values. The formulas for  are derived by setting partial derivatives of SSE with respect to each coefficient to zero.

OLS is an analytical method
### [Gradient Descent](#gradient-descent)

It <mark>iteratively</mark> updates coefficients to minimize error.

Gradient descent is an optimization algorithm used to minimize the cost function in linear regression by iteratively adjusting the model parameters (coefficients). Here's how it works with linear regression:

1. **Initialize Parameters**: Start with initial guesses for the coefficients (weights), typically set to zero or small random values.
2. **Compute Predictions**: Use the current coefficients to make predictions for the dependent variable $\hat{y}$ using the linear regression model: $\hat{y} = b_0 + b_1x_1 + b_2x_2 + \ldots + b_nx_n$
3. **Calculate the Cost Function**:Compute the loss function, SSE.
4. **Compute the Gradient**: Calculate the gradient of SSE function with respect to each coefficient. The gradient is a vector of partial derivatives indicating the direction and rate of change of the cost function:
     $$\frac{\partial J}{\partial b_j} = -\frac{2}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i) x_{ij}$$
   Here, $x_{ij}$ is the value of the $j$-th feature for the $i$-th observation.
5. **Update the Coefficients**: Adjust the coefficients in the opposite direction of the gradient to reduce the cost function. This is done using a [learning rate](#learning-rate)$\alpha$, which controls the size of the steps taken: $b_j = b_j - \alpha \frac{\partial J}{\partial b_j}$

6. **Iterate** & Converge Repeat steps 2 to 5 until the cost function converges to a minimum or a predefined number of iterations is reached. The algorithm converges when the changes in the cost function or the coefficients become very small, indicating that the minimum has been reached.


# [Model Evaluation](#model-evaluation)

- [R squared](#r-squared)
- [Adjusted R squared](#adjusted-r-squared) takes into account the number of variables.
- [F-statistic](#f-statistic) overall significance of model (lower worse).
- [Feature Selection](#feature-selection) Use P>|t| column to decide whether to keep variable less that 0.05
- [p-values in linear regression in sklearn](#p-values-in-linear-regression-in-sklearn)

![Pasted image 20240117145455 1.png|500](../content/images/Pasted%20image%2020240117145455%201.png|500) ![Pasted image 20240124135607.png|500](../content/images/Pasted%20image%2020240124135607.png|500)
### Impact of Extra Variables on Intercept:

When additional variables are introduced, it can impact the intercept ($b_0$) in the linear regression model. The intercept is the value of $y$ when all independent variables ($x_1, x_2, \ldots, x_n$) are zero. The presence of extra variables can affect the baseline value of the dependent variable.


# Linked List {#linked-list}


A **linked list** is a linear data structure in which elements (called **nodes**) are linked together using pointers. Unlike arrays, linked lists do not store elements in contiguous memory locations; instead, each node contains:

1. **Data** – The actual value stored in the node.
2. **Pointer (or Reference)** – A reference to the next node in the sequence.

### Types of Linked Lists:

1. **Singly Linked List** – Each node has a pointer to the next node only.
2. **Doubly Linked List** – Each node has pointers to both the previous and next nodes.
3. **Circular Linked List** – The last node points back to the first node, forming a circular structure.
    - Circular **singly** linked list: Last node points to the first node.
    - Circular **doubly** linked list: Last node points to the first node, and first node points back to the last.

### Advantages of Linked Lists:

- **Dynamic size** – Unlike arrays, they do not have a fixed size.
- **Efficient insertions/deletions** – Adding or removing elements does not require shifting (unlike arrays).
- **Efficient memory utilization** – Memory is allocated as needed.

### Disadvantages of Linked Lists:

- **Extra memory usage** – Each node requires additional storage for pointers.
- **Slower access time** – No direct access like arrays; traversal is required.

```python
class Node:
    """A node in a singly linked list."""
    def __init__(self, data):
        self.data = data  # Store data
        self.next = None  # Pointer to the next node (initially None)

class LinkedList:
    """A simple singly linked list."""
    def __init__(self):
        self.head = None  # Initialize the list as empty

    def append(self, data):
        """Add a new node at the end of the list."""
        new_node = Node(data)
        if not self.head:  # If the list is empty
            self.head = new_node
            return
        last = self.head
        while last.next:  # Traverse to the last node
            last = last.next
        last.next = new_node  # Link the last node to the new node

    def display(self):
        """Print the linked list elements."""
        current = self.head
        while current:
            print(current.data, end=" -> ")
            current = current.next
        print("None")  # Indicate the end of the list

    def delete(self, key):
        """Delete a node by value."""
        current = self.head

        # If the node to be deleted is the head
        if current and current.data == key:
            self.head = current.next
            current = None
            return

        prev = None
        while current and current.data != key:
            prev = current
            current = current.next

        if current is None:  # If the key was not found
            return

        prev.next = current.next
        current = None

# Example usage
ll = LinkedList()
ll.append(10)
ll.append(20)
ll.append(30)
ll.display()  # Output: 10 -> 20 -> 30 -> None

ll.delete(20)
ll.display()  # Output: 10 -> 30 -> None
```

# Load Balancing {#load-balancing}

Load balancing is a technique used to distribute incoming network traffic across multiple servers. This helps ensure both reliability and performance by preventing any single server from becoming overwhelmed with too much traffic.



# Local Interpretable Model Agnostic Explanations {#local-interpretable-model-agnostic-explanations}



LIME explains individual predictions <mark>by approximating the model locally</mark> with an interpretable model and calculating the feature importance based on the surrogate model.

### Key Points

- **Purpose**: LIME focuses on explaining individual predictions by approximating the model locally using a simpler, interpretable model (like linear regression).
  
- **How it Works**: 
  - For a given prediction, LIME generates perturbed samples (e.g., by modifying input features).
  - It observes how the predictions change, thus inferring feature importance for that specific instance.

- **Use Cases**: Useful for understanding why a specific decision was made in complex black-box models.

- **Advantage**: 
  - LIME can work with any model type.
  - It is relatively easy to apply to tabular, text, and image data.

- **Scenario**: 
  - A healthcare provider uses a deep learning model to classify whether patients have a high or low risk of heart disease based on several health metrics, such as cholesterol levels, age, and blood pressure.
- 
  - **LIME Explanation**: LIME is used to explain why the model flagged a specific patient as high-risk. By perturbing the input data (e.g., altering cholesterol levels and re-running the prediction), LIME shows that the patient’s high cholesterol level and advanced age are the main reasons for the high-risk classification. This makes it easier for the healthcare provider to justify the decision to recommend lifestyle changes or further medical tests.

### Example Code

To use LIME for feature importance, you can use the LIME package:

```python
import lime
import lime.lime_tabular

# Create a LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names)

# Explain a single prediction
explanation = explainer.explain_instance(X_test[0], model.predict_proba)
explanation.show_in_notebook()
```


# Logical Model {#logical-model}

Logical Model
   - Customer: CustomerID, Name, Email
   - Order: OrderID, OrderDate, CustomerID
   - Book: BookID, Title, Author
   - Order-Book Relationship: OrderID, BookID

Logical Model
   - Details the attributes of each data entity.
   - Specifies relationships without depending on a specific database management system.

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
![Pasted image 20240124095916.png](../content/images/Pasted%20image%2020240124095916.png)



$$\mathbb{N}$$

# Logistic Regression Does Not Predict Probabilities {#logistic-regression-does-not-predict-probabilities}

In logistic regression, the model predicts the <mark>odds of an event happening rather than directly predicting probabilities.</mark> The odds are defined as:

$$ \text{Odds} = \frac{P(\text{success})}{P(\text{failure})} = \frac{p}{1-p} $$

  where $p$ is the probability of success. The log-odds (or logit function) is the natural logarithm of the odds:
  $$ \text{Log-Odds} = \ln\left(\frac{p}{1-p}\right) = b_0 + b_1 x $$
  This <mark>transformation makes the relationship between the independent variables and the dependent variable linear,</mark> allowing logistic regression to estimate the parameters $b_0$ and $b_1$.

Resources:
- [Explanation of log odds](https://www.youtube.com/watch?v=ARfXDSkQf1Y)
- [Explanation of log odd function](https://www.youtube.com/watch?v=fJ53tIDbvTM)

[What is the difference between odds and probability](#what-is-the-difference-between-odds-and-probability)

# Logistic Regression {#logistic-regression}


<mark>Logistic regression models the log-odds of the probability as a linear function of the input features.</mark>

It models the probability of an input belonging to a particular class using a logistic (sigmoid) function.

The model establishes a decision boundary (threshold) in the feature space.

Logistic regression is best suited for cases where the decision boundary is approximately linear in the feature space.

Logistic [Regression](#regression)  can be used for [Binary Classification](#binary-classification)tasks.

### Related Notes:
- [Logistic Regression Statsmodel Summary table](#logistic-regression-statsmodel-summary-table)
- [Logistic Regression does not predict probabilities](#logistic-regression-does-not-predict-probabilities)
- [Interpreting logistic regression model parameters](#interpreting-logistic-regression-model-parameters)
- [Model Evaluation](#model-evaluation)
- To get [Model Parameters](#model-parameters) use [Maximum Likelihood Estimation](#maximum-likelihood-estimation)

In [ML_Tools](#ml_tools), see:
- [Regression_Logistic_Metrics.ipynb](#regression_logistic_metricsipynb)
## Key Concepts of Logistic Regression

### Logistic Function (Sigmoid Function)

Logistic regression models the probability that an input belongs to a particular class using the logistic (sigmoid) function. This function maps any real-valued input into the range (0,1), representing the probability of belonging to the positive class (usually class 1).

The sigmoid function is defined as:  
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$  
where  
$$ z = \mathbf{w} \cdot \mathbf{x} + b $$  
Thus, the logistic regression model is given by:  
$$ P(y=1 \mid \mathbf{x}) = \sigma(z) $$  

### Log odds: Transforming from continuous to 0-1

Logistic regression is based on the <mark>log-odds</mark> (logit) transformation, which expresses probability in terms of odds:

$$ \text{Odds} = \frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})} $$

Taking the natural logarithm of both sides gives the logit function:

$$ \log \left(\frac{P(y=1 \mid \mathbf{x})}{1 - P(y=1 \mid \mathbf{x})} \right) = \mathbf{w} \cdot \mathbf{x} + b $$

This equation shows that <mark>logistic regression models the log-odds of the probability as a linear function of the input features.</mark>

### Decision Boundary

- Similar to [Support Vector Machines](#support-vector-machines), logistic regression defines a decision boundary that separates the two classes.
- The logistic function determines the probability of a data point belonging to a specific class. If this probability exceeds a given <mark>threshold</mark> (typically 0.5), the model assigns the point to the positive class; otherwise, it is classified as negative.

### [Binary Classification](#binary-classification)

- Logistic regression is primarily used for binary classification tasks, where the target variable has only two possible values (e.g., "0" and "1").
- It can handle multiple independent variables (features) and assigns probabilities to the target classes based on the feature values.
- Examples include:

### No Residuals

- Unlike [Linear Regression](#linear-regression), logistic regression does not compute standard residuals.
- Instead, [Model Evaluation](#model-evaluation) is performed by comparing predicted probabilities with actual class labels using metrics such as accuracy, precision, recall, and the [Confusion Matrix](#confusion-matrix).

### Also see:

Related terms:
- Cost function for logistic regression
- Gradient computation in logistic regression
- Regularized logistic regression
- Cost function for regularized logistic regression

Logistic regression can be extended to handle non-linear decision boundaries through:
- Polynomial features to capture more complex relationships.
- Regularization techniques to improve generalization.

[Explaining logistic regression](https://www.youtube.com/watch?v=Iju8l2qgaJU)


# Logistic regression in sklearn & Gradient Descent

sklearn's Logistic Regression implementation does not use [Gradient Descent](#gradient-descent) by default. Instead, it uses more sophisticated optimization techniques depending on the solver specified. These solvers are more efficient and robust for finding the optimal parameters for logistic regression. Here's a summary:

### [Optimisation function](#optimisation-function): Solvers in sklearn's Logistic Regression**

1. **`lbfgs` (default in many cases)**:
    - Stands for Limited-memory Broyden–Fletcher–Goldfarb–Shanno.
    - It's a quasi-Newton method, which approximates the second derivatives (Hessian matrix) to find the minimum of the cost function efficiently.
      
2. **`liblinear`**:
    - Uses the coordinate descent method for optimization.
    - Ideal for small datasets or when `penalty='l1'` is used.
      
3. **`sag` (Stochastic Average Gradient)**:
    - An iterative solver similar to stochastic gradient descent (SGD) but averages gradients over all samples.
    - Efficient for large datasets.
      
4. **`saga`**:
    - An improved version of `sag`, supporting both `l1` and `l2` penalties.
    - Suitable for sparse and large datasets.
      
5. **`newton-cg`**:
    - Uses the Newton method with conjugate gradients.
    - Efficient for datasets with many features.

# Looker Studio {#looker-studio}

Looker studio is [Google](#google) version of [PowerBI](#powerbi), but its free.
### Connectors to data

Can connect to data sources i.e:
- [standardised/GSheets|GSheets](#standardisedgsheetsgsheets)
- [PostgreSQL](#postgresql)

### Data Modelling

Data models are called Blends

[Dashboard with Relational Database in Looker Studio Data Blending and Modeling](https://www.youtube.com/@virtual_school)

# Loss Function {#loss-function}


Loss functions are used in training machine learning models. Also known as a [cost function](#cost-function), error function, or objective function. Serves as a metric for [model evaluation](#model-evaluation).

Purpose: <mark>Measure predictive accuracy</mark>: Measures the difference between predicted and actual values. That is they measure how well a model's predictions match the actual target values by quantifying the error between the predicted output and the true output. 

Goal: <mark>To be minimized</mark>: The primary goal during model training is to minimize this loss, improving accuracy of predictions on unseen data.

Used during training to adjust [model parameters](#model-parameters) and during evaluation to assess model performance.

### Examples

- **[Mean Squared Error](#mean-squared-error) (MSE)**: Commonly used for [regression](#regression) tasks.
- **[Cross Entropy](#cross-entropy)**: Often used for [classification](#classification) tasks.

Resources
- [Video Explanation](https://www.youtube.com/watch?v=-qT8fJTP3Ks)
- [Loss versus Cost function](#loss-versus-cost-function)


# Loss Versus Cost Function {#loss-versus-cost-function}

In machine learning, the terms "loss function" and "cost function" are often used interchangeably, but they can have slightly different meanings depending on the context:

1. [Loss function](#loss-function): This typically refers to the function that measures the error for a single training example. It quantifies how well or poorly the model is performing on that specific example - data point. Common examples include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.

2. [Cost Function](#cost-function): This is generally used to refer to the average of the loss function over the entire training dataset. It provides a measure of how well the model is performing overall. 
   
   The cost function is what is minimized during the training process to improve the model's performance, see [Model Optimisation|Optimisation](#model-optimisationoptimisation). Used with the parameters to determine the best ones.

[Model parameters vs hyperparameters](#model-parameters-vs-hyperparameters)


# Lambda Architecture {#lambda-architecture}


Lambda architecture is a data-processing architecture designed to handle massive quantities of data by taking advantage of both batch and stream-processing methods. [Data Streaming](#data-streaming)

This approach to architecture attempts to balance <mark>latency, throughput, and fault tolerance</mark> using batch processing to provide comprehensive and accurate views of batch data, while simultaneously using real-time stream processing to provide views of online data. 

The two view outputs may be joined before the presentation. The rise of lambda architecture is correlated with the growth of big data, real-time analytics, and the drive to mitigate the latencies of [MapReduce](term/map%20reduce.md).

Lambda architecture is a [design pattern](#design-pattern) for processing large volumes of data by combining both batch and stream processing methods. It aims to provide a comprehensive and efficient way to handle big data by addressing the needs for low-latency data processing, high throughput, and fault tolerance. 

[Batch Processing](#batch-processing)

Lambda architecture is particularly useful in scenarios where <mark>both historical</mark> and <mark>real-time data insights</mark> are crucial, such as in financial services, telecommunications, and online retail. It allows organizations to leverage the strengths of both batch and stream processing to meet diverse data processing needs.

### Components of Lambda Architecture

1. **Batch Layer**:
   - **Purpose**: To process large sets of historical data in batches.
   - **Functionality**: It computes results from all available data, ensuring accuracy and completeness.
   - **Tools**: Often uses distributed processing frameworks like Hadoop or Spark for batch processing.
   - **Output**: Produces a batch view, which is a complete and accurate dataset that can be queried.

2. **Speed Layer**:
   - **Purpose**: To process real-time data streams with low latency.
   - **Functionality**: It provides immediate insights by processing data as it arrives.
   - **Tools**: Utilizes stream processing frameworks like Apache Storm, Apache Flink, or Spark Streaming.
   - **Output**: Generates a real-time view that reflects the most recent data.

3. **Serving Layer**:
   - **Purpose**: To merge and serve the results from both the batch and speed layers.
   - **Functionality**: It combines the batch view and real-time view to provide a unified, queryable dataset.
   - **Tools**: Databases or data stores optimized for fast reads, such as Cassandra or HBase, are often used.

### How Lambda Architecture Works

- **Data Ingestion**: Data is ingested into both the batch and speed layers simultaneously.
- **Batch Processing**: The batch layer processes data in large volumes, typically with higher latency, to ensure accuracy and completeness.
- **Stream Processing**: The speed layer processes data in real-time, providing low-latency updates.
- **Data Serving**: The serving layer combines outputs from both layers, allowing users to query the most up-to-date and accurate data.

### Benefits of Lambda Architecture

- **Fault Tolerance**: By separating batch and real-time processing, the architecture can handle failures more gracefully.
- **Scalability**: It can scale to handle large volumes of data by leveraging distributed processing frameworks.
- **Flexibility**: Supports both historical and real-time data processing, making it suitable for a wide range of applications.

### Challenges

- **Complexity**: Maintaining two separate processing paths (batch and speed) can increase system complexity.
- **Data Consistency**: Ensuring consistency between batch and real-time views can be challenging.
- **Maintenance**: Requires more effort to maintain and update due to its dual-layer nature.



# Learning Rate {#learning-rate}


### Description

The learning rate is a  [Hyperparameter](#hyperparameter) in machine learning that <mark>determines the step size at which a model's parameters are updated during training</mark>. It plays a significant role in the optimization process, particularly in algorithms like [Gradient Descent](#gradient-descent) which are used to minimize the [Loss function](#loss-function).

### Key Points about Learning Rate:

1. **Parameter Updates**:
   - During training, the model's parameters (such as weights and biases in neural networks) are adjusted iteratively to minimize the loss function.
   - The learning rate controls how much the parameters are changed in response to the estimated error each time the model weights are updated.

2. **Impact on Training**/ Convergence
   - A high learning rate can lead to faster convergence but <mark>risks overshooting</mark> the optimal solution, potentially causing the model to diverge.
   - A low learning rate ensures more stable and precise convergence but may result in slow training and can get stuck in local minima. A lower learning rate makes the model more robust but requires more iterations to converge.
   - 
1. **Tuning**:
   - The learning rate is a hyperparameter that needs careful tuning. It can be adjusted manually or through automated hyperparameter optimization techniques like [standardised/Optuna](#standardisedoptuna). 
   - The optimal learning rate depends on various factors, including the dataset, model complexity, and the specific optimization algorithm used.

3. **Practical Considerations**:
   - It's common to start with a moderate learning rate and adjust based on the model's performance during training.
   - Techniques like learning rate schedules or adaptive learning rate methods (e.g., [Adam Optimizer](#adam-optimizer)) can dynamically adjust the learning rate during training to improve convergence.


This impacts the efficiency of [Gradient Descent](#gradient-descent)

Effects occur if too small (takes long), or too large (over shoots missing the minima).

What happens if you are at a local minima? Then no change.

![Pasted image 20241216204925.png](../content/images/Pasted%20image%2020241216204925.png)

# Lemmatization {#lemmatization}


Lemmatization is the process of <mark>reducing a word to its base or root</mark> form, known as the "lemma." 

Unlike stemming, which simply cuts off word endings, lemmatization considers the context and morphological analysis of the words. 

It ensures that the root word is a valid word in the language. <mark>For example, the words "running," "ran," and "runs" would all be lemmatized to "run."</mark> 

This process helps in normalizing text data for natural language processing tasks by grouping together different forms of a word.