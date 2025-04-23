# B

## Table of Contents
* [B-tree](#b-tree)
* [BERT Pretraining of Deep Bidirectional Transformers for Language Understanding](#bert-pretraining-of-deep-bidirectional-transformers-for-language-understanding)
* [BERT](#bert)
* [BERTScore](#bertscore)
* [Backpropagation](#backpropagation)
* [Bag of words](#bag-of-words)
* [Bag_of_Words.py](#)
* [Bagging](#bagging)
* [Bandit_Example_Fixed.py](#bandit_example_fixedpy)
* [Bandit_Example_Nonfixed.py](#bandit_example_nonfixedpy)
* [Bash](#)
* [Batch Normalisation](#batch-normalisation)
* [Batch Processing](#batch-processing)
* [Bellman Equations](#bellman-equations)
* [Benefits of Data Transformation](#)
* [Bernoulli](#bernoulli)
* [Bias and variance](#bias-and-variance)
* [Big Data](#big-data)
* [Big O Notation](#big-o-notation)
* [BigQuery](#bigquery)
* [Binary Classification](#binary-classification)
* [Binder](#binder)
* [Boosting](#boosting)
* [Bootstrap](#bootstrap)
* [Boxplot](#boxplot)
* [Business observability](#business-observability)
* [business intelligence](#business-intelligence)



# B Tree {#b-tree}



# Bert Pretraining Of Deep Bidirectional Transformers For Language Understanding {#bert-pretraining-of-deep-bidirectional-transformers-for-language-understanding}




# Bert {#bert}


BERT (<mark>Bidirectional Encoder Representations from [Transformer](#transformer)</mark>) is used in [NLP](#nlp)processing, developed by [Google](#google). 

Introduced in the paper "[BERT Pretraining of Deep Bidirectional Transformers for Language Understanding](#bert-pretraining-of-deep-bidirectional-transformers-for-language-understanding)" in 2018. 

It is forward & backward looking in the context.

BERT is a stack of encoders -learning context.

Input [Vector Embedding|embedding](#vector-embeddingembedding):
- [Positional Encoding](#positional-encoding): passes location info to encoder
- Sentence embeddings: differences between sentences
- Token embeddings

Training of BERT:
- Masked Language modelling (hiding words)
- Next Sentence Prediction

Fine tuning ([Transfer Learning](#transfer-learning)) BERT model:
- New output layer dependent
 
Resources:
- [What is BERT and how does it work? | A Quick Review](https://www.youtube.com/watch?v=6ahxPTLZxU8&list=PLcWfeUsAys2my8yUlOa6jEWB1-QbkNSUl&index=12)

### What is BERT?

- BERT is based on the [Transformer](#transformer) architecture and utilizes a bidirectional approach, meaning it considers the <mark>context of a word based on both its left and right surroundings in a sentence.</mark> This allows BERT to capture nuanced meanings and relationships between words more effectively than unidirectional models

- Pre-training and Fine-tuning/[Transfer Learning](#transfer-learning) techniques. It learns to predict masked words in sentences (Masked Language Model) and to determine if one sentence follows another (Next Sentence Prediction).
### What is BERT Used For?

1. Text Classification: Assigning categories or labels to text documents, such as sentiment analysis or topic classification.

2. Named Entity Recognition ([Named Entity Recognition|NER](#named-entity-recognitionner)): Identifying and classifying entities (e.g., names, organizations, locations) within text.

3. Question Answering: Providing answers to questions based on a given context or passage of text.

4. Text [Summarisation](#summarisation): Generating concise summaries of longer documents while retaining key information.

5. Language Translation: Assisting in translating text from one language to another.

6. [Sentence Similarity](#sentence-similarity) :Measuring the similarity between sentences, which can be useful for tasks like paraphrase detection or duplicate question identification.



# Bertscore {#bertscore}



# Backpropagation {#backpropagation}


>[!Summary]  
> Backpropagation is an essential algorithm in the training of neural networks and iteratively correcting its mistakes. It involves a process of calculating the gradient of the loss function $L(\theta)$ concerning each weight in the network, allowing the system to update its weights via [Gradient Descent](#gradient-descent). 
> 
> This process helps minimize the difference between predicted outputs and actual target values. Mathematically, the chain rule of calculus is employed to propagate errors backward through the network.
> 
Each layer in the network computes a partial derivative that is used to adjust the weights. This iterative approach continues until a convergence criterion is met, typically when the change in loss falls below a threshold.
>
>The backpropagation algorithm is critical in [Supervised Learning](#supervised-learning), where labeled data is used to train models to recognize patterns.

>[!Breakdown]  
> Key Components:  
> - **Algorithm**: Gradient Descent  
> - **Mathematical Foundation**: Chain Rule for derivatives  
> - **Metrics**: Loss function (e.g., Mean Squared Error, Cross-Entropy)

>[!important]  
> - Gradient descent uses $\nabla L(\theta) = \frac{\partial L}{\partial \theta}$ to iteratively minimize the loss.  
> - Backpropagation optimizes deep learning models by adjusting weights based on error gradients.

>[!attention]  
> - The method is computationally expensive for deep networks due to the need to compute gradients for each layer.  
> - Vanishing/exploding gradients in deep layers can prevent proper weight updates.

>[!Example]  
> A feed-forward neural network trained on image classification data uses backpropagation to minimize cross-entropy loss. The gradient of the loss is calculated layer by layer, adjusting weights through an optimization algorithm like Adam.

>[!Follow up questions]  
> - How does backpropagation compare with other optimization algorithms such as Newton’s method or evolutionary strategies?  
> - What role does [Regularisation](#regularisation) play in addressing overfitting when using backpropagation in deep neural networks?

>[!Related Topics]  
> - Gradient Descent Optimizers ([Adam Optimizer](#adam-optimizer), RMSprop)  
> - [vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem)


# Backpropgation

Backpropgation is used to calc the gradient of the loss function with respect to the model parameters, when there are a lot of parameters - i.e in a neural network.
Simple example of backpropgation

Simple example of computation graph
Computation graph - calcing derivatives?
Use sympy to calculate derivatives for the loss function.

> The steps in backprop   
>Now that you have worked through several nodes, we can write down the basic method:\
> working right to left, for each node:
>- calculate the local derivative(s) of the node
>- using the chain rule, combine with the derivative of the cost with respect to the node to the right.   

The 'local derivative(s)' are the derivative(s) of the output of the current node with respect to all inputs or parameters.

Example of using sympy to calculate derivatives for the loss function. Use `diff`, `subs`
```python
from sympy import symbols, diff
```

## [Sympy](#sympy)

# Bag Of Words {#bag-of-words}


In [ML_Tools](#ml_tools) see: [Bag_of_Words.py](#bag_of_wordspy)

In the context of natural language processing (NLP), the Bag of Words (BoW) model is a simple and commonly used <mark>method for text representation</mark>. It converts text data into numerical form by treating each <mark>document as a collection of individual words, disregarding grammar and word order</mark>. Here's how it works:

1. Vocabulary Creation: A vocabulary is created from the entire corpus, which is a list of all unique words appearing in the documents.

2. Vector Representation: Each document is represented as a vector, where each element corresponds to a word in the vocabulary. The value of each element is typically the count of occurrences of the word in the document.

3. Simplicity and Limitations: While BoW is easy to implement and useful for tasks like text classification, it has limitations. It ignores word order and context, and can result in large, sparse vectors for large vocabularies.

Despite its simplicity, BoW can be effective for certain NLP tasks, especially when combined with other techniques like [TF-IDF](#tf-idf) to weigh the importance of words.

Takes key terms of a text in normalised <mark>unordered</mark> form.

`CountVectorizer` from scikit-learn to convert a collection of text documents into a matrix of token counts.

```python
#Need normalize_document
from sklearn.feature_extraction.text import CountVectorizer

# Using CountVectorizer with the custom tokenizer
bow = CountVectorizer(tokenizer=normalize_document)
bow.fit(corpus)  # Fitting text to this model
print(bow.get_feature_names_out())  # Key terms
```

Represent each sentence by a vector of length determined by get_feature_names_out. representing the tokens contained.

### Summary of What the Script Does:

1. It takes a dataset of text (movie reviews in this case) and processes it to remove HTML tags, non-alphabetic characters, and stopwords.
2. It transforms the cleaned text into numerical features using the **Bag of Words** model, where each word in the reviews is counted and represented as a feature.
3. It prints a sample of the top features (words) that were extracted from the reviews.

This is a typical text preprocessing pipeline used to prepare textual data for machine learning models.

# Bagging {#bagging}


# Overview:

Bagging, short for Bootstrap Aggregating, is an [Model Ensemble](#model-ensemble) technique designed to improve the stability and accuracy of machine learning algorithms. 

It works by <mark>training multiple instances of the same learning algorithm on different subsets of the training data</mark> and then <mark>combining their predictions.</mark>

### How Bagging Works:

1. **Bootstrap Sampling**: Bagging involves creating multiple subsets of the training data by sampling with replacement. This means that each subset, or "bootstrap sample," is drawn randomly from the original dataset, and some data points may appear multiple times in a subset while others may not appear at all.

2. **Parallel Training**: Each bootstrap sample is used to train a separate instance of the same base learning algorithm. These models are trained independently and in parallel, which makes bagging computationally efficient.

3. **Combining Predictions**: Once all models are trained, their predictions are combined to produce a final output. For regression tasks, this is typically done by <mark>averaging</mark> the predictions. For classification tasks, <mark>majority voting</mark> is used to determine the final class label.

### Key Concepts of Bagging:

- **Reduction of [Overfitting](#overfitting)**: By averaging the predictions of multiple models, bagging reduces the variance and helps prevent overfitting, especially in high-variance models like decision trees.

- **Diversity**: The use of different subsets of data for each model introduces diversity among the models, which is crucial for the success of ensemble methods.

- **Parallelization**: Since each model is trained independently, bagging can be easily parallelized, making it scalable and efficient for large datasets.
### Example of Bagging:

**Random Forest**: A well-known example of a bagging technique is the [Random Forests](#random-forests) algorithm. 

It uses decision trees as base models and combines their predictions to improve accuracy and robustness. 

Each tree in a random forest is trained on a different bootstrap sample of the data, and the final prediction is made by averaging the outputs (for regression) or majority voting (for classification).
# Further Understanding

### Advantages of Bagging:

- **Increased Accuracy**: By combining multiple models, bagging often achieves higher accuracy than individual models.
- **Robustness**: Bagging is less sensitive to overfitting, especially when using high-variance models like decision trees.
- **Flexibility**: It can be applied to various types of base models and is not limited to a specific algorithm.

### Challenges of Bagging:

- **Complexity**: While bagging reduces overfitting, it can increase the complexity of the model, making it harder to interpret.
- **Computational Cost**: Training multiple models can be computationally intensive, although this can be mitigated by parallel processing.


# Bandit_Example_Fixed.Py {#bandit_example_fixedpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/Bandit_Example_Fixed.py



# Bandit_Example_Nonfixed.Py {#bandit_example_nonfixedpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/Bandit_Example_Nonfixed.py

#### Automation Scripts

In [ML_Tools](#ml_tools), see: [Bash_folder](#bash_folder)
#### **Basic Commands**

1. **Show Current Directory**: 
   ```bash
   pwd
   ```
2. **Display Contents of a Text File**: 
   ```bash
   cat filename.txt
   ```
3. **Search for a Word in a File**: 
   ```bash
   grep "word" filename.txt
   ```
4. **Replace Text in a File (Output Only)**: 
   ```bash
   sed 's/old/new/g' filename.txt
   ```

### Writing and Running a Bash Script

1. **Create a Script**: 
   ```bash
   nano hello.sh
   ```
   Add:
   ```bash
   #!/bin/bash
   echo "Hello, $(whoami)! Welcome to Bash scripting!"
   ```
   Save and exit: **Ctrl + O**, **Enter**, **Ctrl + X**.

2. **Make the Script Executable**: 
   ```bash
   chmod +x hello.sh
   ```

3. **Run the Script**: 
   ```bash
   ./hello.sh
   ```

### Useful Bash Automation Tips

- **Clear Screen**: 
   ```bash
   clear
   ```
- **Keyboard Shortcut**: **Ctrl + L**.
- **Clear Screen and Command History**: 
   ```bash
   clear && history -c
   ```
- **Reset Terminal**: 
   ```bash
   reset
   ```

### Managing Command History

1. **Clear Current Session’s History**: 
   ```bash
   history -c
   ```
2. **Save History to a Custom File**: 
   ```bash
   history > my_session_history.txt
   ```
3. **Clear and Remove Saved History**: 
   ```bash
   history -c
   > ~/.bash_history
   ```
4. **Start a Fresh Bash Session**: 
   ```bash
   exec bash
   ```

#### **Example: Conditional Execution**
```bash
if [ -f filename.txt ]; then
  echo "File exists."
else
  echo "File does not exist."
fi
```



# Batch Normalisation {#batch-normalisation}

Links:
- [Batch normalization | What it is and how to implement it](https://www.youtube.com/watch?v=yXOMHOpbon8&list=PLcWfeUsAys2nPgh-gYRlexc6xvscdvHqX&index=2)

Can be used to handle [vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem) and [Overfitting](#overfitting) problems within [Neural network](#neural-network).

First note:
[Normalisation vs Standardisation](#normalisation-vs-standardisation)

How does Batch normalisation work?

Batch normalisation works by first standardising the inputs, then scales linearly - coefficients determined through training. This occurs between each layer.

Outcomes of this process:
- epochs take longer, but less epochs are required.

Benefits:
- Batch normalisation occurs at each layer, so do not need separate normalisation step for input data.
- What about bias? We do not need bias in BN.

![Pasted image 20241219071904.png](../content/images/Pasted%20image%2020241219071904.png)


### Example: 

```python
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(X_train_full, y_train_full) , (X_test, y_test) = mnist.load_data()

plt.imshow(X_train_full[12], cmap=plt.get_cmap('gray' ))
X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X test = x test/255

model = keras.models.Sequential([
keras.layers.Flatten(input_shape=[28,28]),
keras.layers.Dense(300, activation = "relu"),
keras.layers.Dense(100, activation = "relu"),
keras.layers.Dense(10, activation = "softmax")])

```

Introducing BN into this model.

Do you put BN before or after a activation function? Author of Paper suggests before.
```python
# Dont need as have BN now
# X valid, X train = X_train_full[ :5000] / 255, X_train_full[5000:]/255
# y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
# X test = X test/255

model = keras.models.Sequential ([
keras.layers.Flatten(input_shape=[28,28]),
keras.layers.BatchNormalization(), # normalisation layer.
keras.layers.Dense(300,use_bias=False),
keras.layers.BatchNormalization(),
keras.layers.Activation('relu'),
keras.layers.Dense(100,use_bias=False), I
keras.layers.BatchNormalization(),
keras.layers.Activation('relu'),
keras.layers.Dense(10, activation = "softmax")

])
```



# Batch Processing {#batch-processing}


**Batch Processing** is a technique used to handle and process large datasets efficiently. It works by breaking the data into smaller chunks and processing them together in a single batch.

[Apache Spark](#apache-spark) is the leading technology for batch processing, offering scalable and distributed data processing. It can handle unmanageable data sizes by using parallelism and [Distributed Computing](#distributed-computing)

A key concept in batch processing is **MapReduce**:
  - **Map**: Splits the data into smaller, manageable pieces for parallel processing.
  - **Reduce**: Aggregates the processed data results from the individual tasks.
  - **Order**: The order of Map and Reduce steps is flexible; the primary focus is on splitting and then aggregating data.

Batch processing is widely supported by cloud infrastructures like **Amazon EMR** and **[Databricks](#databricks)**, which provide scalable environments for running batch jobs.




[Batch Processing](#batch-processing)
   **Tags**: #data_processing, #data_workflow

# Bellman Equations {#bellman-equations}


[What are the Bellman equations that are used in RL?](#what-are-the-bellman-equations-that-are-used-in-rl)

Equations here may not be accurate.

In reinforcement learning, Bellman's equations are fundamental to understanding how agents make decisions to maximize rewards over time. They are used to describe the relationship between the value of a state and the values of its successor states. There are two main types of Bellman's equations:

1. Bellman Equation for State Value Function (V):
   - This equation expresses the value of a state as the expected return starting from that state and following a particular policy. It is defined as:
     $$
     V(s) = \sum_{a} \pi(a|s) \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma V(s')]
     $$
   - Here, \(V(s)\) is the value of state \(s\), \(\pi(a|s)\) is the policy (probability of taking action \(a\) in state \(s\)), \(P(s'|s, a)\) is the transition probability to state \(s'\) from state \(s\) taking action \(a\), \(R(s, a, s')\) is the reward received, and \(\gamma\) is the discount factor.

2. Bellman Equation for Action Value Function (Q):
   - This equation expresses the value of taking an action in a given state under a particular policy. It is defined as:
     $$
     Q(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \sum_{a'} \pi(a'|s') Q(s', a')]
     $$
   - Here, \(Q(s, a)\) is the value of taking action \(a\) in state \(s\), and the other terms are similar to those in the state value function.

Bellman's equations are used in dynamic programming methods like Value Iteration and Policy Iteration to find optimal policies and value functions. They provide a recursive decomposition of the value functions.

## Benefits of [Data Transformation](#data-transformation)  

- Efficiency: Faster query performance.  
- [Interoperability](#interoperability): Converting data into the required format for target systems.  
- Enrichment: Adding contextual data for better insights.  
- [Data Quality](#data-quality): Validating, cleansing, and deduplicating data.

# Bernoulli {#bernoulli}



# Bias And Variance {#bias-and-variance}


### Related to [Overfitting](#overfitting)

Ways to Reduce Bias and Variance:
- [Regularisation](#regularisation)
- [Boosting](#boosting)
- [Bagging](#bagging)

What is Bias in Machine Learning?  
Bias occurs when a model produces <mark>consistently unfair or inaccurate results</mark>, usually caused during training due to design choices.

What Does High Bias Mean for a Machine Learning Model?  
High bias refers to a situation where a model has a strong and often <mark>simplistic assumption</mark> about the underlying data, leading to underfitting.

It is biased to the data.

What is the Variance of a Machine Learning Model?  
Variance measures how much a <mark>model's predictions change when trained on different subsets</mark> of the training data. It indicates how much the model overfits the training data.

What is the Difference Between Bias and Variance in Machine Learning?

- Bias: The error that occurs when the model cannot learn the true relationship between input and output variables.
- Variance: The error that arises when the model is <mark>too sensitive</mark> to the training data and does not generalize well to new data.

Explain the Bias-Variance Trade-off in the Context of Model Complexity:

The bias-variance trade-off describes the relationship between model complexity and performance. 
- High bias (underfitting) occurs when a model is too simple, leading to poor performance on both training and test data. 
- High variance (overfitting) happens when a model is overly complex, performing well on training data but poorly on unseen data.

# Big Data {#big-data}


The concept of Big Data revolves around datasets that are too large or complex to be managed using traditional data processing techniques. It’s characterized by four main attributes, commonly referred to as the <mark>Four V’s:</mark>

- Volume: The sheer amount of data being generated, often in terabytes, petabytes, or even exabytes.
- Variety: The diversity in data types, including structured, semi-structured, and unstructured data (e.g., text, images, videos).
- Velocity: The speed at which data is generated and needs to be processed in real-time or near-real-time.
- Veracity: The uncertainty or quality of the data, addressing issues like noise, biases, or incomplete data.

Big Data Technologies
- [Apache Spark](#apache-spark)
- [Hadoop](#hadoop)
- [Scala](#scala)
- [Databricks](#databricks)

Handling big data involves:
- Distributed storage systems/ [Data Storage](#data-storage): Ensuring that data is split and stored across multiple machines for redundancy and speed.
- Processing frameworks: Using tools like [Apache Spark|Spark](#apache-sparkspark) or Hadoop to process data efficiently in parallel.
- Cloud platforms: Leveraging cloud infrastructure (e.g., Azure, AWS, Google Cloud) to scale resources dynamically based on workload.






[Big Data](#big-data)
   **Tags**: #big_data, #data_processing

# Big O Notation {#big-o-notation}


Big-O Notation is an analysis of the algorithm using [Big – O asymptotic notation](https://www.geeksforgeeks.org/analysis-of-algorithms-set-3asymptotic-notations/).  

Mostly related to computing rather than storage

Doing things not exponentially, such as copying the same data many times, will save lots of performance and money.

We can express algorithmic complexity using the big-O notation. For a problem of size N:
-   A constant-time function/method is “order 1” : O(1)
-   A linear-time function/method is “order N” : O(N)
-   A quadratic-time function/method is “order N squared” : O(N^2) 


# Bigquery {#bigquery}

cloud-based [Data Warehouse](#data-warehouse)

BigQuery is a fully managed, serverless data warehouse offered by [Google](#google) Cloud Platform (GCP). It is designed to handle large-scale data analytics and allows users to run fast SQL queries on massive datasets. 

1. **Serverless Architecture:** BigQuery is serverless, meaning users do not need to manage any infrastructure. Google handles the provisioning of resources, scaling, and maintenance, allowing users to focus on analyzing data.

2. **Scalability:** BigQuery can scale to handle petabytes of data, making it suitable for large datasets and complex queries.

3. **SQL Support:** BigQuery supports standard SQL, making it accessible to users familiar with SQL syntax. It also offers extensions for advanced analytics.

4. **Real-Time Analytics:** BigQuery can ingest streaming data and perform real-time analytics, enabling users to gain insights from data as it arrives.

5. **Integration:** BigQuery integrates seamlessly with other Google Cloud services, such as Google Cloud Storage, Google Sheets, and Google Data Studio, as well as third-party tools for data visualization and ETL (Extract, Transform, Load).

6. **Machine Learning:** BigQuery ML allows users to build and deploy machine learning models directly within BigQuery using SQL, without needing to move data to another platform.

7. **Security and Compliance:** BigQuery provides robust security features, including data encryption, identity and access management, and compliance with various industry standards.

8. **Cost-Effective:** BigQuery uses a pay-as-you-go pricing model, where users are charged based on the amount of data processed by queries and the amount of data stored.

# Binary Classification {#binary-classification}

Binary classification is a type of [Classification](#classification) task that involves predicting one of two possible classes or outcomes. It is used in scenarios where the goal is to categorize data into two distinct groups, such as spam vs. not spam in email filtering or disease vs. no disease in medical diagnosis.

# Binder {#binder}


https://mybinder.org/



# Boosting {#boosting}


Boosting is a type of [Model Ensemble](#model-ensemble) in machine learning that focuses on improving the accuracy of predictions by building a <mark>sequence of models</mark>.    Each subsequent model focuses on correcting the errors made by the previous ones.

It combines [Weak Learners](#weak-learners) (models that are slightly better than random guessing) to create a strong learner. 

### Key Concepts of Boosting:

1. Sequential Learning: Boosting involves training models sequentially. Each new model is trained to correct the errors made by the previous models. This means that the models are not independent of each other; instead, <mark>each model is built on the mistakes of the previous ones.</mark>

2. Focus on Misclassified Data: As models are trained in sequence, more emphasis is placed on the data points that were misclassified by earlier models. This helps the ensemble model to gradually improve its performance by focusing on the difficult-to-classify instances.

3. [Weak Learners](#weak-learners): Boosting combines multiple weak learners, which are models that perform slightly better than random guessing. By combining these weak learners, boosting creates a strong learner that has improved accuracy.

4. Examples of Boosting Algorithms: Some well-known boosting algorithms include [Ada boosting](#ada-boosting), [Gradient Boosting](#gradient-boosting), and [XGBoost](#xgboost). Each of these algorithms has its own approach to boosting, but they all share the core principle of sequentially improving model performance.

### Advantages of Boosting:

- Increased Accuracy: By focusing on the errors of previous models, boosting can significantly improve the accuracy of predictions.
- Flexibility: Boosting can be applied to various types of base models and is not limited to a specific algorithm.
- Robustness: Boosting can handle complex datasets and is effective in reducing bias and variance.

### Challenges of Boosting:

- Complexity: Boosting models can be more complex and computationally intensive than single models.
- [Interpretability](#interpretability): The final model may be harder to interpret compared to simpler models like decision trees.

# Bootstrap {#bootstrap}

sampling with replacement from an original dataset.



# Boxplot {#boxplot}


A boxplot, also known as a whisker plot, is a standardized way of displaying the distribution of data based on a five-number summary: minimum, first quartile (Q1), median, third quartile (Q3), and maximum. It can also highlight outliers in the dataset.
## Key Components

Uses:
- Identifying [standardised/Outliers](#standardisedoutliers).
- Understanding the spread and skewness of the data [Distributions](#distributions).
- Comparing distributions across different categories.
- Need to remove then in order to do [Data Cleansing](#data-cleansing).

Components:
- **Minimum:** The smallest data point excluding outliers.
- **First Quartile (Q1):** The median of the lower half of the dataset.
- **Median (Q2):** The middle value of the dataset.
- **Third Quartile (Q3):** The median of the upper half of the dataset.
- **Maximum:** The largest data point excluding outliers.
- **Outliers:** Data points that fall outside 1.5 times the interquartile range (IQR) above Q3 or below Q1.

## Implementing Boxplot in Python

You can create a boxplot in Python using libraries like Matplotlib and Seaborn. Here's how you can do it:

## Implementation

```python
import matplotlib.pyplot as plt

# Sample data
data = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# Create a boxplot
plt.boxplot(data)

# Add title and labels
plt.title('Boxplot Example')
plt.ylabel('Values')

# Show plot
plt.show()
```
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

# Create a boxplot
sns.boxplot(data=data)

# Add title and labels
plt.title('Boxplot Example')
plt.ylabel('Values')

# Show plot
plt.show()
```

# Business Observability {#business-observability}


Business [Model Observability|observability](#model-observabilityobservability) refers to the ability to gain insights into the internal state and performance of a business through the continuous monitoring and analysis of data. 

It involves collecting, analyzing, and visualizing data from various sources to understand how different parts of the business are functioning and to identify areas for improvement. 

Business observability aims to provide a comprehensive view of operations, customer interactions, and other critical aspects to enable data-driven decision-making.

It helps businesses to detect issues early, optimize operations, enhance customer experiences, and drive growth and innovation.

Key components of business observability include:

1. **[Data Collection](#data-collection)**: Gathering data from various sources such as customer interactions, sales transactions, operational processes, and external market conditions.

2. **Monitoring**: Continuously tracking key performance indicators (KPIs) and metrics to ensure that the business is operating efficiently and effectively.

3. **Analysis**: Using analytical tools and techniques to interpret the data, identify patterns, and uncover insights that can inform strategic decisions.

4. **[Data Visualisation](#data-visualisation)**: Presenting data in an accessible and understandable format, such as dashboards and reports, to facilitate quick comprehension and action by stakeholders.

5. **Feedback Loops**: Implementing mechanisms to use insights gained from observability to make adjustments and improvements in business processes and strategies.



# Business Intelligence {#business-intelligence}


Business intelligence (BI) leverages software and services to [transform data](Data%20Transformation.md) into actionable insights that inform an organization’s business decisions. 

The new term is [Data Engineer](Data%20Engineer.md). The language of a BI engineer is [SQL](SQL.md).

## Goals of BI
BI should produce a simple overview of your business, boost efficiency, and automate repetitive tasks across your organization. In more detail:


  * **[rollup](#rollup) capability** - (data) [Visualization](term/analytics.md) over the most important [KPIs][2] (aggregations) - like a cockpit in an airplane which gives you the important information at one glance.

  * **Drill-down possibilities** - from the above high-level overview drill down the very details to figure out why something is not performing as planned. **Slice-and-dice or pivot your data from different angles.

  * **[Single Source of Truth](#single-source-of-truth)** - instead of multiple spreadsheets or other tools with different numbers, the process is automated and done for all unified. Employees can talk about the business problem instead of the various numbers everyone has. Reporting, budgeting, and forecasting are automatically updated and consistent, accurate, and in timely manner.

  * **Empower users**: With the so-called self-service BI, every user can analyze their data instead of only BI or IT persons.


