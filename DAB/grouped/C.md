# C

## Table of Contents
* [CI-CD](#ci-cd)
* [CRUD](#crud)
* [Career Interest](#career-interest)
* [Casual Inference](#casual-inference)
* [CatBoost](#catboost)
* [Central Limit Theorem](#central-limit-theorem)
* [Chain of thought](#chain-of-thought)
* [Change Management](#change-management)
* [Checksum](#checksum)
* [Chi-Squared Test](#)
* [Choosing a Threshold](#choosing-a-threshold)
* [Choosing the Number of Clusters](#choosing-the-number-of-clusters)
* [Class Separability](#class-separability)
* [Classification Report](#classification-report)
* [Classification](#classification)
* [Claude](#claude)
* [Click_Implementation.py](#click_implementationpy)
* [Cloud Providers](#cloud-providers)
* [Clustering](#clustering)
* [Clustering_Dashboard.py](#clustering_dashboardpy)
* [Clustermap](#)
* [Code Diagrams](#code-diagrams)
* [Columnar Storage](#columnar-storage)
* [Command Line](#command-line)
* [Command Prompt](#command-prompt)
* [Common Security Vulnerabilities in Software Development](#common-security-vulnerabilities-in-software-development)
* [Common Table Expression](#common-table-expression)
* [Communication Techniques](#communication-techniques)
* [Communication principles](#communication-principles)
* [Comparing LLM](#comparing-llm)
* [Components of the database](#components-of-the-database)
* [Computer Science](#computer-science)
* [Concatenate](#concatenate)
* [Conceptual Model](#conceptual-model)
* [Concurrency](#concurrency)
* [Confidence Interval](#confidence-interval)
* [Confusion Matrix](#confusion-matrix)
* [Continuous Delivery - Deployment](#continuous-delivery---deployment)
* [Continuous Integration](#continuous-integration)
* [Converting categorical variables to a dummy indicators](#converting-categorical-variables-to-a-dummy-indicators)
* [Convolutional Neural Networks](#convolutional-neural-networks)
* [Correlation vs Causation](#correlation-vs-causation)
* [Correlation](#correlation)
* [Cosine Similarity](#cosine-similarity)
* [Cost Function](#cost-function)
* [Covariance](#covariance)
* [Covering Index](#covering-index)
* [Cron jobs](#cron-jobs)
* [Cross Entropy](#cross-entropy)
* [Cross Validation](#cross-validation)
* [Cross_Entropy.py](#cross_entropypy)
* [Cross_Entropy_Single.py](#cross_entropy_singlepy)
* [Crosstab](#crosstab)
* [Cryptography](#cryptography)
* [Current challenges within the energy sector](#current-challenges-within-the-energy-sector)
* [cleaning terminal path](#cleaning-terminal-path)
* [conceptual data model](#conceptual-data-model)



# Ci Cd {#ci-cd}

**CI/CD** stands for **[Continuous Integration](#continuous-integration)** and **[Continuous Delivery/Deployment](#continuous-deliverydeployment)**. It is a set of practices aimed at streamlining and accelerating the [Software Development Life Cycle](#software-development-life-cycle). The main goals of CI/CD are to improve software quality, reduce integration issues, and deliver updates to users more frequently and reliably.

Tools and Technologies
- [Gitlab](#gitlab)
- [Docker](#docker)



# Crud {#crud}

Create,Read,Update,Delete.

# Career Interest {#career-interest}


This is a portal to notes that I find relevant to my career:

# Casual Inference {#casual-inference}

missing data problem



# Catboost {#catboost}


CatBoost is a [Gradient Boosting](#gradient-boosting) library developed by Yandex, designed to handle [categorical](#categorical) features efficiently and provide robust performance with minimal [Hyperparameter|Hyperparameter tuning](#hyperparameterhyperparameter-tuning)

It is particularly useful in scenarios where datasets contain a significant number of categorical variables.

#### Key Advantages

1. Handling Categorical Features: 
   - CatBoost natively processes categorical features without the need for extensive preprocessing like one-hot encoding, which simplifies the workflow and reduces the risk of introducing errors during data preparation.

2. Robustness to Overfitting:
   - It employs techniques such as ordered boosting and per-feature scaling to reduce overfitting, making it a reliable choice for complex datasets.

3. Performance:
   - CatBoost offers competitive performance with minimal hyperparameter tuning, making it suitable for quick experimentation and deployment.

### Implementing CatBoost in Python

To implement CatBoost in Python, you need to install the CatBoost library and then follow these steps:

#### Step 1: Install CatBoost

You can install CatBoost using pip:
```bash
pip install catboost
```
#### Step 2: Import Necessary Libraries
```python
import catboost as cb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```

#### Step 3: Prepare Your Data

Assume you have a dataset with features `X` and target `y`. Split the data into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
#### Step 4: Identify Categorical Features

Identify the indices of categorical features in your dataset:

```python
categorical_features_indices = [0, 1, 2]  # Example indices of categorical features
```
#### Step 5: Create a CatBoost Pool

Create a Pool object for the training data, specifying the categorical features:

```python
train_pool = Pool(data=X_train, label=y_train,cat_features=categorical_features_indices)
test_pool = Pool(data=X_test, label=y_test, cat_features=categorical_features_indices)
```

#### Step 6: Initialize and Train the Model

Initialize the CatBoostClassifier and fit it to the training data:

```python
model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.1, loss_function='Logloss', verbose=100)
model.fit(train_pool)
```

#### Step 7: Make Predictions and Evaluate

Make predictions on the test set and evaluate the model's performance:

```python
y_pred = model.predict(test_pool)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```



# Central Limit Theorem {#central-limit-theorem}


The Central Limit Theorem states that the distribution of the sum (or average) of a large number of independent, identically distributed random variables approaches a normal distribution, regardless of the original distribution.

[Why is the Central Limit Theorem important when working with small sample sizes](#why-is-the-central-limit-theorem-important-when-working-with-small-sample-sizes)

### Key Points

- **Mean of Sampling Distribution:** The mean of the sampling distribution is equal to the mean of the original population.
- **Variance of Sampling Distribution:** The variance of the sampling distribution is the population variance divided by the sample size (\(n\)), making it \(n\) times smaller.
- **Applicability:** The CLT applies when calculating the sum or average of many variables, such as the sum of rolled numbers when rolling dice.

### Importance

- The CLT allows us to assume normality for various variables, which is crucial for:
  - Confidence intervals
  - Hypothesis testing
  - Regression analysis

[Central Limit Theorem](#central-limit-theorem)
**Explain the concept of the Central Limit Theorem.**;; 


<!--SR:!2024-01-26,3,250-->

# Chain Of Thought {#chain-of-thought}

**Chain of Thought (CoT) reasoning**

Asking sequenced questions that guide someone (or yourself) through a reasoning path is a core technique in problem-solving and teaching. Examples:

- "What is the known information?"
- "What is being asked?"
- "What patterns can we observe?"
- "What similar problems have we solved before?"

Used in in AI systems is a cognitive-inspired framework that improves the performance of large [language models](#language-models) (LLMs) by explicitly guiding the AI through intermediate reasoning steps.

Advantages of Chain of Thought:
- **Improved [Interpretability](#interpretability)**: Since the model outputs intermediate steps, it's easier for humans to understand how the final answer was reached.
- **Better Performance on Complex Tasks**: CoT allows the model to handle multi-step reasoning more effectively.
- **Easier Debugging**: If there's an error in reasoning, it can be spotted at a specific step in the chain, which aids in model fine-tuning and debugging.

Related to:
- [Model Ensemble](#model-ensemble)

# Change Management {#change-management}


Change management is a structured approach to transitioning individuals, teams, and organizations from a current state to a desired future state. It involves
- preparing, 
- supporting,
- and helping people to adopt change in order to drive organizational success and outcomes. 

Effective change management helps
- minimize resistance, 
- improves engagement, 
- and increases the likelihood of successful outcomes.

The process typically includes:

1. **Planning**: Identifying the need for change, defining the change, and developing a strategy to implement it.

2. **Communication**: Clearly explaining the reasons for the change, the benefits, and the impact on the organization and its people.

3. **Training and Support**: Providing the necessary training and resources to help employees adapt to the change.

4. **Knowledge Sharing**: Provide training and resources to help teams understand best practices for data quality.

5. **Implementation**: Executing the change plan while managing any resistance or challenges that arise.

6. **Monitoring and Evaluation**: Assessing the effectiveness of the change and making adjustments as needed to ensure successful adoption.

7. **Sustainability**: Ensuring that the change is maintained over time and becomes integrated into the organization's culture and operations.

Why change fails:
- Change is hard, identify the pain points.  
- Resistance is why change fails, due to loss aversion, uncertainty, unexpected change when not bought in.  

How we can accomplish change:
- Story telling will help.  
- Introduce a hook i.e. can we reduce the processing time for tasks by X amount.
- Put ourselves in a better position for tomorrow.

# Checksum {#checksum}

A checksum is a value calculated from a data set that is used to verify the integrity of that data. It acts as a fingerprint for the data, allowing systems to detect errors or alterations that may occur during storage, processing, or transmission.

When data is sent or stored, a checksum is generated based on the contents of the data. This checksum is then sent or stored alongside the data. Upon retrieval or receipt, the checksum is recalculated from the data and compared to the original checksum. If the two checksums match, it indicates that the data has remained unchanged and is likely intact. If they do not match, it suggests that the data may have been corrupted or tampered with.

Checksums are commonly used in various applications, such as:

- **File transfers**: To ensure that files are not corrupted during transfer.
- **[Data storage](#data-storage)**: To verify that data has not changed over time.
- **Networking**: To check the integrity of packets sent over a network.
### Example of a Checksum Calculation

1. **Original Data**: Let's say we have the string "Hello, World!".
   
2. **Checksum Calculation**: A common method for calculating a checksum is to sum the ASCII values of each character in the string. 

   - ASCII values:
     - H = 72
     - e = 101
     - l = 108
     - l = 108
     - o = 111
     - , = 44
     - (space) = 32
     - W = 87
     - o = 111
     - r = 114
     - l = 108
     - d = 100
     - ! = 33

   - Sum of ASCII values:
     $$ 72 + 101 + 108 + 108 + 111 + 44 + 32 + 87 + 111 + 114 + 108 + 100 + 33 =  1,  2,  0 $$

   - Let's say we take the modulo 256 of the sum to get the checksum:
     $$ 1,  2,  0 \mod 256 =  1,  2,  0 $$

3. **Sending Data**: The original data "Hello, World!" is sent along with the checksum value of 1, 2, 0.

4. **Receiving Data**: Upon receiving the data, the receiver calculates the checksum again using the same method.

5. **Verification**: If the calculated checksum matches the received checksum (1, 2, 0), the data is considered intact. If it does not match, it indicates that the data may have been corrupted during transmission.

This is a basic example, and in practice, checksums can be computed using more complex [algorithms](#algorithms) (like CRC32, MD5, or SHA-1) to provide better error detection and  [Security](#security). 



## Chi-Squared Test

The Chi-squared test is used to determine if there is a significant association between categorical variables. It assesses whether the observed frequencies in a contingency table differ from the expected frequencies, assuming the data is independent.

# Choosing A Threshold {#choosing-a-threshold}

The optimal threshold depends on the specific problem and the desired trade-off between different types of errors:

1. Manual Selection: Based on domain expertise or prior knowledge, choose a threshold that seems reasonable.
2. Receiver Operating Characteristic ([ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic)) Curve Analysis: Plot the true positive rate (TPR) against the false positive rate (FPR) for different threshold values. The optimal threshold often lies near the "elbow" of the ROC curve, where a small increase in FPR results in a significant increase in TPR.
3. [Precision-Recall Curve](#precision-recall-curve) Analysis: Plot the precision against the recall for different threshold values. The optimal threshold often lies near the "elbow" of the precision-recall curve, where a small decrease in precision results in a significant increase in recall.
4. [Cost-Sensitive Analysis](#cost-sensitive-analysis): Assign different costs to different types of errors (e.g., false positives vs. false negatives) and choose the threshold that minimizes the total cost.

# Choosing The Number Of Clusters {#choosing-the-number-of-clusters}

The optimal number of clusters ([clustering](#clustering)) depends on the data and the desired level of [granularity](#granularity). Here are some common approaches:

1. Elbow Method: [WCSS and elbow method](#wcss-and-elbow-method): Plot the within-cluster sum of squares (WCSS) as a function of the number of clusters. The optimal number of clusters is often the point where the WCSS starts to decrease slowly.
2. [Silhouette Analysis](#silhouette-analysis): Calculate the silhouette coefficient for each data point, which measures how similar a data point is to its own cluster compared to other clusters. The optimal number of clusters 1 is often the one that maximizes the average silhouette coefficient.T

# Class Separability {#class-separability}


If you have a perfectly balanced dataset (unlike [Imbalanced Datasets](#imbalanced-datasets)) but still experience poor [classification](#classification) [accuracy](#accuracy), class separability might be an issue due to the following reasons:

1. **Overlapping Classes**: The features of different classes may overlap significantly, making it difficult for the model to distinguish between them. If the decision boundaries are not well-defined, the model may struggle to classify instances correctly.

2. **Complex Decision Boundaries**: The underlying relationship between the features and the classes may be complex, requiring a more sophisticated model to capture the nuances. If the model is too simple, it may not be able to learn the necessary patterns.

3. **Noise in the Data**: If there is a significant amount of noise or irrelevant features in the dataset, it can obscure the true signal, leading to poor classification performance despite balanced class representation.

4. **Insufficient Feature Representation**: The features used for classification may not adequately represent the underlying characteristics that differentiate the classes. This can lead to a lack of separability, even in a balanced dataset.

5. **Model Overfitting or Underfitting**: If the model is overfitting, it may perform well on training data but poorly on unseen data. Conversely, if it is underfitting, it may not capture the complexity of the data, leading to poor accuracy.

Addressing these issues may require exploring different [feature engineering](#feature-engineering) techniques, selecting more appropriate models, or adjusting hyperparameters to improve class separability.

# Classification Report {#classification-report}

The `classification_report` function in `sklearn.metrics` is used to evaluate the performance of a classification model. It provides a summary of key metrics for each class, including precision, recall, F1-score, and support.

## Function Signature

```python
sklearn.metrics.classification_report(
    y_true, 
    y_pred, 
    , 
    labels=None, 
    target_names=None, 
    sample_weight=None, 
    digits=2, 
    output_dict=False, 
    zero_division='warn'
)
```
Parameters:
- `y_true`: Array of true labels.
- `y_pred`: Array of predicted labels.
- `labels`: (Optional) List of label indices to include in the report.
- `target_names`: (Optional) List of string names for the labels.
- `sample_weight`: (Optional) Array of weights for each sample.
- `digits`: Number of decimal places for formatting output.
- `output_dict`: If `True`, return output as a dictionary.
- `zero_division`: Sets the behavior when there is a zero division (e.g., 'warn', 0, 1).

### Metrics Explained

- [Precision](#precision): The ratio of correctly predicted positive observations to the total predicted positives. It indicates the quality of the positive class predictions.
  
- [Recall](#recall) (Sensitivity): The ratio of correctly predicted positive observations to all actual positives. It measures the ability of a model to find all relevant cases.

- [F1 Score](#f1-score): The weighted average of precision and recall. It is a better measure than accuracy for imbalanced classes.

- Support: The number of actual occurrences of the class in the specified dataset.

## Resources

In [ML_Tools](#ml_tools) see: [Evaluation_Metrics.py](#evaluation_metricspy)

[official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html).


![Pasted image 20240404163858.png|500](../content/images/Pasted%20image%2020240404163858.png|500)




# Classification {#classification}


Classification is a type of [Supervised Learning](#supervised-learning) in machine learning, where the algorithm learns from labeled data to predict which category or class a new, unlabeled data point belongs to. The goal is to assign the correct label to input data based on patterns learned from the training set.

## Examples of Classifiers

Classifier: A model used for classification tasks, predicting discrete labels or categories. For example, determining whether an email is spam or not, or identifying the species of a flower based on its features. This contrasts with a Regressor ([Regression](#regression)), which predicts continuous values.

[Naive Bayes](#naive-bayes)

[Decision Tree](#decision-tree)

[Support Vector Machines](#support-vector-machines)

[K-nearest neighbours](#k-nearest-neighbours)

[Neural network](#neural-network)

[Model Ensemble](#model-ensemble)

## Choosing a Classifier Algorithm

1. Data Characteristics: Some algorithms work better on structured data, while others perform better on unstructured data.
2. Problem Complexity: Simple classifiers for straightforward problems, complex models for intricate tasks.
3. Model Performance: Consider accuracy and speed requirements.
4. Model [Interpretability](#interpretability): Some models, like decision trees, are easier to interpret, while others, like neural networks, can be more challenging.
5. Model Scalability: Large datasets need scalable models like SVM or Naive Bayes.
6. Model Flexibility: Algorithms like KNN are flexible when the data distribution is unknown.
## Use Cases of Classification

1. Object Recognition: Classifying objects in images (e.g., identifying a cat or a dog).
2. Spam Filtering: Classifying emails as either spam or legitimate.
3. Medical Diagnosis: Using patient symptoms and test results to classify diseases.


# Claude {#claude}


Claude is better for code and uses Artifact for tracking code changes.

Claude is crazy see: https://youtu.be/RudrWy9uPZE?t=473

# Click_Implementation.Py {#click_implementationpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Click_Implementation.py

This script implements a command-line interface (CLI) tool using Python's `click` library. The CLI allows users to interact with a JSON file, enabling them to view keys, retrieve values, and update key-value pairs.

## Functionality Overview

1. CLI Initialization (`cli` function)
    - Serves as the main command group.
    - Accepts a JSON file (`document`) as an argument.
    - Reads the JSON file and stores its content in `ctx.obj`, making it accessible to all subcommands.

2. Displaying Keys (`show_keys` command)
    - Lists all top-level keys in the JSON document.

3. Retrieving a Value (`get_value` command)
    - Accepts a key as an argument.
    - Prints the corresponding value if the key exists; otherwise, prints `"Key not found"`.

4. Updating a Value (`update_value` command)
    - Requires `-k/--key` (key to update) and `-v/--value` (new value).
    - Updates the key’s value in memory.
    - Saves the updated JSON data back to the file.

## Example Usage

### 1. Viewing Keys

```sh
python script.py data.json show_keys
```

Example Output (if `data.json` contains `{"name": "Alice", "age": 30}`):

```
Keys: ['name', 'age']
```

### 2. Retrieving a Value

```sh
python script.py data.json get_value name
```

Output:

```
Alice
```

### 3. Updating a Value

```sh
python script.py data.json update_value -k name -v Bob
```

Modifies `data.json` to:

```json
{
    "name": "Bob",
    "age": 30
}
```



# Cloud Providers {#cloud-providers}


Among the biggest cloud providers are [AWS](https://aws.amazon.com/), [Microsoft Azure](https://azure.microsoft.com/), [Google Cloud](https://cloud.google.com/). 

Whereas [Databricks](#databricks) ( [Databrick](https://www.databricks.com/)) and [Snowflake](https://www.snowflake.com/) provide dedicated [Data Warehouse](#data-warehouse)and [Data Lakehouse|Lakehouse](#data-lakehouselakehouse) solutions

## Features

[Scaling Server](#scaling-server)
[Load Balancing](#load-balancing)
[Memory Caching](#memory-caching)


# Clustering {#clustering}


Clustering involves grouping a set of data points into subsets or clusters based on inherent patterns or similarities. It is an [Unsupervised Learning](#unsupervised-learning)technique used for tasks like customer segmentation and [standardised/Outliers|anomalies](#standardisedoutliersanomalies) detection. The primary goal of clustering is to organize data by grouping similar items.

## Applications

- Customer Segmentation: Group customers with similar purchasing behavior or demographics for targeted marketing.
- Image Segmentation: Group pixels in an image based on color or texture to identify objects or regions.
- [Anomaly Detection](#anomaly-detection): Identify clusters of normal behavior to detect anomalies that deviate significantly from these clusters.
## Methods

- [K-means](#k-means)
- [DBScan](#dbscan)
- [Hierarchical Clustering](#hierarchical-clustering)
- [Gaussian Mixture Models](#gaussian-mixture-models)

## [Interpretability](#interpretability)

 [Feature Scaling](#feature-scaling): Essential for bringing features to the same scale, as clusters may appear distorted without it.
  ```python
  from sklearn.preprocessing import scale
  from sklearn.preprocessing import MinMaxScaler
  ```

Use clustering to find [Correlation](#correlation) between features. Utilize a [Dendrograms](#dendrograms) to visualize the relationship between features.

# Clustering_Dashboard.Py {#clustering_dashboardpy}


https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Clustering_Dashboard.py

## Clustermap

Related to:
[Preprocessing|Preprocess](#preprocessingpreprocess)

- Purpose: Identify which features are most similar using [Dendrograms](#dendrograms).
- Visualization: Regions of color show clustering, similar to a heatmap.
- Functionality: Performs clustering on both rows and columns.

Requirements: Input should be numerical; data needs to be scaled.
  ```python
  import seaborn as sns
  sns.clustermap(x_scaled, cmap='mako', standard_scale=0)  # 0 for rows, 1 for columns
  ```
## Resources
- [Video Explanation](https://youtu.be/crQkHHhY7aY?t=149)
- [Seaborn Clustermap Documentation](https://seaborn.pydata.org/generated/seaborn.clustermap.html)

# Code Diagrams {#code-diagrams}

[Documentation & Meetings](#documentation--meetings)

There are class diagrams showing the hierarchy of classes [Classes](#classes) (Object orientated). 

Done in [Mermaid](#mermaid).

Overall [Architecture Diagram](#architecture-diagram): showing how software components interact.

[Sequence diagram](#sequence-diagram) of componets interact. 

Sequence diagraph: how the componets interact 
Architecture diagram : main componts fitting together



# Columnar Storage {#columnar-storage}

A database storage technique that stores <mark>data by columns</mark> rather than rows, 

Useful for read-heavy operations and <mark>large-scale data analytics</mark>, as it enables the retrieval of specific columns without the need to access the entire row. 

Columnar Storage Example (Analytical Workloads)**:

| `order_id`  | `customer_id` | `order_date` | `order_amount` |
|-------------|---------------|--------------|----------------|
| 1           | 101           | 2024-10-01   | $100           |
| 2           | 102           | 2024-10-02   | $150           |
| 3           | 103           | 2024-10-03   | $200           |
In **columnar storage**, the data would be stored by columns, like:
- `customer_id`: [101, 102, 103]

If you're querying for the total sales (`order_amount`) in a specific period, only the `order_amount` and `order_date` columns are accessed. 


Use case: **Data Analytics/OLAP (Online Analytical Processing)**
- Running a query to get the **total sales for October** only needs to scan the `order_amount` and `order_date` columns, rather than scanning entire rows, faster [Querying](#querying)

# Command Line {#command-line}


The command line is a text-based interface used to interact with a computer's operating system or software. It allows users to execute commands, run scripts, and perform various tasks.

[PowerShell](#powershell)

[Powershell vs Bash](#powershell-vs-bash)

[Bash](#bash)

[Command Prompt](#command-prompt)

# Command Prompt {#command-prompt}


Command Prompt (cmd) is a command-line interpreter on Windows systems that allows users to execute commands to perform various basic tasks. Below are some common tasks that can be performed in cmd, along with examples:

Related to:
- [Bash](#bash)

## 1. Navigating the File System

- **Changing Directories:**
  ```cmd
  cd C:\path\to\directory
  ```
  Changes the current directory to `C:\path\to\directory`.

- **Listing Files and Directories:**
  ```cmd
  dir
  ```
  Lists the files and directories in the current directory.

## 2. Managing Files and Directories

- **Creating a Directory:**
  ```cmd
  mkdir newfolder
  ```
  Creates a new directory named `newfolder`.

- **Deleting a Directory:**
  ```cmd
  rmdir /s /q newfolder
  ```
  Deletes the directory `newfolder` and its contents. The `/s` flag removes all directories and files in the specified directory, and the `/q` flag runs the command quietly without asking for confirmation.

- **Copying Files:**
  ```cmd
  copy C:\source\file.txt D:\destination\
  ```
  Copies `file.txt` from the `C:\source` directory to the `D:\destination` directory.

- **Renaming Files:**
  ```cmd
  ren oldfile.txt newfile.txt
  ```
  Renames `oldfile.txt` to `newfile.txt`.

- **Deleting Files:**
  ```cmd
  del file.txt
  ```
  Deletes `file.txt`.

## 3. Viewing and Managing System Information

- **Viewing IP Configuration:**
  ```cmd
  ipconfig
  ```
  Displays the current network configuration.

- **Viewing System Information:**
  ```cmd
  systeminfo
  ```
  Provides detailed system information including OS version, hardware details, and network configurations.

## 4. Managing Processes

- **Viewing Running Processes:**
  ```cmd
  tasklist
  ```
  Lists all currently running processes.

- **Killing a Process:**
  ```cmd
  taskkill /F /PID 1234
  ```
  Terminates the process with the Process ID (PID) `1234`. The `/F` flag forces the process to terminate.

## 5. Networking Commands

- **Pinging a Server:**
  ```cmd
  ping www.example.com
  ```
  Sends ICMP Echo Request packets to the specified host and displays the response.

- **Tracing Route to a Server:**
  ```cmd
  tracert www.example.com
  ```
  Traces the route packets take to the specified host.

## 6. Batch File Scripting

- **Creating and Running a Simple Batch File:**
  - Create a file named `example.bat` with the following content:
    ```cmd
    @echo off
    echo Hello, World!
    pause
    ```
  - Run the batch file:
    ```cmd
    example.bat
    ```
  This batch file prints "Hello, World!" to the console and waits for the user to press a key before closing.

## 7. Environment Variables

- **Viewing Environment Variables:**
  ```cmd
  set
  ```
  Displays all current environment variables and their values.

- **Setting an Environment Variable:**
  ```cmd
  set MYVAR=Hello
  ```
  Sets an environment variable `MYVAR` with the value `Hello`.

## 8. Disk Operations

- **Checking Disk Usage:**
  ```cmd
  chkdsk C:
  ```
  Checks the file system and file system metadata of the C: drive for logical and physical errors.

- **Formatting a Disk:**
  ```cmd
  format D: /FS:NTFS
  ```
  Formats the D: drive with the NTFS file system.

## 9. Echoing Messages

- **Displaying a Message:**
  ```cmd
  echo Hello, World!
  ```
  Prints `Hello, World!` to the console.

## 10. Redirecting Output

- **Redirecting Command Output to a File:**
  ```cmd
  dir > output.txt
  ```
  Redirects the output of the `dir` command to `output.txt`.

These examples illustrate some of the basic functionalities of Command Prompt. While cmd is less powerful compared to [PowerShell](#powershell), it remains useful for simple file system navigation, file management, and running legacy scripts.



# Common Security Vulnerabilities In Software Development {#common-security-vulnerabilities-in-software-development}


[Security](#security) vulnerabilities can be encountered and mitigated in [Software Development Portal](#software-development-portal).

In this not describe potential security risks in their applications.

Useful Tools
- [tool.bandit](#toolbandit)
## Examples

### Command Injection

General Description: Command injection is a security vulnerability that occurs when an attacker is able to execute arbitrary commands on the host operating system via a vulnerable application. This typically happens when user input is improperly handled and passed to a system shell.

Example: 
The `dangerous_subprocess` function uses `subprocess.call` with `shell=True`, which can lead to command injection if user input is not properly sanitized.
  ```python
  import subprocess
  def dangerous_subprocess(user_input):
      subprocess.call(user_input, shell=True)
  ```
Mitigation:
  - Avoid using `subprocess.call` with `shell=True`. Use `subprocess.run` or `subprocess.call` with a list of arguments.
  - Validate and sanitize user inputs ([Input is Not Properly Sanitized](#input-is-not-properly-sanitized)). 

### Hardcoded Password

General Description: Hardcoded passwords refer to credentials that are embedded directly in the source code. This practice is insecure as it exposes sensitive information and makes it difficult to change passwords without modifying the code.

Example:
The `hardcoded_password` function contains a hardcoded password, which is a common security issue.
  ```python
  def hardcoded_password():
      password = "123456"
      return password
  ```
  
Mitigation:
  - Use environment variables or configuration files to store sensitive information.
  - Consider using a secrets management tool.

### Use of `eval`

General Description: The `eval` function in Python evaluates a string as a Python expression. If not properly controlled, it can execute arbitrary code, leading to security vulnerabilities.

Example:
  - The `unsafe_eval` function uses `eval`, which can execute arbitrary code if the input is not controlled.
  ```python
  def unsafe_eval(user_input):
      return eval(user_input)
  ```
Mitigation:
  - Avoid using `eval`. Use safer alternatives like `ast.literal_eval`.
  - Ensure input is strictly controlled and sanitized.

### Insecure Deserialization

General Description: Insecure deserialization occurs when untrusted data is used to reconstruct objects. This can lead to arbitrary code execution, data tampering, or other malicious activities.

Example:
  - The `insecure_deserialization` function uses `pickle.loads`, which can be exploited if untrusted data is deserialized.
  ```python
  import pickle
  def insecure_deserialization(data):
      return pickle.loads(data)
  ```
Mitigation:
  - Avoid using `pickle` for untrusted data. Use safer formats like JSON ([Why JSON is Better than Pickle for Untrusted Data](#why-json-is-better-than-pickle-for-untrusted-data)).
  - Ensure data is from a trusted source.

### [SQL Injection](#sql-injection)

### Cross-Site Scripting (XSS)

**General Description**: Cross-Site Scripting (XSS) is a security vulnerability that allows an attacker to inject malicious scripts into content from otherwise trusted websites. It occurs when an application includes untrusted data in a web page without proper validation or escaping.

**Example**:
- The `display_user_input` function directly inserts user input into HTML, which can lead to XSS if the input is not properly sanitized.
```html
<div>
	<%= user_input %>
</div>
```
**Mitigation**:
- Escape user input before rendering it in HTML.
- Use security libraries or frameworks that automatically handle escaping

# Common Table Expression {#common-table-expression}


A Common Table Expression (CTE) is a temporary named result set that you can reference within a SELECT, INSERT, UPDATE, or DELETE statement. 

The CTE can also be used in a [Views](#views). Serve as temporary views for a single [Querying|Queries](#queryingqueries).

```sql
WITH cte_query AS
(SELECT … subquery ...)
SELECT main query ... FROM/JOIN with cte_query ...
```

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/ExplorationsSQLite/Utilities/Common_Table_Expression.ipynb
### Non-Recursive CTE

The non-recursive are simple where CTE is used to <mark>avoid SQL duplication</mark> by referencing a name instead of the actual SQL statement. See [Views](#views) simplification usage.

```sql
WITH avg_per_store AS
  (SELECT store, AVG(amount) AS average_order
   FROM orders
   GROUP BY store)
SELECT o.id, o.store, o.amount, avg.average_order AS avg_for_store
FROM orders o
JOIN avg_per_store avg
ON o.store = avg.store;
```

### Recursive CTE

CTEs can be used in [Recursive Algorithm](#recursive-algorithm). The recursive query calls itself until the query satisfied the condition. In a recursive CTE, we should provide a where condition to terminate the recursion.

A recursive CTE is useful in querying hierarchical data such as organization charts where one employee reports to a manager or multi-level bill of materials when a product consists of many components, and each component itself also consists of many other components.

```sql
WITH levels AS (
  SELECT
    id,
    first_name,
    last_name,
    superior_id,
    1 AS level
  FROM employees
  WHERE superior_id IS NULL
  UNION ALL
  SELECT
    employees.id,
    employees.first_name,
    employees.last_name,
    employees.superior_id,
    levels.level + 1
  FROM employees, levels
  WHERE employees.superior_id = levels.id
)
 
SELECT *
FROM levels;
```


# Communication Techniques {#communication-techniques}


## Overview

Using these structured communication bridges can  enhance clarity and engagement, especially in spontaneous or high-stakes discussions.

Tips for Using Communication Bridges
1. Start Small: Begin by integrating 2-3 bridges that feel natural to you.
2. Observe Reactions: Notice how listeners respond when you clarify changes, summarize key points, or highlight actions.
3. Practice Consistency: Make these bridges a regular part of your speaking style.

[Speak More Clearly: 8 Precise Steps to Improve Communication](https://www.youtube.com/watch?v=Tc5dCLE_GP0)

### 1. Context Bridge

- Purpose: Aligns everyone by setting the context before diving into details.
- How to Use: Start with phrases like:
  - "At a high level..."
  - "This is our goal..."
  - <mark>"The main problem is..."</mark>
- Effect: Helps to focus thoughts and prevents initial rambling.

### 2. Change Bridge

- Purpose: Emphasizes shifts, trends, or significant moments in the discussion.
- How to Use: Use phrases that highlight changes, such as:
  - "Here's the before, and here's the after..."
  - "We’re shifting from X to Y..."`
  - "We are at a tipping point..."
- Effect: Grabs attention by making the change clear.

### 3. Insight Bridge

- Purpose: Shares deeper insights or unique perspectives, creating "aha" moments.
- How to Use: Key phrases include:
  - "Counterintuitively, ..."
  - "Here's what most people miss..."
  - "The deeper insight is..."
  - "The key point here is..."
- Effect: Signals that you’ve thought deeply, which moves the conversation forward.

### 4. Analysis Bridge

- Purpose: Anchors discussion in evidence, keeping it grounded.
- How to Use: Reference specific data points or comparisons with:
  - "The evidence shows..."
  - "The data indicates..."
  - "When we compared X and Y..."
- Effect: Focuses on facts, minimizing loss of direction.

### 5. Logical Transition Bridge

- Purpose: Provides a clear flow in the conversation, avoiding confusion.
- How to Use: Classic transitions include:
  - "First, second, third..."
  - "This leads to..."
  - "On the other hand..."
- Effect: Helps listeners follow along without losing the thread.

### 6. Summary Bridge

- Purpose: Ensures that key messages stay clear, especially in long discussions.
- How to Use: Frequently summarize main points with phrases like:
  - "The bottom line is..."
  - <mark>"If you remember one thing, it’s this..."</mark>
  - "To bring it back to the goal..."
- Effect: Reinforces the main message throughout the discussion.

### 7. Refinement Bridge

- Purpose: Allows for clarification or expansion of ideas as needed.
- How to Use: Rephrase or elaborate with:
  - "Let me break this down further..."
  - "Another way of looking at it is..."
  - "A useful analogy might be..."
- Effect: Clarifies complex points, helping everyone understand the core message.

### 8. Action Bridge

- Purpose: Concludes with actionable steps, defining the next moves.
- How to Use: Conclude with statements like:
  - <mark>"Our immediate priority is..."</mark>
  - "Here’s what we’ll do next..."
  - "The deliverables are..."
- Effect: Ends discussions with clear direction and accountability.


# Communication Principles {#communication-principles}


![Pasted image 20240916075433.png](../content/images/Pasted%20image%2020240916075433.png)

![Pasted image 20240916075439.png](../content/images/Pasted%20image%2020240916075439.png)



# Comparing Llm {#comparing-llm}


Use lmarena.ai as a bench marking tool. 

[LLM](#llm)

web dev arena

text to image leader board



# Components Of The Database {#components-of-the-database}

[Fact Table](#fact-table) in main table that [Dimension Table](#dimension-table) connect to them.

![Obsidian_CSP0FnAVD1.png](../content/images/Obsidian_CSP0FnAVD1.png)

# Computer Science {#computer-science}


[Algorithms](#algorithms)



# Concatenate {#concatenate}



# Conceptual Model {#conceptual-model}

Conceptual Model
   - Entities: Customer, Order, Book
   - Relationships: Customers place Orders, Orders include Books



Conceptual Model
   - Focuses on high-level business requirements.
   - Defines important data entities and their relationships.
   - Tools: [ER Diagrams](#er-diagrams), ER Studio, DbSchema.

# Concurrency {#concurrency}

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Transactions/Concurrency.ipynb

# Confidence Interval {#confidence-interval}


A confidence interval is a range of values, derived from sample data, that is likely to contain the true population parameter. It is associated with a confidence level, such as 95%, indicating the probability that the interval captures the true parameter.

Key Points
- **Confidence Level:** The likelihood that the interval includes the true parameter (e.g., 95%).
- **Purpose:** Quantifies the uncertainty of an estimate, providing a range rather than a single value.
### Example
- A 95% confidence interval for a mean of (50, 60) suggests that, in repeated sampling, 95% of such intervals would contain the true mean.


# Confusion Matrix {#confusion-matrix}


A Confusion Matrix is a table used to evaluate the performance of a [Classification](#classification) model. It provides a detailed breakdown of the model's predictions across different classes, showing the number of true positives, true negatives, false positives, and false negatives.
## Purpose

- The confusion matrix helps identify where the classifier is making errors, indicating where it is "confused" in its predictions.
## Structure


![Pasted image 20240120215414.png](../content/images/Pasted%20image%2020240120215414.png)

## Structure

- True Positives (TP): Correctly predicted positive instances.
- False Positives (FP): Incorrectly predicted positive instances (Type 1 error).
- True Negatives (TN): Correctly predicted negative instances.
- False Negatives (FN): Incorrectly predicted negative instances (Type 2 error).

## Metrics

- [Accuracy](#accuracy): The overall percentage of correct predictions. In this case, the accuracy is 78.3%.
- [Precision](#precision): The ratio of true positives to all positive predictions (including both TPs and FPs). In this case, the precision for class 0 is 85.7% and the precision for class 1 is 66.4%.
- [Recall](#recall): The ratio of true positives to all actual positive cases (including both TPs and FNs). In this case, the recall for class 0 is 80.6% and the recall for class 1 is 74.1%.
- [F1 Score](#f1-score): A harmonic average of precision and recall. In this case, the F1-score for class 0 is 83.0% and the F1-score for class 1 is 70.0%.
- [Specificity](#specificity)
- [Recall](#recall)

## Further Examples
![Pasted image 20240116205937.png|500](../content/images/Pasted%20image%2020240116205937.png|500)

![Pasted image 20240116210541.png|500](../content/images/Pasted%20image%2020240116210541.png|500)

## Example Code

```python
from sklearn.metrics import confusion_matrix

# Assuming y_train and y_train_pred are your true and predicted labels
conf_matrix = confusion_matrix(y_train, y_train_pred)
print(conf_matrix)
```

Example Output:

```
array([[377, 63],
       [ 91, 180]], dtype=int64)
```

# Continuous Delivery   Deployment {#continuous-delivery---deployment}

Continuous Delivery
   - Ensures that code changes are automatically prepared for a release to production.
   - Builds, tests, and releases are automated, but the deployment is manual.

Continuous Deployment:
   - Extends continuous delivery by automating the deployment process.
   - Every change that passes the automated tests is deployed to production automatically.

[Model Deployment](#model-deployment)

A continuous integration and continuous deployment (CI/CD) pipeline is **a series of steps that must be performed in order to deliver a new version of software**

# Continuous Integration {#continuous-integration}

   - Developers frequently integrate code into a shared repository.
   - Automated builds and tests are run to detect issues early.
   - Encourages smaller, more manageable code changes.

# Converting Categorical Variables To A Dummy Indicators {#converting-categorical-variables-to-a-dummy-indicators}



# Convolutional Neural Networks {#convolutional-neural-networks}


Convolutional networks, or CNNs, are specialized [Deep Learning](#deep-learning) architectures designed for processing data with grid-like structures, such as images. 

They use convolutional layers with learnable filters to extract spatial features from the input data. The convolutional operation involves sliding these filters across the input, performing element-wise multiplications and summations to create feature maps. 

CNNs are particularly effective for image classification, object detection, and image segmentation tasks.

Primarily used in image recognition and processing tasks. CNNs use convolutional layers to automatically detect spatial patterns in images, like edges and textures.

Pooling:

The idea of pooling in convolutional neural networks is to do two things:
- Reduce the number of parameters in your network (pooling is also called “down-sampling” for this reason)
- To make feature detection ([Feature Extraction](#feature-extraction)) more robust by making it more impervious to scale and orientation changes
- shrink multiple data to single points.

![Pasted image 20241006124829.png|500](../content/images/Pasted%20image%2020241006124829.png|500)

![Pasted image 20241006124735.png|500](../content/images/Pasted%20image%2020241006124735.png|500)


# Correlation Vs Causation {#correlation-vs-causation}

What is the meaning of [Correlation](#correlation) does not imply causation?

Correlation measures the statistical association between two variables, while causation implies a cause-and-effect relationship. 


- **Correlation**: Indicates an association between variables but does not imply that changes in one variable cause changes in the other.
- **Causation**: Suggests a direct cause-and-effect relationship between variables, requiring experimentation to establish.


# Correlation {#correlation}


Use in understanding relationships between variables in data analysis. 

While it helps identify associations, it's important to remember that <mark>correlation does not imply causation.</mark> 

Visualization tools like heatmaps and clustering can aid in identifying and interpreting these relationships effectively.

- What is Correlation?: A measure of the strength and direction of the relationship between two variables.
### Description

- Correlation measures the relationship between two variables, indicating how they change together. It ranges from -1 to 1:

  - -1: Perfect negative correlation
  - 0: No correlation
  - 1: Perfect positive correlation

### Key Points

- [Correlation vs Causation](#correlation-vs-causation): Correlation does not imply causation. While correlation highlights associations, causation establishes a direct influence.
- Significance: Correlation values < -0.5 or > +0.5 are considered significant.
- Impact of Outliers: [standardised/Outliers](#standardisedoutliers) can distort correlation results.
- Standardization: Correlation is a standardized version of [Covariance](#covariance).

### Model Preparation

[Feature Selection](#feature-selection):
  - Identify features correlated with the target. If all are correlated, keep all.
  - For features correlated with each other, consider dropping one to avoid redundancy.
  - If two features are highly correlated with the target, both can be retained.

If two variables are strongly positively correlated, it often makes sense to drop one of them to simplify the model. This is because <mark>highly correlated variables can introduce redundancy</mark>, leading to [multicollinearity](#multicollinearity) in regression models.

By removing one of the correlated variables, you can:

1. Reduce Complexity: Simplifying the model by reducing the number of predictors can make it easier to interpret and manage.
2. Improve Stability: Reducing multicollinearity can lead to more stable and reliable coefficient estimates.
3. Enhance Performance: In some cases, removing redundant features can improve the model's predictive performance by reducing overfitting.

However, it's important to ensure that the variable you choose to keep is the one that is more relevant or has a stronger theoretical justification for inclusion in the model. 

### Viewing Correlations

- Use [Heatmap](#heatmap) or [Clustering](#clustering) to visualize correlations between features.

### Example Code

To find the correlation between two features:

```python
df['var1', 'target'](#var1-target).groupby(['var1'], as_index=False).mean().sort_values(by='target', ascending=False)
```


# Cosine Similarity {#cosine-similarity}

Cosine similarity is a [Metric](#metric) used to measure how similar two vectors are by calculating the cosine of the angle between them. It ranges from -1 to 1.where 1 indicates identical orientation, 0 indicates orthogonality, and -1 indicates opposite orientation. 

Cosine similarity is commonly used in
- text analysis, 
- information retrieval, 
- recommendation systems to compare document similarity, user preferences, or item features.

In [Binary Classification](#binary-classification), cosine similarity can be used as a feature to help distinguish between two classes. For instance, in text classification tasks, you might represent documents as vectors using techniques like [TF-IDF](#tf-idf). 


# Cost Function {#cost-function}

The concept of a Cost Function is central to [Model Optimisation](#model-optimisation), particularly in training models.

A cost function, also known as a loss function or error function, is a mathematical function used in optimization and machine learning to measure the difference between predicted values and actual values. It quantifies the error or "cost" of a model's predictions. The goal of many machine learning algorithms is to minimize this cost function, thereby improving the accuracy of the model. Common examples of cost functions include Mean Squared Error (MSE) for regression tasks and Cross-Entropy Loss for classification tasks.

1. Relation to [Loss Function](#loss-function): The cost function is related to the loss function. While the loss function measures the error for a single data point, the cost function typically aggregates these errors over the entire dataset, often by taking an average. See [Loss versus Cost function](#loss-versus-cost-function).

3. Parameter Space ([Model Parameters](#model-parameters)) and Surface Plotting: By plotting the cost function over the parameter space, you can visualize how different parameter values affect the cost. This surface can have various peaks and valleys representing different levels of error.

4. [Gradient Descent](#gradient-descent): This is an optimization algorithm used to find the minimum of the cost function. By iteratively adjusting the parameters in the direction that reduces the cost, gradient descent helps in finding the optimal parameters for the model.

5. Caveats: The cost function is dependent on the dataset and may not always have an explicit formula. This means that the shape of the cost function surface can vary greatly depending on the data, and finding the global minimum can be challenging.



![Pasted image 20241216202825.png|500](../content/images/Pasted%20image%2020241216202825.png|500)

![Pasted image 20241216202917.png|500](../content/images/Pasted%20image%2020241216202917.png|500)


**[Reward Function](#reward-function)**: Mentioned as the opposite of a cost function, typically used in [Reinforcement learning](#reinforcement-learning) to indicate the desirability of an outcome.

# Covariance {#covariance}



In statistics, covariance is a measure of the degree to which two random variables change together. It indicates the direction of the linear relationship between the variables. Specifically, covariance can be defined as follows:

- **Positive Covariance**: If the covariance is positive, it means that as one variable increases, the other variable tends to also increase. Conversely, if one variable decreases, the other variable tends to decrease as well.
  
- **Negative Covariance**: If the covariance is negative, it indicates that as one variable increases, the other variable tends to decrease, and vice versa.

- **Zero Covariance**: A covariance close to zero suggests that there is no linear relationship between the two variables.

The formula for calculating the covariance between two random variables $X$ and $Y$ is given by:

$$
\text{Cov}(X, Y) = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})
$$

where:
- $X_i$ and $Y_i$ are the individual sample points,
- $\bar{X}$ and $\bar{Y}$ are the means of $X$ and $Y$ respectively,
- $n$ is the number of data points.

Covariance is used in:
- in the calculation of [correlation](#correlation) coefficients 
- and in multivariate statistics, such as in [Gaussian Mixture Models](#gaussian-mixture-models) where it helps describe the shape and orientation of the data distribution.

# Covering Index {#covering-index}

Like an [Database Index|Index](#database-indexindex) but for partial indexes?

# Cron Jobs {#cron-jobs}



# Cross Entropy {#cross-entropy}


Cross entropy is a [Loss function](#loss-function) used in [Classification](#classification) tasks, particularly for [categorical data](#categorical-data). The cross entropy loss function is particularly effective for multi-class classification problems, where the goal is to assign an input to one of several categories. 

<mark>Cross entropy measures confidence.</mark>

Cross entropy works by measuring the (difference/loss) <mark>dissimilarity between two probability distributions</mark>: the true distribution (actual class labels) and the predicted distribution (model's output probabilities). 

Fit of Predictions:
- A low cross entropy loss means the predicted probabilities are close to the true labels (e.g., assigning high probability to the correct class).
- A high loss indicates significant divergence, meaning the model's predictions are inaccurate or uncertain.

By minimizing cross entropy, the model learns to produce probability distributions that closely match the true class distributions, thereby improving its classification <mark>accuracy</mark>.

1. Probability Distributions: In a classification task, the model outputs a probability distribution over the possible classes for each input. For example, in a three-class problem, the model might output probabilities like [0.7, 0.2, 0.1] for classes A, B, and C, respectively.

2. True Labels: The true class label is represented as a one-hot encoded vector. If the true class is A, the vector would be [1, 0, 0].

3. Cross Entropy Calculation calculates the loss by comparing the predicted probabilities with the true labels. The formula for cross entropy loss $L$ for a single instance is:

   $$ L = -\sum_{i=1}^{N} y_i \log(p_i)$$

   where:
   - $N$ is the number of classes.
   - $y_i$ is the true label (1 if the class is the true class, 0 otherwise).
   - $p_i$ is the predicted probability for class $i$.

2. Interpretation: The cross entropy loss increases as the predicted probability diverges from the actual label. If the model assigns a high probability to the correct class, the loss is low. Conversely, if the model assigns a low probability to the correct class, the loss is high.

3. Optimization: During training, the model's parameters are adjusted to minimize the cross entropy loss across all training examples. This process helps the model improve its predictions over time.

## Where is it used

Cross entropy is widely used in classification for several reasons:

Probabilistic Modeling:
    - It directly aligns with the goals of probabilistic classifiers, as it measures how well the predicted probability distribution matches the true distribution.
    
Focus on Confidence:
    - Encourages the model to assign higher probabilities to the correct classes, improving not just accuracy but also confidence in predictions.

Optimization Efficiency:
    - Cross entropy is smooth and convex for logistic regression-like models, enabling efficient gradient-based optimization.

Multi-Class Support:
    - Works seamlessly in multi-class scenarios where the true labels are one-hot encoded and predictions are probability distributions.

### Implementation 

In [ML_Tools](#ml_tools) see: 
- [Cross_Entropy_Single.py](#cross_entropy_singlepy)
- [Cross_Entropy.py](#cross_entropypy)
- [Cross_Entropy_Net.py](#cross_entropy_netpy)





# Cross Validation {#cross-validation}


Cross-validation is a statistical technique used in machine learning to <mark>assess how well a model will generalize</mark> to an independent dataset. It is a crucial step in the model-building process because it helps ensure that the model is not [overfitting](#overfitting) or underfitting the training data.

- Cross-validation is a technique used in machine learning and statistics to evaluate the performance ([Model Optimisation](#model-optimisation)) of a predictive model.
- It provides a robust evaluation by splitting the training data into smaller chunks and training the model multiple times.
- K-Fold Cross-Validation: Involves dividing the dataset into \( k \) equal-sized subsets (called "folds") and using each fold as a validation set once, while the remaining \( k-1 \) folds are used for training.
- The model's performance is averaged across all \( k \) folds to provide a more robust estimate of its generalization performance.
### Common Variations

- K-Fold Cross-Validation: The most common method, where the data is split into \( k \) folds and the model is trained \( k \) times, each time using a different fold as the validation set.
- Stratified K-Fold: Ensures each fold has a similar proportion of class labels, important for imbalanced datasets.
- Repeated K-Fold: Repeats the process multiple times with different random splits for more robust results.
- Leave-One-Out Cross-Validation (LOOCV): Each data point is used once as a test set while the rest serve as the training set.

### How Cross-Validation Fits into Building a Machine Learning Model

1. [Model Evaluation](#model-evaluation): Used to evaluate the performance of different models or algorithms to choose the best one.
2. [Hyperparameter](#hyperparameter) Tuning: Provides a reliable performance metric for each set of hyperparameters.
3. [Model Validation](#model-validation): Ensures consistent performance across different subsets of data.
4. [Bias and variance](#bias-and-variance) tradeoff: Helps in understanding the tradeoff between bias and variance, guiding the choice of model complexity.

Advantages:
- Reduced Bias: Offers a more reliable performance estimate compared to using a single validation set.
- Efficient Data Use: All data is used for both training and validation.
- Prevents Overfitting: By evaluating on multiple folds, it can detect if the model is overfitting to the training data.
### Choosing \( k \)

- Common values: 5 or 10
- Higher \( k \) leads to more accurate estimates but increases computation time.
- Consider dataset size and complexity when choosing \( k \).

### Code Implementation

In [ML_Tools](#ml_tools) see:
- [KFold_Cross_Validation.py](#kfold_cross_validationpy)

### Cross-Validation Strategy in [Time Series](#time-series)

All notebooks use cross-validation based on `TimeSeriesSplit` to ensure proper evaluation of performance with no [Data Leakage](#data-leakage). This method ensures that training and test data are split while maintaining the chronological order of the data.

# Cross_Entropy.Py {#cross_entropypy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Selection/Model_Evaluation/Classification/Cross_Entropy.py
### Generalized Script Description:

1. **Dataset**: Uses the Iris dataset from `sklearn` to classify flower species.
2. **Preprocessing**: One-hot encodes the target labels and splits the data into training and testing sets.
3. **Model**: Trains a multinomial logistic regression model to predict probabilities for each class.
4. **Cross Entropy Calculation**: Computes cross entropy loss for all predictions in the test set.
5. **Visualization**: Plots a histogram to show the distribution of loss values across the test samples.
6. **Summary Statistics**: Outputs mean, median, maximum, and minimum loss values for analysis.

This approach provides insight into the model's performance by analyzing the spread and typical values of cross entropy loss over multiple predictions.

### Strengths:

1. **Real-World Dataset**: The Iris dataset is well-known and intuitive, making it easier to follow and validate the results.
2. **Generalization**: The script calculates the cross entropy loss for multiple predictions, demonstrating the loss function in a real-world, multi-class classification scenario.
3. **Insights Through Visualization**: The histogram of losses provides a clear picture of how well the model performs across different test samples.
4. **Statistical Summary**: The inclusion of mean, median, max, and min loss values gives a quick overview of the model's performance.
5. **Numerical Stability**: The small epsilon value in the log computation ensures stability when dealing with probabilities close to zero.
6. **Reproducibility**: Using `sklearn`'s preprocessing and modeling tools ensures that the example is easy to replicate.

### Possible Enhancements:

1. **Alternative Models**: Incorporating another model (e.g., a neural network) could showcase the versatility of cross entropy in various settings.
2. **Analysis of Misclassifications**: Add a breakdown of where the model performed poorly and why (e.g., confusion matrix analysis).
3. **Feature Exploration**: Include visualizations or explanations of feature importance to show how the model makes decisions.
4. **Comparative Losses**: Compare cross entropy loss with other loss functions (e.g., mean squared error) to highlight its advantages in classification.

**Distribution Insights**:

- The histogram of loss values shows how well the model performs across the test dataset.
    - A **narrow distribution** around a low value suggests consistent, accurate predictions.
    - A **wide or skewed distribution** indicates variability in the model's performance, with some instances being predicted poorly.

### [Mean Squared Error](#mean-squared-error) versus [Cross Entropy](#cross-entropy)

- **When Comparison Makes Sense**:
    - MSE can highlight how "far off" the predicted probabilities are in terms of magnitude but doesn’t account for the probabilistic nature of classification tasks.
    - Comparing cross entropy with MSE can show:
        - How the model performs when considering confidence (cross entropy).
        - How the model performs when focusing on numerical proximity (MSE).
        
- **Insights Gained**:
    - If cross entropy is low but MSE is high, it might indicate that the model predicts probabilities close to the correct class but has poor numerical calibration for other classes.
    - If both are high, the model is likely underperforming across the board.


# Cross_Entropy_Single.Py {#cross_entropy_singlepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Selection/Model_Evaluation/Classification/Cross_Entropy_Single.py
## Example

Let's consider a three-class classification problem with classes A, B, and C. Suppose we have a single data point with the true class label being A. The true label in one-hot encoded form would be [1, 0, 0].

Assume the model predicts the following probabilities for this data point:

- Probability of class A: 0.7
- Probability of class B: 0.2
- Probability of class C: 0.1

The predicted probability vector is [0.7, 0.2, 0.1].

To calculate the cross entropy loss for this example, we use the formula:

$L = -\sum_{i=1}^{N} y_i \log(p_i)$

Substituting the values:

- For class A: $y_1 = 1$ and $p_1 = 0.7$
- For class B: $y_2 = 0$ and $p_2 = 0.2$
- For class C: $y_3 = 0$ and $p_3 = 0.1$

The cross entropy loss $L$ is calculated as:

$L = -(1 \cdot \log(0.7) + 0 \cdot \log(0.2) + 0 \cdot \log(0.1))$

$L = -(\log(0.7))$

$L \approx -(-0.3567) = 0.3567$

So, the cross entropy loss for this example is approximately 0.3567. This value represents the penalty for the model's predicted probabilities not perfectly matching the true class distribution. The lower the loss, the better the model's predictions align with the true labels.

### Script Description:

1. **Cross Entropy Function**: Computes the cross entropy loss given true labels and predicted probabilities.
2. **True and Predicted Probabilities Visualization**: Bar plots display the true one-hot encoded labels and the predicted probability distribution.
3. **Cross Entropy Loss Calculation**: Prints the loss value for a sample data point.
4. **Loss Curve**: A line graph shows how the loss changes as the predicted probability for the true class increases.

# Crosstab {#crosstab}

Used to compute a simple cross-tabulation of two (or more) factors. It is particularly useful for computing frequency tables. Here's an example:

```python
# Sample DataFrame
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Subcategory': ['X', 'X', 'Y', 'Y', 'X']
})

# Cross-tabulation of 'Category' and 'Subcategory'
crosstab = pd.crosstab(df['Category'], df['Subcategory'])
print(crosstab)
```

Input
```
  Category Subcategory
0        A           X
1        B           X
2        A           Y
3        B           Y
4        A           X
```

Output:
```
Subcategory  X  Y
Category         
A            2  1
B            1  1
```

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/reshaping.ipynb

# Cryptography {#cryptography}


Cryptography is the foundation of digital [Security](#security), enabling privacy and secure communication over the internet.

Examples are implemented in [Node.JS](#nodejs) (using `crypto` module) and are written in [JavaScript](#javascript).

Resources:
- [7 Cryptography Concepts EVERY Developer Should Know](https://www.youtube.com/watch?v=NuyzuNBFWxQ)
- https://fireship.io/lessons/node-crypto-examples/
## [Hash](#hash) (Chop and mix)

A hashing function takes an input of any length and outputs a fixed-length value, ensuring:

- The same input always produces the same output.
- It is computationally expensive to reverse the hash.
- It has a low probability of collisions.

### Create a Hash in Node.js

```javascript
const { createHash } = require('crypto');

function hash(str) {
    return createHash('sha256').update(str).digest('hex');
}

let password = 'hi-mom!';
const hash1 = hash(password);
console.log(hash1);

password = 'hi-mom';
const hash2 = hash(password);
console.log(hash1 === hash2 ? '✔️ Good password' : '❌ Password does not match');
```

## Salting

Salting strengthens hashes by appending a random string before hashing, preventing attacks using precomputed hash tables.

### Password Salt with Scrypt in Node.js

```javascript
const { scryptSync, randomBytes, timingSafeEqual } = require('crypto');

function signup(email, password) {
    const salt = randomBytes(16).toString('hex');
    const hashedPassword = scryptSync(password, salt, 64).toString('hex');
    users.push({ email, password: `${salt}:${hashedPassword}` });
}

function login(email, password) {
    const user = users.find(v => v.email === email);
    if (!user) return 'login fail!';
    
    const [salt, key] = user.password.split(':');
    const hashedBuffer = scryptSync(password, salt, 64);
    const match = timingSafeEqual(hashedBuffer, Buffer.from(key, 'hex'));
    return match ? 'login success!' : 'login fail!';
}

const users = [];
signup('foo@bar.com', 'pa$$word');
console.log(login('foo@bar.com', 'password'));
```

## HMAC (Hash-based Message Authentication Code)

HMAC combines a hash with a secret key, ensuring authenticity and integrity.

### HMAC in Node.js

```javascript
const { createHmac } = require('crypto');

const password = 'super-secret!';
const message = '🎃 hello jack';

const hmac = createHmac('sha256', password).update(message).digest('hex');
console.log(hmac);
```

## Symmetric Encryption

Symmetric encryption uses the same key to encrypt and decrypt data.

### Symmetric Encryption in Node.js

```javascript
const { createCipheriv, randomBytes, createDecipheriv } = require('crypto');

const message = 'i like turtles';
const key = randomBytes(32);
const iv = randomBytes(16);
const cipher = createCipheriv('aes256', key, iv);
const encryptedMessage = cipher.update(message, 'utf8', 'hex') + cipher.final('hex');

const decipher = createDecipheriv('aes256', key, iv);
const decryptedMessage = decipher.update(encryptedMessage, 'hex', 'utf-8') + decipher.final('utf8');
console.log(`Decrypted: ${decryptedMessage}`);
```

## Keypairs

Keypairs consist of a public key (shared) and a private key (kept secret) for secure communication.

### Generate an RSA Keypair in Node.js

```javascript
const { generateKeyPairSync } = require('crypto');

const { privateKey, publicKey } = generateKeyPairSync('rsa', {
  modulusLength: 2048,
  publicKeyEncoding: { type: 'spki', format: 'pem' },
  privateKeyEncoding: { type: 'pkcs8', format: 'pem' },
});

console.log(publicKey);
console.log(privateKey);
```

## Asymmetric Encryption

Asymmetric encryption encrypts with a public key and decrypts with a private key, securing communication over networks.

### RSA Encryption in Node.js

```javascript
const { publicEncrypt, privateDecrypt } = require('crypto');
const { publicKey, privateKey } = require('./keypair');

const secretMessage = 'Confidential message';
const encryptedData = publicEncrypt(publicKey, Buffer.from(secretMessage));
console.log(encryptedData.toString('hex'));

const decryptedData = privateDecrypt(privateKey, encryptedData);
console.log(decryptedData.toString('utf-8'));
```

## Signing

Signing verifies the authenticity of a message by hashing it and encrypting the hash with a private key.

### RSA Signing in Node.js

```javascript
const { createSign, createVerify } = require('crypto');
const { publicKey, privateKey } = require('./keypair');

const data = 'this data must be signed';
const signer = createSign('rsa-sha256');
signer.update(data);
const signature = signer.sign(privateKey, 'hex');
console.log(signature);

const verifier = createVerify('rsa-sha256');
verifier.update(data);
const isVerified = verifier.verify(publicKey, signature, 'hex');
console.log(isVerified);
```

# Current Challenges Within The Energy Sector {#current-challenges-within-the-energy-sector}


[Current challenges within the energy sector](#current-challenges-within-the-energy-sector) related to reinforcement learning and that can be progressed with recent technological advances



# Cleaning Terminal Path {#cleaning-terminal-path}


https://www.youtube.com/watch?v=18hUejOK0qk

```cmd
prompt $g
```

### powershell
```powershell
$profile

microsfot_Powershell_profile have

function prompt{
$p = -path
"$p> "
}
```

getting the script working 

https://stackoverflow.com/questions/41117421/ps1-cannot-be-loaded-because-running-scripts-is-disabled-on-this-system



# Conceptual Data Model {#conceptual-data-model}

