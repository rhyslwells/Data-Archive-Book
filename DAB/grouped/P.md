# P

## Table of Contents
* [PCA Explained Variance Ratio](#pca-explained-variance-ratio)
* [PCA Principal Components](#pca-principal-components)
* [PCA-Based Anomaly Detection](#pca-based-anomaly-detection)
* [PCA_Analysis.ipynb](#pca_analysisipynb)
* [PCA_Based_Anomaly_Detection.py](#pca_based_anomaly_detectionpy)
* [PDF++](#pdf)
* [PDP and ICE](#pdp-and-ice)
* [Page Rank](#page-rank)
* [Pandas Dataframe Agent](#pandas-dataframe-agent)
* [Pandas Pivot Table](#pandas-pivot-table)
* [Pandas Stack](#pandas-stack)
* [Pandas join vs merge](#pandas-join-vs-merge)
* [Pandas](#pandas)
* [Pandas_Common.py](#pandas_commonpy)
* [Pandas_Stack.py](#pandas_stackpy)
* [Parametric tests](#parametric-tests)
* [Parquet](#parquet)
* [Part of speech tagging](#part-of-speech-tagging)
* [Percentile Detection](#percentile-detection)
* [Performance Dimensions](#performance-dimensions)
* [Performance Drift](#performance-drift)
* [Physical Model](#physical-model)
* [Plotly](#plotly)
* [Poetry](#poetry)
* [Policy](#policy)
* [Positional Encoding](#positional-encoding)
* [PostgreSQL](#postgresql)
* [PowerBI](#powerbi)
* [PowerShell](#powershell)
* [Powerquery](#powerquery)
* [Powershell versus cmd](#powershell-versus-cmd)
* [Powershell vs Bash](#powershell-vs-bash)
* [Precision or Recall](#precision-or-recall)
* [Precision-Recall Curve](#precision-recall-curve)
* [Precision](#precision)
* [Prediction Intervals](#prediction-intervals)
* [Preprocessing](#preprocessing)
* [Prevention Is Better Than The Cure](#prevention-is-better-than-the-cure)
* [Primary Key](#primary-key)
* [Principal Component Analysis](#principal-component-analysis)
* [Probability in other fields](#probability-in-other-fields)
* [Problem Definition](#problem-definition)
* [Prompt Engineering](#prompt-engineering)
* [Proportion Test](#)
* [Publish and Subscribe](#publish-and-subscribe)
* [Pull Request Template](#)
* [Push-Down](#push-down)
* [PyCaret](#pycaret)
* [PyGraphviz](#pygraphviz)
* [PySpark](#pyspark)
* [PyTorch](#pytorch)
* [Pycaret_Anomaly.ipynb](#pycaret_anomalyipynb)
* [Pycaret_Example.py](#pycaret_examplepy)
* [Pydantic](#pydantic)
* [Pydantic.py](#pydanticpy)
* [Pydantic_More.py](#pydantic_morepy)
* [Pyright vs Pydantic](#pyright-vs-pydantic)
* [Pyright](#pyright)
* [Pytest](#)
* [Python Click](#python-click)
* [Python](#python)
* [Pytorch vs Tensorflow](#pytorch-vs-tensorflow)
* [p values](#p-values)
* [p-values in linear regression in sklearn](#p-values-in-linear-regression-in-sklearn)
* [parametric vs non-parametric models](#parametric-vs-non-parametric-models)
* [parametric vs non-parametric tests](#parametric-vs-non-parametric-tests)
* [parsimonious](#parsimonious)
* [pd.Grouper](#pdgrouper)
* [pdoc](#pdoc)
* [pmdarima](#pmdarima)
* [programming languages](#programming-languages)



# Pca Explained Variance Ratio {#pca-explained-variance-ratio}

[PCA Explained Variance Ratio](#pca-explained-variance-ratio)
- The variance explained by each principal component is printed using `pca.explained_variance_ratio_`.
- The sum of the explained variances is calculated and printed, which should ideally equal 1 (indicating that all variance in the data is accounted for by the principal components).

The **explained variance** refers to how much of the total variance in the original dataset is captured or explained by each principal component (PC) in Principal Component Analysis (PCA).

### In the context of PCA:
    
2. **Explained Variance**: After performing PCA, each principal component corresponds to a certain amount of variance in the original data. The **explained variance** of a principal component is the proportion of the total variance in the original dataset that is captured by that component.
    
3. **Explained Variance Ratio**: This is the proportion of the total variance explained by each principal component. It is calculated as:
    
    Explained Variance Ratio of PCi=Variance explained by PCiTotal variance of the dataset\text{Explained Variance Ratio of PC}_i = \frac{\text{Variance explained by PC}_i}{\text{Total variance of the dataset}}
    
    It tells us how much of the total variance in the data is accounted for by each component. For example, if the first principal component explains 70% of the total variance, then the explained variance ratio for that component would be 0.70.
    
4. **Cumulative Explained Variance**: If we sum up the explained variance ratios of the first few principal components, we can assess how many components are needed to explain a significant portion of the total variance. For example, if the first two components explain 90% of the variance, it means that we can reduce the dataset's dimensionality by keeping just those two components without losing much information.
    

### Example:

In PCA applied to the Iris dataset:

```python
pca.explained_variance_ratio_
```

This might return something like:

```
[0.924, 0.053, 0.018, 0.005]
```

This means:

- The first principal component explains 92.4% of the total variance.
- The second principal component explains 5.3% of the variance.
- The third and fourth components explain very little (1.8% and 0.5%, respectively).

Summing these ratios gives the total variance explained by the first few components. In this case, the first two components explain over 97% of the variance in the dataset, meaning the remaining components contribute very little additional information.

### Why it matters:

- **Dimensionality Reduction**: PCA is used to reduce the number of dimensions (features) in the dataset. The goal is to retain as much variance as possible while reducing the number of dimensions. By selecting the components with the highest explained variance, we can reduce the dataset's complexity without sacrificing much information.
    
- **Data [interpretability](#interpretability)**: The explained variance helps us understand how important each component is in representing the dataset. If the first few components explain most of the variance, we can focus on them for analysis and modeling.
    

# Pca Principal Components {#pca-principal-components}

The principal components (or the new axes that explain the most variance) are stored in `pca.components_` and displayed as a DataFrame for easier reading

## Interpretating

See [PCA_Analysis.ipynb](#pca_analysisipynb)

![Pasted image 20250317093551.png](../content/images/Pasted%20image%2020250317093551.png)

### How to Interpret the PCA Heatmap

This heatmap represents the principal component loadings, which show how strongly each original feature contributes to each principal component (PC).

### Understanding the Heatmap Content

- Rows = Principal Components (PCs)
    
    - Each row corresponds to a principal component (e.g., PC1, PC2, PC3, etc.).
    - PC1 captures the most variance, PC2 captures the second-most, and so on.
- Columns = Original Features
    
    - Each column represents an original feature from the dataset (e.g., `sepal length`, `sepal width`, etc.).
- Cell Values = Loadings
    
    - Each cell contains a loading coefficient, which tells us how much that feature contributes to the corresponding principal component.
    - The values range from -1 to 1:Close to 1 → The feature strongly contributes to that PC in the positive direction, ect.
### Key Insights from the [Heatmap](#heatmap)

1. Which features are most important for each PC?
    - Look for the largest absolute values in each row.
    - Example: If `sepal length` has a high positive value in PC1, it means PC1 is largely influenced by sepal length.

2. Feature Groupings & Correlations
    - Features with similar values in a PC vary together in the data.
    - Example: If `sepal length` and `sepal width` have similar values in PC1, they might be [Correlation](#correlation) correlated in the dataset.

3. Interpreting the First Few Principal Components
    - PC1 often represents the main pattern in the data (e.g., overall size of the iris flowers).
    - PC2 might represent a different pattern (e.g., a contrast between petal and sepal size).
    - Together, PC1 and PC2 often explain the majority of variance in the dataset.

### Example Interpretation (Hypothetical Output)

| Sepal Length | Sepal Width | Petal Length | Petal Width |       |
| ------------ | ----------- | ------------ | ----------- | ----- |
| PC1          | 0.70        | -0.40        | 0.85        | 0.75  |
| PC2          | -0.60       | 0.80         | -0.35       | -0.45 |
PC1 Interpretation: `Petal length` and `sepal length` have high positive loadings, meaning PC1 mainly captures flower size.`Sepal width` has a negative loading, meaning flowers with large sepals tend to have smaller widths.

PC2 Interpretation: `Sepal width` has the highest positive loading, while `sepal length` has a negative loading, suggesting that PC2 contrasts width vs. length.

### How to Use This Information?

[Feature Selection](#feature-selection): If one PC captures most of the variance, we can reduce dimensionality and keep only the most important PCs.



# Pca Based Anomaly Detection {#pca-based-anomaly-detection}

For implementation, see: [ML_Tools](#ml_tools):
- [PCA_Based_Anomaly_Detection.py](#pca_based_anomaly_detectionpy)

# Pca_Analysis.Ipynb {#pca_analysisipynb}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/PCA/PCA_Analysis.ipynb

This script performs Principal Component Analysis (PCA) on the Iris dataset to reduce its dimensionality while preserving key variance. 

See also [Principal Component Analysis|PCA](#principal-component-analysispca)

Summary:

1. Load and Preprocess Data
    - Loads the Iris dataset and extracts features and target labels.
    - Scales the data to standardize feature ranges.

2. Apply PCA (3 Components)
    - Fits PCA to the scaled data and transforms it into three principal components.
    - Stores the transformed data in a DataFrame with species labels.
    
3. Analyze PCA Loadings & Variance
    - Computes and stores PCA loadings (weights of original features in principal components).
    - Computes explained variance and cumulative variance to assess PCA effectiveness.

4. Visualizations
    - Explained variance: Bar plot of individual and cumulative variance contributions.
    - PCA Scores: 3D scatter plots of transformed data, colored by species.
    - PCA Loadings: 3D scatter plot showing feature contributions to principal components.
    - Heatmap: Displays PCA component weights for feature importance analysis.

5. Additional Full PCA Analysis
    - Computes and prints explained variance for all components.
    - Uses Seaborn to generate a heatmap of PCA component contributions.

# Pca_Based_Anomaly_Detection.Py {#pca_based_anomaly_detectionpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/PCA/PCA_Based_Anomaly_Detection.py

# Pdf++ {#pdf}

https://www.youtube.com/watch?v=4dU6WXULSqg

# Pdp And Ice {#pdp-and-ice}

link:

https://scikit-learn.org/1.5/modules/partial_dependence.html#h2009

[interpretability|interpretable](#interpretabilityinterpretable)



Regression example:

![Pasted image 20241204203338.png](../content/images/Pasted%20image%2020241204203338.png)

Categorical Example

![Pasted image 20241204203413.png](../content/images/Pasted%20image%2020241204203413.png)



# Page Rank {#page-rank}

PageRank is an algorithm originally developed by Larry Page and Sergey Brin (founders of Google) to rank web pages in search engine results. It measures the relative importance of each node (e.g., webpage) in a directed graph based on the structure of incoming links.
###  Intuition

The core idea is:
> A node is important if many other important nodes link to it.

PageRank simulates a “random surfer” who clicks on links at random:
- With probability $d$ (typically 0.85), the surfer follows a link from the current page.
- With probability $1 - d$, the surfer jumps to a random page.

### Mathematical Formulation

Given a graph with $N$ nodes, the PageRank of node $i$ is defined recursively as:

$$
PR(i) = \frac{1 - d}{N} + d \sum_{j \in \text{In}(i)} \frac{PR(j)}{L(j)}
$$

Where:
- $d$ is the damping factor (usually 0.85),
- $\text{In}(i)$ is the set of nodes linking to $i$,
- $L(j)$ is the number of outbound links from node $j$.



### Implementation (using NetworkX)

```python
import networkx as nx

G = nx.DiGraph()
G.add_edges_from([
    ('A', 'B'), ('B', 'C'), ('C', 'A'), ('A', 'D'), ('D', 'C')
])

pagerank_scores = nx.pagerank(G, alpha=0.85)
for node, score in pagerank_scores.items():
    print(f"{node}: {score:.4f}")
```



### 📊 Use Cases
- Graph-based NLP: Keyword extraction (e.g., TextRank).

#graph #data_visualization

# Pandas Dataframe Agent {#pandas-dataframe-agent}

Example:
https://github.com/AssemblyAI/youtube-tutorials/tree/main/pandas-dataframe-agent


Follow:

https://www.youtube.com/watch?v=ZIfzpmO8MdA&list=PLcWfeUsAys2kC31F4_ED1JXlkdmu6tlrm&index=7

Can as pandas questions to a dataframe. 

Types of questions:
- what is the max value of "col1"

# Pandas Pivot Table {#pandas-pivot-table}


Pivot Table: Summarize Data
```python
df = pd.DataFrame({'A': ['foo', 'foo', 'bar'], 'B': ['one', 'two', 'one'], 'C': [1, 2, 3]})
pivot_table = df.pivot_table(values='C', index='A', columns='B', aggfunc='sum')
```

Relevant links:
- https://pandas.pydata.org/docs/reference/api/pandas.pivot_table.html

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/pivot_table.ipynb

# Pandas Stack {#pandas-stack}


Tool for reshaping data, particularly when you need to pivot a DataFrame ([Pandas Pivot Table](#pandas-pivot-table)) from a wide format to a long format. 

See:
- [Pandas_Common.py](#pandas_commonpy)
- [Pandas_Stack.py](#pandas_stackpy)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/reshaping.ipynb
#### Why Use `stack`? ([Data Transformation](#data-transformation))

Data Reshaping:
   - <mark>Wide to Long Format</mark>: Convert a DataFrame from a wide format to a long format, which is often preferred for statistical models and visualizations.

Handling Multi-Index DataFrames:
   - Simplifying Structure: Move the inner level of a column MultiIndex to the row index, simplifying the DataFrame's structure.

[Data Cleansing](#data-cleansing):
   - Aggregation and Operations: Facilitate data cleaning by allowing aggregation or operations across columns in a more manageable long format.

Preparing Data for Grouping or Aggregation ([Groupby](#groupby)):
   - Ease of Grouping: Simplify group-by operations and aggregations on data with columns representing different categories or time periods.
#### Example of Using `stack`

Consider the following example to illustrate how `stack` works:

```python
import pandas as pd

# Sample DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# Using stack
stacked_df = df.stack()
print("\nStacked DataFrame:")
print(stacked_df)
```

Output:
```
Original DataFrame:
   A  B  C
0  1  4  7
1  2  5  8
2  3  6  9

Stacked DataFrame:
0  A    1
   B    4
   C    7
1  A    2
   B    5
   C    8
2  A    3
   B    6
   C    9
dtype: int64
```

In this example:
- The original DataFrame has three columns ('A', 'B', 'C') and three rows.
- After stacking, the DataFrame is transformed into a Series with a MultiIndex/[Multi-level index](#multi-level-index). The outer level of the index corresponds to the original DataFrame’s row index, and the inner level corresponds to the original column labels.

#### When Not to Use `stack`

- Wide Format Requirements: If your analysis or processing requires a wide format, such as some [Machine Learning Algorithms](#machine-learning-algorithms), stacking may not be appropriate.
- Complexity: If stacking makes the data too complex to manage or understand, it might be better to keep the original structure.
- Simplicity: When the current structure of your DataFrame already suits your analysis needs, stacking may be unnecessary.

# Pandas Join Vs Merge {#pandas-join-vs-merge}

In pandas, both `.join()` and `pd.merge()` are used to combine DataFrames, but they differ in **syntax**, **defaults**, and **use cases**.

[Merge](#merge) is better than Join.

|Feature|`df.join()`|`pd.merge()`|
|---|---|---|
|**Default key**|Uses index of caller and index/column of other|Requires explicit column(s) to merge on|
|**Syntax style**|Method on a DataFrame|Function (`pd.merge(left, right)`)|
|**Column join**|Must specify `on=` and use one column from each|Can merge on multiple columns|
|**Index join**|Default behavior (index-to-index)|Requires `left_index=True`, `right_index=True`|
|**Suffixes**|`lsuffix`, `rsuffix`|`suffixes=('_x', '_y')`|
|**Complex joins**|Not well-suited|Supports full SQL-style joins|
|**Use case**|Simple joins on index or one column|Complex joins with control over join behavior|
#data_cleaning #data_integration 

# Pandas {#pandas}


In [ML_Tools](#ml_tools) see:
- [Pandas_Common.py](#pandas_commonpy)

Areas:
- [Handling Missing Data](#handling-missing-data) 
- [Data Selection](#data-selection)
- [Data Transformation](#data-transformation)








# Pandas_Common.Py {#pandas_commonpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Pandas_Common.py

# Pandas_Stack.Py {#pandas_stackpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Pandas_Stack.py

# Parametric Tests {#parametric-tests}



# Parquet {#parquet}


A **Parquet file** is a **columnar storage file format** specifically designed for storing large amounts of data efficiently. It is commonly used in [Big Data](#big-data) ecosystems due to its optimised performance for both storage and querying.

[Data Storage](#data-storage)
### Key Features of Parquet:

1. **Columnar Storage Format:**
    - Data is stored **column by column** instead of row by row.
    - This design is highly efficient for analytical queries that access only specific columns, reducing the amount of data read.
      
2. **Optimised for Big Data:**
    - Parquet is designed for distributed systems like Apache [Hadoop](#hadoop), [Apache Spark](#apache-spark), and other big data processing tools.
      
3. **Compression:**
    - It supports multiple compression algorithms (e.g., Snappy, GZIP) for reducing file size while maintaining fast read and write performance.
      
4. **[Schema Evolution](#schema-evolution):**
    - Parquet supports flexible schemas, allowing fields to be added or modified without breaking compatibility with older data.
      
5. **Efficient [Metadata Handling](#metadata-handling):**
    - Metadata is stored along with the data, making it easier to retrieve and query information about the dataset without scanning the entire file.

### Advantages of Parquet:

1. **Improved Query Performance:**
    - Since data is stored column-wise, queries that require only a few columns read less data compared to row-based formats like CSV.
      
2. **Lower Storage Costs:**
    - Built-in compression and columnar storage **significantly reduce file size.**
      
3. **Compatibility:**
    - Parquet is compatible with most big data processing tools, such as Hadoop, Spark, Hive, Presto, and AWS Athena.

1. **Efficient I/O:**
    - Parquet’s columnar format minimises disk I/O, making it faster to process large datasets.

---

### When to Use Parquet:
- **Analytical Workloads:** Ideal for scenarios where you need to perform aggregations, filtering, or processing large datasets.
- **Big Data Processing:** Frequently used with tools like Spark, Hive, and Presto in data lakes.
- **Cloud Storage:** Supported by [Cloud Providers](#cloud-providers) platforms.

### Example Use Case:

Imagine a dataset with 10 million rows and 100 columns.

- If you query just 3 columns in a row-based format (e.g., CSV), you must read all 100 columns for every row.
- In Parquet, only the 3 relevant columns are read, significantly improving performance.

---

### File Structure:

- **Row Group:** Data is divided into chunks called row groups, enabling efficient data retrieval.
- **Columns:** Each column in a row group is stored together for fast access.
- **Footer:** Contains metadata, such as schema definitions and row group locations, allowing quick navigation of the file.

---

### How to Work with Parquet in Python:

You can use libraries like **pandas** or **pyarrow**:

```python
import pandas as pd

# Write a DataFrame to a Parquet file
df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df.to_parquet('example.parquet', engine='pyarrow')

# Read a Parquet file into a DataFrame
df_read = pd.read_parquet('example.parquet')
print(df_read)
```

---

In summary, Parquet is an efficient, compact, and scalable file format ideal for big data analytics and storage, providing faster performance and reduced costs.

# Part Of Speech Tagging {#part-of-speech-tagging}

Part of speech tagging : assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text

Part-of-speech tagging involves assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text
```python
from nltk import pos_tag
pos_tag(temp[:20])
```
will get outputs such as [('history', 'NN'), ('poland', 'NN'), ('roots', 'NNS'), ('early', 'JJ').

# Percentile Detection {#percentile-detection}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Outliers/outliers_percentile.py

# Performance Dimensions {#performance-dimensions}


 Efficiency & Performance  
- [Cost-efficiency](#cost-efficiency): Ensuring that the solutions used are cost-effective and provide value for money.  
- [Speed](#speed): The ability to process and analyze data quickly to meet business needs.  
- [Performance](#performance): Optimizing query performance by organizing data in a way that supports efficient retrieval.

 Flexibility & Scalability  
- [Flexibility](#flexibility): The capability to adapt to changing requirements and data sources.  
- [Scalability](#scalability): Ensuring the system can handle increasing volumes of data without performance degradation.  Can handle large volumes of data, making it suitable for enterprise-level data warehousing.  
- [Reusability](#reusability): Designing components that can be reused across different projects or processes.  
- [Latency](#latency): Minimizing delays in data processing and retrieval.  

 Usability & Accessibility  
- [Simplicity](#simplicity): Keeping the system easy to manage and understand, reducing complexity.  
- [Usability](#usability): Providing a clear and intuitive structure that is easy for business users to understand and navigate.  
- [Accessibility](#accessibility): Ensuring that data is accessible when needed by authorized users.

 Data Quality & Integrity  
- [Data Integrity](#data-integrity): Ensuring data accuracy and consistency.  
- [Data Quality](#data-quality): Maintaining high-quality data for reliable insights.  
- [Availability](#availability): Ensuring that data is available when required by authorized users.  

 Interoperability  
- [Interoperability](#interoperability): Ensuring that different systems and tools can work together seamlessly.  
- [Data Compatibility](#data-compatibility): Ensuring data is compatible with different systems.




# Performance Drift {#performance-drift}


Not [Data Drift](#data-drift)

 **TL;DR**. Data drift is a change in the input data. Concept drift is a change in input-output relationships. Both often happen simultaneously.

 Performance drift refers to the <mark>gradual decline in a machine learning model's accuracy</mark> or effectiveness over time as the underlying data distribution changes. 
 
 This phenomenon occurs when the real-world data that the model is applied to differs from the data it was trained on. Mathematically, this is often represented by a shift in the joint distribution $P(X, Y)$ of the features $X$ and target variable $Y$. 
 
 Performance drift can occur due to <mark>*concept drift* (</mark>when the relationship between inputs and outputs changes) or <mark>*covariate shift*</mark> (when the distribution of the inputs changes). The model’s prediction error increases, leading to suboptimal decisions or predictions.

 Key Components:  
 - **Concept drift**: Changes in the relationship between inputs and outputs, $P(Y|X)$.  
 - **Covariate shift**: <mark>Change in the input data distribution, $P(X)$.</mark>  
 - **Model monitoring** [Data Observability|monitoring](#data-observabilitymonitoring): Continuous assessment of a model’s accuracy over time to detect drift.  
 - **Retraining**: Updating the model with new data to restore performance.

Important
 - Performance drift results from data distribution shifts, leading to increased prediction errors.  
 - Monitoring and retraining are key strategies to address performance drift in real-world applications.

 - A lack of continuous monitoring can result in undetected <mark>model performance degradation.</mark>  
 - Overfitting a model to the original data without considering future data can accelerate drift.

Example
 In a credit scoring model, performance drift may occur if consumer spending habits change due to an economic recession. The model trained on pre-recession data will perform poorly on post-recession data as the input patterns ($P(X)$) and the relationship between inputs and outputs ($P(Y|X)$) shift.

Questions
 - How can adaptive learning techniques help mitigate the effects of performance drift?  
 - What statistical methods can be used to detect early signs of concept drift in production models?

Related Topics
 - Model retraining strategies  
 
Images

![Pasted image 20250113072251.png](../content/images/Pasted%20image%2020250113072251.png)

# Physical Model {#physical-model}

Physical Model

(for a SQL database):
   ```sql
   CREATE TABLE Customer (
       CustomerID INT PRIMARY KEY,
       Name VARCHAR(100),
       Email VARCHAR(100)
   );

   CREATE TABLE Order (
       OrderID INT PRIMARY KEY,
       OrderDate DATE,
       CustomerID INT,
       FOREIGN KEY (CustomerID) REFERENCES Customer(CustomerID)
   );

   CREATE TABLE Book (
       BookID INT PRIMARY KEY,
       Title VARCHAR(100),
       Author VARCHAR(100)
   );

   CREATE TABLE OrderBook (
       OrderID INT,
       BookID INT,
       PRIMARY KEY (OrderID, BookID),
       FOREIGN KEY (OrderID) REFERENCES Order(OrderID),
       FOREIGN KEY (BookID) REFERENCES Book(BookID)
   );
   ```



Physical Model
   - Implements the logical model in a specific database system.
   - Includes table structures, columns, data types, and constraints.

# Plotly {#plotly}



# Poetry {#poetry}


Modern version of setting up dependencies instead of requirements.txt ([dependency manager](#dependency-manager))

Primary Purpose: Poetry is a [dependency manager](#dependency-manager) and packaging tool for Python projects.
    
Main Features:
    - Dependency management: Allows you to declare, install, and lock dependencies in the `pyproject.toml` file.
    - Package management: Can help package your Python project for distribution (e.g., publishing to PyPI).
    - Virtual environment management: Poetry automatically creates and manages a virtual environment for your project.
    - Version management: Ensures that all your project dependencies use compatible versions through its `poetry.lock` file, similar to `npm` or `yarn` in the JavaScript ecosystem.
    - Built-in publishing: Simplifies the process of publishing your Python package to PyPI.

Ideal Use Case:
    - If you need to manage dependencies for a Python project, create virtual environments, and ensure reproducibility (using `poetry.lock`).
    - If you're developing a Python package that you want to distribute or manage versions for, Poetry is a great choice.

Use:

```cmd
pip install poetry
poetry init
poetry add numpy
```



# Policy {#policy}


In [reinforcement learning](#reinforcement-learning) (RL), a **policy** is a strategy or a rule that defines the actions an agent takes in a given state to achieve its goals. It essentially <mark>maps states of the environment to actions that the agent should take when in those states.</mark>

Policies are used in RL as they determine the behavior of the agent. 

The goal of many RL algorithms is to find an optimal policy that maximizes the cumulative reward the agent receives over time. This involves balancing exploration (trying new actions) and exploitation (using known actions that yield high rewards).

### Key Concepts

**On-Policy vs. Off-Policy**:
  - **On-Policy**: The agent learns the value of the policy it is currently following. An example is the [SARSA](#sarsa) algorithm, which updates its policy based on the actual actions taken by the agent.
  - **Off-Policy**: The agent learns the value of the optimal policy, regardless of the actions it actually takes. [Q-Learning](#q-learning) is an example of an off-policy algorithm, as it updates its policy based on the best possible action in the next state, not necessarily the action taken.

**Conservatism**:
  - Some policies, like those in SARSA, are more conservative in their updates. This means they are more cautious and adapt to uncertainties in the environment, making them suitable for environments where [exploration](#exploration) and [exploitation](#exploitation) need to be balanced carefully.

### Example of a Policy

Consider a simple grid world where an agent can move up, down, left, or right to reach a goal. A policy in this context could be a set of rules that tells the agent to always move towards the goal if it is visible, or to explore randomly if the goal is not visible.

For instance, in a 3x3 grid where the goal is at position (2, 2), a simple policy might be:

- If the agent is at (0, 0), move right.
- If the agent is at (0, 1), move right.
- If the agent is at (0, 2), move down.
- Continue this pattern until the agent reaches the goal at (2, 2).

This policy can be represented as a table or a function that maps each state (grid position) to an action (move direction).

|State (Position)|Action|
|---|---|
|(0, 0)|Move Right|
|(0, 1)|Move Right|
|(0, 2)|Move Down|
|(1, 0)|Move Right|
|(1, 1)|Move Right|
|(1, 2)|Move Down|
|(2, 0)|Move Right|
|(2, 1)|Move Right|
|(2, 2)|Goal Reached|


# Positional Encoding {#positional-encoding}



# Postgresql {#postgresql}


#### Installation
[How to set up a Postgres database on your Windows 10 PC](https://www.youtube.com/watch?v=4J0V3AaiOns)

In [Tableau](#tableau) can connect to a database here.

There are plugins.

Spatial objects?

Connections to database tables: hosted

#### Connecting

[Adding a database to PostgreSQL](#adding-a-database-to-postgresql)

#### PGAdmin
pgadmin tools: to check system information

![Pasted image 20250329081752.png](../content/images/Pasted%20image%2020250329081752.png)

# Powerbi {#powerbi}



[tutorial](https://www.youtube.com/watch?v=TmhQCQr_DCA)

Business analytics tool for data visualization and reporting.

# Powershell {#powershell}


Why is Powershell better than [Command Prompt|cmd](#command-promptcmd)?

PowerShell is often considered better than Command Prompt (cmd) for several reasons:

1. **Object-Oriented**: PowerShell is built on the [.NET](#net) framework and works with objects rather than plain text. This allows for more complex data manipulation and easier handling of outputs.

2. **Powerful Scripting Capabilities**: PowerShell supports advanced scripting features, including functions, loops, and error handling, making it more suitable for automation and complex tasks.

3. **Access to .NET Framework**: PowerShell can leverage the full power of the .NET framework, allowing users to utilize a vast array of libraries and functionalities.

4. **Cmdlets**: PowerShell uses cmdlets, which are specialized .NET classes designed to perform specific functions. This makes it easier to perform tasks compared to the simpler commands in cmd.

5. **Remote Management**: PowerShell has built-in capabilities for remote management, allowing users to manage multiple systems from a single console.

6. **Pipeline Support**: PowerShell allows for the use of pipelines to pass objects between cmdlets, enabling more efficient and powerful command chaining.

7. **Integrated Help System**: PowerShell includes a robust help system that can be accessed directly from the command line, making it easier to learn and use.

8. **Cross-Platform**: PowerShell Core (now known as PowerShell 7) is cross-platform, meaning it can run on Windows, macOS, and Linux, unlike cmd, which is Windows-only.

## Scripts

PowerShell interacts with several types of scripts and scripting languages, including:

1. **PowerShell Scripts (.ps1)**: These are the primary script files used in PowerShell. They contain a series of PowerShell commands and can automate tasks.

2. **[Batch Files](#batch-files) ( or .cmd)**: PowerShell can execute traditional Windows batch files, allowing for integration with legacy scripts.

3. **VBScript (.vbs)**: PowerShell can run VBScript files, which can be useful for interacting with older systems or applications that rely on VBScript.

4. **Windows Management Instrumentation (WMI)**: PowerShell can interact with WMI scripts to manage and monitor system resources.

5. **.NET Scripts**: Since PowerShell is built on the .NET framework, it can execute .NET code and interact with .NET assemblies.

6. **Python and Other Scripting Languages**: PowerShell can call scripts written in other languages (like Python) using the appropriate command-line interfaces.

7. **JSON and XML**: PowerShell can parse and manipulate JSON and XML data, which are often used in configuration files and data exchange.


# Powerquery {#powerquery}


[How to normalise a merged table](#how-to-normalise-a-merged-table)


# Powershell Versus Cmd {#powershell-versus-cmd}


PowerShell and Command Prompt (cmd) are both command-line interfaces available on Windows systems, but they differ significantly in their capabilities, syntax, and scripting abilities. Here are the key differences and examples that highlight their distinct features:

### [PowerShell](#powershell):

1. **Object-Oriented Shell:**
   - PowerShell is designed around the concept of objects rather than text streams like cmd. This makes it more powerful for scripting and automation tasks.
   - **Example:** Getting detailed information about files:
     ```powershell
     Get-ChildItem | Select-Object Name, Length, LastWriteTime
     ```
     This command retrieves file objects and selects specific properties (Name, Length, LastWriteTime).

2. **Extensive Commandlets (Cmdlets):**
   - PowerShell includes a wide range of cmdlets for performing specific tasks, such as managing Active Directory, working with files, or interacting with web services.
   - **Example:** Restarting a service:
     ```powershell
     Restart-Service -Name "serviceName"
     ```
     This cmdlet restarts a service named "serviceName".

3. **Advanced Scripting and Automation:**
   - PowerShell supports advanced scripting features, including loops, conditional statements, functions, and error handling.
   - **Example:** Checking if a file exists:
     ```powershell
     if (Test-Path -Path "C:\path\to\file.txt") {
         Write-Output "File exists."
     } else {
         Write-Output "File does not exist."
     }
     ```
     This script uses `Test-Path` cmdlet to check if a file exists and then outputs a message accordingly.

4. **Integration with .NET Framework:**
   - PowerShell can leverage .NET Framework libraries and assemblies directly within scripts.
   - **Example:** Using .NET Framework classes:
     ```powershell
     [System.IO.File]::ReadAllText("C:\path\to\file.txt")
     ```
     This line reads the entire content of a text file using .NET Framework's `File` class.

### [Command Prompt](#command-prompt) (cmd):

1. **Text-Based Command Line:**
   - Cmd operates primarily with text-based commands and outputs, lacking PowerShell's object-oriented approach.
   - **Example:** Listing files in a directory:
     ```
     dir /b
     ```
     This command lists all files in the current directory in a bare format.

2. **Limited Built-in Commands:**
   - Cmd has a more limited set of built-in commands (compared to PowerShell's cmdlets), focusing on basic system commands.
   - **Example:** Copying files:
     ```
     copy C:\source\file.txt D:\destination\
     ```
     This command copies a file from `C:\source` to `D:\destination`.

3. **Batch Scripting:**
   - Cmd uses batch files (.bat) for scripting, which are simpler and less flexible compared to PowerShell scripts.
   - **Example:** Simple batch file to copy files:
     ```
     @echo off
     copy C:\source\file.txt D:\destination\
     ```
     This batch script copies a file without displaying command prompt output.

4. **Direct Command Execution:**
   - Commands in cmd are executed directly without the pipeline and object manipulation features of PowerShell.
   - **Example:** Renaming a file:
     ```
     ren oldfile.txt newfile.txt
     ```
     This command renames a file from `oldfile.txt` to `newfile.txt`.

### Choosing Between PowerShell and cmd:

- **Use PowerShell When:**
  - You need to work with complex data structures or objects.
  - Automation and scripting tasks require advanced features like loops, conditions, and error handling.
  - Integration with .NET Framework or other external libraries is necessary.

- **Use Command Prompt (cmd) When:**
  - Performing basic system tasks or operations.
  - Working with simple text-based outputs.
  - Using legacy batch scripts or when PowerShell is unavailable.

In summary, PowerShell offers a more versatile and powerful environment for scripting, automation, and administrative tasks on Windows systems, while cmd remains useful for straightforward commands and basic system interactions.

# Powershell Vs Bash {#powershell-vs-bash}


The choice between [PowerShell](#powershell) and [Bash](#bash) largely depends on the user's needs and the environment in which they are working. Here are some considerations for each:

### PowerShell
- **Windows Integration**: PowerShell is deeply integrated with Windows and is ideal for managing Windows systems and applications.
- **Object-Oriented**: As mentioned earlier, PowerShell works with objects, making it easier to manipulate data and interact with .NET applications.
- **Remote Management**: It has strong capabilities for remote management of Windows systems.
- **Scripting**: PowerShell's scripting capabilities are robust, making it suitable for complex automation tasks.

### Bash
- **Unix/Linux Environment**: Bash is the default shell for many Unix and Linux systems, making it the go-to choice for system administrators and developers in those environments.
- **Simplicity**: Bash scripts are often simpler and more straightforward for basic tasks, especially for file manipulation and text processing.
- **Community and Resources**: There is a vast amount of community support, tutorials, and resources available for Bash, especially in the open-source community.
- **Cross-Platform**: While traditionally associated with Unix/Linux, Bash can also be used on Windows through [Windows Subsystem for Linux](#windows-subsystem-for-linux) (WSL) or [Git](#git) Bash.

# Precision Or Recall {#precision-or-recall}


[Precision](#precision) and [Recall](#recall) are two fundamental metrics used to evaluate the performance of a [Classification](#classification) model, particularly in binary classification tasks. They are related through a trade-off: improving one often comes at the expense of the other.

Key Differences:
- [Precision](#precision) focuses on the quality of positive predictions.
- [Recall](#recall) focuses on the ability to identify all relevant positive instances

Trade-off:
- It is challenging to optimize both precision and recall simultaneously. Improving precision by reducing false positives may lead to an increase in false negatives, thereby reducing recall, and vice versa.
- However, it's important to balance recall with precision. A model with high recall might also have a higher rate of false positives (non-spam emails incorrectly marked as spam), which can lead to important emails being missed.

Task Dependency Example:
- The choice between prioritizing precision or recall is task-dependent. In a spam classification task, it might be more important to avoid moving important emails to the spam folder (high precision) than to occasionally allow spam emails into the inbox (lower recall). Thus, precision is prioritized over recall in this context.


![Pasted image 20240116211130.png](../content/images/Pasted%20image%2020240116211130.png)


# Precision Recall Curve {#precision-recall-curve}


A [precision](#precision)-[recall](#recall) curve is a graphical representation used to evaluate the performance of a [Binary Classification](#binary-classification) model, particularly in scenarios where the classes are imbalanced. It plots [precision](#precision) (the positive predictive value) against [recall](#recall) (the true positive rate) for different threshold values.

Overall, precision-recall curves are a valuable tool for assessing the tradeoffs between precision and recall, helping to choose the optimal threshold for classification based on the specific requirements of the task.
### Resources

In [ML_Tools](#ml_tools) see: [ROC_PR_Example.py](#roc_pr_examplepy)

[Sklearn Link](https://scikit-learn.org/1.5/auto_examples/model_selection/plot_precision_recall.html)
### Precision Recall Curve:

Plot: The curve is generated by varying the threshold for classifying a positive instance and plotting the corresponding precision and recall values. Each point on the curve represents a precision recall pair at a specific threshold.

Interpretation: 
- Be cautious of the class [Distributions](#distributions) impact on the curve's shape. A curve that appears favorable might still represent poor performance if the positive class is very rare.
- A steep precision-recall curve indicates that the model maintains high precision across a range of recall values, which is desirable. Conversely, a more gradual curve might suggest a trade-off between precision and recall.
- A model with high precision and high recall is considered to perform well. However, there is often a tradeoff between [Precision or Recall](#precision-or-recall).
- The area under the precision-recall curve (not the same as [AUC](#auc)) is a single scalar value that summarizes the performance of the model. A higher AUCPR indicates better model performance.

Use Cases: precision-recall curves are particularly useful in situations where the positive class is rare or when the cost of false positives and false negatives is different. They provide more insight than [ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic) curves in these scenarios because they focus on the performance of the positive class.
 
![Pasted image 20241231172749.png](../content/images/Pasted%20image%2020241231172749.png)

## Other features

Multi-Class Scenarios
- **Adapting for Multi-Class Problems**: Precision-recall curves can be extended to multi-class classification problems by using strategies like one-vs-rest (OvR). In this approach, a separate precision-recall curve is computed for each class, treating it as the positive class while considering all other classes as negative.

Comparison with [ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic)
- **When to Use Precision-Recall Curves**: Precision-recall curves are particularly useful over ROC curves in scenarios with highly [imbalanced datasets](#imbalanced-datasets). They provide a more informative picture of a model's performance by focusing on the positive class, which is often the minority class in imbalanced datasets.


# Precision {#precision}


**Precision Score** is a metric used to evaluate the [Accuracy](#accuracy) of a [Classification](#classification) model, specifically focusing on the positive class.

<mark>How many retrieved items are relevant?</mark>

This metric indicates the accuracy of positive predictions. The formula for precision is:

$$\text{Precision} = \frac{TP}{TP + FP}$$

where:
- **TP (True Positives):** The number of correctly predicted positive instances.
- **FP (False Positives):** The number of instances incorrectly predicted as positive.

Importance:
- Precision is crucial in scenarios where the **cost of false positives is high**, such as in spam detection or medical diagnosis. It helps in understanding how many of the predicted positive instances are actually positive.

![Pasted image 20241222091831.png](../content/images/Pasted%20image%2020241222091831.png)

## Related Concepts

- [Classification Report](#classification-report) & [Precision or Recall](#precision-or-recall)






# Prediction Intervals {#prediction-intervals}


Prediction intervals estimate the range within which a future observation from the same distribution is likely to fall, with a specified confidence level.

**Formula**:

$$\bar{x} \pm t_{\alpha/2, n-1} \cdot s \cdot \sqrt{1 + \frac{1}{n}}$$

**Where**:
- $\bar{x}$: Sample mean
- $s$: Sample standard deviation
- $n$: Sample size
- $t_{\alpha/2, n-1}$: t-critical value for the chosen confidence level

**Notes**:
- Prediction intervals are always **wider** than a confidence interval for the mean.
- They use the t-distribution due to sample uncertainty.
- The interval is centered around $\bar{x}$ but accounts for:
    - **Estimation error** of the mean
    - **Natural variability** of new values

**Use Cases**:
- Forecasting where a new measurement is likely to fall.
- Risk assessment and operational thresholds.


# Preprocessing {#preprocessing}


### Data Preprocessing

Data Preprocessing refers to the overall process of cleaning and transforming raw data into a format that is suitable for analysis and modelling. This includes a variety of tasks, such as:

Useful:
- [Data Cleansing](#data-cleansing)
- [Data Transformation](#data-transformation)

Others:
- [Data Collection](#data-collection)
- [Data Reduction](#data-reduction)

End goal:
- [EDA](#eda)

### Feature Preprocessing

Feature preprocessing refers to the process of transforming raw data into a clean data set for learning models, after Data Preprocessing. This step is crucial for improving model performance and ensuring accurate predictions

2. **[Feature Scaling](#feature-scaling)**: Normalizing or standardizing features to ensure they are on a similar scale. Normalization and Scaling: Adjusting the range of features, often using techniques like min-max scaling or z-score normalization, to ensure that all features contribute equally to the model.

4. [Feature Selection](#feature-selection): Identifying and retaining the most relevant features that contribute to the predictive power of the model, often using statistical tests or model-based approaches.

5. [Dimensionality Reduction](#dimensionality-reduction): Reducing the number of features while preserving important information, using techniques like Principal Component Analysis (PCA).

6. [Feature Engineering](#feature-engineering): Creating new features from existing data to improve model performance, often based on domain knowledge.





# Prevention Is Better Than The Cure {#prevention-is-better-than-the-cure}

To ensure data products are effective essential to prioritize prevention over remediation of [Data Quality](#data-quality)
### Prevention
Preventing data quality issues is the most effective strategy. This involves identifying and addressing potential problems at the source, ensuring that data is accurately entered from the beginning.

### Remediation
When data quality issues do arise, organizations should implement remediation strategies, including:
- **[Data Observability](#data-observability) Tools**: Monitor data quality continuously to detect issues early.
- **Alerting Systems**: Notify stakeholders when data quality problems are identified.
- **Complex ETL Processes**: Manage data effectively to minimize errors.
- **Trust Building**: Address the erosion of trust that can result from poor data quality.

### Consequences of Poor Data Quality
Failing to address data quality issues can lead to significant opportunity costs and hinder the ability to meet business goals. The sooner these issues are resolved, the cheaper and easier it is to manage them.

### Motivating and Maintaining Data-Driven Value

To foster a culture of data quality, it is essential to motivate data producers by demonstrating the value of high-quality data. 

Effective [Change Management](#change-management) is vital for maintaining data quality. This includes: Clear Communication to ensure all stakeholders are informed about data quality standards.

### Addressing Data Quality Issues

To effectively handle data quality issues, organizations should focus on:
1. **Detecting**: Identify issues as they arise through user reports, failed tests, or monitoring alerts.
2. **Understanding**: Analyze the root causes of data quality problems.
3. **Fixing**: Implement solutions to correct identified issues.
4. **Reducing**: Minimize the occurrence of future data quality problems.

### Questions for Consideration
**Q:** What if data producers are not part of the data team but are business users (e.g., entering data into Google Sheets or Excel) with naming convention issues?  
**A:** Encourage these users to apply the same data quality patterns by establishing agreements on data structure and implementing alerting and automated change management processes.



# Primary Key {#primary-key}

A primary key (PK) is a unique identifier for each record in a database table.

- **Uniqueness**: No two records can have the same primary key value.
- **Non-null**: A primary key cannot contain null values; every record must have a valid primary key.
- **Immutability**: Ideally, the primary key should not change over time, as it serves as a stable reference for the record

For example, an ISBN serves as a primary key for books, uniquely identifying each book in the database

# Principal Component Analysis {#principal-component-analysis}


PCA is a tool for [Dimensionality Reduction](#dimensionality-reduction) in [Unsupervised Learning](#unsupervised-learning) to reduce the dimensionality of data. 

It transforms the original data into a new coordinate system defined by the principal components, which are <mark>orthogonal vectors</mark> that capture the most [Variance](#variance) in the data.

It helps simplify models, enhances [interpretability](#interpretability), and improves computational efficiency by transforming data into a lower-dimensional space while <mark>retaining the most significant variance</mark>, and reducing noise.


### How PCA Works

- <mark>Linear Technique</mark>: PCA is a linear technique, meaning it assumes that the relationships between features are linear. This distinguishes it from methods like [Manifold Learning](#manifold-learning) which can capture non-linear relationships.

- Principal Components: PCA identifies principal components, which are linear combinations of the original features. These components are ordered by the amount of variance they capture from the data, with the first principal component capturing the most variance, the second capturing the second most, and so on.
### Comparison with Other Techniques

- [t-SNE](#t-sne): Unlike PCA, t-SNE is a non-linear technique used for visualization, preserving local structures in high-dimensional data.
- [Manifold Learning](#manifold-learning): Techniques like Isomap and Locally Linear Embedding (LLE) are designed to capture non-linear structures, which PCA might miss due to its linear nature
### Code Implementation

In [ML_Tools](#ml_tools) see:
- [PCA-Based Anomaly Detection](#pca-based-anomaly-detection)
- [PCA_Analysis.ipynb](#pca_analysisipynb)

### Related terms

- [PCA Explained Variance Ratio](#pca-explained-variance-ratio)
- [PCA Principal Components](#pca-principal-components)

# Probability In Other Fields {#probability-in-other-fields}



# Problem Definition {#problem-definition}


# What is involved:

Clearly articulate the problem you're trying to solve and the outcomes you expect.

## Follow up questions
What assumption can we make based on the problem?

# What kind of questions are good to ask?

**Business Context:**

- What are the desired outcomes and how would success be measured?
- What are the limitations and feasibility of using machine learning in this context?

**2. Data Availability and Quality:**

- What data is available in quantity and quality and relevant to the problem?
- What is the format and structure of the data?

**3. Feature Engineering and Model Selection:**

- What are the key features or variables that might be predictive of the desired outcome?
- What type of machine learning model might be best suited for this problem (e.g., classification, regression, [Clustering](#clustering))?

**4. Evaluation and Deployment:**

- How will we evaluate the performance of the machine learning model?
- What metrics will be used to measure success?



# Prompt Engineering {#prompt-engineering}


Prompt engineering is a technique in the field of natural language processing (NLP), particularly when working with large language models (LLMs). 

It involves designing and optimizing input prompts to get the most relevant and accurate responses from these models. 

Techniques like [prompt retrievers](#prompt-retrievers), which include systems like UPRISE and DaSLaM, enhance the ability to retrieve and generate contextually appropriate prompts.

Prompt engineering aims to guide LLMs toward producing desired outputs while minimizing ambiguity. 

### Key Takeaways

- Prompt engineering optimizes input to improve LLM responses.
- Techniques like prompt retrievers (e.g., UPRISE, DaSLaM) enhance prompt effectiveness.
- Quality prompts reduce ambiguity and guide model outputs.
- Applications span multiple industries, enhancing user interaction and content generation.

### Key Components Breakdown

**Methods**: 
  - **Prompt Design**: Crafting specific, clear prompts to guide model responses.
  - **Prompt Retrieval**: Utilizing systems like UPRISE and DaSLaM to find effective prompts based on context.
  
**Concepts**:
  - **Contextualization**: Understanding the context in which prompts are used to improve relevance.
  - **Iterative Testing**: Continuously refining prompts based on model performance.

**Algorithms**:
  - **Retrieval-Augmented Generation (RAG)**: Combines retrieval of relevant documents with generative responses.
  - **Few-Shot Learning**: Providing examples within prompts to guide model behavior.

### Concerns, Limitations, or Challenges
- **Ambiguity**: Poorly designed prompts can lead to vague or irrelevant responses.
- **Dependence on Training Data**: LLMs may produce biased or inaccurate outputs based on their training data.
- **Complexity**: Designing effective prompts requires a deep understanding of both the model and the task.

### Example
For instance, if a user wants to generate a summary of a scientific article, a poorly constructed prompt like "Summarize this" may yield unsatisfactory results. In contrast, a well-engineered prompt such as "Provide a concise summary of the key findings and implications of the following article on climate change" is likely to produce a more relevant and informative response.

### Follow-Up Questions
1. [What are the best practices for evaluating the effectiveness of different prompts](#what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts)
2. [How can prompt engineering be integrated into existing NLP workflows to enhance performance](#how-can-prompt-engineering-be-integrated-into-existing-nlp-workflows-to-enhance-performance)

### Proportion Test

The proportion test is used to compare proportions between groups. It can be categorized into:
- **One-Sample Proportion Test**: Compares the proportion of successes in a single sample to a known population proportion.
- **Two-Sample Proportion Test**: Compares the proportions of successes between two independent samples.

# Publish And Subscribe {#publish-and-subscribe}


The **Publish-Subscribe (Pub-Sub) model** is a messaging pattern that enables real-time data distribution by decoupling message producers from consumers. This architecture is widely used in [data streaming](#data-streaming), [Event-Driven Architecture](#event-driven-architecture), and [Distributed Computing](#distributed-computing).

It can help in designing more efficient and [Scalability](#scalability) and data processing architectures.

### Core Components

This model ensures that multiple consumers can receive the same data without requiring direct connections between producers and consumers, allowing for a more scalable and flexible system.

- **Producers**: Entities or applications that generate data and publish messages to specific channels known as **Topics**. Each Topic represents a category or stream of data.

- **Consumers**: Applications or services that subscribe to Topics to receive messages. They process incoming data in real-time, enabling immediate action or analysis.
### Importance in Data Streaming

Ensures continuous data flow, in contrast to [batch processing](#batch-processing), which collects and processes data in groups at scheduled intervals. Streaming applications benefit from Pub-Sub by:

- Enabling real-time analytics and monitoring
- Supporting event-driven architectures
- Improving scalability by decoupling message producers from consumers

### Questions for Consideration

- If you’re working with a streaming dataset, why might [batch processing](#batch-processing) not be suitable, and what alternatives would you consider?
- How does the decoupling of producers and consumers improve scalability in large-scale data systems?
- What are the trade-offs between a Pub-Sub model and a point-to-point messaging system?

### Example: [Apache Kafka](#apache-kafka)

1. **Producers**: In a Kafka setup, a producer could be a web application that generates user activity events, such as clicks, page views, or purchases. This application publishes these events to a specific topic, for example, "user-activity".

2. **Topics**: The "user-activity" topic acts as a channel where all user activity events are sent. Multiple producers can publish messages to this topic without needing to know about the consumers.

3. **Consumers**: Various applications or services can subscribe to the "user-activity" topic to receive real-time updates. For instance:
   - An analytics service that processes user activity data to generate insights.
   - A notification service that sends alerts based on specific user actions (e.g., sending a welcome email after a user signs up).
   - A monitoring service that tracks user engagement metrics.

### Workflow:
- When a user interacts with the web application, the producer generates an event and publishes it to the "user-activity" topic.
- All subscribed consumers receive this event simultaneously, allowing them to process the data in real-time.
- This decoupling means that the producer does not need to know how many consumers are listening or what they are doing with the data.



## Tl;dr

_1-liner if the context of the change is long_

## Context

_A few sentences on the high level context for the change. Link to relevant design docs or discussion._

## This Change

_What this change does in the larger context. Specific details to highlight for review. Include UI change screenshots or videos if applicable:_

- _Callout 1_
    
- _Callout 2_
    
- _Callout 3_
    

## Test Plan

_Go over how you plan to test it. Your test plan should be more thorough the riskier the change is. For major changes, I like to describe how I E2E tested it and will monitor the rollout._

## Links

- _link to ticket_
    
- _link to design doc_
    
- _link to design_
    

## Checklist

- Pull request title is succinct with [tiny] if it’s extra small
    
- Describes the problem
    
- Describes the solution (screenshots included if UI changes)
    
- Has a test plan
    
- Contains links to any context (Slack, Figma, JIRA ticket, etc.)
    
- Code is self reviewed for readability, approach, and edge cases
    
- Lines changed that may require additional explanation are annotated with an explanation
    
- Change is ideally < 500 lines if possible. < 150 is ideal.

# Push Down {#push-down}


Query pushdown aims to execute as much work as possible in the source databases. 

Push-downs or query pushdowns push transformation logic to the source database. This reduces to store data physically and transfers them over the network. 

For example, a [semantic layer](semantic%20layer.md) or [Data Virtualization](Data%20Virtualization.md) translates the transformation logic into [SQL](SQL.md) [Querying|queries](#queryingqueries) and sends the SQL queries to the database. The source database runs the SQL queries to process the transformations.

Pushdown optimization increases mapping performance when the source database can process transformation logic faster than the semantic layer itself. 


# Pycaret {#pycaret}

PyCaret is an open-source, low-code Python library designed to simplify machine learning workflows. 

It allows users to build, evaluate, and deploy machine learning models with minimal coding and effort. 

PyCaret provides an end-to-end solution for automating repetitive tasks in machine learning, such as 
- [Preprocessing](#preprocessing),
- model training,
- [hyperparameter](#hyperparameter) tuning
- [Model Deployment|Deployment](#model-deploymentdeployment)

### Implementation

See: https://pycaret.gitbook.io/docs/get-started/quickstart

Resources: https://github.com/pycaret/pycaret/tree/master

In [ML_Tools](#ml_tools) see: 
- [Pycaret_Example.py](#pycaret_examplepy)
### Key Features of PyCaret

1. Ease of Use: PyCaret is designed to be beginner-friendly, enabling users to build models without deep expertise in coding.
2. Modular Design: PyCaret supports various machine learning tasks through its modular APIs:
    - Classification: `pycaret.classification`
    - Regression: `pycaret.regression`
    - Clustering: `pycaret.clustering`
    - Anomaly Detection: `pycaret.anomaly`
    - NLP: `pycaret.nlp`
    - Time Series Forecasting: `pycaret.time_series`
3. Automated Machine Learning (AutoML): PyCaret automates data preprocessing, feature engineering, model selection, and [hyperparameter tuning](#hyperparameter-tuning).
4. Integration: PyCaret integrates well with other Python libraries, such as Pandas, NumPy, and Plotly.
5. Model Evaluation and Comparison: [Model Selection](#model-selection): It provides an easy way to compare multiple models and their performance metrics in a single function call.
6. Deployment [Model Deployment](#model-deployment): Facilitates the deployment of trained models using tools like Flask, FastAPI, or Microsoft Power BI.

### Notes

Object or functional APIs



### Advantages of PyCaret

- Time-Saving: Reduces the coding and time required to build machine learning pipelines.
- Quick prototyping of machine learning models.
- Educational purposes for teaching machine learning concepts.
- Rapid development of machine learning solutions for business problems.

# Pygraphviz {#pygraphviz}

PyGraphviz
Interface: Thin wrapper around the C Graphviz API.
Better integration with NetworkX, especially with graphviz_layout.

Advantages:

Native Graphviz object model (AGraph).

Seamless conversion between NetworkX graphs and Graphviz objects.

Supports advanced Graphviz features and layout options.

Limitations:

Requires Graphviz development libraries to be installed (can be hard to set up on Windows).

Slightly more complex installation due to C bindings.

Example with NetworkX:

python
from networkx.drawing.nx_agraph import graphviz_layout
pos = graphviz_layout(G, prog="dot")

# Pyspark {#pyspark}



<mark style="background: #FF5582A6;">Python API </mark>for [Apache Spark](#apache-spark), a <mark>distributed processing framework</mark> for big data analysis and machine learning on clusters.

Part of [Apache Spark](#apache-spark)

[Directed Acyclic Graph (DAG)](#directed-acyclic-graph-dag)

Interlinked with [SQL](#sql) queriers

[Tutorial](https://www.youtube.com/watch?v=WyZmM6K7ubc)

Can run local.

Similar to [Pandas](#pandas) 





# Pytorch {#pytorch}


[Text Generation With LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/text-generation-lstm-recurrent-neural-networks-python-keras/) want for [PyTorch](#pytorch)

Open-source [Deep Learning](#deep-learning) framework with dynamic computational graphs, emphasizing flexibility and research. Similar to [Tensorflow](#tensorflow).

Framework pieces:
- torch: a general purpose array library similar to [Numpy](#numpy) that can do computations on GPU when the tensor type is cast to (torch.cuda.TensorFloat)
- torch.autograd: a package for building a computational graph and automatically obtaining gradients
- torch.nn: a [Neural network|Neural Network](#neural-networkneural-network) library with common layers and [Loss function](#loss-function)
- torch.optim: an optimization package with common optimization algorithms like [Stochastic Gradient Descent](#stochastic-gradient-descent)

# Basics of [PyTorch](#pytorch)

### Tensors arrays 

PyTorch uses tensors, which are similar to NumPy arrays, but with GPU acceleration.

```python
import torch
# Creating a tensor
x = torch.tensor([2.0, 3.0, 4.0])
# Tensor operations
y = x + 2
z = x * 3
print(y)  # Output: tensor([4., 5., 6.])
print(z)  # Output: tensor([ 6.,  9., 12.])
```

### Automatic Differentiation  
PyTorch can compute gradients automatically with `autograd`.

```python
# Create tensor with gradient tracking (input value)
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) #`requires_grad=True`, which tells PyTorch to track all operations performed on this tensor. T

# Perform some operations
y = x ** 2 #formula
z = y.sum() #14

# Compute gradients
z.backward() #backpropagation (the chain rule of calculus). Since `z` is a scalar, PyTorch can compute the gradients of `z` with respect to each element in `x`.

# Print gradients (dy/dx)
print(x.grad)  # Output: tensor([2., 4., 6.])
```

The gradient is calculated by the chain rule:
- For $y = x^2$, the derivative of $y$ with respect to $x$ is $\frac{dy}{dx} = 2x$.
- ==Since `z` is the sum of the elements of `y`, the gradient of `z` with respect to each element in `x` is $\frac{dz}{dx_i} = 2x_i$ for each element in `x`.<mark>

</mark>z is a formula but calculates the derivative wrt x, so the derivative of y with x.==

So, the gradients for each element in `x` are:
$\frac{dz}{dx} = [2 \times 1.0, 2 \times 2.0, 2 \times 3.0] = [2.0, 4.0, 6.0]$

The gradients are stored in `x.grad`. After calling `z.backward()`, `x.grad` contains the derivative of `z` with respect to each element of `x`, which is `[2.0, 4.0, 6.0]`

Gradient Computation (Summary):
- The gradient $\frac{dz}{dx}$ represents how much the output `z` changes for a small change in `x`. In our case, if we slightly increase `x_1`, `x_2`, or `x_3`, the change in `z` can be predicted using these gradients.
- This gradient information is used to update weights in neural networks during training. For example, in optimization algorithms like ([Stochastic Gradient Descent](#stochastic-gradient-descent)), these gradients are used to adjust model parameters in the direction that minimizes the loss function.
###### Confusion: total derivative is not the partial derivates

The confusion here comes from the distinction between **partial derivatives** (for each element in a tensor) and **total derivatives**. Let me clarify this.

In the context of:

```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
z.backward()
```

- `x` is a tensor: $ x = [1.0, 2.0, 3.0] $
- `y = x^2` computes element-wise squares: $ y = [1.0^2, 2.0^2, 3.0^2] = [1.0, 4.0, 9.0] $
- `z = y.sum()` adds the elements in `y`: $ z = 1.0 + 4.0 + 9.0 = 14.0 $

Now, you're asking about the derivative of `z` with respect to `x`, specifically whether **`dz/dx` should be the sum of the derivatives** and equal to 4.

**Derivative with Respect to Each Element (Partial Derivative)**

When you calculate the gradient of `z` with respect to `x`, you're computing the **partial derivatives** of `z` with respect to each element in `x`. The function `z` depends on each element of `x` individually.

For each element $x_i $ in `x`, we are calculating:

$\frac{\partial z}{\partial x_i} = \frac{\partial (x_1^2 + x_2^2 + x_3^2)}{\partial x_i} = 2x_i$

This gives:

$\frac{\partial z}{\partial x} = [2x_1, 2x_2, 2x_3] = [2 \times 1.0, 2 \times 2.0, 2 \times 3.0] = [2.0, 4.0, 6.0]$

These are the gradients stored in `x.grad` after calling `z.backward()`.

**Total Derivative vs Partial Derivatives**

If we are talking about **partial derivatives**, we get a gradient for each individual component of `x`:

- $ \frac{\partial z}{\partial x_1} = 2.0$
- $ \frac{\partial z}{\partial x_2} = 4.0$
- $ \frac{\partial z}{\partial x_3} = 6.0$

These partial derivatives form the gradient vector: `[2.0, 4.0, 6.0]`.

**Why Is It Not Just 4?**

If you're thinking of the total derivative, that would be different from what we are calculating here. **The sum of the derivatives of `z` with respect to all components of `x`** is:

$
\sum_{i=1}^{3} \frac{\partial z}{\partial x_i} = 2 + 4 + 6 = 12
$

However, this total derivative is not what we are computing here. **We are computing the partial derivatives for each element of `x` separately**, which results in the gradient vector `[2.0, 4.0, 6.0]`.

In Summary:
- **Gradient (`x.grad`)**: A vector of partial derivatives of `z` with respect to each element in `x`, giving us `[2.0, 4.0, 6.0]`.
- **Sum of Gradients**: The sum of the elements of the gradient vector is `12`, but that’s not the gradient of `z` with respect to `x` as a whole—it's just a summation of the partial derivatives.
### Basic [Neural network|Neural Network](#neural-networkneural-network) Implementation 

A simple feedforward network using PyTorch.

```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
    #- **`forward()` method**: This defines the forward pass, i.e., how the input data is transformed as it moves through the network. In this case, the input `x` is passed through the linear layer.
        return self.fc(x)

# Create the network
net = SimpleNet()

# Create input tensor
x = torch.tensor([1.0, 2.0](#10-20))

# Forward pass
output = net(x)

print(output)  # Output: a tensor from the linear layer
```

**Input Tensor and Forward Pass**
```python
x = torch.tensor([1.0, 2.0](#10-20))
output = net(x)
```
- **`torch.tensor([1.0, 2.0](#10-20))`**: This defines a 2D input tensor with two features, which corresponds to the two input nodes in the network.
- **`net(x)`**: This performs a **forward pass**, feeding the input tensor `x` into the network. The linear layer applies the learned weights and bias to compute the output.

**Output**
The output of the network is a tensor from the linear layer, which corresponds to the result of the operation $y = w_1 \cdot x_1 + w_2 \cdot x_2 + b$, where:
- $w_1$ and $w_2$ are the learned weights for each input feature.
- $b$ is the learned bias.

[Use Cases for a Simple Neural Network Like](#use-cases-for-a-simple-neural-network-like)

### Training a Simple Model  
An example of training a linear model with

- [Gradient Descent](#gradient-descent):
- [Loss function](#loss-function): [Ordinary Least Squares](#ordinary-least-squares)
- [Stochastic Gradient Descent|SGD](#stochastic-gradient-descentsgd):

Training of [Linear Regression](#linear-regression) model. This model find best w,b in $y=wx+b$.

```python
import torch
import torch.optim as optim

# Data
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# Model
model = nn.Linear(1, 1) #**`nn.Linear(1, 1)`** defines a simple linear model with **one input feature** and **one output**. This model has two parameters:

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # stochastic gradient descent

# Training loop
for epoch in range(100):
    # Forward pass #- **Forward pass**: For each epoch, the model makes predictions (`y_pred`) by passing the input `x` through the linear model.
    y_pred = model(x)
    loss = criterion(y_pred, y)# - **Loss calculation**: The loss is computed by comparing `y_pred` with the true values `y` using the MSE loss function.

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward() #- **Backward pass**: The gradients of the loss with respect to the model’s parameters are computed using `loss.backward()`. This step calculates how much each parameter needs to be adjusted to reduce the loss.
    optimizer.step() #- **Optimization step**: The optimizer (`SGD`) updates the model’s parameters (`w` and `b`) based on the computed gradients.

    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Output the trained model's parameters
print(list(model.parameters()))
```

### Moving Tensors to GPU  

Summary:
- speed up training e.g. of [Neural network|Neural Network](#neural-networkneural-network)
- use [Parallelism](#parallelism) for simultaneous calculations
- GPU can do larger batches of computations, better on the memory, better for [Gradient Descent](#gradient-descent) estimations
- 

PyTorch makes it easy to move computations to a GPU.

```python
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a tensor and move it to the GPU
x = torch.tensor([1.0, 2.0, 3.0]).to(device)

print(x)  # Output: tensor([1., 2., 3.], device='cuda:0') (if GPU is available)
```

Both the **model** and the **data** are moved to the GPU (`device='cuda'`). All computations, including the forward pass, loss calculation, backward pass, and optimizer step, happen on the GPU.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN().to(device)  # Move model to GPU
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input data and labels (move them to GPU)
inputs = torch.randn(64, 1, 28, 28).to(device)  # 64 images of 28x28 pixels
labels = torch.randint(0, 10, (64,)).to(device)  # Random labels for 64 images

# Forward pass (computation happens on the GPU)
outputs = model(inputs)
loss = criterion(outputs, labels)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()  # Compute gradients on the GPU
optimizer.step()  # Update model weights

print("Training step completed on:", device)
```

# Pycaret_Anomaly.Ipynb {#pycaret_anomalyipynb}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Outliers/Pycaret_Anomaly.ipynb



# Pycaret_Example.Py {#pycaret_examplepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/Pycaret/Pycaret_Example.py



# Pydantic {#pydantic}

Pydantic is a Python library used for [data validation](#data-validation) and settings management using Python type annotations.

It provides a way to define data models with type hints, and it automatically validates and parses input data to ensure it matches the specified types. 

Pydantic is often used in applications where data integrity is crucial, such as in web APIs, configuration management, and data processing pipelines. It helps developers catch errors early by enforcing type constraints and providing clear error messages when data does not conform to the expected format.

In [ML_Tools](#ml_tools) see:
- [Pydantic.py](#pydanticpy)
- [Pydantic_More.py](#pydantic_morepy)
### What Pydantic Does:

1. **[Data Validation](#data-validation) and Parsing**  
    Pydantic takes raw input data (e.g., dictionaries, JSON) and validates it against the [type checking](#type-checking) types and constraints defined in a `BaseModel`. If the input data doesn't match the requirements (e.g., wrong type, missing fields, invalid values), it raises a `ValidationError`.
    
    Example:
    
    ```python
    from pydantic import BaseModel
    
    class User(BaseModel):
        id: int
        name: str
    user = User(id="1", name="Alice")  # Automatically parses "1" to integer.
    ```
    
2. **Automatic Type Conversion**  
    Pydantic can coerce compatible types into the expected type. For example, if a field expects an `int` but receives a string `"123"`, it will try to convert it to an integer.
    
3. **Error Messaging**  
    If validation fails, Pydantic provides detailed error messages explaining what went wrong, making debugging easier.
    
4. **Nested and Complex Data Models**  
    Pydantic supports nested models and complex data structures, enabling you to handle hierarchical data easily.
    
5. **Settings Management**  
    Pydantic can load configuration from environment variables or other sources using its `BaseSettings` class, making it handy for managing application settings.
    
6. **Serialization and Deserialization**  
    Pydantic models support converting data into [JSON](#json) or dictionaries, making it easy to work with web APIs or store validated data.

### Key Advantages of Pydantic:

1. **Type-Safe Programming**:  
    By relying on Python’s type hints, Pydantic promotes better coding practices and helps prevent runtime errors.
    
2. **Ease of Use**:  
    Pydantic abstracts a lot of the boilerplate code you'd write manually for validating and parsing data.
    
3. **Error Reporting**:  
    Pydantic provides clear and structured error messages, making [debugging](#debugging) simpler.
    
4. **Interoperability**:  
    It works well with libraries like [FastAPI](#fastapi), where it powers request/response validation and serialization.

### Use Cases:

1. **Web APIs**:  
    Validating incoming HTTP requests and outgoing responses (e.g., with FastAPI).
2. **Data Processing**:  
    Ensuring raw input data from files or APIs meets requirements before processing.
3. **Configuration Management**:  
    Validating and loading application settings from environment variables or files.
4. **Data Pipelines**:  
    Verifying the integrity of data as it moves through pipeline stages.

### Analogy to Summarize:

Think of Pydantic as a **data traffic cop**. It stands at the intersection where raw data enters your application, ensuring that:

- The data is well-formed.
- It complies with rules you’ve set (type, format, constraints).
- It’s transformed into the expected structure (if possible).

By using Pydantic, you focus on defining the rules, and it ensures the data fits them—saving you from writing repetitive validation code.

### Is Pydantic [Object-Oriented Programming](#object-oriented-programming) (OOP)?

While Pydantic uses classes and inheritance (features of OOP), it is **not purely OOP in intent or design**. Instead, it is:

- **Data-centric**: Focused on defining and validating data structures rather than encapsulating behavior like traditional OOP.
- **Declarative**: Pydantic models are [declarative](#declarative) in nature. You define the "shape" of your data (fields and their types) and rely on Pydantic to handle validation, parsing, and serialization.

#### Differences from Typical OOP:

- **Behavior vs. Structure**:  
    Traditional OOP often centers on defining behavior (methods) alongside data. Pydantic, on the other hand, prioritizes defining and validating data.
- **State Management**:  
    In OOP, objects encapsulate their state and methods for interacting with it. Pydantic models are more lightweight and focused on validation, not managing stateful objects.

#### Similarities to OOP:

- **Class-Based Models**:  
    Pydantic models are Python classes, and you can use inheritance, encapsulation, and even add methods to your models.
- **Reusability**:  
    You can define base models and extend them, similar to class inheritance in OOP.

# Pydantic.Py {#pydanticpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Pydantic.py
### Explanation:

- **BaseModel**: This is the base class for creating data models in Pydantic. You define your model by subclassing `BaseModel` and specifying fields with type annotations.
- **Optional**: Used to indicate that a field is optional.
- **List**: Used to specify a list of items, in this case, a list of strings for friends.
- **Validator**: A custom validator is used to enforce additional constraints, such as ensuring the age is positive.
- **ValidationError**: This exception is raised when the input data does not conform to the model's constraints.

This script demonstrates how Pydantic can be used to validate and parse data, ensuring it meets the specified types and constraints.





# Pydantic_More.Py {#pydantic_morepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Pydantic_More.py
### Key Features Demonstrated in the Script:

1. **Nested Models:** Use of the `Friend` model inside the `User` model.
2. **Custom Validators:** Validating `age` and `email` fields with specific logic.
3. **Dynamic Defaults:** Using `datetime.now` for `created_at`.
4. **Field Aliases:** Supporting different key names during parsing and serialization.
5. **Configuration Options:** Stripping whitespace and enabling strict typing.
6. **Model Inheritance:** Extending the `User` model to create an `AdminUser` model.
7. **Parsing Raw Data:** Demonstrating `parse_raw` for JSON strings.

# Pyright Vs Pydantic {#pyright-vs-pydantic}

While [Pyright](#pyright) and [Pydantic](#pydantic) serve different roles in Python development, they complement each other well. 

Pyright helps ensure that the code adheres to <mark>type constraints before execution</mark>, while Pydantic ensures that the <mark>data being processed</mark> adheres to the expected types and formats during runtime. 

### Key Differences

1. **Purpose**:
   - **Pyright** is aimed at improving code quality through static analysis and type checking.
   - **Pydantic** is focused on runtime [data validation](#data-validation), ensuring that the data conforms to specified types and constraints.

2. **Functionality**:
   - **Pyright** checks for type errors and enforces type hints during development, preventing potential issues before the code is executed.
   - **Pydantic** validates and parses data at runtime, providing clear error messages when data does not conform to the expected format.

3. **Use Cases**:
   - **Pyright** is beneficial in any Python project where type safety is desired, especially in large codebases.
   - **Pydantic** is particularly useful in applications that require data validation, such as web frameworks (e.g., [FastAPI](#fastapi)) and data processing pipelines.

### Key Similarities

4. **Type Annotations**:
   - Both utilize Python's type hints to define and enforce types, promoting better coding practices and reducing runtime errors.

5. **Error Handling**:
   - Both tools provide mechanisms for error reporting, although they do so at different stages (compile-time for Pyright and runtime for Pydantic).

6. **Improving Code Quality**:
   - Both contribute to overall code quality and maintainability, albeit through different approaches—Pyright through static analysis and Pydantic through runtime validation.




# Pyright {#pyright}


Pyright is a <mark>static type checker</mark> for Python that enhances code reliability by enforcing type constraints <mark>at compile-time.</mark>

It utilizes type hints to identify potential errors, such as type mismatches, before runtime, thereby improving code robustness. 

Pyright significantly reduces runtime errors by enforcing type constraints at compile-time.

The use of type hints in Pyright improves code readability and maintainability, serving as [Documentation & Meetings](#documentation--meetings) for function signatures.

### Related Topics

- Type inference in programming languages
- The role of type systems in [functional programming](#functional-programming)
- [Debugging](#debugging)
- [Maintainable Code](#maintainable-code)
- [type checking](#type-checking)

### Follow up questions

- How does the inclusion of Pyright impact the performance of large-scale Python applications?
- What are the trade-offs between using Pyright and other static type checkers in terms of accuracy and speed?

### **`@pytest.fixture` Explanation**

`@pytest.fixture` is a decorator in `pytest` used to define reusable test setup functions. It allows tests to use shared resources without redundant code.

#### **Example & Usage**

python

Copy code

`import pytest  @pytest.fixture def sample_data():     return {"name": "John Doe", "age": 30}      def test_example(sample_data):     assert sample_data["name"] <mark> "John Doe"     assert sample_data["age"] </mark> 30`

🔹 **How it works:**

- The function `sample_data()` is decorated with `@pytest.fixture`, making it a fixture.
- The test function `test_example()` receives `sample_data` as an argument.
- `pytest` automatically provides the fixture data when running the test.

#### **Why use fixtures?**

- Avoids repetitive setup code.
- Ensures clean test environments.
- Can handle resource management (e.g., opening/closing database connections, creating temporary files).

# Python Click {#python-click}


Python Click, or "Command Line Interface Creation Kit," is a library for building command-line interfaces (CLIs). It supports arbitrary nesting of commands, automatic help page generation, and lazy loading of subcommands. 

In [ML_Tools](#ml_tools) see: [Click_Implementation.py](#click_implementationpy)
## Installation

To install Click, use pip:

```sh
pip install click
```

## Creating a Command Group

Click uses groups to organize related commands. A group serves as a container for multiple commands.

```python
import click
import json

@click.group("cli")
@click.pass_context
@click.argument("document")
def cli(ctx, document):
    """An example CLI for interfacing with a document"""
    with open(document) as _stream:
        _dict = json.load(_stream)
    ctx.obj = _dict

def main():
    cli(prog_name="cli")

if __name__ == '__main__':
    main()
```

Running `python script.py --help` generates an automatic help page.

## Adding Commands

Commands can be added to a Click group using the `@<group>.command` decorator.

### Checking Context Object

```python
import pprint

@cli.command("check_context_object")
@click.pass_context
def check_context(ctx):
    pprint.pprint(type(ctx.obj))
```

### Custom Pass Decorator

A pass decorator allows passing specific objects through context.

```python
pass_dict = click.make_pass_decorator(dict)
```

### Retrieving Keys from Context

```python
@cli.command("get_keys")
@pass_dict
def get_keys(_dict):
    keys = list(_dict.keys())
    click.secho("The keys in our dictionary are", fg="green")
    click.echo(click.style(str(keys), fg="blue"))
```

### Retrieving a Specific Key

```python
@cli.command("get_key")
@click.argument("key")
@click.pass_context
def get_key(ctx, key):
    if key in ctx.obj:
        pprint.pprint(ctx.obj[key])
    else:
        click.echo(f"Key '{key}' not found in document.", err=True)
```

### Arbitrary Nesting of Commands

```python
@cli.command("get_summary")
@click.pass_context
def get_summary(ctx):
    ctx.invoke(get_key, key="summary")
```

## Adding Optional Parameters

Optional parameters can be defined using the `@click.option` decorator.

```python
@cli.command("get_results")
@click.option("-d", "--download", is_flag=True, help="Download the result to a JSON file")
@click.option("-k", "--key", help="Specify a key from the results")
@click.pass_context
def get_results(ctx, download, key):
    results = ctx.obj.get('results', [])
    if key:
        results = {key: sum(entry.get(key, 0) for entry in results)}
    if download:
        filename = f"{key or 'results'}.json"
        with open(filename, 'w') as w:
            json.dump(results, w)
        click.echo(f"File saved to {filename}")
    else:
        pprint.pprint(results)
```

## Using `@click.pass_obj`

`@click.pass_obj` passes only `ctx.obj` instead of the full context.

```python
@cli.command("get_text")
@click.option("-s", "--sentences", is_flag=True, help="Return sentences")
@click.option("-p", "--paragraphs", is_flag=True, help="Return paragraphs")
@click.option("-d", "--download", is_flag=True, help="Download as JSON file")
@click.pass_obj
def get_text(_dict, sentences, paragraphs, download):
    results = _dict.get('results', [])
    text = {} if paragraphs else {'text': ''}
    for idx, entry in enumerate(results):
        if paragraphs:
            text[idx] = entry.get('text', '')
        else:
            text['text'] += entry.get('text', '')
    if sentences:
        text = {i: s for i, s in enumerate(text.get('text', '').split('.')) if s}
    pprint.pprint(text)
    if download:
        filename = "paragraphs.json" if paragraphs else "text.json"
        with open(filename, 'w') as w:
            json.dump(text, w)
        click.echo(f"File saved to {filename}")
```

## Handling User Input

Click provides `@click.prompt` to interact with users.

```python
@cli.command("prompt_user")
@click.pass_context
def prompt_user(ctx):
    name = click.prompt("Enter your name")
    age = click.prompt("Enter your age", type=int)
    click.echo(f"Hello {name}, you are {age} years old!")
```

## Handling Confirmation

Use `@click.confirm` to get user confirmation before proceeding.

```python
@cli.command("confirm_action")
@click.pass_context
def confirm_action(ctx):
    if click.confirm("Do you want to proceed?"):
        click.echo("Proceeding with action...")
    else:
        click.echo("Action canceled.")
```

## Conclusion

Python Click simplifies CLI creation with its decorators and built-in features like automatic help generation and context passing. This guide provides a foundation for building more advanced command-line applications. Additionally, handling user input and confirmations enhances the interactivity of CLI applications.

# Python {#python}


dynamic language
lower learning, support
object orientated

[Immutable vs mutable](#immutable-vs-mutable)




# Pytorch Vs Tensorflow {#pytorch-vs-tensorflow}



- [Tensorflow](#tensorflow) is widely adopted but pytorch picking up
- Dynamic vs static graph
- Tensorboard is better than [pytorch](#pytorch) visualization
- Plain tensorflow looks pretty much like a library
- Abstraction is better in pytorch, even data parallelism
- Tf.contrib, [keras](#keras) to rescue

# P Values {#p-values}


A p-value is a measure of the evidence against a null hypothesis.
 p-values indicate whether an effect exists
Used in [Feature Selection](#feature-selection)

# P Values In Linear Regression In Sklearn {#p-values-in-linear-regression-in-sklearn}


# Question How to include [p values](#p-values) in sklearn for a [Linear Regression](#linear-regression)? 

import scipy.stats as stat.

You can modify the class of LinearRegression() from sklearn to include them

C:\Users\RhysL\Desktop\DS_Obs\1_Inbox\Work\Udemy\Part_5_Advanced_Statistical_Methods_(Machine_Learning)\multiple_linear_regression\sklearn - How to properly include p-values.ipynb

# What is f_regression and why can it compute p values?

from sklearn.feature_selection 
import f_regression
p_values = f_regression(x,y)[1]
p_values

[link](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html)

We will look into: f_regression
f_regression finds the F-statistics for the *simple* regressions created with each of the independent variables
In our case, this would mean running a simple linear regression on GPA where SAT is the independent variable
and a simple linear regression on GPA where Rand 1,2,3 is the indepdent variable
The limitation of this approach is that it doe<mark>s not take into account the mutual effect of the two features</mark>

f_regression(x,y)

There are two output arrays
The first one contains the F-statistics for each of the regressions
The second one contains the p-values of these F-statistics

outputs:
(array([56.04804786, 0.17558437]), array([7.19951844e-11, 6.76291372e-01]))

# Parametric Vs Non Parametric Models {#parametric-vs-non-parametric-models}


### Parametric Models

In [Statistics](#statistics)

**Definition:** Models  that summarize data with a <mark>set of parameters of fixed size, regardless of the number of data points.</mark>

**Characteristics:**
  - Assumes a specific form for the function mapping inputs to outputs (e.g., linear regression assumes a linear relationship).
  - Requires estimation of a finite number of parameters.
  - Generally faster to train and predict due to their simplicity.
  - Risk of underfitting if the model assumptions do not align well with the data.

  **Examples:** 
  - Linear [regression](#regression), logistic regression, neural networks (with a fixed architecture), [Bernoulli](#bernoulli)

### Non-parametric Models

- **Definition:** Models that do not assume a fixed form for the function mapping inputs to outputs and can grow in complexity with more data.
- **Characteristics:**
  - Do not make strong assumptions about the underlying data distribution.
  - Can adapt to the data's complexity, potentially capturing more intricate patterns.
  - Generally require more data to make accurate predictions.
  - Risk of overfitting, especially with small datasets, as they can model noise in the data.
- **Examples:** K-nearest neighbors, decision trees, [support vector machines](#support-vector-machines) (with certain kernels).

### Key Differences

- **Flexibility:** Non-parametric models are more flexible and can model complex relationships, while parametric models are simpler and rely on assumptions about the data.
- **Data Requirements:** Non-parametric models typically require more data to achieve good performance compared to parametric models.
- **Computation:** Parametric models are usually computationally less intensive than non-parametric models.


# Parametric Vs Non Parametric Tests {#parametric-vs-non-parametric-tests}

[Parametric tests ](#parametric-tests) are statistical tests that make <mark>assumptions about the distribution</mark> of the data. For example, a t-test assumes that the data is normally distributed. Non-parametric tests do not make assumptions about the distribution of the data. Parametric tests are generally more powerful than non-parametric tests, but they are only valid if the data meets the [Statistical Assumptions](#statistical-assumptions) of the test.

[Non-parametric tests ](#non-parametric-tests) are less powerful than parametric tests, but they can be used on any type of data, regardless of the distribution.


# Parsimonious {#parsimonious}

**Parsimonious** refers to a principle in [Model Selection](#model-selection) and statistical modeling that emphasizes <mark>simplicity</mark>. In the context of regression and other statistical models, a parsimonious model is one that explains the data with the fewest possible parameters or predictors while still providing a good fit.

A parsimonious model is one that achieves a good balance between simplicity and explanatory power.
### Key Points about Parsimonious Models:

1. **Simplicity**: A parsimonious model avoids unnecessary complexity. It uses only the essential variables that contribute meaningfully to the prediction or explanation of the outcome.

2. **Avoiding [Overfitting](#overfitting)**: By keeping the model simple, a parsimonious approach helps prevent overfitting, where a model learns the noise in the training data rather than the underlying pattern. Overfitting can lead to poor generalization to new, unseen data.

3. **Interpretability**: Simpler models are often easier to interpret and understand. This is particularly important in fields where explaining the model's decisions is crucial, such as healthcare or finance.

4. **Balance**: The goal is to strike a balance between model accuracy and complexity. A parsimonious model should provide a good fit to the data without being overly complicated.



# Pd.Grouper {#pdgrouper}

`pd.Grouper` is a utility in pandas used with `.groupby()` to flexibly group data by a specific column, often useful for time-based grouping, multi-index grouping, or applying custom frequency aggregation.

See:
- [Groupby](#groupby)
- [Multi-level index](#multi-level-index)

### Why Use `pd.Grouper`?
- Allows more readable and declarative code when working with time-indexed data.
- Supports multi-index groupings without restructuring your data.
- Enables resampling-like grouping without setting the index.
### Syntax
```python
pd.Grouper(key=None, level=None, freq=None, axis=0, sort=False)
```
### Parameters
- `key`: The column name to group by.
- `level`: For MultiIndex, the level to group by.
- `freq`: Used to group time-series data (e.g., `'D'` for daily, `'M'` for monthly).
- `axis`: Default is 0 (rows).
- `sort`: Whether to sort the result.






# Pdoc {#pdoc}

[PDOC](https://pdoc.dev/) is a documentation generator specifically designed for Python projects. Here are some key features and details:

1. **Automatic Documentation**: It scans your Python code and automatically generates documentation based on the docstrings you include in your code. This means that as long as you write clear comments and descriptions in your code, pdoc can create documentation without much extra work.

2. **Modern and Clean Design**: The output documentation is visually appealing and easy to navigate. It uses a modern design that enhances readability, making it user-friendly for anyone who needs to understand your code.

3. **Customization Options**: While pdoc generates documentation automatically, it also allows for some customization. You can configure settings to adjust how the documentation looks and what content is included.

4. **Markdown Support**: pdoc supports Markdown, which means you can use Markdown syntax in your docstrings to format your documentation with headings, lists, links, and more.

5. **Easy Integration**: It can be easily integrated into your development workflow, allowing you to generate documentation as part of your build process or whenever you need it.

6. **No Manual Guides Required**: With pdoc, you can avoid the tedious task of writing extensive documentation manually. Instead, you can focus on writing code, and pdoc will handle the documentation generation for you.

Once you generate the HTML files using [pdoc](#pdoc), you have several options for what to do with them:

1. **Local Viewing**: You can open the generated HTML files directly in your web browser to view the documentation locally. This is useful for personal reference or for sharing with a small team.

2. **Hosting on a Web Server**: You can upload the generated HTML files to a web server to make the documentation accessible to a wider audience. This is a common practice for open-source projects or any project where you want to share documentation with users or collaborators.

3. **Integrating with Project Repositories**: If you're using version control systems like Git, you can include the generated documentation in your repository. This way, anyone who clones the repository can access the documentation easily.

4. **Publishing to Documentation Platforms**: You can publish the HTML files to documentation hosting platforms like Read the Docs, GitHub Pages, or similar services. These platforms often provide additional features like versioning, search functionality, and easy navigation.

5. **Archiving**: You can keep the generated HTML files as part of your project archive for future reference. This is useful for maintaining a history of your documentation as your project evolves.

6. **Sharing with Stakeholders**: If you are working on a project with stakeholders or clients, you can share the HTML documentation with them to provide insights into the project's structure and functionality.

---
To explicitly tell pdoc to document the local `scripts` directory, you need to prepend `./` to the directory name.

Here’s how to do it:

1. **Open your terminal**.
2. **Navigate to the directory** where your `scripts` folder is located (which seems to be `C:\Users\RhysL\Desktop\Auto_YAML`).
3. **Run the pdoc command** with the `-o` option and prepend `./` to the `scripts` directory name:

   ```bash
   pdoc -o docs ./scripts
   ```

This command tells pdoc to generate documentation for the local `scripts` directory and save the output in the `docs` folder.

After running this command, you should find the generated documentation in the `docs` folder, which you can then open in your web browser.

## what to do next

1. **View Locally**: Open the HTML files in your web browser to view the documentation. You can start by opening the `index.html` file in the `docs` folder. This file typically serves as the main entry point for your documentation

2. **Host on a Web Server**: If you want to make the documentation accessible online, you can upload the `docs` folder to a web server. This could be a personal website, a cloud storage service that supports HTML hosting, or a documentation hosting platform like GitHub Pages or Read the Docs.
    
3. **Integrate into a Project Repository**: If you're using version control (like Git), you can include the `docs` folder in your repository. This way, anyone who clones the repository can easily access the documentation.

# Pmdarima {#pmdarima}

Helps find [Model Parameters](#model-parameters) for [ARIMA](#arima) models

[Forecasting_AutoArima.py](#forecasting_autoarimapy)

# Programming Languages {#programming-languages}

