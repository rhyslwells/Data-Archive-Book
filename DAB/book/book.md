<a id="turning-a-flat-file-into-a-database"></a>
# Turning A Flat File Into A Database {#turning-a-flat-file-into-a-database}


## Summary:

1. Read and Clean the Data: Load the data from the Excel sheet and clean it.
2. Split the Data: Separate the data into two DataFrames, one for customers and one for orders.
3. Create Tables: Create the SQLite tables with appropriate foreign key relationships.
4. Insert Data: Insert the cleaned and separated data into the respective tables.
5. Verify [Foreign Key](#foreign-key): Ensure that the foreign key relationships are valid.

## Example Data Structure

### Combined Excel Data (`Sheet1`):

| order_id | order_date | customer_id | customer_name | contact_name | country | amount |
|----------|------------|-------------|---------------|--------------|---------|--------|
| 1        | 2024-01-15 | 101         | John Doe      | Jane Doe     | USA     | 100.50 |
| 2        | 2024-02-20 | 102         | Alice Smith   | Bob Smith    | Canada  | 200.00 |
| 3        | 2024-03-10 | 101         | John Doe      | Jane Doe     | USA     | 150.75 |
| 4        | 2024-04-05 | 103         | Michael Brown | Sarah Brown  | UK      | 250.00 |

## Steps to Process and Split Data

### Step 1: Import Libraries and Read Data

```python
import pandas as pd
import sqlite3

# Example data for demonstration purposes
data = {
    'order_id': [1, 2, 3, 4],
    'order_date': ['2024-01-15', '2024-02-20', '2024-03-10', '2024-04-05'],
    'customer_id': [101, 102, 101, 103],
    'customer_name': ['John Doe', 'Alice Smith', 'John Doe', 'Michael Brown'],
    'contact_name': ['Jane Doe', 'Bob Smith', 'Jane Doe', 'Sarah Brown'],
    'country': ['USA', 'Canada', 'USA', 'UK'],
    'amount': [100.50, 200.00, 150.75, 250.00]
}

# Create a DataFrame from the example data
df = pd.DataFrame(data)
```

### Step 2: Clean Data

```python
# Example cleaning function
def clean_data(df):
    df.dropna(inplace=True)
    df.columns = [col.strip().replace(" ", "_").lower() for col in df.columns]
    return df

# Clean the data
df = clean_data(df)
```

### Step 3: Split Data into Customers and Orders

```python
# Extract unique customers
customers_df = df['customer_id', 'customer_name', 'contact_name', 'country'](#customer_id-customer_name-contact_name-country).drop_duplicates()

# Extract orders
orders_df = df['order_id', 'order_date', 'customer_id', 'amount'](#order_id-order_date-customer_id-amount)
```

### Step 4: Create Tables with Foreign Keys in SQLite

```python
# Connect to SQLite database (or create it)
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Enable foreign key support
cursor.execute("PRAGMA foreign_keys = ON")

# Create 'customers' table
cursor.execute('''
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    customer_name TEXT NOT NULL,
    contact_name TEXT,
    country TEXT
)
''')

# Create 'orders' table with a foreign key referencing 'customers'
cursor.execute('''
CREATE TABLE IF NOT EXISTS orders (
    order_id INTEGER PRIMARY KEY,
    order_date TEXT,
    customer_id INTEGER,
    amount REAL,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
)
''')

# Commit changes
conn.commit()
```

### Step 5: Insert Data into Tables

```python
# Insert data into 'customers' table
customers_df.to_sql('customers', conn, if_exists='append', index=False)

# Insert data into 'orders' table
orders_df.to_sql('orders', conn, if_exists='append', index=False)

# Commit changes and close the connection
conn.commit()
conn.close()
```

### Verification: Ensure Foreign Key Relationships

```python
conn = sqlite3.connect('data.db')
cursor = conn.cursor()

# Query to check if all customer_id in orders table exist in customers table
cursor.execute('''
SELECT order_id
FROM orders
WHERE customer_id NOT IN (SELECT customer_id FROM customers)
''')

invalid_orders = cursor.fetchall()
conn.close()

if invalid_orders:
    print("Invalid foreign key references found:", invalid_orders)
else:
    print("All foreign key references are valid.")
```



<a id="types-of-computational-bugs"></a>
# Types Of Computational Bugs {#types-of-computational-bugs}

Each of these types of bugs can have significant impacts on software functionality and performance, and understanding them is crucial for effective [Debugging](#debugging) and software development.
### Types of Computational Bugs
1. **Cumulative Rounding Error**: This occurs when small rounding errors accumulate over time, potentially leading to significant inaccuracies. An example is the Vancouver Stock Exchange issue.
2. **Cascades**: These are bugs that trigger a series of failures or errors in a system.
3. **Integer Overflow**: This happens when an arithmetic operation attempts to create a numeric value that is outside the range that can be represented with a given number of bits.
4. **Backend Issues**: These are problems that occur on the server-side of an application, affecting its functionality or performance.
### Additional Types of Computational Bugs

5. **Off-by-One Error**: This is a common programming error where an iterative loop iterates one time too many or one time too few. For example, iterating over an array with incorrect bounds can lead to accessing an out-of-bounds index.

6. **Null Pointer Dereference**: Occurs when a program attempts to access or modify data through a null pointer, leading to crashes or undefined behavior. For instance, trying to access an object method without checking if the object is null.

7. **Race Condition**: This happens when the behavior of software depends on the sequence or timing of uncontrollable events, such as threads accessing shared resources. An example is two threads modifying a shared variable simultaneously without proper synchronization.

8. **Memory Leak**: Occurs when a program fails to release memory that is no longer needed, leading to reduced performance or system crashes. This is common in languages like C++ where manual memory management is required.

9. **Buffer Overflow**: This happens when a program writes more data to a buffer than it can hold, potentially leading to data corruption or security vulnerabilities. An example is a classic stack buffer overflow attack.

10. **Logic Error**: This is a bug where the program compiles and runs but produces incorrect results due to a flaw in the algorithm or logic. For example, using the wrong formula to calculate a result.

11. **Deadlock**: Occurs when two or more processes are unable to proceed because each is waiting for the other to release resources. This is common in multithreaded applications.

12. **Syntax Error**: These are errors in the code that violate the rules of the programming language, preventing the code from compiling or running. For example, missing a semicolon in languages like Java or C++.

13. **Concurrency Issues**: These arise when multiple processes or threads execute simultaneously, leading to unpredictable results if not managed correctly. Examples include data races and inconsistent data states.

14. **Configuration Error**: Occurs when software is incorrectly configured, leading to unexpected behavior or failures. An example is a misconfigured database connection string.



<a id="types-of-database-schema"></a>
# Types Of Database Schema {#types-of-database-schema}

There are several types of database schemas commonly used in data warehousing and database design.

[Star Schema](#star-schema)

[Snowflake Schema](#snowflake-schema)

Galaxy Schema (or Fact Constellation Schema):
   - This schema consists of multiple fact tables that share dimension tables. It is useful for complex data models that require analysis across different business processes. The galaxy schema allows for more flexibility in querying and reporting.

[Normalised Schema](#normalised-schema)

Denormalized Schema:
   - A denormalized schema combines data from multiple tables into fewer tables to improve query performance. This approach is often used in data marts and data warehouses where read performance is prioritized over write performance.

Entity-Relationship Model ([ER Diagrams](#er-diagrams))
   - This is a conceptual schema that represents the data and its relationships in a graphical format. It is often used during the design phase of a database to visualize how entities (tables) relate to one another.

Columnar Schema:
   - In a [Columnar Storage](#columnar-storage) database, data is stored in columns rather than rows. This schema is optimized for read-heavy operations and analytical queries, making it suitable for data warehousing applications. Examples include Apache Cassandra and Google BigQuery.



<a id="types-of-neural-networks"></a>
# Types Of Neural Networks {#types-of-neural-networks}

Types of [Neural network](#neural-network):

[Feed Forward Neural Network](#feed-forward-neural-network)

[Convolutional Neural Networks](#convolutional-neural-networks)

[Recurrent Neural Networks](#recurrent-neural-networks)

[Generative Adversarial Networks](#generative-adversarial-networks)

[Transformer](#transformer)


<a id="typescript"></a>
# Typescript {#typescript}


Superset of JavaScript adding static typing and object-oriented features for building large-scale applications.

<a id="typical-output-formats-in-neural-networks"></a>
# Typical Output Formats In Neural Networks {#typical-output-formats-in-neural-networks}

The output format of a [Neural network](#neural-network) is largely determined by the specific task it is designed to perform.
## Classification

### [Binary Classification](#binary-classification)

 Single Output Node: This involves a single output node with a value between 0 and 1, representing the probability of the input belonging to the positive class.

Example: A spam classifier might output a value close to 1 for a spam email and a value close to 0 for a legitimate email.

### Multiclass [Classification](#classification)

 Multiple Output Nodes: Each class has its own output node, with values typically between 0 and 1, representing the probability of the input belonging to that class. These probabilities often sum to 1.

Example: An image classifier for different types of animals (cat, dog, bird) might output a vector like [0.2, 0.7, 0.1], indicating a 70% probability of the image being a dog.

## [Regression](#regression)

Single Output Node: This involves a single output node representing a continuous value.

Example: A neural network predicting house prices would output a single value representing the predicted price.

## SequencetoSequence Tasks

 Sequence of Outputs: The output is often represented as a list or a tensor.
 
Example: A neural machine translation model would output a sequence of words or subword units in the target language.

Example Applications
- **Machine Translation:** Converts a sentence from one language to another.
- **Text Summarization:** Generates a concise summary from a longer text.
- **Speech Recognition:** Transcribes spoken language into written text.

## Generative Tasks (e.g., Image Generation, Music Composition)

Data in the Same Format as the Input: The output is typically in the same format as the input data.
 
Example: An image generation model might output a tensor representing a generated image.

See [Generative AI](#generative-ai)
## Key Considerations

[Activation Function](#activation-function): The choice of activation function in the output layer can significantly influence the output format. 


Loss Functions: The [loss function](#loss-function) used during training also guides the output format. For example, binary crossentropy is commonly used for binary classification, while mean squared error is often used for regression.

<a id="ubuntu"></a>
# Ubuntu {#ubuntu}

Ubuntu is a popular open-source operating system based on the [Linux](#linux) kernel. It is designed to be user-friendly:

1. **Desktop Environment**: Ubuntu provides a graphical user interface (GUI) that makes it accessible to users who may not be familiar with command-line interfaces. It is often used as a desktop operating system for personal computers.

2. **Server Use**: Ubuntu Server is a version of Ubuntu designed for server environments. It is commonly used for hosting websites, applications, and databases due to its stability and security.

3. **Development**: Many developers use Ubuntu for software development because it supports a wide range of programming languages and development tools. It is also compatible with various software libraries and frameworks.

4. **Education**: Ubuntu is often used in educational institutions for teaching computer science and programming due to its open-source nature and the availability of free software.

5. **Customization**: Being open-source, Ubuntu allows users to customize their operating system according to their needs. Users can modify the source code, install different desktop environments, and choose from a variety of applications.

6. **Community Support**: Ubuntu has a large community of users and developers who contribute to its development and provide support through forums, documentation, and tutorials.



<a id="uml"></a>
# Uml {#uml}




https://www.drawio.com/

https://www.reddit.com/r/SoftwareEngineering/comments/133iw7n/is_there_any_free_handy_tool_to_create_uml/

https://plantuml.com/



<a id="unittest"></a>
# Unittest {#unittest}


### **`@patch` (from `unittest.mock`) Explanation**

`@patch` is used to replace objects/functions with mock versions during tests. It is part of Python’s `unittest.mock` module.

#### **Example & Usage**

python

Copy code

`from unittest.mock import patch  def fetch_data():     """Simulated function that fetches data from an API"""     return "Real Data"  @patch("__main__.fetch_data", return_value="Mocked Data") def test_fetch_data(mock_fetch):     assert fetch_data() == "Mocked Data"`

🔹 **How it works:**

- `@patch("__main__.fetch_data", return_value="Mocked Data")` replaces `fetch_data()` with a mocked version returning `"Mocked Data"`.
- Inside the test, `fetch_data()` will **always** return `"Mocked Data"` instead of calling the real function.

#### **Why use `@patch`?**

- Prevents tests from making actual API/database calls.
- Speeds up testing by mocking expensive operations.
- Allows control over return values and side effects.

---

### **Your Case:**

- **`@pytest.fixture`** is used to provide reusable test data (`mock_files`).
- **`@patch`** is used to:
    - Mock file operations (`builtins.open`, `os.walk`).
    - Mock function calls (`process_file`, `log_action`, `write_updated_file`).
    - Prevent real file modifications while testing.

<a id="univariate-vs-multivariate"></a>
# Univariate Vs Multivariate {#univariate-vs-multivariate}

Single feature versus multiple features



<a id="unstructured-data"></a>
# Unstructured Data {#unstructured-data}


>[!Important]
> Unstructured data is data that does not conform to a data model and has no easily identifiable structure. 

Unstructured data cannot be easily used by programs, and is difficult to analyze. Examples of unstructured data could be the contents of an <mark>email, contents of a word document, data from social media, photos, videos, survey results</mark>, etc.
## An example of unstructured data

An simple example of unstructured data is a string that contains interesting information inside of it, but that has not been formatted into a well defined schema. An example is given below:

|               |  **UnstructuredString**|
|---------| -----------|
|Record 1| "Bob is 29" |
|Record 2| "Mary just turned 30"|

## Unstructured vs structured data

In contrast with unstructured data, [structured data](term/structured%20data.md) refers to data that has been formatted into a well-defined schema. An example would be data that is stored with precisely defined columns in a relational database or excel spreadsheet. Examples of structured fields could be age, name, phone number, credit card numbers or address. Storing data in a structured format allows it to be easily understood and queried by machines and with tools such as  [SQL](#sql).

  



<a id="unsupervised-learning"></a>
# Unsupervised Learning {#unsupervised-learning}


Unsupervised learning is a type of machine learning where the algorithm is trained on data without explicit labels or predefined outputs. 

Unsupervised learning involves discovering hidden patterns in data without predefined labels. It is valuable for exploratory data analysis, [Clustering](#clustering), and [Isolated Forest](#isolated-forest).

The goal is to find hidden patterns, relationships, or structures in the data. Unlike supervised learning, which uses labeled input-output pairs, unsupervised learning relies solely on input data, allowing the algorithm to uncover insights independently.

### Key Concepts

1. No Labeled Data: There is no ground truth or correct output associated with the input data.
2. Data Patterns: The algorithm identifies inherent structures, clusters, or associations within the dataset.
3. Objective: The primary objective is to explore the data and organize it to reveal underlying patterns.

### Common Types of Unsupervised Learning

#### [Clustering](#clustering)

Description: The algorithm groups similar data points together based on their features.

Example: Customer segmentation in marketing, where a clustering algorithm divides customers into groups based on purchasing behavior, demographics, or browsing history.

Popular Algorithms:
  - [K-means](#k-means): Divides the data into \( k \) clusters, where each data point belongs to the nearest cluster.
  - Hierarchical Clustering
  - [DBScan](#dbscan)
  - [Support Vector Machines](#support-vector-machines)
  - [K-nearest neighbours](#k-nearest-neighbours)

#### [Dimensionality Reduction](#dimensionality-reduction)

Description: Reduces the number of input variables (features) while preserving as much information as possible. This is helpful for high-dimensional data, where visualization and analysis become challenging.

Popular Algorithms:
  - [Principal Component Analysis](#principal-component-analysis) 

#### [Isolated Forest](#isolated-forest)

Description: Identifies [standardised/Outliers](#standardisedoutliers) or unusual data points that don’t conform to the expected pattern in the dataset.

Example: Detecting fraudulent credit card transactions by identifying transactions that deviate significantly from typical spending patterns.

Mechanism: Works by randomly partitioning the data and identifying [standardised/Outliers|anomalies](#standardisedoutliersanomalies) as points that can be isolated quickly.


<a id="untitled-1"></a>
# Untitled 1 {#untitled-1}




<a id="untitled-2"></a>
# Untitled 2 {#untitled-2}




<a id="untitled"></a>
# Untitled {#untitled}



<a id="use-cases-for-a-simple-neural-network-like"></a>
# Use Cases For A Simple Neural Network Like {#use-cases-for-a-simple-neural-network-like}

Scenarios where a simple [Neural network|Neural Network](#neural-networkneural-network) work like this might be useful:

**[Regression](#regression) with Multiple Features**
If you have multiple input features and you want to predict a continuous output, this network can learn the appropriate weights for each feature. For instance:
- Predicting **fuel efficiency** of a car based on features like engine size, horsepower, and weight.
- Predicting **sales** based on multiple factors like marketing spend, seasonality, and economic indicators.

**[Binary Classification](#binary-classification)**
With slight modification (e.g., adding a **Sigmoid activation** to the output layer), you could use this network for binary classification tasks. For example:
- Classifying whether an email is **spam** or not based on features like word frequency and sender information.
  
**Multi-Feature [Time Series Forecasting](#time-series-forecasting)**
If you have time series data with multiple variables, you can feed it into this simple network to predict future values based on past trends. For instance:
- Predicting **stock prices** based on multiple features like historical prices, trading volume, and economic data.

**Training and Optimization (Next Steps)**
The provided code only defines the network and performs a **forward pass**, but to use this model for real-world tasks, you would need to:
- **Define a loss function** (e.g., Mean Squared Error for regression or Cross-Entropy Loss for classification).
- **Train the network** using an optimizer like **Stochastic Gradient Descent (SGD)**, **Adam**, or another optimization algorithm.
- **Backpropagate** the gradients to update the model’s weights using gradient descent.

<a id="use-of-rnns-in-energy-sector"></a>
# Use Of Rnns In Energy Sector {#use-of-rnns-in-energy-sector}



For energy data problems, many **interpretable machine learning algorithms** can be applied in place of or alongside RNNs. These models offer transparency, making it easier to understand the relationships between features and predictions, which is critical in areas like energy management, where interpretability can be as important as accuracy.

For each of the energy data questions that RNNs might solve, **interpretable alternatives** [Machine Learning Algorithms](#machine-learning-algorithms): such as **linear regression**, **decision trees**, **random forests**, and **ARIMA** models can be employed. These models provide **transparency** by revealing which features (e.g., weather, demand) influence predictions the most, making them suitable for stakeholders who need clear explanations of the decisions made by the model.

### [Demand forecasting](#demand-forecasting)
   - **Algorithms**:
     - **Linear Regression**: Can model simple linear relationships between energy consumption and time (e.g., daily/seasonal trends).
     - **Decision Trees**: Provides clear if-then rules for predicting future energy usage based on historical consumption, time of day, and other factors.
     - **Random Forests**: An ensemble of decision trees that provides better accuracy than individual trees while still being interpretable using feature importance.
     - **[Gradient Boosting](#gradient-boosting) (GBM)**: Can be used with feature importance or [SHapley Additive exPlanations|SHAP](#shapley-additive-explanationsshap) values to understand which factors (e.g., time, weather) drive energy demand.
   
   - **Why**: These models allow for clear interpretation of how factors like temperature, time of day, and previous energy use contribute to predictions.



### 2. **Renewable Energy Generation Prediction**
   - **Algorithms**:
     - **Linear Regression**: For simple relationships, like the effect of sunlight hours or wind speed on energy generation.
     - **Support Vector Machines (SVM)**: Can create interpretable linear boundaries when predicting renewable energy outputs, with clear separation of factors (e.g., wind speed thresholds).
     - **Random Forests**: Offers feature importance metrics that explain which weather factors are most important for predicting energy generation.
     - **GBM**: Using [SHapley Additive exPlanations|SHAP](#shapley-additive-explanationsshap) values or feature importance to interpret the impact of weather variables on the energy output.

   - **Why**: These algorithms can provide insights into the key weather conditions driving renewable energy generation and give transparent predictions for decision-making.



### 3. **Energy Price Forecasting**
   - **Algorithms**:
     - **ARIMA (AutoRegressive Integrated Moving Average)**: A traditional time series forecasting method that models linear relationships in energy prices over time.
     - **Linear Regression**: Can model the impact of factors like demand, supply, and historical prices in an interpretable way.
     - **Decision Trees**: Easy to interpret and can show thresholds where prices change based on inputs like demand or fuel costs.
     - **XGBoost**: Provides interpretability through SHAP values or feature importance, explaining which market factors (e.g., demand, fuel prices) drive price changes.

   - **Why**: These algorithms offer interpretable insights into what drives price fluctuations, making them useful for energy market analysis and trading.



### 4. **Anomaly Detection in Energy Consumption**
   - **Algorithms**:
     - **Isolation Forests**: Specifically designed for anomaly detection and provides interpretable results by isolating outliers.
     - **k-Nearest Neighbors (k-NN)**: Can flag anomalies by comparing new consumption data to known normal consumption patterns, with simple explanations of "closeness" to typical patterns.
     - **Logistic Regression**: Can be used to classify energy consumption data into "normal" and "anomalous" categories based on clear feature contributions.
     - **One-Class SVM**: A linear model that can classify whether energy usage deviates from typical patterns.

   - **Why**: These interpretable algorithms can identify unusual patterns in energy data, providing clear reasons (e.g., thresholds exceeded) for flagging certain periods as anomalous.



### 5. **Load Balancing and Optimization**
   - **Algorithms**:
     - **Linear Programming (Optimization)**: Provides interpretable rules for how energy should be distributed across the grid to minimize costs and prevent overloads.
     - **Decision Trees**: Can clearly show the impact of different factors (e.g., region, time of day) on grid load, and thresholds for balancing loads.
     - **Rule-Based Systems**: Set explicit rules for load balancing based on historical data and real-time demand, offering full transparency.
   
   - **Why**: These interpretable models can assist grid operators in understanding which regions or time periods contribute most to load imbalances and suggest corrective actions.



### 6. **Customer Energy Usage Profiling**
   - **Algorithms**:
     - **k-Means Clustering**: Can group customers into distinct profiles based on energy usage patterns, with each cluster representing a clear profile (e.g., high-energy consumers, off-peak users).
     - **Decision Trees**: Can predict customer profiles based on historical usage data and explain which features (e.g., time of usage, appliance usage) define each profile.
     - **Logistic Regression**: Can be used to classify customers into different segments based on usage characteristics, providing clear coefficient-based interpretations.

   - **Why**: These models provide transparency into what factors drive a customer’s energy usage profile, which is essential for creating personalized recommendations.



### 7. **Demand Response Optimization**
   - **Algorithms**:
     - **Linear Programming (Optimization)**: Provides interpretable solutions for when and where to implement demand response programs to minimize peak energy use.
     - **Decision Trees**: Can clearly define rules for when demand response should be triggered based on time of day, weather, and current load.
     - **k-Nearest Neighbors (k-NN)**: Can identify similar past scenarios where demand response was implemented successfully and explain why the current situation matches.

   - **Why**: These methods give clear, interpretable guidelines for when and how to reduce energy demand during peak times, based on past patterns.



### 8. **Fault Detection in Power Systems**
   - **Algorithms**:
     - **Decision Trees**: Can explain why certain operational conditions (e.g., voltage drops, temperature increases) are likely to lead to faults, with clear rules and thresholds.
     - **Random Forests**: Provides feature importance scores that highlight which factors (e.g., temperature, load) are most indicative of impending faults.
     - **Logistic Regression**: Offers simple, interpretable probabilities for whether a fault will occur, based on key factors like current and voltage.

   - **Why**: Fault detection requires clear, interpretable models that help engineers understand the most important factors leading to equipment failures.



### 9. **Energy Usage Forecasting for Smart Buildings**
   - **Algorithms**:
     - **Multiple Linear Regression**: Can model the relationship between building factors (e.g., temperature, occupancy) and energy usage, offering clear coefficients.
     - **Decision Trees**: Provides an interpretable way to understand which building features (e.g., time of day, external temperature) influence energy consumption the most.
     - **k-Means Clustering**: Can group similar time periods or usage patterns to explain different operational modes of the building.
   
   - **Why**: These algorithms provide interpretable insights into how building features and external factors impact energy consumption, allowing for more efficient energy management.



### 10. **Time Series Forecasting for Energy Production in Microgrids**
   - **Algorithms**:
     - **ARIMA**: Traditional interpretable time series model that predicts future production based on past production data.
     - **Linear Regression**: Can predict energy production based on simple factors like weather data, fuel availability, and historical output.
     - **Decision Trees**: Helps identify which weather or resource factors are most critical for predicting energy production at a given time.

   - **Why**: Time series models like ARIMA are highly interpretable and useful for understanding how different factors contribute to energy production in microgrids.



### 11. **Battery Storage Optimization**
   - **Algorithms**:
     - **Linear Programming (Optimization)**: Provides a clear, interpretable approach to optimizing charge/discharge schedules based on forecasted energy generation and consumption.
     - **Decision Trees**: Can explain when and why batteries should be charged or discharged based on energy production, consumption, and cost factors.
     - **Rule-Based Systems**: Establish clear rules for battery storage optimization, offering fully interpretable decision-making processes.

   - **Why**: Optimizing battery storage requires clear, rule-based or linear models to understand how different variables (e.g., energy prices, consumption) impact storage decisions.







<a id="utilities"></a>
# Utilities {#utilities}



<a id="vacuum"></a>
# Vacuum {#vacuum}



<a id="vanishing-and-exploding-gradients-problem"></a>
# Vanishing And Exploding Gradients Problem {#vanishing-and-exploding-gradients-problem}


[Recurrent Neural Networks|RNN](#recurrent-neural-networksrnn)



[vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem)
In standard RNNs, the difficulty lies in retaining useful information over long sequences due to the exponential decrease in the gradient values, which results in poor learning of long-term dependencies.

<a id="variance"></a>
# Variance {#variance}

Variance in a dataset is a statistical measure that represents the degree of spread or dispersion of the data points around the mean (average) of the dataset. 

It quantifies how much the individual data points differ from the mean value. 

A higher variance indicates that the data points are more spread out from the mean, while a lower variance indicates that they are closer to the mean. 

Variance is calculated as the average of the squared differences between each data point and the mean.

See also:
[Boxplot](#boxplot)
[Distributions](#distributions)

**Variance**:
- Measures how much a single variable deviates from its mean.
- For variable $X$, variance is: 
$$
\text{Var}(X) = \frac{1}{n}\sum_{i=1}^n (x_i - \mu_X)^2
$$
- Variance determines the **spread** of data along a particular dimension.

<a id="vector-database"></a>
# Vector Database {#vector-database}

## Overview

Vector databases are specialized systems designed to handle and manage [Vector Embedding](#vector-embedding). 

As most real-world data is unstructured, such as text, images, and audio, vector databases play a  role in organizing and querying this data effectively.

Why use them:
- data is [unstructured data](#unstructured-data) i.e image
- 

## Key Features

- **Vector Embeddings**: At the core, vector databases store embeddings generated by machine learning models. These embeddings transform complex data into fixed-size vectors that encapsulate semantic information.
- **Similarity Search**: By leveraging the geometric properties of vector spaces, vector databases can quickly identify similar items. This is achieved by measuring distances (e.g., cosine similarity, Euclidean distance) between vectors.
- **Indexing Methods**: Various indexing techniques, such as HNSW (Hierarchical Navigable Small World) graphs, IVF (Inverted File), and PQ (Product Quantization), are employed to optimize search speed and accuracy. Allows faster searching.

## Querying Vectors

To query vectors, users typically specify a target vector and a similarity metric. 

The database then retrieves vectors that are closest to the target, based on the chosen metric. This process is crucial for applications like recommendation systems, where finding similar items is essential.

## Use Cases
1. **Long-term Memory for [LLM](#llm)**: Vector databases can store vast amounts of contextual information, enhancing the memory and retrieval capabilities of large language models (LLMs). Implemented using [Langchain](#langchain).
2. Rank and Recommendation system using nearest neighbours.
3. **Semantic Search**: Unlike traditional keyword-based search, semantic search understands the context and meaning, providing more relevant results. This is particularly useful in natural language processing (NLP) applications.
4. **Similarity Search**: Beyond text, vector databases support similarity searches for multimedia data, enabling applications in image recognition, audio analysis, and video retrieval.

## Options
Several vector database solutions are available, each with unique features and optimizations:
- **Pincone**: Known for its scalability and ease of integration with machine learning workflows.
- **Weaviate**: Offers a semantic graph database with built-in vector search capabilities.
- **Chroma**: Focuses on simplicity and performance for embedding-based applications.
- **Redis**: Provides vector search capabilities through its modules, suitable for real-time applications.
- **Qdrant**: Designed for high-performance vector search with a focus on scalability.
- **Milvus**: An open-source solution optimized for handling large-scale vector data.
- **Vespa**: Combines vector search with traditional search capabilities, ideal for complex applications.

## Related Concepts
- **[standardised/Vector Embedding](#standardisedvector-embedding)**: The process of converting data into vector form, capturing its semantic essence.
- **[standardised/Vector Embedding](#standardisedvector-embedding)**: A specific type of vector embedding used in NLP to represent words in a continuous vector space.
- **[Semantic Relationships](#semantic-relationships)**: A search technique that leverages the meaning and context of queries and data to deliver more relevant results.
- [Cosine Similarity](#cosine-similarity)
### Resources

[Vector Databases simply explained! (Embeddings & Indexes)](https://www.youtube.com/watch?v=dN0lsF2cvm4&list=PLcWfeUsAys2kC31F4_ED1JXlkdmu6tlrm)



<a id="vector-embedding"></a>
# Vector Embedding {#vector-embedding}


Vector Embedding is a technique used in machine learning and [NLP](#nlp) to represent data in a continuous vector space. This representation captures the [Semantic Relationships](#semantic-relationships) of data, such as words or sentences, allowing similar items to be positioned close to each other in the vector space.

### Key Concepts

- Data Compression: Embeddings compress data into a lower-dimensional space, making it easier to process and analyze. This is particularly useful for high-dimensional data like text or images.
  
- Semantic Similarity: In the embedding space, similar items are positioned close to each other. This proximity reflects semantic similarity, meaning that items with similar meanings or characteristics have similar vector representations.

1. [Dimensionality Reduction](#dimensionality-reduction): Words are represented in a lower-dimensional space compared to traditional methods like one-hot encoding, resulting in more efficient computations.

2. [Semantic Relationships](#semantic-relationships): Words with similar meanings or contexts are located close to each other in the vector space. For example, "king" and "queen" might be closer to each other than "king" and "apple."

![Pasted image 20241015211934.png](../content/images/Pasted%20image%2020241015211934.png)

4. Contextual Understanding: (Vector) Word embeddings capture the context in which words appear, allowing models to understand nuances and relationships in language.

Popular methods for generating vector (word) embeddings include:
- [Word2vec](#word2vec),
- GloVe, 
- FastText.
- [spaCy](#spacy)

### Types of Similarity Measures

- Euclidean Distance
- [Cosine Similarity](#cosine-similarity)

### Applications

- [Language Models](#language-models): Vector embeddings are widely used in language models to represent words, phrases, or sentences, enabling models to understand and generate human language more effectively.
- [Attention mechanism](#attention-mechanism): Embeddings are often used with attention mechanisms to enhance model performance in tasks like translation, summarization, and question answering.

### Example Use Cases

- Word Embeddings: Techniques like Word2Vec and GloVe create word embeddings that capture semantic relationships between words, enabling tasks like word similarity and analogy solving.
- Sentence Embeddings: Models like [BERT](#bert) and Sentence Transformers generate embeddings for entire sentences, facilitating tasks like sentiment analysis and semantic search.

### Visualizations

- [t-SNE](#t-sne): A technique for visualizing high-dimensional data, often used to display word embeddings in a two-dimensional space.




![Pasted image 20241015211844.png](../content/images/Pasted%20image%2020241015211844.png)


## Implementation

How to do vector embeddings in [PyTorch](#pytorch) that show [Semantic Relationships](#semantic-relationships) between terms.

In [ML_Tools](#ml_tools) see: [Vector_Embedding.py](#vector_embeddingpy)

## Related Terms

[How to search within a graph](#how-to-search-within-a-graph)
[How would you decide between using TF-IDF and Word2Vec for text vectorization](#how-would-you-decide-between-using-tf-idf-and-word2vec-for-text-vectorization)
[embeddings for OOV words](#embeddings-for-oov-words)



<a id="vector_embeddingpy"></a>
# Vector_Embedding.Py {#vector_embeddingpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/NLP/Vector_Embedding.py

### Explanation of the Script

1. **Vocabulary and Embedding Layer**:
    - Terms are mapped to indices using a dictionary.
    - The embedding layer learns continuous vector representations for these terms.
      
2. **Cosine Similarity**:
    - The cosine similarity function measures how similar two terms are in the embedding space. Higher values indicate closer relationships.
      
3. **Visualization**:
    - Embeddings are plotted in a 2D space to show semantic relationships. Terms with similar meanings (e.g., "king" and "queen") are expected to cluster together.
      
4. **t-SNE for Dimensionality Reduction**:
    - If the embedding dimension is higher than 2, t-SNE can reduce it to 2D for visualization while preserving semantic relationships.

### Outputs

1. **Cosine Similarities**:
    - Pairwise similarity scores between terms to quantify their semantic closeness.
      
2. **Visualization**:
    - A scatter plot showing the positions of terms in the embedding space.

<a id="vectorisation"></a>
# Vectorisation {#vectorisation}




[Link](https://www.youtube.com/watch?v=uvTL1N02f04&list=PLkDaE6sCZn6FNC6YRfRQc_FbeQrF8BwGI&index=24)

![Pasted image 20241217204829.png|500](../content/images/Pasted%20image%2020241217204829.png|500)
Numpy dot is better than for loop and summing.
Why does it run faster?
A: Designed to run in parallel 

Sequentially versus simultaneously in parallel.

Related concepts:
- [Numpy](#numpy)
- [gpu](#gpu)

<a id="vectorized-engine"></a>
# Vectorized Engine {#vectorized-engine}

## Vectorized Engine

A modern database query execution engine designed to optimize data processing by leveraging vectorized operations and SIMD (Single Instruction, Multiple Data) capabilities of modern CPUs. Vectorized engines, such as  [DuckDB](#duckdb), process data in large blocks or batches using SIMD instructions, allowing for improved parallelism, cache locality, and reduced overhead compared to traditional row-at-a-time processing engines, using [Columnar Storage](#columnar-storage).

<a id="vercel"></a>
# Vercel {#vercel}




<a id="view-use-case"></a>
# View Use Case {#view-use-case}

## View Use Case

### Scenario

A company wants to generate monthly performance reports for its employees. The performance data is spread across multiple tables, including `employees`, `departments`, and `performance_reviews`. Instead of writing complex queries every time a report is needed, the company can create a view that simplifies data retrieval.

### Step 1: Define the Tables

Assume we have the following tables:

- **employees**: Contains employee details.
  - `employee_id`
  - `name`
  - `department_id`

- **departments**: Contains department details.
  - `department_id`
  - `department_name`

- **performance_reviews**: Contains performance review data.
  - `review_id`
  - `employee_id`
  - `review_score`
  - `review_date`

### Step 2: Create a View

To simplify the reporting process, we create a view that joins these tables and aggregates the performance scores:

```sql
CREATE VIEW employee_performance AS
SELECT 
    e.employee_id,
    e.name,
    d.department_name,
    AVG(pr.review_score) AS average_score
FROM 
    employees e
JOIN 
    departments d ON e.department_id = d.department_id
JOIN 
    performance_reviews pr ON e.employee_id = pr.employee_id
GROUP BY 
    e.employee_id, e.name, d.department_name;
```

### Step 3: Query the View

Now, whenever the HR department needs to generate a performance report, they can simply query the `employee_performance` view:

```sql
SELECT * FROM employee_performance WHERE average_score >= 4.0;
```

This query retrieves all employees with an average performance score of 4.0 or higher, making it easy to identify top performers.

### Benefits of Using Views in This Use Case

1. **Simplification**: The view encapsulates complex joins and aggregations, allowing HR to retrieve performance data without needing to understand the underlying table structure.

2. **Reusability**: The view can be reused for different reports, such as quarterly reviews or department-specific performance assessments.

3. **Maintainability**: If the logic for calculating performance scores changes, the HR team only needs to update the view definition, not every individual query.

4. **Data Consistency**: All reports generated from the view will be consistent, as they rely on the same underlying logic for calculating average scores.

5. **Security**: If sensitive employee data needs to be protected, the view can be designed to exclude certain columns, ensuring that only necessary information is accessible.

<a id="views"></a>
# Views {#views}


Views are virtual tables defined by SQL [Querying|Query](#queryingquery) that <mark>simplify complex data representation.</mark> They can remove unnecessary columns, aggregate results, partition data, and secure sensitive information.


In [DE_Tools](#de_tools) see:
https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Viewing/Viewing.ipynb

Basic Usage:
- Simplification
- Aggregation (using GROUP)
- [Common Table Expression](#common-table-expression)
- Securing data: can give all values in a field the same value.

Advanced Usage
- Temporary Views: Exist only for the <mark>duration of the database connection.</mark>
- [Common Table Expression](#common-table-expression): Serve as temporary views for a single query.
- [Soft Deletion](#soft-deletion): Use views and triggers to mark records as deleted without physically removing them from the table.

Related topics:
- [View Use Case](#view-use-case)

## Why Use Views?

1. **Simplification and Abstraction**:
   - Views encapsulate complex queries, allowing users to interact with data without needing to understand the underlying structure. This simplifies data retrieval by hiding complexity.

2. **Security**:
   - Views restrict access to specific data by granting users access to views instead of underlying tables, which can help protect sensitive information. Note: Access controls may vary by database system (e.g., not available in [SQLite](#sqlite)).

3. **Reusability and Maintainability**:
   - Define complex queries once in a view and reuse them across multiple applications, simplifying maintenance when logic changes.

4. **Data Consistency and Integrity**:
   - Ensure consistent data presentation across applications and users by encapsulating business logic for uniform calculations.

5. **Performance Optimization**:
   - While regular views do not inherently improve performance, materialized views can enhance performance by storing precomputed results.

6. **Logical Data Independence**:
   - Provide a layer of abstraction between physical data storage and access methods, allowing [Database Schema|schema](#database-schemaschema) changes without affecting view users.

7. **Aggregation and Partitioning**:
   - Views can be used to calculate and store aggregated results (e.g., average ratings) and organize data by specific criteria (e.g., years or categories).

<a id="violin-plot"></a>
# Violin Plot {#violin-plot}


An extension of a [Boxplot](#boxplot) showing the data distribution. Useful when comparing distributions, skewness.

```python
data = [...]  # Your data
sns.violinplot(data=data, color="purple", fill="lightblue", scale="area")
plt.show()
```



<a id="virtual-environments"></a>
# Virtual Environments {#virtual-environments}


[Setting up virtual env](https://www.youtube.com/watch?v=yG9kmBQAtW4)

For windows (need to not be in a venv before del)
```cmd
rmdir /s /q venv
python -m venv venv
venv\Scripts\activate
```

pip freeze > requirements.txt

.gitignore 
https://www.youtube.com/watch?v=_vejzukmn4s

Remember to set python interpreter

Related terms:
- [Poetry](#poetry)

<a id="wcss-and-elbow-method"></a>
# Wcss And Elbow Method {#wcss-and-elbow-method}


USE: WCSS (within-cluster sum of squares)

WCSS is a measure developed within the ANOVA framework. It gives a very good idea about the different distance between different clusters and within clusters, thus providing us a rule for deciding the appropriate number of clusters.

The plot will resemble an "elbow," and the goal is to find the point where the decrease in WCSS slows down, forming an elbow-like shape.

Elbow numbers are the point where the rate of decrease in WCSS starts to flatten out

The rationale behind the elbow method is that 

Rationale: as you increase the number of clusters (K), the WCSS will generally decrease because each cluster becomes smaller. However, there is a point where the addition of more clusters provides diminishing returns in terms of reducing WCSS. The elbow point represents a good balance between capturing the variance in the data and avoiding excessive fragmentation.

## Code


```python

# Use WCSS and elbow method
# number of clusters
wcss=[]
start=2
end=10
# Create all possible cluster solutions with a loop
for i in range(start,end):
    # Cluster solution with i clusters
    kmeans = KMeans(i)
    # Fit the data
    kmeans.fit(df_scaled)
    # Find WCSS for the current iteration
    wcss_iter = kmeans.inertia_
    # Append the value to the WCSS list
    wcss.append(wcss_iter)

  

# Create a variable containing the numbers from 1 to 6, so we can use it as X axis of the future plot

number_clusters = range(start,end)

# Plot the number of clusters vs WCSS

plt.plot(number_clusters,wcss)

# Name your graph

plt.title('The Elbow Method')
# Name the x-axis
plt.xlabel('Number of clusters')
# Name the y-axis
plt.ylabel('Within-cluster Sum of Squares')

# Identify the elbow numbers (there may be more than one thats best)
elbow_nums=[4,5,6,7,8]
```

plotting
```python
# function to give scatter for each elbow number
    
def scatter_elbow(X, elbow_num, var1, var2):
    """
    Apply clustering with elbow method and plot a scatter plot with cluster information.

    Parameters:
    - X: DataFrame, input data for clustering
    - elbow_num: int, number of clusters determined by elbow method
    - var1, var2: str, names of the variables for the scatter plot

    Returns:
    None (plots the scatter plot)
    """
    # Apply [clustering](#clustering) with elbow number
    kmeans = KMeans(elbow_num)
    kmeans.fit(X)

    # Add cluster information
    identified_clusters = kmeans.fit_predict(X)
    X['Cluster'] = identified_clusters

    # Plot
    plt.scatter(X[var1], X[var2], c=X['Cluster'], cmap='rainbow')
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.title(f"{elbow_num}-Clustering for {var1}-{var2}")
    plt.show()
# Example usage:
# scatter_elbow(data, elbow_num, 'var1', 'var2')
for elbow_num in elbow_nums:
    scatter_elbow(df, elbow_num, var1, var2)
```

<a id="weak-learners"></a>
# Weak Learners {#weak-learners}


Weak learners are simple models that perform slightly better than random guessing. They are often used as the building blocks in [Model Ensemble](#model-ensemble) methods to create a strong predictive model.

## Characteristics

- **Simplicity:** Weak learners are typically simple models, such as [Decision Tree](#decision-tree) stumps, which split the data based on a single feature.
- **Performance:** Individually, they may not perform well, but when combined, they can produce a powerful ensemble model.

## Role in Model Ensembling

Weak learners are a crucial component of [Model Ensemble](#model-ensemble) techniques, such as boosting and bagging, where multiple weak learners are combined to improve overall model performance.

## Learning Rate

- The [learning rate](#learning-rate) is a [Hyperparameter](#hyperparameter)that controls the contribution of each weak learner to the final ensemble model.
- A smaller learning rate means that each weak learner has a smaller impact, often requiring more learners to achieve good performance.


<a id="web-feature-server-wfs"></a>
# Web Feature Server (Wfs) {#web-feature-server-wfs}

[GIS](#gis)

### Web Feature Server (WFS)

**Purpose**: WFS is designed to serve raw geographic features (vector data) over the web.

**Functionality**:
- **Feature-Based**: It delivers geographic features (such as points, lines, and polygons) and their associated attribute data in formats like GML (Geography Markup Language).
- **Interactivity**: Allows clients to query and retrieve specific features, perform spatial and attribute queries, and even support transactions (e.g., inserting, updating, deleting features).
- **Data Access**: Provides access to the actual data behind the map, enabling more detailed and customized analysis and processing compared to image-based services.
- **Standardization**: Also standardized by the OGC, ensuring compatibility and interoperability across various GIS applications and systems.

<a id="web-map-tile-service-wmts"></a>
# Web Map Tile Service (Wmts) {#web-map-tile-service-wmts}

[GIS](#gis)

### Web Map Tile Service (WMTS)

**Purpose**: WMTS is designed to serve pre-rendered, cached image tiles of maps. 

**Functionality**:
- **Tile-Based**: It serves map images as small, fixed-size tiles, usually in a format such as PNG or JPEG.
- **Performance**: By using cached tiles, WMTS can quickly deliver map images, making it highly efficient for applications requiring fast map rendering, like web mapping applications.
- **Scalability**: The tile-based approach allows for easy scaling and efficient handling of high load, as the same tiles can be reused for multiple requests.
- **Standardization**: It is standardized by the Open Geospatial Consortium (OGC), ensuring interoperability between different systems and software.

<a id="webpages-relevant"></a>
# Webpages Relevant {#webpages-relevant}

Using bookmarks:
#### [Time Series](#time-series)

https://aeturrell.com/blog/posts/time-series-explosion/?utm_source=substack&utm_medium=email

https://otexts.com/fpp3/?utm_source=substack&utm_medium=email#

<a id="what-algorithms-or-models-are-used-within-the-energy-sector"></a>
# What Algorithms Or Models Are Used Within The Energy Sector {#what-algorithms-or-models-are-used-within-the-energy-sector}





<a id="what-algorithms-or-models-are-used-within-the-telecommunication-sector"></a>
# What Algorithms Or Models Are Used Within The Telecommunication Sector {#what-algorithms-or-models-are-used-within-the-telecommunication-sector}





<a id="what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts"></a>
# What Are The Best Practices For Evaluating The Effectiveness Of Different Prompts {#what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts}



<a id="what-can-abm-solve-within-the-energy-sector"></a>
# What Can Abm Solve Within The Energy Sector {#what-can-abm-solve-within-the-energy-sector}



[Agent-Based Modelling](#agent-based-modelling)

energy systems analysis

<a id="what-is-the-difference-between-odds-and-probability"></a>
# What Is The Difference Between Odds And Probability {#what-is-the-difference-between-odds-and-probability}





<a id="what-is-the-role-of-gradient-based-optimization-in-training-deep-learning-models"></a>
# What Is The Role Of Gradient Based Optimization In Training Deep Learning Models. {#what-is-the-role-of-gradient-based-optimization-in-training-deep-learning-models}





<a id="when-and-why-not-to-us-regularisation"></a>
# When And Why Not To Us Regularisation {#when-and-why-not-to-us-regularisation}

While regularization is tool to combat overfitting, it is not a always useful. It is crucial to consider the model's 
- complexity,
- the quality and 
- quantity of data, 
- and the appropriateness of the regularization parameters

to ensure effective performance on validation data. If your model is performing well on [training data](#training-data) but poorly on [validation data](#validation-data), regularization might not always solve this issue for several reasons:

1. **Underfitting**: While regularization aims to reduce overfitting, it can also lead to underfitting if the penalty is too strong. This occurs when the model becomes too simplistic and fails to capture the underlying patterns in the data, resulting in poor performance on both training and validation datasets.

2. **Model Complexity**: Regularization primarily addresses the complexity of the model. If the model architecture itself is not suitable for the task (e.g., too simple or inappropriate for the data distribution), regularization won't help improve performance. The model may still struggle to learn the necessary features, leading to poor validation performance.

3. **Insufficient Data**: If the training dataset is small or not representative of the validation dataset, regularization may not compensate for the lack of data. The model might learn noise or irrelevant patterns from the training data, which regularization cannot correct.

4. **Improper Regularization Parameter ($\lambda$)**: The effectiveness of regularization depends on the choice of the regularization parameter $\lambda$. If $\lambda$ is set too high, it can overly penalize the model's parameters, leading to underfitting. Conversely, if it's too low, it may not sufficiently reduce overfitting.

5. **Feature Interaction**: Regularization techniques like $L_1$ and $L_2$ may not effectively capture complex interactions between features. If the relationships in the data are intricate, regularization alone may not improve the model's ability to generalize.

6. **Validation Set Issues**: The validation set itself may not be representative of the problem space, or it may contain noise or outliers that affect the model's performance. Regularization won't address these issues if the validation data is flawed.



<a id="why-and-when-is-feature-scaling-necessary"></a>
# Why And When Is Feature Scaling Necessary {#why-and-when-is-feature-scaling-necessary}

[Feature Scaling](#feature-scaling) is useful for models that use distances like [Support Vector Machines|SVM](#support-vector-machinessvm) and [K-means](#k-means)
### When Scaling Is Unnecessary

1. **Tree-based Algorithms:**
    - Algorithms like [Decision Tree](#decision-tree), [Random Forests](#random-forests), and Gradient Boosted Trees are invariant to feature scaling because they split data based on thresholds, not distances.
    - Example: Splits are determined by feature values, not their magnitude.
      
2. **Data with Uniform Scales:**
    - If all features have the same range or are already normalized (e.g., percentages), scaling may not be required.
      


<a id="why-does-increasing-the-number-of-models-in-a-ensemble-not-necessarily-improve-the-accuracy"></a>
# Why Does Increasing The Number Of Models In A Ensemble Not Necessarily Improve The Accuracy {#why-does-increasing-the-number-of-models-in-a-ensemble-not-necessarily-improve-the-accuracy}


Increasing the number of models in an ensemble ([Model Ensemble](#model-ensemble)) does not always lead to improved accuracy due to several limiting factors:

- **Convergence of Predictions**: Additional models may lead to similar predictions, resulting in minimal changes to the overall output.
- **Limited Data Representation**: If the dataset is noisy or incomplete, more models will only aggregate existing noise without capturing new patterns.
- **Diminishing Returns**: Each new model contributes less unique information, and performance is ultimately limited by the irreducible error in the data.
- **Increased Complexity**: More models increase computational costs and training times without necessarily improving accuracy.
- **Overfitting Risk**: Adding complex models can lead to overfitting, where the ensemble learns noise instead of underlying patterns.

<a id="why-does-label-encoding-give-different-predictions-from-one-hot-encoding"></a>
# Why Does Label Encoding Give Different Predictions From One Hot Encoding {#why-does-label-encoding-give-different-predictions-from-one-hot-encoding}

Label Encoding and One-Hot Encoding give different predictions because they represent categorical variables in fundamentally different ways. 

- **Label Encoding** might cause issues by implying an ordinal relationship between categories, leading to biased predictions.
- **One-Hot Encoding** prevents this by treating categories independently, resulting in more accurate predictions when there's no natural order among the categories.

### **Label Encoding:**

- **How It Works**: Label Encoding assigns an integer value to each unique category in a feature. For example, if you have three towns: `['West Windsor', 'Robbinsville', 'Princeton']`, Label Encoding would convert them into numerical values like this:
    - West Windsor → 0
    - Robbinsville → 1
    - Princeton → 2
- **Interpretation in the Model**: When you use Label Encoding, the model interprets the numbers as continuous values, meaning it sees a numeric relationship between them (i.e., "Princeton" might be considered numerically higher than "West Windsor" and closer to "Robbinsville"). This can cause issues if the numeric values don’t have any ordinal relationship.

### **One-Hot Encoding:**

- **How It Works**: One-Hot Encoding creates a separate binary (0 or 1) column for each unique category. For example, the three towns would be represented as:
    - West Windsor → [1, 0, 0]
    - Robbinsville → [0, 1, 0]
    - Princeton → [0, 0, 1]
- **Interpretation in the Model**: One-Hot Encoding treats each category as a separate binary feature and does not impose any ordinal relationship between them. This means the model doesn’t assume that one category is greater or lesser than another. Each category is treated independently.

### **Key Differences in Predictions:**

1. **Ordinal vs. Non-Ordinal Data Representation**:
    - With **Label Encoding**, the model might treat "Robbinsville" (encoded as 1) as closer to "West Windsor" (encoded as 0) than "Princeton" (encoded as 2), even though these categories don't have any inherent numerical relationship. This can lead the model to incorrectly infer relationships based on these numeric values.
    - With **One-Hot Encoding**, no such relationship is assumed. Each category is represented as a vector of 0s and 1s, and the model treats them as distinct entities, preventing any assumptions about their order.

2. **Model Interpretation**:
    - **Label Encoding** introduces an implicit ordinal relationship (e.g., 0 < 1 < 2) that can influence the model, especially for linear models like Linear Regression, which assumes that the input features are on a similar scale. This may lead to inappropriate relationships in the regression model.
    - **One-Hot Encoding** avoids this issue by using binary columns for each category, effectively preventing the model from assuming an ordinal relationship between the categories.

3. **Feature Space**:
    - **Label Encoding** results in a single feature column for the categorical variable.
    - **One-Hot Encoding** expands the feature space, creating as many columns as there are categories. In the case of a categorical feature with many unique values, this can significantly increase the dimensionality of the model.

### Why Predictions Differ:
- In Label Encoding, a linear regression model might learn that "Robbinsville" is numerically closer to "West Windsor" than "Princeton," and this might distort the predictions.
- In One-Hot Encoding, the model treats each category independently, leading to different relationships being learned (if any) between the categories and the target variable.

### Example:

Let's assume you are predicting house prices, and you're using a linear regression model where the `town` feature is the only predictor (along with some other features like `area`).

- **With Label Encoding**:
    - The model will interpret the encoded numeric values (0 for West Windsor, 1 for Robbinsville, and 2 for Princeton) and might incorrectly assume a relationship such as: "Princeton" (2) is somehow numerically "higher" than "West Windsor" (0), which doesn’t reflect any meaningful relationship.
    - This can lead to biased coefficients and, therefore, inaccurate predictions.

- **With One-Hot Encoding**:
    - The model will learn the effect of each category (West Windsor, Robbinsville, and Princeton) as a separate feature, with no assumption of ordinality.
    - This often results in more accurate predictions, especially when categorical features have no inherent order.


<a id="why-does-the-adam-optimizer-converge"></a>
# Why does the Adam Optimizer converge {#why-does-the-adam-optimizer-converge}

### Why the Adam Optimizer Converges

The Adam optimizer is able to efficiently handle sparse gradients and adaptively adjust learning rates. The convergence of Adam, often observed as a flattening of the cost function, can be attributed to several factors inherent to its design and the characteristics of the dataset being used.

The convergence of the Adam optimizer, resulting in a stable cost value, is a product of its adaptive learning rate, regularization effects, numerical stability mechanisms, and the dataset's characteristics. 
#### 1. Convergence to Local Minimum or Saddle Point

**Adaptive Learning Rate:** Adam adjusts the learning rate for each parameter individually, allowing it to quickly converge towards a local minimum or saddle point. This adaptability helps in navigating complex cost landscapes but may also cause the optimizer to plateau at a local minimum rather than reaching the global minimum. This plateauing effect can result in a stable final cost value, such as 0.146.

**Gradient Behaviour:** As Adam converges, the gradients often become small or near zero, slowing down the learning process and leading to a flattened cost curve.

#### 2. Learning Rate

**Impact of Learning Rate (\(\alpha\)):** The choice of learning rate significantly influences convergence speed. A larger learning rate might cause overshooting, while a smaller one might lead to slow convergence. Common values like 0.001, 0.005, 0.01, and 0.1 are used to balance these effects.

**Stability in Suboptimal Regions:** If the learning rate is not optimal, Adam might get stuck in a suboptimal region, such as a local minimum or saddle point, resulting in a stable cost value.

#### 3. Regularization

**L2 Regularization:** The inclusion of L2 regularization helps prevent overfitting but also affects the optimization process by slightly increasing the final cost value. The observed cost value is a balance between error reduction and the regularization penalty.

#### 4. Numerical Stability

**Moment Estimates:** Adam uses first and second moment estimates (mean of gradients and squared gradients) to update parameters. As these estimates improve, the magnitude of updates decreases, leading to smaller changes in the cost function and eventual flattening.

**Epsilon for Stability:** The epsilon parameter ensures numerical stability by preventing division by very small values, which can also lead to reduced update steps when squared gradients are small.

#### 5. Dataset Characteristics

**Simplicity of the Dataset:** The characteristics of the dataset, such as its simplicity or complexity, can influence convergence. In a simple dataset with few features, the optimizer might reach a plateau quickly due to the limited complexity of the problem.

#### 6. Final Cost Comparison

**Reasonable Solution:** The stable cost value, such as 0.146, indicates that Adam has found a reasonable solution given the dataset and optimizer settings. It reflects a balance between minimizing error and applying regularization.




<a id="why-is-named-entity-recognition-ner-a-challenging-task"></a>
# Why Is Named Entity Recognition (Ner) A Challenging Task {#why-is-named-entity-recognition-ner-a-challenging-task}

Named Entity Recognition (NER) is considered a challenging task for several reasons:

1. **Ambiguity**: Entities can be ambiguous, meaning the same word or phrase can refer to different entities depending on the context. For example, "Washington" could refer to a city, a state, or a person. Disambiguating these entities requires a deep understanding of context.

2. **Variability in Language**: Natural language is highly variable and can include slang, idioms, and different syntactic structures. This variability makes it difficult for NER models to consistently identify entities across different texts.

3. **Named Entity Diversity**: Entities can take many forms, including names, organizations, locations, dates, and more. Each type may have different characteristics, requiring the model to adapt to various patterns.

4. **Lack of Annotated Data**: High-quality annotated datasets are crucial for training NER models. However, creating such datasets can be time-consuming and expensive, leading to limited training data for certain domains or languages.

5. **Multilingual Challenges**: NER systems often struggle with multilingual texts, where the same entity may be represented differently in different languages. This adds complexity to the recognition process.

6. **Nested Entities**: In some cases, entities can be nested within each other (e.g., "The University of California, Berkeley"). Recognizing such nested structures can be particularly challenging for NER systems.

7. **Domain-Specific Language**: Different domains (e.g., medical, legal, technical) may have specific terminologies and entities that general NER models may not recognize effectively without domain-specific training.

<a id="why-is-the-central-limit-theorem-important-when-working-with-small-sample-sizes"></a>
# Why Is The Central Limit Theorem Important When Working With Small Sample Sizes {#why-is-the-central-limit-theorem-important-when-working-with-small-sample-sizes}

The [Central Limit Theorem](#central-limit-theorem) (CLT) is particularly important for data scientists working with small sample sizes. It enables the use of various statistical methods, and helps in making valid inferences about the population from limited data.

1. **Assumption of Normality**: The CLT states that the sampling [Distributions|distribution](#distributionsdistribution) of the sample means will approximate a normal distribution, regardless of the underlying population distribution, as long as the sample size is sufficiently large. 
2. 
3. This is crucial for data scientists because many statistical methods and tests (such as t-tests, ANOVA, and regression analysis) rely on the [assumption of normality](#assumption-of-normality). Even with small sample sizes, the CLT provides a foundation for making inferences about the population.

4. **Confidence Intervals and [Hypothesis Testing](#hypothesis-testing)**: The CLT enables data scientists to construct confidence intervals and perform hypothesis tests even when the sample size is small. By using the sample mean and the standard error (which is derived from the sample size), data scientists can estimate the range within which the true population mean is likely to fall, and test hypotheses about population parameters.

5. **Reduction of Variability**: The variance of the sampling distribution decreases as the sample size increases, which means that larger samples provide more reliable estimates of the population mean. For small sample sizes, the CLT helps data scientists understand the potential variability in their estimates and make more informed decisions based on their data.

6. **Practical Application**: In many real-world scenarios, obtaining large samples may not be feasible due to time, cost, or logistical constraints. The CLT allows data scientists to work with smaller samples while still applying statistical techniques that assume normality, thus broadening the scope of analysis.

7. **Robustness of Results**: The CLT provides a theoretical justification for the robustness of statistical methods. Even if the original data is not normally distributed, the means of sufficiently large samples will tend to be normally distributed, allowing for more reliable conclusions.


<a id="why-json-is-better-than-pickle-for-untrusted-data"></a>
# Why Json Is Better Than Pickle For Untrusted Data {#why-json-is-better-than-pickle-for-untrusted-data}

**JSON vs. [Pickle](#pickle)**:

1. **Security**:
   - **JSON**: JSON is a text-based data format that is inherently safer for handling untrusted data. It <mark>only</mark> supports basic data types like strings, numbers, arrays, and objects, which reduces the risk of executing arbitrary code.
   - **Pickle**: Pickle is a Python-specific binary serialization format that <mark>can serialize and deserialize complex Python objects.</mark> However, it can <mark>execute arbitrary code</mark> during deserialization, making it unsafe for handling untrusted data.

2. **Interoperability (the ability of computer systems or software to exchange and make use of information)**:
   - **JSON**: JSON is language-agnostic and widely used across different programming environments, making it ideal for data interchange between systems.
   - **Pickle**: Pickle is specific to Python, which limits its use in cross-language applications.

3. **Readability**:
   - **JSON**: Being a text format, JSON is human-readable and easy to debug.
   - **Pickle**: Pickle produces binary data, which is not human-readable and harder to debug.

For these reasons, JSON is preferred over Pickle when dealing with untrusted data, as it minimizes security risks and offers better interoperability and readability.

<a id="why-type-1-and-type-2-matter"></a>
# Why Type 1 And Type 2 Matter {#why-type-1-and-type-2-matter}

Type I and Type II errors are used in evaluating the performance of classification models, and understanding their differences is essential for interpreting model results effectively.

![Pasted image 20250312064809.png](../content/images/Pasted%20image%2020250312064809.png)

### Type I Error (False Positive)
- **Definition**: A Type I error occurs when the model incorrectly predicts the positive class. In other words, it identifies a negative instance as positive.
- **Example**: If a model predicts that an email is spam (positive) when it is actually not spam (negative), this is a Type I error.
- **Consequences**: Type I errors can lead to unnecessary actions or consequences, such as misclassifying legitimate emails as spam, which may result in important messages being missed.

### Type II Error (False Negative)
- **Definition**: A Type II error occurs when the model incorrectly predicts the negative class. This means it fails to identify a positive instance.
- **Example**: If a model predicts that an email is not spam (negative) when it is actually spam (positive), this is a Type II error.
- **Consequences**: Type II errors can lead to missed opportunities or risks, such as allowing spam emails to clutter the inbox or failing to detect a disease in a medical diagnosis scenario.

### Why Both Errors Matter
1. **Impact on Decision-Making**: The consequences of Type I and Type II errors can vary significantly depending on the context. In some applications, such as medical diagnoses, a Type II error (failing to detect a disease) may be more critical than a Type I error (false alarm). Conversely, in fraud detection, a Type I error may lead to unnecessary investigations.

2. **Balancing Precision and Recall**: Understanding these errors helps in balancing precision (the proportion of true positives among all positive predictions) and recall (the proportion of true positives among all actual positives). Depending on the application, one may be prioritized over the other, influencing model tuning and evaluation.

3. **Model Evaluation**: Both types of errors are essential for a comprehensive evaluation of a model's performance. Metrics such as precision, recall, and the F1 score incorporate these errors to provide a more nuanced view of how well the model is performing.

4. **Risk Management**: By analyzing the trade-offs between Type I and Type II errors, practitioners can make informed decisions about model thresholds and operational strategies, ensuring that the model aligns with business or clinical objectives.



<a id="why-use-er-diagrams"></a>
# Why Use Er Diagrams {#why-use-er-diagrams}

[Why use ER diagrams](#why-use-er-diagrams)

Cleaning a dataset before creating an [ER Diagrams](#er-diagrams) is crucial for ensuring accuracy and reliability in your database design

1. [Data Quality](#data-quality): Cleaning the dataset helps identify and rectify errors, inconsistencies, and missing values. This ensures that the data accurately represents the real-world entities and relationships you intend to model.

2. [Normalised Schema](#normalised-schema): Before creating an ER diagram, it's essential to normalize the data, which involves organizing it efficiently to reduce redundancy and dependency. Cleaning the dataset beforehand allows you to identify redundant information and eliminate it, leading to a more streamlined ER diagram.

3. Entity Identification: Through data cleaning, you can properly identify the entities within your dataset. This involves determining which attributes belong to which entity, as well as identifying any composite or derived attributes. Proper entity identification is fundamental to creating an accurate ER diagram.

4. Relationship Clarity: Cleaning the dataset helps clarify the relationships between entities. By ensuring that the data accurately reflects the relationships between different entities, you can create a more precise ER diagram that accurately represents the connections between various elements.

5. Data Consistency: [Data Cleansing](#data-cleansing) ensures consistency across the dataset, which is essential for maintaining integrity in the ER diagram. Consistent data allows for clearer identification of relationships and attributes, leading to a more effective database design.

<a id="wikipedia_apipy"></a>
# Wikipedia_Api.Py {#wikipedia_apipy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Wikipedia_API.py

<a id="windows-subsystem-for-linux"></a>
# Windows Subsystem For Linux {#windows-subsystem-for-linux}

[Windows Subsystem for Linux](#windows-subsystem-for-linux) (WSL) is a compatibility layer for running Linux binary executables natively on Windows 10 and Windows 11. It allows users to run a Linux environment directly on Windows without the need for a virtual machine or dual-boot setup. 

Key features of WSL include:

1. **Integration with Windows**: Users can access files from both Windows and the [Linux](#linux) environment seamlessly.
2. **Multiple Distributions**: WSL supports various Linux distributions, such as [Ubuntu](#ubuntu), Debian, and Fedora, which can be installed from the Microsoft Store.
3. **Command-Line Tools**: Users can run Linux command-line tools and applications directly in Windows, making it easier for developers to work in a familiar environment.
4. **Performance**: WSL provides near-native performance for Linux applications, making it suitable for development and testing.


<a id="word2vec"></a>
# Word2Vec {#word2vec}

Word2Vec is a technique for generating vector representations of words. Developed by researchers at Google, it uses a shallow [neural network](#neural-network) to produce [standardised/Vector Embedding|word embedding](#standardisedvector-embeddingword-embedding) that capture [Semantic Relationships](#semantic-relationships) and [syntactic relationships](#syntactic-relationships). Word2Vec has two main architectures:

In [ML_Tools](#ml_tools) see: [Word2Vec.py](#word2vecpy)

1. CBOW (Continuous [Bag of Words](#bag-of-words)):
    - Predicts a target word given its context (neighboring words).
    - Efficient for smaller datasets.
      
2. Skip-Gram:
    - Predicts the context words given a target word.
    - Performs better on larger datasets.

Word2Vec generates dense, continuous vector representations where words with similar meanings are close to each other in the embedding space. For example:

- `vector("king") - vector("man") + vector("woman") ≈ vector("queen")`




<a id="word2vecpy"></a>
# Word2Vec.Py {#word2vecpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/NLP/Word2Vec.py

The script can benefit from **Word2Vec embeddings** by replacing the randomly initialized embeddings with pretrained or trained embeddings generated using Word2Vec. These embeddings provide a meaningful semantic structure that is learned from a corpus of text, enhancing the visualization and [cosine similarity](#cosine-similarity) calculations.

#### Benefits:

1. **Meaningful Relationships**: Words like "king" and "queen" will naturally be closer than "king" and "apple."
2. **Analogy Solving**: Word2Vec supports vector arithmetic to solve word analogies (e.g., "man is to king as woman is to queen").
3. **Improved Visualizations**: The embeddings reflect real-world semantic and syntactic relationships, making the 2D plots more interpretable.

### Further Enhancements

1. **Train Your Word2Vec**:
    - Train embeddings on a custom corpus using `gensim.models.[Word2Vec](#word2vec)` to reflect domain-specific semantics.
    
1. **Hybrid Embeddings**:
    - Combine Word2Vec with other models (e.g., [BERT](#bert) or Sentence [Transformer|Transformers](#transformertransformers)) for tasks requiring contextual understanding.

**Using `glove-wiki-gigaword-100`**:

- A GloVe model with 100-dimensional embeddings trained on the Wikipedia Gigaword dataset.
- Approximate size: ~100MB.

### Expected Outcome

1. **Visualization**:
    - Terms from the same category (e.g., royalty, fruits, animals) will cluster together in the t-SNE plot.
2. **Cosine Similarity**:
    - Similar terms (e.g., "king" and "queen" or "apple" and "orange") will have higher cosine similarity scores.
3. **Semantic Diversity**:
    - The expanded list increases the diversity of semantic relationships and highlights the strength of embeddings in grouping similar concepts.

<a id="wordnet"></a>
# Wordnet {#wordnet}



<a id="wrapper-methods"></a>
# Wrapper Methods {#wrapper-methods}

Used in [Feature Selection](#feature-selection). Wrapper methods are powerful because they directly optimize the performance of the machine learning model by selecting the most informative subset of features. 

1. **Iterative Approach**: Unlike [Filter method](#filter-method), which assess the relevance of features based on statistical properties, wrapper methods <mark>directly involve the machine learning algorithm in the feature selection process.</mark>
   
2. **Subset Selection**: Wrapper methods work by creating different subsets of features from the original dataset and training a model on each subset. These subsets can be combinations of different features or a subset of all features.

3. [Model Evaluation](#model-evaluation): After training a model on each subset of features, the performance of each model is evaluated using a performance metric, such as accuracy, precision, recall, or F1-score, depending on the problem type (classification or regression).

4. **Optimization Criterion**: The goal of wrapper methods is to find the subset of features that maximizes the performance of the machine learning model. This can be achieved by selecting the subset that yields the highest performance metric on a validation set or through cross-validation.

5. **Computational Intensity**: Wrapper methods are computationally intensive because they involve training multiple models for each possible combination of features. As a result, they can be slower and require more computational resources compared to filter methods.

## **Examples of Wrapper Methods**:

   - **Forward Selection**: Starts with an empty set of features and iteratively adds one feature at a time, selecting the feature that improves model performance the most.
    
   - **Backward Elimination**: Begins with all features and iteratively removes one feature at a time, selecting the feature whose removal improves model performance the least.
     
   - **Recursive Feature Elimination (RFE)**: Iteratively removes features from the full feature set based on their importance, as determined by a specified machine learning algorithm.
     
   - **Selection Criteria**: The choice of performance metric and optimization criterion depends on the specific machine learning task and dataset characteristics. It's essential to select a metric that aligns with the goals of the project and to validate the selected subset of features on unseen data.



<a id="xgboost"></a>
# Xgboost {#xgboost}


XGBoost (eXtreme Gradient Boosting) is a highly efficient and flexible implementation of [Gradient Boosting](#gradient-boosting) that is widely used for its accuracy and performance in machine learning tasks.

### How does XGBoost work

It works by building an [Model Ensemble](#model-ensemble) - ensemble of decision trees, where each tree is trained to correct the errors made by the previous ones. Here's a breakdown of how XGBoost works:
### Key Concepts

1. Gradient Boosting Framework:
   - XGBoost is based on the gradient boosting framework, which builds models sequentially. Each new model aims to reduce the errors (residuals) of the combined ensemble of previous models.

2. Decision Trees:
   - XGBoost typically uses decision trees as the base learners. These trees are added one at a time, and existing trees in the model are not changed.

3. Objective Function:
   - The objective function in XGBoost consists of two parts: the loss function and a regularization term.
   - [Loss function](#loss-function): Measures how well the model fits the training data. For regression, this might be mean squared error; for classification, it could be logistic loss.
   - [Regularisation](#regularisation): Helps prevent overfitting by penalizing complex models. XGBoost supports both L1 (Lasso) and L2 (Ridge) regularization.

4. Additive Training:
   - XGBoost adds trees to the model sequentially. Each tree is trained to minimize the loss function, taking into account the errors made by the previous trees.

5. [Gradient Descent](#gradient-descent)
   - The model uses gradient descent to minimize the loss function. It calculates the gradient of the loss function with respect to the model's predictions and uses this information to update the model.

6. [learning rate](#learning-rate) ($\eta$):
   - A parameter that scales the contribution of each tree. A smaller learning rate requires more trees but can lead to better performance.

7. Tree Pruning:
   - XGBoost uses a technique called "max depth" to control the complexity of the trees. It also employs a "max delta step" to ensure that the updates are not too aggressive.

8. [Handling Missing Data](#handling-missing-data)
   - XGBoost can handle missing data internally by learning the best direction to take when a value is missing.

9. Parallel and Distributed Computing:
   - XGBoost is designed to be highly efficient and can leverage parallel and distributed computing to speed up training.

Key Features:
- Tree Splitting: Builds [Decision Tree](#decision-tree) in a level-wise manner, leading to balanced trees and efficient computation.
- Parameters: Key parameters include `eta` (learning rate) and `max_depth` (maximum depth of a tree), which control the model's complexity and learning process.

### Workflow

1. Initialization:
   - Start with an initial prediction, often the mean of the target values for regression or a uniform probability for classification.

2. Iterative Training:
   - For each iteration, compute the gradient of the loss function with respect to the current predictions.
   - Fit a new decision tree to the negative gradient (residuals).
   - Update the model by adding the new tree, scaled by the learning rate.

3. Model Output:
   - The final model is a weighted sum of all the trees, where each tree contributes to the final prediction.

Advantages:
- Accuracy: Known for its high accuracy and robustness across various machine learning tasks.
- [Regularisation](#regularisation): Supports L1 (Lasso) and L2 (Ridge) regularization to prevent overfitting.
- Flexibility: Offers a wide range of hyperparameters for fine-tuning models.

Use Cases:
- Structured Data: Particularly effective for structured data and tabular datasets.
- [Interpretability](#interpretability): Suitable when model interpretability is important.
- [Hyperparameter Tuning](#hyperparameter-tuning): Ideal for scenarios where extensive hyperparameter tuning is feasible.
### Implementing XGBoost in Python

#### Step 2: Import Necessary Libraries

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
#### Step 3: Prepare Your Data

Split your dataset into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

#### Step 4: Convert Data to DMatrix

Convert the data into DMatrix, the optimized data structure used by XGBoost:

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```

#### Step 5: Set Parameters

Define the parameters for the XGBoost model:

```python
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',  # Use 'reg:squarederror' for regression tasks
    'eval_metric': 'logloss'
}
```

#### Step 6: Train the Model

Train the XGBoost model using the training data:
```python
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)
```

#### Step 7: Make Predictions and Evaluate
Make predictions on the test set and evaluate the model's performance:

```python
y_pred = bst.predict(dtest)
y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")
```

# Notes

Set up an example of XGBoost. Plot the paramater space slices "Min_Samples_split", "Max_Depth" vs accuracy.

```python
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
xgb_model.best_itersation
```

<a id="yaml"></a>
# Yaml {#yaml}


Stands for [YAML ain't markup language](https://github.com/yaml/yaml-spec) and is a superset of JSON

- lists begin with a hyphen
- dependent on whitespace / indentation
- better suited for configuration than [Json](#json)

YAML is a data serialization language often used to write configuration files. Depending on whom you ask, YAML stands for yet another markup language, or YAML isn’t markup language (a recursive acronym), which emphasizes that YAML is for data, not documents.

<a id="z-normalisation"></a>
# Z Normalisation {#z-normalisation}


https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Outliers/outliers_z_score.py

Z-normalisation, also known as z-score normalization, is a technique used to standardize the range of independent variables or features of data. 

This process is used in preparing data for [machine learning algorithms](#machine-learning-algorithms), especially those that rely on distance calculations, such as k-nearest neighbors and [gradient descent](#gradient-descent) optimization.

### Why Normalize?

- Consistency Across Features: By normalizing, the peak-to-peak range of each column is reduced from a factor of thousands to a factor of 2-3. This ensures that each feature contributes equally to the distance calculations, preventing features with larger ranges from dominating the results.
  
- Centered Data: The range of the normalized data (x-axis) is centered around zero and roughly +/- 2. This centering is beneficial for algorithms that assume data is normally distributed around zero.

- Improved Learning Rates: Normalization allows for a larger [learning rate](#learning-rate) in [Gradient Descent](#gradient-descent), which can speed up convergence and improve the efficiency of the learning process.

### Z-Score Normalization

Z-score normalization transforms the data so that each feature has:
- A mean of 0
- A standard deviation of 1

To implement z-score normalization, adjust your input values using the formula:

$$x^{(i)}_j = \frac{x^{(i)}_j - \mu_j}{\sigma_j}$$
Where:
- $x^{(i)}_j$ is the value of the feature $j$ for the $i$-th example.
- $\mu_j$ is the mean of all the values for feature $j$.
- $\sigma_j$ is the standard deviation of feature $j$.

The mean and standard deviation are calculated as follows:

$$\mu_j = \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j$$

$$\sigma^{2}_j = \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^{2}$$

Where $m$ is the number of examples.

### Examples


![Pasted image 20241224091151.png](../content/images/Pasted%20image%2020241224091151.png)

See that they are centred around 0.

![Pasted image 20241224091157.png](../content/images/Pasted%20image%2020241224091157.png)

Below we see that its centered around 0 and been brought together.

![Pasted image 20241224091007.png](../content/images/Pasted%20image%2020241224091007.png)


<mark>Rescales the feature values</mark> to a range of [0, 1]. This is useful when you want to ensure that all features contribute equally to the distance calculations.
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)  # Rescales each feature to [0, 1]
```

<a id="z-score"></a>
# Z Score {#z-score}


Z-scores standardize a value relative to a distribution by measuring how many standard deviations it is from the mean. This is useful for [standardised/Outliers|Outliers](#standardisedoutliersoutliers) and [Normalisation](#normalisation).

Definition:  
The Z-score of a value $x$ is given by:
    $$Z = \frac{x - \bar{x}}{s}$$
    
where $\bar{x}$ is the sample mean and $s$ is the sample standard deviation.
    
Interpretation:
- $Z = 0$: The value equals the mean.
- $|Z| > 2$: Indicates a possible outlier (if normality is assumed).
- Z-scores allow comparisons across different distributions.
	
Assumptions:
- Data is approximately normally distributed.
- Useful primarily when comparing existing values to a distribution.

Use Cases:
- Standardizing data for machine learning algorithms.
- Detecting anomalies.
- Ranking or scoring values.

Related terms:
- [Z-Test](#z-test)
- [Z-Normalisation](#z-normalisation)
- [Z-Score](#z-score)

### **2. Modified Z-Score**

- **Formula:**  
    $M = \frac{0.6745 \cdot (X - \text{median})}{\text{MAD}}$
    - $MAD$: Median Absolute Deviation
- **Procedure:**
    - Use this method for datasets with extreme outliers.
    - Points with $M > 3.5$ are typically anomalies.

<a id="z-scores-vs-prediction-intervals"></a>
# Z Scores Vs Prediction Intervals {#z-scores-vs-prediction-intervals}


[Z-Score](#z-score) and [Prediction Intervals](#prediction-intervals) serve different purposes. Z-scores assess existing values within a dataset, while prediction intervals estimate the likely range for future observations.

Use Z-scores to evaluate existing values or standardize. Use prediction intervals to express uncertainty about where a **new** observation is likely to fall.

**Comparison Table**:

|Feature|Z-Score|Prediction Interval|
|---|---|---|
|**Purpose**|Assess deviation from the mean|Forecast future values|
|**Formula**|$Z = \frac{x - \bar{x}}{s}$|$\bar{x} \pm t_{\alpha/2, n-1} \cdot s \cdot \sqrt{1 + \frac{1}{n}}$|
|**Distribution**|Standard Normal (Z)|Student’s t-distribution|
|**Use case**|Outlier detection, normalization|Prediction of new measurements|
|**Width of range**|Based on fixed $\sigma$|Wider—accounts for both sampling error and variability|
|**Needs population $\sigma$?**|Yes (or large $n$ to approximate)|No (uses sample $s$ and $t$ for small $n$)|

<a id="z-test"></a>
# Z Test {#z-test}

The Z-test is a statistical method used to determine if there is a <mark>significant difference between the means of two groups or to compare a sample mean to a known population mean when the population [standard deviation](#standard-deviation) is known</mark>. 

It is typically applied when the sample size is large (usually n > 30).

## Types of Z-tests

1. **One-Sample Z-test**: This test compares the mean of a single sample to a known population mean. It assesses whether the sample mean significantly differs from the population mean.

2. **Two-Sample Z-test**: This test compares the means of two independent samples. It is used when both sample sizes are large and the population variances are known or can be assumed to be equal.

## Characteristics of the Z-distribution

The Z-distribution is a normal distribution with a mean of 0 and a standard deviation of 1. It is symmetric and bell-shaped, which allows for the application of the [Central Limit Theorem](#central-limit-theorem). As sample sizes increase, the distribution of sample means approaches a normal distribution, making the Z-test applicable.

## Assumptions

For the Z-test to be valid, certain assumptions must be met:
- The data should be normally distributed, especially for smaller sample sizes. However, with large samples, the Central Limit Theorem allows for the Z-test to be used even if the data is not perfectly normal.
- The samples should be independent of each other.
- The population standard deviation should be known.

## Test Statistic

The test statistic for the Z-test is calculated using the formula:

$$ Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} $$

where:
- $\bar{X}$ = sample mean
- $\mu$ = population mean (or mean of the second sample in the two-sample test)
- $\sigma$ = population standard deviation
- $n$ = sample size

This formula allows for the comparison of the sample mean to the population mean, standardizing the difference in terms of standard deviations.