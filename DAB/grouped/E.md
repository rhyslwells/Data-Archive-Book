

# Eda {#eda}


Exploratory [Data Analysis](#data-analysis) (EDA) is an approach to analyzing datasets to summarize their main characteristics, often utilizing visual methods. EDA helps users to:

- Understand the Data's Structure: Gain insights into the organization and format of the data.
- Detect Patterns: Identify trends and patterns within the data.
- Decide on Statistical Techniques: Choose appropriate statistical methods by examining [distributions](#distributions) and [correlation](#correlation).
- Select Variables: Determine which variables to include in further analysis.
- Handle [Data Quality](#data-quality): Address issues related to data quality and integrity.
- Spot Anomalies and [standardised/Outliers](#standardisedoutliers): Identify unusual data points that may affect analysis.
- Generate and Test Hypotheses: Formulate hypotheses and validate them using statistical methods.
- Check Assumptions: Verify assumptions through statistical summaries and graphical representations.
### Common Techniques Used in EDA

- Descriptive Statistics: Calculating measures such as mean, median, mode, [standard deviation](#standard-deviation), and percentiles to summarize data.
- [Data Visualisation](#data-visualisation): Using plots and charts like histograms, box plots, scatter plots, and bar charts to visually explore data.
- Correlation Analysis: Assessing relationships between variables using [correlation](#correlation) coefficients and scatter plots.
- [Data Transformation](#data-transformation): Applying transformations to data, such as normalization or log transformation, to better understand its characteristics.
### Implementation 

In [ML_Tools](#ml_tools) see:
- [EDA_Pandas.py](#eda_pandaspy)

# Eda_Pandas.Py {#eda_pandaspy}



# Elt {#elt}


**ELT** (Extract, Load, Transform) is a data integration approach that involves three main steps:

1. **Extract (E)**: Data is extracted from a source system.
2. **Load (L)**: The raw data is loaded into a destination system, such as a data warehouse.
3. **Transform (T)**: Transformation of the data occurs within the destination system after the data has been loaded.

This approach contrasts with the traditional **ETL** (Extract, Transform, Load) method, where data is transformed before reaching the destination. For a detailed comparison, see [ETL vs ELT](#etl-vs-elt)

### Advantages of ELT
The shift from ETL to ELT has been facilitated by several factors:

- **Cost Efficiency**: The decreasing costs of cloud-based storage and computation have reduced the advantages of ETL's pre-loading data transformation.
  
- **Cloud-Based Data Warehouses**: The emergence of cloud-based data warehouses like Redshift, [BigQuery](#bigquery), and Snowflake has made the ELT approach more feasible and efficient.
### Historical Context
Historically, ETL was preferred for reasons that are now less relevant:

- **Cost Savings**: ETL was believed to save costs by filtering out unwanted data before loading. However, this is less significant with modern cloud solutions.
  
- **Complexity Management**: ETL minimizes the complexity of post-loading transformations. Yet, contemporary tools like [dbt](#dbt) simplify this process, making it easier to perform transformations after loading.



[ELT](#elt)
   **Tags**: #data_transformation, #data_integration

# Etl Pipeline Example {#etl-pipeline-example}


[Link](https://www.youtube.com/watch?v=uqRRjcsUGgk)

[Github](https://github.com/syalanuj/youtube/blob/main/de_fundamentals_python/etl.py

#### 1.  Extract using a API

Get data via api or download.

#### 2. Transform 

Put into a pandas df.

#### 3. Load

Save df as a database. Save SQLite database. Save as table.

Run functions


```python
"""
Python Extract Transform Load Example
"""

# %%
import requests
import pandas as pd
from sqlalchemy import create_engine

def extract()-> dict:
    """ This API extracts data from
    http://universities.hipolabs.com
    """
    API_URL = "http://universities.hipolabs.com/search?country=United+States"
    data = requests.get(API_URL).json()
    return data

def transform(data:dict) -> pd.DataFrame:
    """ Transforms the dataset into desired structure and filters"""
    df = pd.DataFrame(data)
    print(f"Total Number of universities from API {len(data)}")
    df = df[df["name"].str.contains("California")]
    print(f"Number of universities in california {len(df)}")
    df['domains'] = [','.join(map(str, l)) for l in df['domains']]
    df['web_pages'] = [','.join(map(str, l)) for l in df['web_pages']]
    df = df.reset_index(drop=True)
    return df["domains","country","web_pages","name"](#domainscountryweb_pagesname)

def load(df:pd.DataFrame)-> None:
    """ Loads data into a sqllite database"""
    disk_engine = create_engine('sqlite:///my_lite_store.db')
    df.to_sql('cal_uni', disk_engine, if_exists='replace')

# %%
data = extract()
df = transform(data)
load(df)


# %%
```

# Etl Vs Elt {#etl-vs-elt}


[ETL](ETL.md) (Extract, Transform, and Load) and [ELT](term/elt.md) (Extract, Load, and Transform) are two paradigms for moving data from one system to another.

==ELT is most friendly for analysts==

The main difference between them is that when an ETL approach is used, data is transformed before it is loaded into a destination system. On the other hand, in the case of ELT, any required transformations are done after the data has been written to the destination and are _then_ done _inside_ the destination -- often by executing SQL commands. The difference between these approaches is easier to understand by a visual comparison of the two approaches. 

The image below demonstrates the ETL approach to [data integration](term/data%20integration.md):

![](etl-tool.png)

While the following image demonstrates the ELT approach to data integration:

![](elt-tool.png)

[ETL](#etl) was originally used for [Data Warehousing](Data%20Warehouse.md) and ELT for creating a [Data Lake](Data%20Lake.md). 

## Disadvantages of ETL compared to ELT

**ETL** has several **disadvantages compared to ELT**, including the following:

- Generally, only transformed data is stored in the destination system, and so ==analysts must know beforehand every way== they are going to use the data, and every report they are going to produce.  
- Modifications to requirements can be costly, and often require re-ingesting data from source systems.
- Every transformation that is performed on the data may obscure some of the underlying information, and analysts only see what was kept during the transformation phase. 
- Building an ETL-based data pipeline is often beyond the technical capabilities of analysts.

# Etl {#etl}


**ETL** (Extract, Transform, Load) is a data integration process that involves moving data from one system to another. It consists of three main stages:

1. **Extract**: Collecting data from various sources, such as databases, APIs, or flat files. This may involve setting up API connections to pull data from multiple sources.

2. **Transform**: Cleaning and converting the data into a usable format. This includes filtering, aggregating, and joining data to create a unified dataset. See [Data Transformation](#data-transformation).

3. **Load**: Inserting the transformed data into a destination system, such as a data warehouse (organized), database, or data lake (unorganized).

### Historical Context
The ETL paradigm emerged in the 1970s and was traditionally preferred for its ability to transform data before it reaches the destination. This approach ensures that data is standardized and passes quality checks, enhancing overall data quality.

### Transition to ELT
In recent years, the data movement paradigm has shifted towards [ELT](#elt) (Extract, Load, Transform). This approach emphasizes keeping raw data accessible in the destination system, allowing for more flexibility in data processing. For a comparison of ETL and ELT, see [ETL vs ELT](#etl-vs-elt).

Reasons for Change to see [ELT](#elt)

### Modern ETL Tools
Current ETL processes are often managed using tools like [Apache Airflow](#apache-airflow), [dagster](#dagster), and [Temporal](term/temporal.md).

### Enhancing the ETL Process
To improve an ETL process, consider the following enhancements:

- **Error Handling**: Implement error handling to manage exceptions and prevent silent failures.
- **Logging**: Include logging to track the process flow and facilitate debugging.
- **Parameterization**: Make scripts flexible by parameterizing file paths and database connections.
- **Data Validation and Cleaning**: Incorporate steps to validate and clean the data.
- **Database Indexing and Constraints**: Optimize database tables with proper indexing and constraints for better performance.
##### Related Notes
- [ETL Pipeline Example](#etl-pipeline-example)
- [ETL vs ELT](#etl-vs-elt)




[ETL](#etl)
   **Tags**: #data_transformation, #data_integration

# Edge Machine Learning Models {#edge-machine-learning-models}

**Edge ML** refers to deploying machine learning models directly on edge devices, such as IoT sensors, smartphones, or embedded systems, instead of relying on cloud-based processing. This is crucial in scenarios requiring low-latency, real-time decision-making, or environments with limited connectivity.

#### Key Characteristics of Edge ML Models:

1. **Low Latency**
   - Models running at the edge can make real-time decisions with minimal delay. This is critical in applications like autonomous vehicles, industrial automation, or real-time health monitoring, where delays can have serious consequences.

2. **Reduced Bandwidth Usage**
   - By processing data locally, edge ML models reduce the need to send large amounts of data to the cloud for analysis. This is particularly valuable in environments with limited or expensive connectivity (e.g., remote locations or bandwidth-constrained networks).
   
3. **Privacy Preservation**
   - Processing sensitive data on-device, instead of sending it to the cloud, enhances privacy and reduces the risk of data breaches. This is important in healthcare, financial services, or any scenario involving personal data.

4. **Energy Efficiency**
   - Edge devices often have limited power resources. As a result, models deployed at the edge need to be optimized for low energy consumption, ensuring they can operate for extended periods without requiring frequent battery replacements or recharging.

#### Common Applications of Edge ML:
   
1. **Autonomous Systems (e.g., Drones, Robots, Vehicles)**
   - Autonomous systems rely on real-time decision-making for navigation, obstacle detection, and control. Edge ML allows these systems to react instantaneously to their surroundings without depending on external servers.

3. **Smart Cities and Industrial IoT**
   - Edge ML powers applications such as **traffic monitoring**, **environmental sensing**, and **predictive maintenance** in smart factories. For example, sensors in factories can use edge models to predict equipment failure before it occurs, ensuring smooth operations without cloud reliance.

#### Challenges in Edge ML:

1. **Model Compression**
   - Since edge devices often have limited storage and computational power, ML models need to be compressed or optimized (e.g., using techniques like quantization, pruning, or knowledge distillation) to run efficiently while maintaining accuracy.

2. **On-Device Model Updates**
   - Keeping models updated without frequent cloud interactions is a challenge. Edge devices need mechanisms for **incremental learning** or efficient updates without disrupting normal operations.
#### Popular Frameworks for Edge ML:

- **TensorFlow Lite**: A lightweight version of TensorFlow, designed to run on mobile and embedded devices.
- **[PyTorch](#pytorch) Mobile**: PyTorch’s framework for deploying ML models on mobile devices.
- **[ONNX](#onnx) Runtime**: Optimized for running machine learning models on various platforms, including edge devices.
- **Edge Impulse**: A platform specifically for building ML models for edge devices, particularly for IoT applications.

Edge ML is driving innovation in industries requiring decentralized, real-time intelligence, enabling devices to make smart decisions locally while minimizing reliance on cloud resources.

# Education And Training {#education-and-training}

Adaptive Learning Systems

- **Overview**: Adaptive learning systems use technology to tailor educational experiences to individual student needs. RL is instrumental in personalizing these systems.
- **Applications**:
    - **Personalized Learning Paths**: RL algorithms can create customized learning paths for students based on their performance, preferences, and engagement levels, adapting content delivery in real-time.
    - **Feedback and Assessment**: Adaptive systems can provide immediate feedback based on student responses, reinforcing concepts through targeted exercises and adjusting difficulty levels as needed.
    - **Engagement Strategies**: By analyzing student interactions, RL can suggest motivational strategies, such as gamification elements or timely reminders, to keep students engaged and motivated.

# Elastic Net {#elastic-net}


This method combines both L1 ([Lasso](#lasso)) and L2 ([Ridge](#ridge)) regularization by adding both absolute and squared penalties to the loss function. It strikes a balance between Ridge and Lasso.

It is particularly useful when you have high-dimensional datasets with highly correlated features.

The Elastic Net loss function is:

    $$\text{Loss} = \text{MSE} + \lambda_1 \sum_{i=1}^{n} |w_i| + \lambda_2 \sum_{i=1}^{n} w_i^2$$
    
where $\lambda_1$ controls the L1 regularization and $\lambda_2$ controls the L2 regularization.

#### Code

```python
from sklearn.linear_model import ElasticNet

# Initialize an Elastic Net model
model = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio controls the L1/L2 mix
model.fit(X_train, y_train)
```

# Embedded Methods {#embedded-methods}

Embedded methods for [Feature Selection](#feature-selection) ==integrate feature selection directly into the model training process.==

Embedded methods provide a convenient and efficient approach to feature selection by seamlessly integrating it into the model training process, ultimately leading to models that are more parsimonious and potentially more interpretable.

1. **Incorporated into Model Training**: Unlike [Filter method](#filter-method) and [Wrapper Methods](#wrapper-methods), which involve feature selection as a separate step from model training, embedded methods perform feature selection simultaneously with model training. This means that feature importance or relevance is determined within the context of the model itself.

2. **Regularization Techniques**: Embedded methods commonly use [Regularisation](#regularisation) techniques to penalize the inclusion of unnecessary features during model training. 

3. **Automatic Feature Selection**: Embedded methods automatically select the most relevant features by learning feature importance during the training process. The model adjusts the importance of features iteratively based on their contribution to minimizing the [Loss function](#loss-function).

4. **Examples of Embedded Methods**:
   - **[Lasso](#lasso) (L1 Regularization)**:
   - [Elastic Net](#elastic-net): Elastic Net combines L1 ([Lasso](#lasso)) and L2 ([Ridge](#ridge)) regularization .
   - **Tree-based Methods**: [Decision Tree](#decision-tree) and ensemble methods like [Random Forests](#random-forests) and [Gradient Boosting](#gradient-boosting) inherently perform feature selection during training by selecting the most informative features at each split node of the tree.

5. **Advantages**:
   - Simplicity: Embedded methods simplify the feature selection process by integrating it into model training, reducing the need for additional preprocessing steps.
   - Efficiency: Because feature selection is performed during model training, embedded methods can be more computationally efficient compared to wrapper methods, which require training multiple models.

6. **Considerations**:
   - Hyperparameter Tuning: Tuning regularization parameters or other model-specific parameters may be necessary to optimize feature selection performance.
   - Model [interpretability](#interpretability): While embedded methods can automatically select features, interpreting the resulting model may be challenging, especially for complex models like ensemble methods.



# Encoding Categorical Variables {#encoding-categorical-variables}


### Overview

Categorical variables need to be converted into numerical representations to be used in models, particularly in [Regression](#regression) analysis. This process is essential for transforming categorical results into a format that algorithms can interpret.

### Label Encoding

This method assigns a unique integer to each category in the variable.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
var1_cat = df['var1']  # Replace df with your DataFrame
var1_encoded = label_encoder.fit_transform(var1_cat)
```
For example, if `df[col]` contains the categories `['apple', 'banana', 'orange']`, the `LabelEncoder` would transform them into `[0, 1, 2]`.

However, keep in mind that this encoding can imply an order or hierarchy in the data, which might not be intended. In some cases, you might want to use `OneHotEncoder` instead, which creates a binary vector for each category.{}

Given a term in the df you can transform it without needing to look up its value.
```python
company="google"
company_n = LabelEncoder().transform([company])
```
### One-Hot Encoding

This technique creates a binary column for each category, allowing the model to treat each category as a separate feature.

```python
from sklearn.preprocessing import OneHotEncoder

binary_encoder = OneHotEncoder(categories='auto')
var1_1hot = binary_encoder.fit_transform(var1_encoded.reshape(-1, 1))
var1_1hot_mat = var1_1hot.toarray()
var1_DF = pd.DataFrame(var1_1hot_mat, columns=['cat1', 'cat2', 'cat3'])  # Adjust column names as needed
var1_DF.head()
```

Understanding OneHotEncoder:

The `OneHotEncoder` from `sklearn.preprocessing` is used to convert categorical integer values into a format that can be provided to machine learning algorithms to do a better job in prediction. It creates a binary column for each category and returns a sparse matrix or dense array.

### Converting All Categorical Variables to Dummies

To convert all categorical variables in a DataFrame to dummy variables, you can use the following loop:

```python
for col in df.columns:
    if df[col].dtype == 'object':
        dummies = pd.get_dummies(df[col], drop_first=False)
        dummies = dummies.add_prefix(f'{col}_')
        df.drop(col, axis=1, inplace=True)
        df = df.join(dummies)
```

Dummy Variable Trap:
When using one-hot encoding, it's important to avoid the **dummy variable trap**, which occurs when one category can be perfectly predicted from the others. To prevent this, you can drop one of the dummy variables, as one column is sufficient to represent a binary choice (0 or 1).

### Alternative Encoding Method
Another way to encode categorical variables is by mapping them directly to integers:

```python
dataset['var1'] = dataset['var1'].map({'A': 0, 'B': 1, 'C': 2}).astype(int)
```
### Related Topics
- **[Regression](#regression)**: Understanding how regression models utilize encoded variables.
- **[Feature Engineering](#feature-engineering)**: Techniques to enhance model performance through better feature representation.
### Overview

- Categorical variables need to be converted into numerical representations for use in models. This is essential for transforming categorical data into a format that algorithms can interpret.

### Methods

- Label Encoding: Assigns a unique integer to each category.
- One-Hot Encoding: Creates a binary column for each category, allowing the model to treat each category as a separate feature.

### Example Code

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

# Label Encoding
label_encoder = LabelEncoder()
var1_encoded = label_encoder.fit_transform(df['var1'])

# One-Hot Encoding
binary_encoder = OneHotEncoder(categories='auto')
var1_1hot = binary_encoder.fit_transform(var1_encoded.reshape(-1, 1))
var1_1hot_mat = var1_1hot.toarray()
var1_DF = pd.DataFrame(var1_1hot_mat, columns=['cat1', 'cat2', 'cat3'])
```

# Energy ABM

- **Complex Systems Understanding**: Energy systems involve numerous stakeholders (producers, consumers, regulators) with diverse interests and behaviors. [Agent-Based Modelling|ABM](#agent-based-modellingabm) helps capture this complexity, providing a clearer picture of system dynamics.
- **Adaptive Behavior**: Agents in ABM can adapt their behavior based on interactions, mirroring how consumers and producers might respond to incentives or changes in the market.
- **Scenario Analysis**: ABM allows for "what-if" analyses, enabling stakeholders to explore different scenarios, such as the impact of implementing new technologies or policies on energy systems.
- **Data-Driven Insights**: With the rise of smart meters and IoT devices, ABM can leverage real-time data to improve model accuracy and relevancy, enhancing decision-making processes.

# Energy Storage {#energy-storage}


## Energy Storage

Battery farms exist.

Stored energy can be traded.

Stored energy can be stored using distributed system such as EV cars.

# Energy {#energy}


Areas of interest:
- [Smart Grids](#smart-grids)
- [Energy Storage](#energy-storage)
- [Demand forecasting](#demand-forecasting)
- [Network Design](#network-design)
- [Energy ABM](#energy-abm)

Questions:
- [How to model to improve demand forecasting](#how-to-model-to-improve-demand-forecasting)
- What patterns can be identified in consumer behavior data to inform energy pricing strategies?
- How can predictive maintenance be implemented using data from smart sensors in energy infrastructure?

**Techniques:**
- **[Differential Equations](#differential-equations)**: Used to model dynamic systems in energy generation and consumption. For example, they can describe the behavior of power systems over time or the thermal dynamics of energy storage systems.
- **[Stochastic Modeling](#stochastic-modeling)**: Involves random variables to model uncertainties in energy production (e.g., variability in solar or wind energy) and consumption.
- [Agent-Based Modelling](#agent-based-modelling)Simulates interactions of agents (consumers, producers, regulators) to understand complex systems and emergent phenomena in energy markets.
- **Time Series Analysis**: Analyzing historical data to forecast future energy demand or production trends.
- **[Regression](#regression) Analysis**: Used to model relationships between different variables, such as energy prices and consumption patterns.
- [Neural network|Neural Network](#neural-networkneural-network) Particularly deep learning, is applied for complex pattern recognition in large datasets, such as detecting anomalies in energy consumption or predicting equipment failures.


Dymanic pricing, incentised load management, local generation 
  
Use green energy if on grid

# Environment Variables {#environment-variables}

Solution 1: Set Environment Variables Permanently (Recommended)
This ensures that environment variables persist across sessions.

On Windows (Permanent)
Open Control Panel → System → Advanced system settings → Environment Variables.

Under System Variables, click New.

Variable Name: PG_USER

Variable Value: postgres

Click New again.

Variable Name: PG_PASSWORD

Variable Value: your_password

Click OK and restart your computer.

Once restarted, Jupyter Notebook should be able to access the variables.

# Epoch {#epoch}


An epoch in machine learning is a single pass through the entire training dataset. The number of epochs, denoted as $N$, determines how many times the data is applied to the model.

Why Use Multiple Epochs?
- **Repetition for Learning:** The data is applied to the model $N$ times to improve learning and accuracy. For example, if $N = 10$, the model will see the entire dataset 10 times.

Example
```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```

- **Epoch 1/10:** Indicates the model is currently on the first epoch out of a total of 10.
- **Batches:** For efficiency, the dataset is divided into smaller groups called 'batches'. In TensorFlow, the default batch size is 32. With 200,000 examples, this results in 6,250 batches.
- **Batch Execution:** The notation `6250/6250` shows the progress of batch execution within the current epoch.



# Epub {#epub}


An **EPUB** (short for *electronic publication*) file is a widely used **open eBook format** that is designed for **reflowable content**, meaning it can adapt its layout to fit various screen sizes—unlike PDFs, which preserve a fixed layout.

### Key Features of EPUB
- **Reflowable Text:** The content adjusts to screen size, font preferences, and orientation. This is ideal for smartphones, tablets, and e-readers like Kobo or Apple Books.
- **HTML + CSS Based:** Internally, an EPUB file is a compressed archive (`.zip`) that contains HTML files, images, stylesheets, metadata, and a manifest.
- **Navigation:** It supports **table of contents**, **internal links**, and **chapters** for easy navigation.
- **Supports Rich Media:** EPUB 3 can include audio, video, interactive elements, and MathML.

### How EPUB Shows “Pages”

EPUB doesn't have fixed "pages" like PDF. Instead:

- The **reading software** (like Apple Books, Calibre, or Kobo) dynamically **splits content into pages** based on screen size, font size, and user settings.
- Pages can vary in number depending on:
  - Device screen resolution
  - Font size or style
  - Margin settings

Because of this, you can't refer to a fixed page number universally across devices.
### EPUB vs PDF

| Feature                | EPUB                                 | PDF                                |
|------------------------|--------------------------------------|------------------------------------|
| Layout                 | Reflowable                           | Fixed                              |
| Usability on small screens | Excellent                         | Poor                               |
| Internal format        | HTML + CSS + XML                     | PostScript-based (binary)          |
| Navigation             | Flexible (TOC, links, metadata)      | Static (can have TOC, but fixed)   |


# Estimator {#estimator}

Given a sample an estimator is a formula that approximates a population parameter i.e feature

# Etlt {#etlt}



>[!important]
> EtLT refers to Extract, “tweak”, Load, [Transform](Data%20Transformation.md), and can be thought of an extension to the [ELT](term/elt.md) approach to [data integration](term/data%20integration.md). 

When compared to ELT, the EtLT approach incorporates an additional light “tweak” (small “t”) transformation, which is done on the data after it is extracted from the source and before it is loaded into the destination.





# Evaluating Language Models {#evaluating-language-models}


The [LMSYS](#lmsys) Chatbot Arena is a platform where various large language models ([LLM](#llm)s), including versions of GPT and other prominent models like LLaMA or Claude, are compared through side-by-side interactions. The performance of these models is evaluated using human feedback based on the quality of their generated responses.

The arena employs several techniques to rank and compare models:

1. **[Elo Rating System](#elo-rating-system)**: Adapted from chess, this system rates models based on their relative performance in head-to-head competitions. When one model's response is preferred over another's, the winning model gains points while the losing model loses points. The rating difference helps determine the strength of models in future predictions. The system adjusts ratings gradually to avoid bias towards more recent results, ensuring stability over time.

2. [Bradley-Terry Model](#bradley-terry-model): This model goes beyond simple win-loss records by taking into account the ==difficulty of the task== and the models' relative strengths. It helps fine-tune the ranking, especially when one model consistently performs better against tougher tasks.

In addition to these ranking systems, users can directly compare LLMs by giving them [Prompting](#prompting) to handle, such as writing articles, answering questions, or performing translations. Human voters then decide which model's output is better, or they can declare a tie if neither response is satisfactory.

These methods ensure continuous improvement of the rankings, providing a transparent and evolving leaderboard of the best generative models, including GPT versions

https://openlm.ai/chatbot-arena/

https://www.analyticsvidhya.com/blog/2024/05/from-gpt-4-to-llama-3-lmsys-chatbot-arena-ranks-top-llms/





# Evaluation Metrics {#evaluation-metrics}


## Description

[Confusion Matrix](#confusion-matrix)
[Accuracy](#accuracy)
[Precision](#precision)
[Recall](#recall)
[Precision or Recall](#precision-or-recall)
[F1 Score](#f1-score)
[Recall](#recall)
[Specificity](#specificity)


![Pasted image 20241222091831.png](./images/Pasted%20image%2020241222091831.png)

## Resources:
[Link to good website describing these](https://txt.cohere.com/classification-eval-metrics/)

In [ML_Tools](#ml_tools) see: [Evaluation_Metrics.py](#evaluation_metricspy)

## Types of predictions

Types of predictions in evaluating models. Also see [Why Type 1 and Type 2 matter](#why-type-1-and-type-2-matter)

**True Positive (TP)**:
- This occurs when the model correctly predicts the positive class. For example, if the model predicts that an email is spam and it actually is spam, that's a true positive.

**False Positive (FP)**:
- Also known as a =="Type I error,"== this occurs when the model incorrectly predicts the positive class. For example, if the model predicts that an email is spam but it is not, that's a false positive.

**True Negative (TN)**:
- This occurs when the model correctly predicts the negative class. For example, if the model predicts that an email is not spam and it actually is not spam, that's a true negative.

**False Negative (FN)**:
- Also known as a =="Type II error,"== this occurs when the model incorrectly predicts the negative class. For example, if the model predicts that an email is not spam but it actually is spam, that's a false negative.

## Evaluation metrics in practice

Having many evaluation metrics is hard to understand and optimise. Sometimes it is best to combine into one.

Use a single number i.e. accuracy or [F1 Score](#f1-score) . 

This speeds up development of ml projects.

In order to use metrics to evaluate a model we can:
- Can combine multiple metrics a formula, i.e. weighted average.
- If there is a metrics we are happy that the model passes a given level then we can have it "==Satisfying==". So the for the given metric it just needs to pass a given level.
- For metrics we are interested in we have it "==Optimising==", the one we want to be the best.
- Setup: Pick N-1 satisfying and 1 optimising.

![Pasted image 20241217073706.png](./images/Pasted%20image%2020241217073706.png)







# Event Driven Events {#event-driven-events}

Events can be stored in a [Data Lake](#data-lake) and analysed to find patterns/predictions.  

[Event Driven Microservices](#event-driven-microservices) allow for [Business observability](#business-observability)

[Monolith Architecture](#monolith-architecture)

[Event Driven Microservices](#event-driven-microservices)

[API Driven Microservices](#api-driven-microservices)

# Event Driven Microservices {#event-driven-microservices}


Event-driven microservices refer to a [software architecture](#software-architecture) pattern where [microservices](#microservices) communicate and coordinate their actions through the production, detection, consumption, and reaction to [Event Driven Events](#event-driven-events). This approach is designed to create a more decoupled and scalable system, where services can operate independently and react to changes in real-time.

Event-driven microservices are particularly useful for applications that require high scalability, real-time processing, and flexibility. 

They are commonly used in domains like e-commerce, IoT, financial services, and any system where real-time data processing and responsiveness are critical. 

However, they also introduce challenges in terms of event schema management, eventual consistency, and debugging, which need to be carefully addressed.

Key characteristics of event-driven microservices include:

1. **Event Producers and Consumers**: In this architecture, services can act as event producers, consumers, or both. An event producer generates events when something of interest happens (e.g., a new order is placed), and event consumers listen for these events to perform actions (e.g., processing the order).

2. **Asynchronous Communication**: Events are typically communicated asynchronously, meaning that the producer does not wait for the consumer to process the event. This allows services to operate independently and improves system responsiveness and scalability.

3. **Event Brokers**: An event broker or message broker (such as Apache Kafka, RabbitMQ, or AWS SNS/SQS) is often used to facilitate the distribution of events between services. The broker decouples producers from consumers, allowing them to evolve independently.

4. **Loose Coupling**: Services are loosely coupled because they do not need to know about each other directly. They only need to know about the events they produce or consume, which reduces dependencies and increases flexibility.

5. **Real-Time Processing**: Event-driven architectures are well-suited for real-time processing and can handle high volumes of events efficiently, making them ideal for applications that require immediate responses to changes.

6. **Scalability and Resilience**: The decoupled nature of event-driven microservices allows for independent scaling of services. If one service fails, it does not necessarily affect others, enhancing the system's resilience.

7. **Event Sourcing and CQRS**: Event-driven architectures often use patterns like event sourcing (storing the state of an entity as a sequence of events) and CQRS (Command Query Responsibility Segregation) to manage data consistency and separation of concerns.



# Event Driven {#event-driven}

Event-driven refers to a ==programming paradigm== or architectural style where the flow of the program is determined by events—changes in state or conditions that trigger specific actions or responses. 

In this model, components of a system communicate through events, which can be generated by user interactions, system changes, or external sources.

### Key Concepts of Event-Driven Architecture:

1. Events: An event is a significant change in state or an occurrence that can trigger a response. For example, a user clicking a button, a file being uploaded, or a sensor detecting a change in temperature.

2. Event Producers: These are components or services that generate events. For instance, a web application might produce events when users perform actions like signing up or making a purchase.

3. Event Consumers: These are components or services that listen for and respond to events. They take action based on the events they receive, such as updating a database or sending a notification.

4. Event Channels: These are the pathways through which events are transmitted from producers to consumers. This can include message queues, event buses, or streaming platforms.

5. Loose Coupling: In an event-driven system, components are often loosely coupled, meaning that producers and consumers do not need to know about each other directly. This allows for greater flexibility and scalability.

Benefits of Event-Driven Architecture:
- [Scalability](#scalability)
- Responsiveness
- Flexibility

### Related topics

- [Event Driven Events](#event-driven-events)
- [Event-Driven Architecture](#event-driven-architecture)
- [Event Driven Microservices](#event-driven-microservices)
- **Tags**: #event_driven, #data_processing

# Event Driven Architecture {#event-driven-architecture}



# Everything {#everything}


Can we search with descriptions ? 

## Tips

use \ to match in paths i.e \playground 

can copy file 

new window crl+ n

Use | to get or search 

search syntax



# Excel & Sheets {#excel--sheets}


##### Links

[Google sheets example folder](https://drive.google.com/drive/folders/1F9GTIK-MARSjl6-BKb1AOID6EoRLe_zk?usp=drive_link)

see [standardised/GSheets|GSheets](#standardisedgsheetsgsheets)

Excel Example folder: Desktop/Example_Examples

## Tools common to Excel and Sheets

##### Vlookup

Table

| **Product ID** | **Product Name** | **Price** |
| -------------- | ---------------- | --------- |
| 1001           | Apple            | $1.00     |
| 1002           | Banana           | $0.50     |
| 1003           | Orange           | $0.80     |

$0.50=VLOOKUP(1002, Table, 3, FALSE)
##### Pivot table


## Excel specific

index-match

index-match-match
##### Excel: Evaluate Formula

**Evaluate Formula** is a feature in Excel that allows you to step through complex formulas to see how Excel calculates the result. This tool is helpful for debugging or understanding how nested formulas work.

**Example:**
Suppose you have a formula like this:
```excel
=IF(SUM(A1:A3) > 10, A4 * 2, A5 + 5)
```
To understand how Excel processes this formula, you can use **Evaluate Formula**. It will break down the formula into steps, showing how the `SUM(A1:A3)` is calculated, followed by the evaluation of the `IF` condition, and finally either the multiplication or addition operation based on the result.

**How to use it:**
1. Select the cell containing the formula.
2. Go to the "Formulas" tab on the ribbon.
3. Click on "Evaluate Formula".
4. Click "Evaluate" to step through each part of the formula.

##### Excel: What-If Analysis

**What-If Analysis** in Excel allows you to explore different scenarios by changing the inputs to your formulas. The three main tools within What-If Analysis are **Scenario Manager**, **Goal Seek**, and **Data Tables**.

- **Scenario Manager**: Lets you save different sets of input values and switch between them to see the impact on the result.
- **Goal Seek**: Finds the required input value to achieve a specific output.
- **Data Tables**: Shows how changing one or two variables affects your formulas.

**Example (Goal Seek):**
Imagine you have a loan payment formula:
```excel
=PMT(interest_rate, number_of_periods, loan_amount)
```
You want to find out what interest rate would result in a monthly payment of $500.

Steps:
1. Enter the formula with an initial guess for the interest rate.
2. Go to "Data" > "What-If Analysis" > "Goal Seek".
3. Set the cell with the formula to a value of 500.
4. Set the interest rate cell as the one to change.
5. Click "OK" and Excel will adjust the interest rate to meet the target.

##### Excel: Forecast Sheets

**Forecast Sheets** use historical data to predict future values. Excel creates a forecast chart based on the pattern in the data, using linear or exponential smoothing models. This is particularly useful for time series data, such as sales or financial data over time.

**Example:**
Suppose you have monthly sales data in a column:

| Month  | Sales |
|--------|-------|
| Jan    | 1000  |
| Feb    | 1100  |
| Mar    | 1200  |
| Apr    | 1300  |
You want to forecast future sales for the next 6 months.

Steps:
1. Select the range of historical data (both months and sales).
2. Go to the "Data" tab on the ribbon.
3. Click "Forecast Sheet".
4. Excel will automatically create a forecast for the future months based on the trend in the data.
5. You can adjust the forecast options (e.g., forecast length, confidence intervals) before creating the sheet.
##### Excel: Consolidate

In Excel, the **Consolidate** feature (found under the **Data** tab) allows you to combine data from multiple ranges or worksheets into a single summary. It is particularly useful when you have data spread across different locations and want to summarize it, such as calculating totals, averages, or other aggregate functions.

Key Features of Consolidate:
- **Multiple Ranges**: You can consolidate data from different ranges, even if they are in separate worksheets or workbooks.
- **Functions**: It provides several functions such as SUM, AVERAGE, COUNT, MIN, MAX, and more to aggregate the data.
- **Labels**: You can use row and column labels to match and consolidate the data correctly.

How to Use Consolidate:
1. **Prepare your data**: Ensure that your data is organized in a table format with similar structures (e.g., same columns and rows across different sheets or ranges).
2. **Navigate to Consolidate**: Go to the **Data** tab and click on **Consolidate** in the **Data Tools** group.
3. **Select Function**: In the Consolidate dialog box, select the function you want to use (e.g., **Sum**, **Average**, etc.).
4. **Add References**: Add the ranges you want to consolidate by clicking on **Add** after selecting the range. You can select ranges from different worksheets or workbooks.
5. **Use Labels**: If your data contains row or column labels, check the boxes for "Use labels in top row" or "Use labels in left column" to consolidate the data correctly based on these labels.
6. **Create links**: If you want the consolidated data to update automatically when the source data changes, check the box for **Create links to source data**.

Benefits:
- Saves time by avoiding manual data entry or copy-pasting from multiple sheets.
- Helps in summarizing large amounts of data quickly.
- Ensures accuracy in consolidation, especially with functions like **Sum** or **Average**.

Example:
Suppose you have sales data in two worksheets as follows:

**Sheet1 (Region A)**:

| Product | Sales |
|---------|-------|
| A       | 100   |
| B       | 200   |
| C       | 300   |

**Sheet2 (Region B)**:

| Product | Sales |
|---------|-------|
| A       | 150   |
| B       | 250   |
| C       | 350   |
You can use the **Consolidate** feature to combine the sales from both regions into a summary table:

**Consolidated Sheet**:

| Product | Sales |
|---------|-------|
| A       | 250   |
| B       | 450   |
| C       | 650   |

In this case, you would use the **Sum** function in the Consolidate dialog to add the sales from the two regions.
##### Excel: Text to Columns

The **Text to Columns** feature in Excel is used to split the data in one column into multiple columns, based on a delimiter (like commas, spaces, or tabs) or a fixed width. This is particularly helpful when you have data combined in a single column and you need to separate it into distinct parts.

Benefits:
- Efficiently splits combined data without manual editing.
- Helps with data cleaning when importing text-based data from sources like CSV files.
- Reduces errors by automating the process of separating values.

Key Features of Text to Columns:
- **Delimiters**: You can split text based on specific delimiters such as commas, spaces, semicolons, or custom characters.
- **Fixed Width**: If the data is in a consistent format, you can split the text based on fixed-width segments.

How to Use Text to Columns:
1. **Select the Data**: Highlight the column or cells that contain the text you want to split.
2. **Navigate to Text to Columns**: Go to the **Data** tab on the ribbon, then click **Text to Columns** in the **Data Tools** group.
3. **Choose the Split Type**:
   - **Delimited**: Choose this option if your data is separated by characters like commas, tabs, or spaces.
   - **Fixed Width**: Choose this if the data is aligned into specific columns with consistent spacing.
4. **Select Delimiters or Set Width**: 
   - For **Delimited**, choose the character that separates your data (e.g., comma, space, semicolon).
   - For **Fixed Width**, manually set where the splits should occur by clicking in the preview window.
5. **Select Destination**: Choose where you want the split data to appear (by default, it will overwrite the original data).
6. **Finish**: Click **Finish** to apply the split.

Example 1: Delimited (Comma-Separated Values)
Imagine you have a list of full names in one column:

| Full Name        |
|------------------|
| John,Smith       |
| Jane,Doe         |
| Robert,Johnson   |

You want to split these names into two separate columns: First Name and Last Name. Here's how you would use **Text to Columns**:
1. Select the column with the full names.
2. Go to **Data** > **Text to Columns**.
3. Choose **Delimited**, then select **Comma** as the delimiter.
4. Click **Finish**. The data will be split into two columns:

| First Name | Last Name |
|------------|-----------|
| John       | Smith     |
| Jane       | Doe       |
| Robert     | Johnson   |

Example 2: Fixed Width
Suppose you have a column with product codes where each section of the code has a fixed length:

| Product Code   |
|----------------|
| 12345ABC67890  |
| 67890DEF12345  |

If the first 5 characters are the product number, the next 3 characters are the product category, and the last 5 characters are the batch number, you can split them based on fixed widths:
1. Select the column with the product codes.
2. Go to **Data** > **Text to Columns**.
3. Choose **Fixed Width**.
4. In the preview window, set the breaks where you want to split the text (after the 5th and 8th characters).
5. Click **Finish**. The data will be split into separate columns like this:

| Product Number | Category | Batch Number |
|----------------|----------|--------------|
| 12345          | ABC      | 67890        |
| 67890          | DEF      | 12345        |

##### Excel: [Data Validation](#data-validation)
- **Restrict Data Types**: You can limit entries to specific data types, such as whole numbers, decimal numbers, dates, or times.
- **Set Input Rules**: You can set conditions (e.g., numbers between 1 and 100, dates in a certain range, or specific text lengths).
- **Create Drop-Down Lists**: You can provide users with a predefined list of options to select from.
- **Custom Validation**: You can use formulas for advanced validation rules.
- **Error Alerts**: You can display custom messages to users when they try to enter invalid data.

How to Use Data Validation:

1. **Select the Cell(s)**: Select the range of cells where you want to apply data validation.
2. **Go to Data Validation**: Navigate to the **Data** tab on the ribbon, and in the **Data Tools** group, click **Data Validation**.
3. **Set Validation Criteria**:
   - In the **Settings** tab, choose the type of data you want to allow (Whole Number, Decimal, List, Date, Time, Text Length, or Custom).
   - Specify the condition (e.g., a range of values or a list of items).
4. **Input Message (Optional)**: In the **Input Message** tab, you can set a message that will appear when the user selects the cell, providing guidance on what they should enter.
5. **Error Alert**: In the **Error Alert** tab, define what happens if invalid data is entered. You can show an error message and choose whether to stop the entry, give a warning, or provide information.

**Restricting Input to a Range of Numbers**
You want to restrict the values in a certain column to only allow whole numbers between 10 and 100.
   
Steps:
1. Select the column or cells where you want to apply the rule.
2. Go to **Data** > **Data Validation**.
3. In the **Settings** tab, choose **Whole Number** from the **Allow** drop-down.
4. Set the **Minimum** to 10 and the **Maximum** to 100.
5. Optionally, create an input message or error alert to inform the user of the valid range.

**Creating a Drop-Down List**
You want users to select from a list of predefined options, such as product categories (e.g., "Electronics", "Furniture", "Clothing").
Steps:
1. Select the cells where the drop-down list should appear.
2. Go to **Data** > **Data Validation**.
3. In the **Settings** tab, choose **List** from the **Allow** drop-down.
4. In the **Source** field, type the list items, separated by commas: `Electronics,Furniture,Clothing`.
5. Click **OK**. Now users will see a drop-down arrow in the cells, allowing them to choose from the options.

**Validating Dates**
You want to ensure that users can only enter dates within a specific range, such as between January 1, 2023, and December 31, 2023.
Steps:
1. Select the cells where the dates will be entered.
2. Go to **Data** > **Data Validation**.
3. In the **Settings** tab, choose **Date** from the **Allow** drop-down.
4. Set the **Start Date** to 1/1/2023 and the **End Date** to 12/31/2023.
5. Click **OK**. Now users will only be able to enter dates within the specified range.

**Custom Validation with Formulas**
You want to ensure that users can only enter text starting with the letter "A".
Steps:
1. Select the cells where the validation should apply.
2. Go to **Data** > **Data Validation**.
3. In the **Settings** tab, choose **Custom** from the **Allow** drop-down.
4. In the **Formula** field, enter:
   ```excel
   =LEFT(A1, 1)="A"
   ```
5. Click **OK**. Now users will only be able to enter text starting with the letter "A".


## Google specific

Xlookup

# Explain Different Gradient Descent Algorithms, Their Advantages, And Limitations. {#explain-different-gradient-descent-algorithms-their-advantages-and-limitations}





# Explain The Curse Of Dimensionality {#explain-the-curse-of-dimensionality}


The **curse of dimensionality** refers to the various phenomena that arise when working with data in high-dimensional spaces.

- **Increased Data ==Sparsity==:** As the number of dimensions grows, the available data becomes increasingly sparse, making it difficult for algorithms to find ==meaningful patterns==. This sparsity can lead to poor generalization performance, as the algorithm might not have enough data points in each region of the input space to learn a robust model.

- ==**Distance Metric Issues:**== In high-dimensional spaces, traditional distance metrics like Euclidean distance become less effective, as the relative difference between the nearest and farthest points diminishes. This can make it difficult for algorithms like k-nearest neighbours to identify meaningful neighbours.

- **Difficulty in Visualization:** Visualizing data beyond three dimensions becomes incredibly challenging, making it difficult to gain insights from the data and understand the behaviour of machine learning models.

### Examples of the Curse of Dimensionality

**Vulnerability of [Ngrams](#ngrams) [Language Models](#language-models):** 

Classical n-gram language models in [NLP](#nlp), which rely on counting the occurrences of word sequences, are particularly vulnerable to the curse of dimensionality. As the vocabulary size and the value of 'n' increase, the number of possible n-grams grows exponentially, making it impossible to observe most of them in even a massive training set.
### Addressing the Curse of Dimensionality

While the curse of dimensionality presents significant challenges, there are techniques to mitigate its effects:

- **[Dimensionality Reduction](#dimensionality-reduction):** Techniques like Principal Components Analysis (PCA), Factor Analysis, and [Multidimensional Scaling](#multidimensional-scaling) (MDS) can reduce the number of features while retaining essential information, making it easier to visualize and analyze data and train machine learning models.

- **[Feature Selection](#feature-selection):** Identifying and selecting the most relevant features for a given task can improve model performance and reduce computational complexity.

- **Distributed Representations:** Using distributed representations, where information is encoded across multiple features rather than a single one, can help overcome the limitations of one-hot encodings in high-dimensional spaces.

- **[Regularisation](#regularisation):** Techniques like weight decay in neural networks can help prevent overfitting and improve generalization performance, particularly in high-dimensional settings

- **[Manifold Learning](#manifold-learning):** Manifold learning methods assume that the data lies on a lower-dimensional manifold embedded in a high-dimensional space. By learning this manifold structure, these methods can reduce dimensionality while preserving nonlinear relationships in the data.




# Exploration Vs. Exploitation {#exploration-vs-exploitation}

One of the major challenges in [Reinforcement learning](#reinforcement-learning) is balancing exploration (trying new actions) and exploitation (choosing the best-known actions). 

The ==epsilon-greedy strategy== is commonly used, where a small probability (epsilon) allows for exploration while primarily exploiting the best-known actions.

# Exploration {#exploration}









# Embeddings For Oov Words {#embeddings-for-oov-words}

Can you find words in a [Vector Embedding|word embedding](#vector-embeddingword-embedding) that where not used to creates the embedding?

Yes, but with important caveats. If a word is not in the [spaCy](#spacy) model’s vocabulary with a vector, then:

### ✅ What you can do

#### Option 1: Filter out words without vectors (what you're doing now)
This is the cleanest option:
```python
if token.has_vector:
    embeddings.append(token.vector)
    valid_words.append(word)
```

#### Option 2: Fallback to character-level embeddings (optional)
If you're using `en_core_web_lg`, spaCy sometimes provides approximate vectors for out-of-vocabulary (OOV) words using subword features. But with `en_core_web_md`, OOV words truly lack vector meaning.

#### Option 3: Use a different embedding model
Use FastText or transformer-based models (e.g., Sentence Transformers), which can produce [embeddings for OOV words](#embeddings-for-oov-words) based on subword information or context.

Example with [FastText](#fasttext) (using gensim):
```python
from gensim.models import KeyedVectors

model = KeyedVectors.load_word2vec_format("cc.en.300.vec")  # or download from FastText
embedding = model.get_vector("unseenword")  # FastText will synthesize it
```

### 💡 Summary

| Approach                     | Handles OOV? | Notes |
|-----------------------------|--------------|-------|
| spaCy `en_core_web_md`      | ❌            | Skips words without vectors (recommended) |
| spaCy `en_core_web_lg`      | ⚠️ Sometimes  | May infer vectors using subword info |
| FastText / GloVe            | ✅            | Good for unseen words |
| Sentence Transformers (BERT)| ✅            | Contextualized, ideal for phrases/sentences |

#NLP #ml_process #ml_optimisation

# Emergent Behavior {#emergent-behavior}

