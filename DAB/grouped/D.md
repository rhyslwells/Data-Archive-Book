# D

## Table of Contents
* [DBScan](#dbscan)
* [DS & ML Portal](#)
* [Dash](#dash)
* [Dashboarding](#dashboarding)
* [Data AI Education at Work](#data-ai-education-at-work)
* [Data Analysis Portal](#data-analysis-portal)
* [Data Analysis](#data-analysis)
* [Data Analyst](#data-analyst)
* [Data Architect](#data-architect)
* [Data Archive Graph Analysis](#data-archive-graph-analysis)
* [Data Cleansing](#data-cleansing)
* [Data Collection](#data-collection)
* [Data Contract](#)
* [Data Distribution](#data-distribution)
* [Data Drift](#data-drift)
* [Data Engineer](#data-engineer)
* [Data Engineering Portal](#data-engineering-portal)
* [Data Engineering Tools](#data-engineering-tools)
* [Data Engineering](#data-engineering)
* [Data Governance](#data-governance)
* [Data Hierarchy of Needs](#data-hierarchy-of-needs)
* [Data Ingestion](#data-ingestion)
* [Data Integration](#data-integration)
* [Data Integrity](#data-integrity)
* [Data Lake](#data-lake)
* [Data Lakehouse](#data-lakehouse)
* [Data Leakage](#data-leakage)
* [Data Lifecycle Management](#data-lifecycle-management)
* [Data Management](#data-management)
* [Data Modelling](#data-modelling)
* [Data Observability](#data-observability)
* [Data Pipeline to Data Products](#data-pipeline-to-data-products)
* [Data Pipeline](#data-pipeline)
* [Data Principles](#data-principles)
* [Data Product](#data-product)
* [Data Quality](#data-quality)
* [Data Reduction](#data-reduction)
* [Data Roles](#data-roles)
* [Data Science](#data-science)
* [Data Scientist](#data-scientist)
* [Data Selection in ML](#data-selection-in-ml)
* [Data Selection](#data-selection)
* [Data Steward](#data-steward)
* [Data Storage](#data-storage)
* [Data Streaming](#data-streaming)
* [Data Terms](#data-terms)
* [Data Transformation with Pandas](#data-transformation-with-pandas)
* [Data Transformation](#data-transformation)
* [Data Validation](#data-validation)
* [Data Virtualization](#data-virtualization)
* [Data Visualisation](#data-visualisation)
* [Data Warehouse](#data-warehouse)
* [Data transformation in Data Engineering](#data-transformation-in-data-engineering)
* [Data transformation in Machine Learning](#data-transformation-in-machine-learning)
* [Database Index](#database-index)
* [Database Management System (DBMS)](#database-management-system-dbms)
* [Database Schema](#database-schema)
* [Database Storage](#database-storage)
* [Database Techniques](#database-techniques)
* [Database](#database)
* [Databricks vs Snowflake](#databricks-vs-snowflake)
* [Databricks](#databricks)
* [Datasets](#datasets)
* [Debugging ipynb](#debugging-ipynb)
* [Debugging](#debugging)
* [Debugging.py](#debuggingpy)
* [Decision Tree](#decision-tree)
* [Deep Learning Frameworks](#deep-learning-frameworks)
* [Deep Learning](#deep-learning)
* [Deep Q-Learning](#deep-q-learning)
* [DeepSeek](#deepseek)
* [Deleting rows or filling them with the mean is not always best](#deleting-rows-or-filling-them-with-the-mean-is-not-always-best)
* [Demand forecasting](#demand-forecasting)
* [Dendrograms](#dendrograms)
* [Design Thinking Questions](#design-thinking-questions)
* [Determining Threshold Values](#determining-threshold-values)
* [DevOps](#devops)
* [Difference between Databricks vs. Snowflake](#difference-between-databricks-vs-snowflake)
* [Difference between snowflake to hadoop](#difference-between-snowflake-to-hadoop)
* [Differentation](#differentation)
* [Digital Transformation](#digital-transformation)
* [Digital twin](#digital-twin)
* [Dimension Table](#dimension-table)
* [Dimensional Modelling](#dimensional-modelling)
* [Dimensionality Reduction](#dimensionality-reduction)
* [Dimensions](#dimensions)
* [Directed Acyclic Graph (DAG)](#directed-acyclic-graph-dag)
* [Directory Structure](#directory-structure)
* [Distillation](#distillation)
* [Distributed Computing](#distributed-computing)
* [Distribution_Analysis.py](#distribution_analysispy)
* [Distributions](#distributions)
* [Docker Image](#docker-image)
* [Docker](#docker)
* [Documentation & Meetings](#)
* [Dropout](#dropout)
* [DuckDB in python](#duckdb-in-python)
* [DuckDB vs SQLite](#duckdb-vs-sqlite)
* [DuckDB](#duckdb)
* [Dummy variable trap](#)
* [dagster](#dagster)
* [data asset](#data-asset)
* [data lineage](#data-lineage)
* [data literacy](#data-literacy)
* [dbt](#dbt)
* [declarative](#declarative)
* [dependency manager](#dependency-manager)



# Dbscan {#dbscan}


**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a [Clustering](#clustering) algorithm that groups together data points <mark>based on density</mark>. It is particularly useful when K-means doesn't work well, such as in datasets with complex shapes or when there are outliers.

- **Used when [K-means](#k-means) doesn't work**: DBSCAN handles datasets with <mark>irregular cluster shapes</mark> and is not sensitive to outliers like K-means.
- **When you have nesting of clusters**: It can identify clusters of varying shapes and sizes without needing to predefine the number of clusters, unlike K-means.
- **Groups core points to make clusters**: DBSCAN identifies core points, which have many nearby points, and groups them together.
- **Can identify [standardised/Outliers](#standardisedoutliers)**: It detects noise points (outliers) that don't belong to any cluster.

### Python Example:

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Create sample data
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Apply DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='plasma')
plt.show()
```

This will cluster the data and visualize it, highlighting core points and marking outliers as separate clusters.

## 🌐 Sources
1. [hex.tech - When and Why To Choose Density-Based Methods](https://hex.tech/blog/comparing-density-based-methods/#:~:text=DBSCAN%20is%20a%20density%2Dbased)
2. [newhorizons.com - DBSCAN vs. K-Means: A Guide in Python](https://www.newhorizons.com/resources/blog/dbscan-vs-kmeans-a-guide-in-python)

### Machine Learning Fundamentals

- [ML_Tools](#ml_tools)
- [Supervised Learning](#supervised-learning)
- [Unsupervised Learning](#unsupervised-learning)
- [Reinforcement learning](#reinforcement-learning)
- [Deep Learning](#deep-learning)

### Model Training and Optimisation

- [Learning rate](#learning-rate)
- [Overfitting](#overfitting)
- [Regularisation](#regularisation)
- [Hyperparameter](#hyperparameter)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Model Optimisation](#model-optimisation)
- [Model Selection](#model-selection)
- [Vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem)

### Feature Engineering and Data Handling

- [Feature Selection](#feature-selection)
- [Feature Engineering](#feature-engineering)
- [Imbalanced Datasets](#imbalanced-datasets)
- [Outliers](#outliers)
- [Anomaly Detection](#anomaly-detection)
- [Multicollinearity](#multicollinearity)
- [Dimensionality Reduction](#dimensionality-reduction)
- [Clustering](#clustering)
### Machine Learning Models

Classification Models

- [Classification](#classification)
- [Binary Classification](#binary-classification)
- [Support Vector Machines](#support-vector-machines)
- [Decision Tree](#decision-tree)
- [Random Forests](#random-forests)
- [K-nearest neighbours](#k-nearest-neighbours)
- [Logistic Regression](#logistic-regression)

Regression Models

- [Regression](#regression)
- [Linear Regression](#linear-regression)

Boosting and Optimisation

- [Gradient Descent](#gradient-descent)
- [Gradient Boosting](#gradient-boosting)
- [XGBoost](#xgboost)

### Deep Learning and Neural Networks
 
- [BERT](#bert)
- [LSTM](#lstm)
- [Recurrent Neural Networks](#recurrent-neural-networks)
- [Transformer](#transformer)
- [Attention mechanism](#attention-mechanism)
- [Neural network](#neural-network)

### Model Evaluation and Metrics

- [Cost Function](#cost-function)
- [Loss function](#loss-function)
- [Cross Entropy](#cross-entropy)
- [Evaluation Metrics](#evaluation-metrics)
- [Model Evaluation](#model-evaluation)
- [Accuracy](#accuracy)
- [Precision](#precision)
- [Recall](#recall)

### Algorithms and Frameworks

- [Machine Learning Algorithms](#machine-learning-algorithms)
- [Optimisation techniques](#optimisation-techniques)
- [Optimisation function](#optimisation-function)
- [Model Ensemble](#model-ensemble)
- [Batch Processing](#batch-processing)
- [Apache Spark](#apache-spark)
- [Sklearn](#sklearn)

### Statistical and Data Analysis Concepts

- [Distributions](#distributions)
- [Statistics](#statistics)
- [Correlation](#correlation)
- [Data Analysis](#data-analysis)
- [Data Quality](#data-quality)
- [Principal Component Analysis](#principal-component-analysis)

### Misc

- [Interpretability](#interpretability)
- [RAG](#rag)

# Dash {#dash}


**Dash** is an open-source framework for building interactive web applications using Python. 

It is particularly well-suited for data visualization and dashboard creation. 

Dash integrates  with popular libraries such as Plotly, Pandas, and NumPy, making it ideal for creating dynamic and interactive visualizations.

In [ML_Tools](#ml_tools) see [Clustering_Dashboard.py](#clustering_dashboardpy)

Key Components of Dash
1. **Dash App**: The main application instance, created using `dash.Dash(__name__)`.
2. **Dash HTML Components (`dash_html_components`)**: Provides wrappers for standard HTML elements (e.g., `html.Div`, `html.H1`).
3. **Dash Core Components (`dash_core_components`)**: Includes interactive UI components like graphs, dropdowns, sliders, and more (e.g., `dcc.Graph`, `dcc.Dropdown`).
4. **Callback Functions**: Used to make components interactive by linking inputs (user actions) to outputs (changes in the UI).
5. **[Plotly](#plotly) Integration**: Dash apps leverage Plotly for creating interactive visualizations.

# Dashboarding {#dashboarding}




[Dash](#dash)
[Streamlit.io](#streamlitio)



# Data Ai Education At Work {#data-ai-education-at-work}


### Introduction

Organizations are increasingly recognizing the importance of integrating data and AI learning into their people strategies. This involves practical steps to ensure employees are equipped with the necessary skills to leverage these technologies effectively.

Integrating data and AI education into organizational strategies is essential for maintaining competitiveness and fostering a culture of continuous learning. By addressing these areas, organizations can better prepare their workforce for the evolving technological landscape.

### Practical Steps for Integration

1. **Access to Training**:
   - Provide clear guidance on how to access training courses.
   - Offer details on accessing training funds and budgets.
   - Partner with training providers to offer relevant courses.

2. **Learning Resources**:
   - Collect and distribute clear and concise training materials.
   - Capture, document, and discuss use cases from staff experiences.
   - Encourage peer-to-peer learning and collaboration.

3. **Organizational Support**:
   - Align training with professional competencies.
   - Foster communication and collaboration with external partners.
   - Encourage staff to experiment with AI tools to enhance efficiency.

4. **Governance and Strategy**:
   - Establish governance and skill strategies before deploying AI.
   - Develop acceptable use policies for AI tools.
   - Recognize that adopting AI is a gradual process requiring leadership support.

### Fostering a Culture of Continuous Learning

1. **Leadership and Culture**:
   - Connect learning initiatives with employee incentives and pay.
   - Allow time and space for learning by alleviating workloads.
   - Protect training time and build networks to showcase use cases.

2. **Mindset and Adaptability**:
   - Promote digital literacy and adaptability among employees.
   - Encourage openness to new ideas and recognize skills beyond the technical team.

### Business Risks of Not Upskilling

1. **Competitive Disadvantage**:
   - Risk of losing competitive and productivity edges.
   - Potential loss of market differentiation as competitors advance.

2. **Staff Retention**:
   - Risk of losing skilled staff uncomfortable with new technologies.
   - Employees may move to companies at the cutting edge of AI.

3. **Operational Challenges**:
   - Inconsistencies in ways of working between partner organizations.
   - Inappropriate use of AI tools by untrained staff.

4. **Productivity and Trust**:
   - AI's potential to enhance productivity is yet to be fully realized.
   - Trust and verifiability of AI systems (black boxes) are crucial for business benefits.


# Data Analysis Portal {#data-analysis-portal}

[Data Analyst](#data-analyst)

[Data Visualisation](#data-visualisation)

# Data Analysis {#data-analysis}

What is it? Usually done with a [Data Analyst](#data-analyst).After processing, data is analyzed to extract meaningful insights and derive value from the data.
### Types of analysis:

Exploration and understanding:
- [EDA](#eda): Involves exploring data sets to find patterns, anomalies, or relationships without having a specific hypothesis in mind. It is often used in the initial stages of data analysis to generate insights.
- Descriptive: <mark>Focuses on summarizing historical data to understand what has happened</mark> in the past. It often involves the use of [Statistics](#statistics) measures and [Data Visualisation](#data-visualisation) tools to present data trends and patterns.
- Diagnostic: <mark>Seeks to understand why something happened</mark>. It involves examining data to identify causes and correlations, often using techniques like data mining and statistical analysis.

Forward looking:
- Predictive: Uses historical data and statistical algorithms to <mark>forecast future outcome</mark>s. It helps in identifying trends and making predictions about future events based on past data.
- Prescriptive: Goes a step further by <mark>recommending actions based on the predictions made</mark>. It uses optimization and simulation algorithms to suggest the best course of action for a given situation.
- Inferential: Makes inferences and predictions about a population based on a sample of data. It often involves [Hypothesis testing](#hypothesis-testing) and [Confidence Interval](#confidence-interval).

# Data Analyst {#data-analyst}

Summary:
- Gathers and processes data to generate reports.
- Communicates insights and findings to management
- Conducts [Data Analysis](#data-analysis).

### Key responsibilities of a data analyst:

- Define Objectives: Clearly outline the goals of the analysis to guide the process.
  
- [Data Collection](#data-collection): Gather relevant data from various sources, ensuring accuracy, completeness, and timeliness.
  
- [Data Cleansing](#data-cleansing): Clean the data to remove errors, duplicates, and inconsistencies for reliable findings.
  
- Data Exploration: Perform exploratory data analysis ([EDA](#eda)) to understand data structure, [Distributions|distribution](#distributionsdistribution), and relationships.
  
- Choose the Right Tools: Utilize appropriate tools and software for analysis, such as Excel, R, Python, SQL, or specialized platforms.
  
- [Statistics](#statistics): Apply various statistical methods and techniques, such as regression analysis, clustering, and [hypothesis testing](#hypothesis-testing).
  
- [Data Visualisation](#data-visualisation): Use visualization techniques to effectively present findings and communicate insights.
  
- Interpret Results: Analyze results in the context of objectives and consider implications for decision-making.
  
- Documentation: Maintain thorough [Documentation & Meetings](#documentation--meetings) of the analysis process, including data sources, methodologies, and findings.
  
- Continuous Learning: Stay updated with the latest tools, techniques, and best practices in the evolving field of data analysis.

# Data Architect {#data-architect}

Data Architect
  - Designs and manages the data infrastructure.
  - Ensures data is stored, organized, and accessible for analysis.

# Data Archive Graph Analysis {#data-archive-graph-analysis}


Use the following to 

[Dataview](#dataview)
[Graph View](#graph-view)

Check out [Graph Analysis Plugin](#graph-analysis-plugin)

Convert Dataview to CSV


# Data Cleansing {#data-cleansing}


Data cleansing is the process of correcting or removing inaccurate, incomplete, or inconsistent data to improve its [Data Quality](#data-quality) for analysis. Involves:

- [standardised/Outliers|Handling Outliers](#standardisedoutliershandling-outliers)
- [Handling Missing Data](#handling-missing-data)
- [Handling Different Distributions](#handling-different-distributions)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Cleaning/Dataframe_Cleaing.ipynb
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Cleaning

Related terms:
- [Data Selection](#data-selection)

Follow-up questions:
- [Deleting rows or filling them with the mean is not always best ](#deleting-rows-or-filling-them-with-the-mean-is-not-always-best)



# Data Collection {#data-collection}

Determine the [Data Quality](#data-quality) and quantity of data required and get it.

[Imbalanced Datasets](#imbalanced-datasets)

### [Data Contract](#data-contract)

pattern to handle schema changes

Pattern to apply to organisation using tools they have.

Tooling:
- [dbt](#dbt)

Data contracts help prevent preventable data issues while increasing collaboration and reducing costs.

A data contract is an agreed interface between
the generators of data and its consumers. It sets
the expectations around that data, defines how
it should be governed, and facilitates the
explicit generation of quality data that meets
the business requirements.

Interfaces:
- [API](#api) for data.

A document to codify what has been agreed.

Q: How does the Data Contract allow for contextual rules? Example the same schema can support multiple products in our org but the DQ rules can be different for different Products
A: Data contracts are particular business. Could use <mark>inheritance</mark> of rules in data contracts - basic template. To get standardisation across products.

[Data Contract](#data-contract)
By establishing a [data contract](#data-contract) and building interfaces based on it, organizations can improve data quality. Implementing structured agreements and automated change management processes can help business users, who may not be data experts, produce higher-quality data ([Data Quality](#data-quality)).

### Images

![Pasted image 20250312163351.png](../content/images/Pasted%20image%2020250312163351.png)

# Data Distribution {#data-distribution}

Data distribution refers to the process of making processed and analyzed data available for downstream applications and systems. 

This can involve supplying data to
- business applications, 
- reporting systems, 
- or other data-driven processes, 
- ensuring that stakeholders

have access to the information they need for decision-making and operations.

# Data Drift {#data-drift}

Data drift refers to changes in the statistical properties of input data that a machine learning (ML) model encounters during production. Such shifts can lead to decreased model performance, as the model may struggle to make accurate predictions on data that differ from its training set. 

Regular monitoring and prompt response to data drift are essential to maintain the effectiveness of ML models in dynamic production environments.

Concepts:

- Data drift involves changes in input data distributions
- Concept drift pertains to alterations in the relationship between inputs and outputs.
- [Performance Drift](#performance-drift) drift relates to changes in model outputs. 

**Training-Serving Skew:** This refers to discrepancies between training data and production data, which can arise from data drift or other factors, leading to performance issues. 

<mark>**Detecting Data Drift:**</mark>

Identifying data drift is crucial for maintaining model accuracy. Techniques include:

- **Statistical Hypothesis Testing:** Assessing whether differences between training and production data distributions are statistically significant.

- **Distance Metrics:** Quantifying the divergence between data distributions using measures like Kullback-Leibler divergence or Kolmogorov-Smirnov tests.

- **Monitoring Summary Statistics:** Regularly reviewing key statistical indicators (e.g., mean, variance) of input features to detect anomalies.

**Addressing Data Drift:**

Once detected, strategies to manage data drift include:

1. **Data Quality Checks:** Ensure that the drift isn't due to data quality issues, such as errors in data collection or processing. 

2. **Investigate the Drift:** Analyze the source and nature of the drift to understand its implications.

3. **Model Retraining:** Update the model using recent data to help it adapt to new patterns.

4. **Model Rebuilding:** In cases of significant drift, it may be necessary to redesign the model architecture or feature engineering processes.

5. **Fallback Strategies:** Implement alternative decision-making processes, such as rule-based systems or human judgment, when the model's reliability is compromised.





 

# Data Engineer {#data-engineer}


The primary responsibility of a data engineer is to take data from its source and make it available for analysis. They focus on
- automating the data collection, 
- processing, 
- and analysis workflows,
- solving how systems manage and handle the flow of data. 

<mark>Develops data pipelines and ensures data flow between systems.</mark>

Resources:
- [Link](https://www.youtube.com/watch?v=qWru-b6m030)

### Key Responsibilities:

1. Infrastructure Design and Maintenance:  
   Data engineers design, build, and maintain the necessary infrastructure to collect, process, and store large amounts of data. This infrastructure is crucial for ensuring data is accessible and usable for analysis and reporting.

2. [Data Pipeline](#data-pipeline): 

3. Support Role:  
   Data engineers act as a bridge between <mark>data producers and consumers</mark>, ensuring smooth and reliable data flow. They support business operations through scalable and efficient [Data Management](#data-management) solutions, contributing indirectly to product delivery and decision-making.

### Core Activities:

What engineers do & interact with: see [Data Engineering Portal](#data-engineering-portal)

Stakeholders they interact with see [Data Roles](#data-roles)

Tools they use: [Data Engineering Tools](#data-engineering-tools)

Tasks They Are Usually Given
  - Project Management: Tracking tasks, bugs, and progress through Azure Boards.
  - Collaboration: Facilitating teamwork with shared repositories and [continuous integration](#continuous-integration) workflows.
  - Continuous Learning: Keeping up-to-date with the latest technologies and updating pipelines due to obsolescence of tech
  - [Documentation & Meetings](#documentation--meetings) and Security: Creating documentation, implementing security measures, and exploring system upgrades for enhanced efficiency.



# Data Engineering Portal {#data-engineering-portal}


Databases manage large data volumes with scalability, speed, and flexibility. Key systems include:

- [MySql](#mysql)
- [PostgreSQL](#postgresql)


They facilitate efficient [CRUD.md](obsidian://open?vault=content&file=standardised%2FCRUD.md) operations and transactional processing ([OLTP.md](obsidian://open?vault=content&file=standardised%2FOLTP.md)), structured by a [Database Schema.md](obsidian://open?vault=content&file=standardised%2FDatabase%20Schema.md) that organizes data into tables and relationships.

## Key Features

- **[Structured Data](#structured-data)**: Organized for efficient CRUD operations, allowing reliable access.
- **Relational Databases**: Use SQL to manage data in tables with relationships expressed through foreign keys and joins, minimizing redundancy.

Structure
- Data is organized into tables (like spreadsheets) with columns (fields) and rows (records), enabling efficient storage and retrieval.

Flexibility
- Databases have a flexible schema that adapts to evolving requirements, unlike static solutions like spreadsheets.

Related Ideas:
- [Spreadsheets vs Databases](#spreadsheets-vs-databases)
- [Database Management System (DBMS)](#database-management-system-dbms)
- [Components of the database](#components-of-the-database)
- [Relating Tables Together](#relating-tables-together)
- [Turning a flat file into a database](#turning-a-flat-file-into-a-database)
- [Database Techniques](#database-techniques)

# Data Engineering Tools {#data-engineering-tools}


  - **Snowflake:** [Cloud](#cloud)-based data warehousing for scalable storage and processing.
  - **Microsoft SQL Server:** [SQL](#sql)-based relational database management.
  - **[Azure](#azure) SQL Database:** Managed relational database service on Azure.
  - **Azure Data Lake Storage:** Scalable storage for big data analytics.
  - **SQL and T-SQL:** Query languages for managing and querying relational databases.
  - **AWS [Amazon S3|S3](#amazon-s3s3):** Storage for data lakes.

[Data Ingestion](#data-ingestion) Tools and Technologies:
- [Apache Kafka](#apache-kafka)
- AWS Kinesis: A cloud service for real-time data processing, enabling the collection and analysis of streaming data.
- Google Pub/Sub: A messaging service that allows for asynchronous communication between applications, supporting real-time data ingestion.

[Data Storage](#data-storage)
Tools: Amazon S3, Google BigQuery, Snowflake.

[dbt](#dbt)

### Tags
- **Tags**: #data_tools, #data_management

# Data Engineering {#data-engineering}


The definition from the [Fundamentals of Data Engineering](https://www.oreilly.com/library/view/fundamentals-of-data/9781098108298/), as it’s one of the most recent and complete: 

> Data engineering is the development, implementation, and maintenance of systems and processes that take in raw data and produce high-quality, consistent information that supports downstream use cases, such as analysis and machine learning. Data engineering intersects security, [Data Management](#data-management), DataOps, data architecture, orchestration, and software engineering.

A [Data Engineer](#data-engineer) today oversees the whole data engineering process, from collecting data from various sources to making it available for downstream processes. The role requires familiarity with the multiple stages of the [Data Engineering Lifecycle](Data%20Lifecycle%20Management.md) and an aptitude for evaluating data tools for optimal performance across several dimensions, including price, speed, flexibility, scalability, simplicity, reusability, and interoperability.

Data Engineering helps also overcome the bottlenecks of [Business Intelligence](term/business%20intelligence.md):
- More transparency as tools are open-source mostly
- More frequent data loads
- Supporting [Machine Learning](Machine%20Learning.md) capabilities 

Compared to existing roles it would be a **software engineering plus business intelligence engineer including big data abilities** as the [Hadoop](term/apache%20hadoop.md) ecosystem, streaming, and computation at scale. Business creates more reporting artifacts themselves but with more data that needs to be collected, cleaned, and updated near real-time and complexity is expanding every day.

With that said more programmatic skills are needed similar to software engineering. **The emerging language at the moment is [Python](term/python.md)** which is used in engineering with tools alike [Apache Airflow](#apache-airflow), [dagster](dagster.md), [Prefect](#prefect) as well as data science with powerful libraries.

As a data engineer, you use mainly [SQL](SQL.md) for almost everything except when using external data from an API. Here you'd use [ELT](term/elt.md) tools or write some [Data Pipeline](#data-pipeline) with the tools mentioned above.

# Data Governance {#data-governance}


[**Data governance**](https://www.talend.com/resources/what-is-data-governance/) **is a collection of processes, roles, policies, standards, and metrics that ensure the effective and efficient use of information in enabling an organization to achieve its goals.**

It establishes the processes and responsibilities that ensure the [Data Quality](Data%20Quality.md) and security of the data used across a business or organization. Data governance defines who can take what action, upon what data, in what situations, and using what methods.

**Data Governance**: Focuses on ensuring that data is managed consistently and adheres to policies, often working in tandem with [Data Observability](#data-observability) to enforce quality standards.

# Data Hierarchy Of Needs {#data-hierarchy-of-needs}


![Pasted image 20241005170237.png|500](../content/images/Pasted%20image%2020241005170237.png|500)

The **Data Hierarchy of Needs** is a framework that outlines the stages required to effectively use data in organizations. It resembles Maslow’s hierarchy, progressing from basic data needs to advanced capabilities:

1. **Data Collection**:  (bottom)
   Start by collecting raw data from various sources, ensuring it's stored securely and reliably.

2. **Data Storage and Access**:  
   Organize and store data so it's easily accessible for those who need it, using databases or data warehouses.

3. **Data Cleaning and Preparation**:  
   Clean, preprocess, and transform data to ensure it’s accurate, consistent, and ready for analysis.

4. **Data Analytics**:  
   Analyze the prepared data to generate insights, identify patterns, and create reports.

5. **Data-Driven Decision Making**:  
   Use the insights from data analytics to inform and improve decision-making across the organization.

6. **Advanced Data Capabilities (AI/ML)**:  (top)
   Once the foundation is in place, apply advanced techniques like machine learning and artificial intelligence for predictive and prescriptive insights.


# Data Ingestion {#data-ingestion}


Data ingestion is the process of collecting and importing raw data from various sources ([Database](#database), [API](#api), [Data Streaming](#data-streaming) services) into a system for processing and analysis, and can be performed in batch and realtime ingestion. The goal is to gather raw data that can be processed and analyzed.

Used for building [Data Pipeline](#data-pipeline)

Challenges
- [Data Quality](#data-quality): Ensuring that the ingested data is accurate, complete, and consistent.
- [Scalability](#scalability): Handling large volumes of data efficiently as the data sources grow.
- [Latency](#latency): Minimizing the delay between data generation and processing, especially in real-time scenarios.

Use Cases:
- Data ingestion is used in various applications, including: [business intelligence](#business-intelligence), [Machine Learning](#machine-learning)

Related to:
- [Data Engineering Tools](#data-engineering-tools)



[Data Ingestion](#data-ingestion)
   **Tags**: #data_collection, #data_management

# Data Integration {#data-integration}


Data integration is the process of combining data from disparate source systems into a single unified view, moving data to a [Single Source of Truth](#single-source-of-truth).

## Manual Integration
Manual integration involves analysts manually logging into source systems, analyzing and/or exporting data, and creating reports. 

### Disadvantages of Manual Integration:
- **Time-consuming**: The process requires significant time investment.
- **Security Risks**: Analysts need access to multiple operational systems.
- **Performance Issues**: Running analytics on non-optimized systems can interfere with their functioning.
- **Outdated Reports**: Data changes frequently, leading to quickly outdated reports.

## [Data Virtualization](#data-virtualization)

Data virtualization is a method that allows access to data without needing to replicate it, providing a unified view of data from multiple sources.

## Application Integration
Application integration links multiple applications to move data directly between them. 

### Methods of Application Integration:
- **Point-to-Point Communications**: Direct connections between applications.
- **Middleware Layer**: Using tools like an Enterprise Service Bus (ESB).
- **Application Integration Tools**: Specialized tools for integrating applications.

### Disadvantages of Application Integration:
- **Data Redundancy**: May result in multiple copies of the same data across systems.
- **Increased Costs**: Managing multiple copies can lead to higher costs.
- **Point-to-Point Traffic**: Can create excessive traffic between systems.
- **Performance Impact**: Executing analytics on operational systems may interfere with their functioning.

# Data Integrity {#data-integrity}

Data integrity refers to the 
- accuracy, 
- consistency, and 
- reliability of data

throughout its lifecycle. It ensures that data remains <mark>unaltered</mark> and <mark>trustworthy</mark>, whether it is being 
- stored, 
- processed, 
- or transmitted. 

Maintaining data integrity involves implementing measures to prevent unauthorized access, corruption, or loss of data.

In the context of [Database](#database) and information systems, data integrity can be enforced through:

1. **Validation Rules**: Ensuring that data entered into a system meets certain criteria.
2. **Access Controls**: Limiting who can view or modify data.
3. **Backups**: Regularly saving copies of data to prevent loss.
4. **Error Checking**: Using [Checksum](#checksum) or [Hash](#hash) to verify data integrity during transmission.



[Data Integrity](#data-integrity)
   **Tags**: #data_quality, #data_management

# Data Lake {#data-lake}


A Data Lake is a storage system with vast amounts of [unstructured data](#unstructured-data) and [structured data](#structured-data), stored as-is, without a specific purpose in mind, that can be built on multiple technologies such as Hadoop, NoSQL, Amazon Simple Storage Service, a relational database, or various combinations and different formats (e.g. Excel, CSV, Text, Logs, etc.).

**Definition**: A repository that <mark>stores diverse data types</mark>, including structured, semi-structured, and unstructured data. If cant fit into a database.

Features:
- **Versatility**: Can accommodate various data formats, including videos, images, documents, and more.
- **Raw Data Storage**: Preserves data in its raw form, suitable for advanced analytics, particularly in machine learning and AI.
- **Data Usability**: Raw data <mark>may require cleaning and transformation for analytical use</mark>, often transferred to databases or data warehouses.
- **Use Case**: Valuable for storing large volumes of raw data, especially in contexts requiring advanced analytics and experimentation.

[unstructured data](#unstructured-data) for predictive modeling and analysis. This leads to the creation of a **data lake**, which stores raw data without predefined schemas. 

The data lake supports the following capabilities:
-   To capture and store raw data at scale for a low cost
-   To store many types of data in the same repository
-   To perform [Data Transformation](Data%20Transformation.md) on the data where the purpose may not be defined
-   To perform new types of data processing
-   To perform single-subject analytics based on particular use cases

Components of a data lake
		1. [Storage Layer](term/storage%20layer%20object%20store.md)
		2. [Data Lake File Format](term/data%20lake%20file%20format.md)
		3. [Data Lake Table Format](term/data%20lake%20table%20format.md) with [Apache Parquet](term/apache%20parquet.md), [Apache Iceberg](term/apache%20iceberg.md), and [Apache Hudi](term/apache%20hudi.md)

# Data Lakehouse {#data-lakehouse}


A Data Lakehouse open [Data Management](#data-management) architecture that combines the flexibility, cost-efficiency, and scale of [Data Lake](Data%20Lake.md) with the data management and ACID transactions of [Data Warehouse](Data%20Warehouse.md) with Data Lake Table Formats ([Delta Lake](term/delta%20lake.md), [Apache Iceberg](term/apache%20iceberg.md) & [Apache Hudi](term/apache%20hudi.md)) that enable Business Intelligence (BI) and Machine Learning (ML) on all data.

A **data lakehouse** is an emerging architectural approach that combines the best features of data lakes and data warehouses to provide a unified platform for storing, processing, and analyzing large volumes of structured and unstructured data. Here’s a breakdown of its key characteristics and benefits:

The data lakehouse architecture represents a significant evolution in [Data Management](#data-management), addressing the limitations of traditional data lakes and [Data Warehouse|Warehouse](#data-warehousewarehouse) by providing a unified platform for all data types.
### Key Characteristics

1. **Unified Storage**:
   - Data lakehouses store data in a single repository, accommodating both structured data (like tables in a database) and unstructured data (like images, videos, and text). This eliminates the need for separate systems, simplifying data management.

2. **Support for Multiple Data Types**:
   - They can handle various data formats, such as **CSV**, **JSON**, **Parquet**, and **Avro**, enabling flexibility in how data is ingested and stored.

3. [ACID Transaction](#acid-transaction):
   - Unlike traditional data lakes, data lakehouses provide [ACID Transaction](#acid-transaction) which ensure reliable data operations and integrity, even in concurrent processing environments.

4. **Schema Enforcement**:
   - Data lakehouses can enforce [Database Schema|schema](#database-schemaschema) at the time of data write, allowing users to define data structures while still benefiting from the flexibility of a data lake.

5. **Performance Optimization**:
   - They incorporate various optimization techniques, such as indexing and caching, to improve query performance and provide faster access to data.

6. **Integration with BI Tools**:
   - Data lakehouses are designed to work seamlessly with business intelligence (BI) tools and data analytics platforms, enabling users to derive insights without needing extensive data preparation.

### Benefits

1. **Cost-Effectiveness**:
   - By merging the functionalities of data lakes and data warehouses, organizations can reduce the costs associated with maintaining separate systems for structured and unstructured data.

2. **Scalability**:
   - Data lakehouses leverage cloud storage solutions, allowing for scalable data storage that can grow with the organization’s needs.

3. **Data Accessibility**:
   - With a unified architecture, data from different sources can be accessed and analyzed together, breaking down silos and fostering a more holistic view of the organization’s data landscape.

4. **Simplified Data Pipelines**:
   - Data lakehouses streamline the data ingestion process, enabling organizations to build more efficient data pipelines that accommodate a variety of data sources.

5. **Support for Advanced Analytics**:
   - They provide a robust foundation for advanced analytics, including machine learning and real-time data processing, allowing organizations to extract actionable insights more effectively.

Platforms that implement the data lakehouse architecture include:
- **Databricks Lakehouse Platform**: Combines data engineering, data science, and BI capabilities with a focus on collaboration.
- **Apache Iceberg**: A high-performance table format for large analytic datasets that supports ACID transactions and schema evolution.




# Data Leakage {#data-leakage}

**Data Leakage** refers to the unintentional inclusion of information in the training data that would not be available in a real-world scenario, leading to overly optimistic model performance. It occurs when the model has access to data it shouldn't during training, such as future information or test data, which can result in misleading evaluation metrics and poor generalization to new data.

# Data Lifecycle Management {#data-lifecycle-management}


This is the comprehensive process of managing data from its initial ingestion to its final use in downstream processes. 

Used for maintaining [data integrity](#data-integrity), optimizing performance, and ensuring that data-driven decisions are based on accurate and timely information. 

Not the same as the [Software Development Life Cycle](#software-development-life-cycle)

Key Stages of Full Lifecycle Management

1. [Data Ingestion](#data-ingestion)
2. [Data Storage](#data-storage)
3. [Preprocessing](#preprocessing)
4. [Data Analysis](#data-analysis)
5. [Data Visualisation](#data-visualisation)
6. [Data Distribution](#data-distribution)

Data engineers must evaluate and select tools and technologies based on several [Performance Dimensions](#performance-dimensions)

# Data Management {#data-management}


Data management involves overseeing processes to maintain data integrity and quality. It includes:

- **Responsibility**: Identifying accountable individuals or teams.
- **Issue Resolution**: Mechanisms for detecting and addressing data-related problems.

Data management ensures that a [Data Pipeline](#data-pipeline) operates efficiently, focusing on monitoring errors, performance issues, and [data quality](#data-quality).

**Tools**:
- [Apache Airflow](#apache-airflow)
- Prefect
- [Dagster](#dagster)

Related Concepts:
- [Database Management System (DBMS)](#database-management-system-dbms)
- [Master Data Management](#master-data-management)
- [Data Distribution](#data-distribution)



[Data Management](#data-management)
   **Tags**: #data_management, #data_quality

# Data Modelling {#data-modelling}


Data modelling is the process of creating a visual representation of a system's data and the relationships between different data elements. 

This helps in organizing and structuring the data so it can be efficiently managed and utilized.

Data modelling ensures that data is logically structured and organized, making it easier to store, retrieve, and manipulate in a database.

Workflow of Data Modeling:
1) [Conceptual Model](#conceptual-model)
2) [Logical Model](#logical-model)
3) [Physical Model](#physical-model)


Types of Modeling:
- Relational: Organizes data into tables.
- Object-Oriented: Focuses on objects and their state changes, e.g., robots in a car factory.
- Entity: Uses [ER Diagrams](#er-diagrams) to represent data entities and relationships.
- Network: An extension of hierarchical models.
- Hierarchical: Organizes data in a tree-like structure.





[Data Modelling](#data-modelling)
   **Tags**: #data_modeling, #database_design

# Data Observability {#data-observability}


Data observability refers to the continuous monitoring and collection of metrics about your data to ensure its [Data Quality](#data-quality), reliability, and availability. 

It covers various aspects, such as data quality, pipeline health, metadata management, and infrastructure performance. By tracking key metrics and [standardised/Outliers|anomalies](#standardisedoutliersanomalies), it helps detect issues like data freshness problems, schema changes, or pipeline failures before they impact downstream processes or users.
### Categories of Observability

Auto-profiling Data:

Automatically tracks data attributes, such as row count, column types, data distributions, and schema changes.
 - Bigeye: Provides ML-driven threshold tests and automatic alerts when data drifts beyond expected ranges.
 - Datafold: Integrates with GitHub to run data diffs between environments, offering insights into differences between datasets during development.
 - Monte Carlo: Enterprise-focused with data lake integrations for comprehensive observability.
 - Metaplane: Offers a high level of configuration and both out-of-the-box and custom tests.

Pipeline Testing:

Ensures that data transformation pipelines are functioning correctly by verifying the quality and accuracy of data as it moves through different stages.
 - Great Expectations: An open-source tool that allows you to define tests and automatically generate documentation for those tests, promoting transparency in data quality checks.
 - Soda: Offers pipeline testing with the flexibility of a self-hosted option for more control over data quality monitoring.
 - [dbt](#dbt)tests: Integrated with [dbt](#dbt) Core and dbt Cloud, allowing testing during the transformation process in a dbt project.

 Infrastructure Monitoring:
 
Monitors the health and performance of the underlying data infrastructure, such as databases, pipelines, and servers, to prevent failures and bottlenecks.
 - DataDog: Provides deep monitoring capabilities, including for Airflow, containers, and custom metrics, allowing visibility at various layers of the data stack.

### Managing Metadata

Managing metadata is critical for observability, as it provides context and lineage for your data. Metadata can include:

- Technical Metadata: Information about the dataset’s structure, such as table schema, data types, and column descriptions.
- Operational Metadata: Information about the dataset’s freshness, when it was last updated, and the number of records processed.
- Business Metadata: Describes the meaning of data, such as field definitions and business rules, helping stakeholders understand the context and usage of the dataset.

How to Manage Metadata:

- Manual Documentation: Teams may manually document metadata, but this can be prone to human error and inconsistency.
- Automated Metadata Management: Many modern data tools, such as data catalogs (e.g., Atlan, Alation), automatically track and manage metadata, offering insights into data lineage, schema changes, and data usage.
- Integration with Data Pipelines: Tools like dbt also generate metadata about transformations, which can be included in downstream monitoring systems to ensure consistency and traceability.

[Data Observability](#data-observability)
- Tracking the issues.
- Alerting and ensuring data owners fix it.

# Data Pipeline To Data Products {#data-pipeline-to-data-products}


The journey from [Data Pipeline](#data-pipeline) to [Data Product](#data-product) involves transforming raw data into valuable insights or applications that can be used to drive business decisions. This process typically includes several stages, each with its own set of tasks and objectives.

Read more on [Data Orchestration Trends: The Shift From Data Pipelines to Data Products](https://airbyte.com/blog/data-orchestration-trends).
### Workflow

1. **Define Objectives**:
   - Understand the business goals and what insights or products are needed.

2. **Design the Pipeline**:
   - Plan the architecture and select appropriate tools for each stage of the pipeline.

3. **Implement and Test**:
   - Build the pipeline, ensuring data flows smoothly from ingestion to product delivery.
   - Test for accuracy, performance, and reliability.

4. **Deploy and Monitor**:
   - Deploy the pipeline in a production environment.
   - Continuously monitor for performance and make adjustments as needed.

5. **Iterate and Improve**:
   - Gather feedback and refine the pipeline and products to better meet business needs.
### Example

Imagine a retail company wants to create a recommendation system for its online store:

1. **Data Ingestion**: Collect customer browsing and purchase data from the website.
2. **Data Processing**: Clean and transform the data to identify patterns in customer behavior.
3. **Data Storage**: Store the processed data in a data warehouse for easy access.
4. **Data Analysis**: Use machine learning algorithms to analyze the data and generate recommendations.
5. **Data Visualization**: Create dashboards to visualize customer trends and recommendation performance.
6. **Data Products**: Deploy the recommendation system on the website to enhance customer experience.



# Data Pipeline {#data-pipeline}


A data pipeline is a series of processes that automate the movement and transformation of data from various sources to a destination where it can be stored, analyzed, and used to generate insights. 

It ensures that data flows smoothly and efficiently through different stages, maintaining data quality and [Data Integrity](#data-integrity).

By implementing a data pipeline, organizations can automate data workflows, reduce manual effort, and ensure timely and accurate data delivery for decision-making.
### Workflow

1. [Data Ingestion](#data-ingestion)
2. [Data Transformation](#data-transformation)
3. [Data Storage](#data-storage)
4. [Preprocessing|Data Preprocessing](#preprocessingdata-preprocessing)
5. [Data Management](#data-management)
#### Other steps:

Design:
   - Define the objectives and requirements of the data pipeline.
   - Choose appropriate tools and technologies.

Development:
   - Build the pipeline components and integrate them into a cohesive system.

Testing:
   - Validate the pipeline to ensure data accuracy and performance.

Deployment:
   - Deploy the pipeline in a production environment.

Monitoring and Maintenance:
   - Continuously monitor the pipeline and make necessary adjustments to improve performance and reliability.

### Related Notes

- [Data Pipeline to Data Products](#data-pipeline-to-data-products)




[Data Pipeline](#data-pipeline)
   **Tags**: #data_workflow, #data_management

# Data Principles {#data-principles}


Data principles are essential for ensuring that data is managed, used, and maintained effectively and ethically.

1. [Data Quality](#data-quality) Ensure data is accurate, complete, reliable, and up-to-date. High-quality data is crucial for making informed decisions.

2. [Data Governance](#data-governance): Establish clear policies and procedures for data management, including roles and responsibilities, to ensure [data integrity](#data-integrity) and compliance with regulations.

3. Data Privacy: Protect personal and sensitive information by adhering to privacy laws and regulations, such as GDPR or CCPA, and implementing appropriate security measures.

4. Data Security: Safeguard data against unauthorized access, breaches, and other security threats through encryption, access controls, and regular [security](#security) audits.

5. Data Accessibility: Ensure that data is easily accessible to those who need it while maintaining appropriate security and privacy controls. This includes providing the necessary tools and training for data access.

6. Data Transparency: Maintain transparency about data collection, usage, and sharing practices. This helps build trust with stakeholders and ensures accountability.

7. Data Consistency: Standardize data formats and definitions across the organization to ensure consistency and interoperability.

8. Data Stewardship: Assign data stewards to oversee [data management](#data-management) practices, ensuring data quality, compliance, and proper usage.

9. [Data Lifecycle Management](#data-lifecycle-management) Manage data throughout its lifecycle, from creation and storage to archiving and deletion, ensuring that data is retained only as long as necessary.

10. Ethical Data Use: Use data ethically and responsibly, considering the potential impact on individuals and society. Avoid biases and ensure fairness in data-driven decisions.

11. Data [Documentation & Meetings](#documentation--meetings): Maintain thorough documentation of data sources, definitions, and processes to facilitate understanding and reproducibility.

12. Data Sharing and Collaboration: Encourage data sharing and collaboration within and across organizations to maximize the value of data, while respecting privacy and security constraints.
    
13. DRY

Related:
- [Performance Dimensions](#performance-dimensions)

# Data Product {#data-product}


A data product is

"a product that facilitates an end goal through data".

Delivering the final output, which could be dashboards, reports, or machine learning models. For example Recommendation systems or predictive analytics dashboards.

It applies more product thinking, whereas the "Data Product" essentially is a dashboard, report, and table in a [Data Warehouse](Data%20Warehouse.md) or a Machine Learning model.

Sometimes Data Products are also called [data asset](#data-asset).

# Data Quality {#data-quality}


Data quality is the process of ensuring that data meets established expectations. High-quality data is crucial for effective decision-making and analysis.

**Definition**: Data quality refers to the <mark>accuracy, consistency, and reliability of data.</mark> It is essential for maintaining trust in data-driven processes and outcomes. 

**Importance**: The principle of "garbage in, garbage out" highlights that poor-quality data leads to poor model performance.

Related terms:
- [Data Observability](#data-observability)
- [Change Management](#change-management)
- [Prevention Is Better Than The Cure](#prevention-is-better-than-the-cure)


Related terms:
- [Data Observability](#data-observability)
- [Data Contract](#data-contract)
- [Change Management](#change-management)

# Data Reduction {#data-reduction}

Reducing the volume of data through techniques:

[Dimensionality Reduction](#dimensionality-reduction)

[Sampling](#sampling): Use subsets of data for training to speed up the process and address issues like imbalanced data representation.

Remove features with zero or low [variance](#variance) and redundant features to improve model performance.

# Data Roles {#data-roles}

A data team is a specialized group within an organization responsible for managing, analyzing, and leveraging data to drive business decisions and strategies. 

The team collaborates across various functions to ensure data integrity, accessibility, and usability.

## Key Roles and Responsibilities

| Role                         | Focus Area                    | Key Responsibilities                                                                           |
| ---------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------- |
| **[Data Steward](#data-steward)**         | [Data quality](#data-quality) & governance | Enforces data policies, resolves [data quality](#data-quality) issues, manages metadata.                    |
| **[Data Governance](#data-governance) Team** | Policy & compliance           | Defines [data management](#data-management) rules, ensures regulatory adherence.                               |
| **[Data Engineer](#data-engineer)**        | Data infrastructure           | Builds data pipelines, integrates data sources, and ensures data flow.                         |
| **[Data Scientist](#data-scientist)**       | [Data analysis](#data-analysis) & modeling  | Utilizes BI tools, analyzes data, develops and deploys ML models.                              |
| **[ML Engineer](#ml-engineer)**          | Machine learning              | Configures and optimizes ML models, monitors performance in production.                        |
| **[Data Architect](#data-architect)**       | Data architecture             | Designs and manages data infrastructure, ensures data accessibility.                           |
| **[Data Analyst](#data-analyst)**         | Reporting & visualization     | Gathers and processes data, generates reports, communicates insights using tools like Tableau. |

#### Other Stakeholders
- **Business Analysts:** Ensure data is structured and accessible for analysis and reporting.
- **Senior Stakeholders and Business Ambassadors:** Communicate requirements, progress, and solutions to align with business goals.
- **Software Engineers and Data Teams:** Coordinate on data production and integration processes.

# Data Science {#data-science}


A field that uses the [Scientific Method](#scientific-method), algorithms, and systems to <mark>extract knowledge</mark> and insights from structured and [unstructured data](#unstructured-data). It combines techniques from [statistics](#statistics), computer science, and domain expertise to analyze and interpret complex data sets, enabling informed decision-making and predictive modeling.

Resources:
- https://scikit-learn.org/stable/auto_examples/index.html


# Data Scientist {#data-scientist}

Data Scientist
  - Utilizes [Business Intelligence](#business-intelligence) (BI) tools to analyze data.
  - Works with data lakes to extract insights.
  - Develops and deploys production Machine Learning (ML) models for predictions.

# Data Selection In Ml {#data-selection-in-ml}

When selecting data for machine learning models, several important considerations can significantly impact the model's performance/[Model Optimisation](#model-optimisation) and the insights you can derive from it. Here are key factors to consider:

1. Relevance:
   - Ensure that the features (input variables) you select are relevant to the problem you are trying to solve. Irrelevant features can introduce noise and reduce model accuracy.

2. Quality: [Data Quality](#data-quality)
   - Assess the quality of the data, including checking for missing values, outliers, and errors. Poor quality data can lead to inaccurate models.

3. Quantity:
   - Consider the size of your dataset. More data can lead to better models, but it also requires more computational resources. Ensure you have enough data to train your model effectively.

4. Balance: [Imbalanced Datasets](#imbalanced-datasets)
   - Check for [Imbalanced Datasets|class imbalance](#imbalanced-datasetsclass-imbalance) in classification problems. An imbalanced dataset can bias the model towards the majority class. Techniques like resampling, synthetic data generation, or using different evaluation metrics can help address this.

5. Feature Distribution: [Distributions](#distributions)
   - Analyze the distribution of your features. Features with skewed [distributions](#distributions) may need transformation ([Data Transformation](#data-transformation)) (e.g., log transformation) to improve model performance.

6. [Correlation](#correlation):
   - Examine the correlation between features. Highly correlated features can lead to [multicollinearity](#multicollinearity), which can affect model stability and interpretability. Consider removing or combining correlated features.

7. Dimensionality: [Dimensionality Reduction](#dimensionality-reduction)
   - High-dimensional data can lead to overfitting. Techniques like [feature selection](#feature-selection), dimensionality reduction (e.g., PCA), or regularization can help manage this.

8. Temporal Considerations:
- For time series data, ensure that the temporal order is maintained. Avoid data leakage by ensuring that future information is not used in training.

9. Domain Knowledge:
   - Leverage domain expertise to select features that are known to be important for the problem. This can guide feature engineering and selection.

10. Data Leakage:
  - Be cautious of [Data Leakage](#data-leakage), where information from the test set is inadvertently used in training. This can lead to overly optimistic performance estimates.

11. Scalability:
- Consider the scalability of your data selection process. As datasets grow, ensure that your methods can handle larger volumes efficiently.

# Data Selection {#data-selection}


Data selection is a crucial part of data manipulation and analysis. Pandas provides several methods to select data from a DataFrame.

In [DE_Tools](#de_tools)  we explore how to do Data Selection with Pandas
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/selection.ipynb

Related:
- [Data Selection in ML](#data-selection-in-ml)
## Examples
### Selecting Columns

You can select a single column from a DataFrame using either bracket notation or dot notation:

```python
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
column_a = df['A']  # or df.A
```
### Selecting Rows by Index

To select rows by their index position, you can use slicing:
```python
rows_0_to_2 = df[0:3]  # Selects the first three rows
```
### Selecting Rows by Date Range

If your DataFrame has a DateTime index, you can select rows within a specific date range:

```python
date_rng = pd.date_range(start='2013-01-01', end='2013-01-06', freq='D')
df = pd.DataFrame(date_rng, columns=['date'])
df.set_index('date', inplace=True)
selected_dates = df['2013-01-02':'2013-01-04']
```
### Label-based Selection

Use `.loc` or `.at` to select rows by label:

```python
df = pd.DataFrame({'Weather': ['Sunny', 'Rain', 'Cloudy'], 'Temp': [30, 22, 25]})
df.set_index('Weather', inplace=True)
rain_row = df.loc['Rain']  # or df.at['Rain']
```
### Position-based Selection

Use `.iloc` or `.iat` to select rows by position:

```python
third_row = df.iloc[2]  # Selects the third row
specific_value = df.iat[1, 1]  # Selects the value at row 1, column 1
```
### Conditional Selection

Create a new DataFrame based on a condition:
```python
df_new = df[df['var1'] >= 999]  # Selects rows where 'var1' is greater than or equal to 999
```
The condition `df["var1"] >= 999` creates a boolean Series that filters the rows of `df`.



# Data Steward {#data-steward}

A **Data Steward** is responsible for ensuring the quality, integrity, and governance of an organization's data assets. They act as a bridge between business users, IT teams, and data governance policies, ensuring that data is well-defined, accurate, and used appropriately.

### **Key Responsibilities of a Data Steward**

1. **Data Quality Management** – Ensuring data accuracy, completeness, consistency, and reliability across systems.
2. **Metadata Management** – Documenting data definitions, relationships, and lineage.
3. **Data Governance Compliance** – Implementing policies, standards, and best practices for data handling.
4. **Master Data Management (MDM)** – Managing critical business data entities like customers, products, and suppliers.
5. **Collaboration with Stakeholders** – Acting as a liaison between business units, data engineers, and data governance teams.
6. **Issue Resolution** – Identifying and resolving data-related issues such as duplicates, missing values, and inconsistencies.
7. **Data Security & Privacy** – Ensuring compliance with regulations (e.g., GDPR, HIPAA) by monitoring access and usage.

### **Why is a Data Steward Important?**

- Enhances **data trustworthiness**, leading to better decision-making.
- Reduces **data inconsistencies** and errors in analytics and reporting.
- Supports **regulatory compliance** and risk management.
- Enables **efficient data integration** across systems and departments.





[Data Steward](#data-steward)
  - Responsible for [data governance](#data-governance) and quality.
  - Ensures that data policies and standards are adhered to across the organization.
  - Acts as a liaison between data users and IT to facilitate [data management](#data-management).

# Data Storage {#data-storage}


Data storage is a fundamental aspect of [Data Engineering](#data-engineering), influencing processes such as 
- (occurring after [Data Ingestion](#data-ingestion))
- [Data Transformation](#data-transformation)
- [Querying](#querying)
- [data management](#data-management).

Storing the [Data Transformation](#data-transformation) data in a database or [Data Warehouse](#data-warehouse) for easy access and analysis.
## Types of Storage

Data storage encompasses various methods and technologies for storing, retrieving, and managing data. The choice of storage method significantly impacts <mark>data retrieval efficiency</mark> and consistency

| Storage Type                                 | Description                                                                                           |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| [storage layer object store\|Object Store](#storage-layer-object-storeobject-store) | The gold standard for data lakes, ideal for unstructured data such as images, audio, and text.        |
| [Database](#database)                                 | The most widely deployed database globally is [SQLite](#sqlite). Suited for transaction recording.           |
| [NoSQL](#nosql)                                    |                                                                                                       |
| [Data Warehouse](#data-warehouse)                           | Excels in analytics and reporting.                                                                    |
| [Data Lake](#data-lake)                                | Offers versatility for storing raw data, particularly beneficial for advanced analytics applications. |
## Follow-Up Questions
- How do different data storage methods impact data retrieval speed in large datasets?
- What are the trade-offs between using relational versus [NoSQL](#nosql) databases in specific applications?
## Related Resources
- [Cloud Providers](#cloud-providers)
- [Amazon S3](#amazon-s3)
- [Data Governance](#data-governance)
- [Data Engineering Tools](#data-engineering-tools)



# Data Streaming {#data-streaming}


Data Streaming is used for real-time data processing, allowing continuous flow and processing of data as it arrives. This is different from [batch processing](#batch-processing), which handles data in chunks.

The key to data streaming is the [Publish and Subscribe](#publish-and-subscribe)
  
[Apache Kafka](#apache-kafka)

Example:
  - Companies like Netflix use Kafka to handle billions of messages daily, powering real-time recommendations, analytics, and user activity tracking.

[Alternatives to Batch Processing](#alternatives-to-batch-processing)



[Data Streaming](#data-streaming)
   **Tags**: #data_workflow

# Data Terms {#data-terms}



# Data Transformation With Pandas {#data-transformation-with-pandas}


Using [pandas](#pandas) we can do the following:


- [Merge](#merge)
- [Concatenate](#concatenate)
- [Joining Datasets](#joining-datasets) 
- [Pandas join vs merge](#pandas-join-vs-merge)
- [Multi-level index](#multi-level-index)

- [Aggregation](#aggregation)

- [Pandas Stack](#pandas-stack)
- [Crosstab](#crosstab)

A summary of transformations steps can be helpful:

|Step|Operation|Result|
|---|---|---|
|1|`set_index`|Rows get hierarchical keys|
|2|`stack`|Wide → long with 3-level row index|
|3|`reset + extract`|Parse variable names into fields|
|4|`pivot`|Tidy format with metric columns|
|5|`unstack`|Wide format with MultiIndex columns|

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Investigating/Transformation

Related terms:

### Split-Apply-Combine

![Pasted image 20250323081817.png](../content/images/Pasted%20image%2020250323081817.png)


# Data Transformation {#data-transformation}


Data transformation is the process of converting data from one format to another. 

Data transformation may involve:  
- [Data Cleansing](#data-cleansing)
- [Structuring and organizing data](#structuring-and-organizing-data)
- [Aggregation](#aggregation)
- [Data Selection](#data-selection)
- [Joining Datasets](#joining-datasets)
- [Normalisation of data](#normalisation-of-data)
- [Normalised Schema](#normalised-schema)

Others:
- Sorting: Arranging data in a logical order.  
- Validating: Ensuring data integrity and accuracy.  
- Data Type Conversion: Changing data types (e.g., converting strings to integers).  
- Schema Normalization: Ensuring a consistent data structure for efficiency.  

Related:
- [Data Transformation with Pandas](#data-transformation-with-pandas)  
- [Data transformation in Data Engineering](#data-transformation-in-data-engineering)
- [Data transformation in Machine Learning](#data-transformation-in-machine-learning)
- [Benefits of Data Transformation](#benefits-of-data-transformation)



# Data Validation {#data-validation}

Data Validation:

- **Error Prevention**: It ensures data accuracy by preventing incorrect or inappropriate data entries.
- **Consistent Data Entry**: Helps maintain consistency across large datasets by controlling what users can input.
- **Efficiency**: By providing drop-down lists or constraints, it reduces the chances of manual errors.
- **Better [Data Quality](#data-quality): Validating input ensures that your data is clean and ready for analysis or reporting without requiring additional checks.
- [type checking](#type-checking)
- [TypeScript](#typescript)

--- 
[Pydantic](#pydantic)

# Data Virtualization {#data-virtualization}

Organizations may also consider adopting a data virtualization solution to integrate their data. 

In this type of [data integration](#data-integration), data from multiple sources is left in place and is <mark>accessed</mark> via a virtualization layer so that it <mark>_appears_</mark> as a single data store. 

This virtualization layer makes use of adapters that translate queries executed on the virtualization layer into a format that each connected source system can execute. 

The virtualization layer then combines the responses from these source systems into a single result. This data integration strategy is sometimes used when a BI tool like Tableau needs to access data from multiple data sources.

One disadvantage of data virtualization is that analytics workloads are executed on operational systems, which could interfere with their functioning. Another disadvantage is that the virtualization layer may act as a bottleneck on the performance of analytics operations.

# Data Visualisation {#data-visualisation}


Data visualization involves presenting data in a visual format, enabling stakeholders to quickly grasp insights and make informed decisions. Effective visualization tools include dashboards and reports.

Can generate reports using:
- [Tableau](#tableau)
- [PowerBI](#powerbi)
- [Looker Studio](#looker-studio)

# Data Warehouse {#data-warehouse}


A Data Warehouse (DWH) is a centralized repository designed for [Querying](#querying) and analysis, storing large volumes of structured data from various sources within an organization. It supports reporting and decision-making by providing a consolidated view of data.
### Key Features

[Data Ingestion](#data-ingestion) Integration: Combines data from diverse sources (e.g., transactional databases, CRM systems) into a single repository, ensuring consistency.
  
Subject-Oriented: Organizes data around key business areas (e.g., sales, finance) rather than operational processes.

Non-Volatile: Data remains unchanged once entered, preserving historical data for long-term analysis.

Time-Variant: Stores data with a time dimension, enabling historical analysis and trend identification.

### Components

Data Sources: Internal (e.g., ERP systems) and external (e.g., market research data) origins of data.

[ETL](#etl)

[Data Storage](#data-storage)

Metadata/[Documentation & Meetings](#documentation--meetings): Information about the data, including definitions and transformation rules, aiding in data management.

Access Tools: Tools for querying and analyzing data, such as SQL clients and business intelligence tools.

##### Resources
- [Designing a Data Warehouse](https://www.youtube.com/watch?v=patBYUGwsHE)
- [Why a Data Warehouse?](https://www.youtube.com/watch?v=jmwGNhUXn_o)


# Data Transformation In Data Engineering {#data-transformation-in-data-engineering}

Data transformation in [Data Engineering](#data-engineering) is a key step in data pipelines, often part of:  

- [ETL (Extract, Transform, Load)](ETL.md) [ETL](#etl): Data is transformed before loading into the target system.  
- [ELT (Extract, Load, Transform)](term/elt.md) [ELT](#elt): Data is loaded first, then transformed for analysis.  
- EtLT (Extract, “tweak”, Load, Transform: A hybrid approach combining elements of ETL and ELT.  

Related:
- [ETL vs ELT](#etl-vs-elt)for a comparison.

# Data Transformation In Machine Learning {#data-transformation-in-machine-learning}

Transforming raw data into a meaningful format is necessary for building effective models.  

- [Supervised Learning](#supervised-learning): Annotating datasets with correct labels (e.g., labeling images of apples vs. other fruits).  
- Manual & Automated Labeling: Using human annotators or leveraging existing labeled datasets (e.g., Google reCAPTCHA).  
- Feature Scaling & Encoding: Applying normalization and encoding to categorical variables.  
- [Encoding Categorical Variables](#encoding-categorical-variables): Converting categorical data into numerical format for machine learning models.  

# Database Index {#database-index}


In [DE_Tools](#de_tools) see: 
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Indexing/Indexing.ipynb

Related terms:
- [Covering Index](#covering-index)
- Partial Index (Index with where clause)

Indexing is a technique used to <mark>speed up data retrieval</mark> in [database](#database). It achieves this by creating a separate structure, known as an index, that organizes specific columns of data for faster access. Better than scanning.

Commonly created on <mark>primary keys</mark> (unique for item) and foreign keys.

Indexes can also be created across multiple tables to enhance the performance of complex queries, especially those that involve joins. A special type of index, called a <mark>covering index</mark>, includes all the necessary data within the index itself, further improving efficiency.

Example:  
For instance, creating an index on the "title" column in the "movies" table can significantly reduce the time it takes to execute [Querying|queries](#queryingqueries) that search for movie titles.

## Using Indexes

Keep in mind that indexes consume additional storage space.

Creating Indexes: To improve search performance, create indexes on relevant columns. For example:
  ```sql
  CREATE INDEX idx_title ON movies(title);
  ```

Analyzing Queries: Use the `EXPLAIN QUERY PLAN` command to check if a query is utilizing an index effectively.

Dropping Indexes: If an index is no longer needed, it can be removed using:
  ```sql
  DROP INDEX idx_title;
  ```

## Space and Time Trade-offs

Space: Indexes require extra storage because they are built using B-Trees, which are hierarchical data structures.
Time: While indexes speed up data retrieval, creating and updating them can slow down data insertion and modification processes.

## How Indexes Work

- Data Structure: Indexes typically use a [B-tree](#b-tree) data structure, which allows for efficient searching.
- Node Structure: A B-tree organizes data into nodes, where each node contains links to the corresponding rows in the table. The data is sorted, enabling quick access.
- Search Mechanism: When searching, a binary search method is employed. This involves checking the middle of the data and deciding which side to search next, taking advantage of the ordered nature of B-trees for efficiency.


# Database Management System (Dbms) {#database-management-system-dbms}


A **Database Management System** (DBMS) is software that allows you to interact with and manage databases.
Easiest to use:
- [SQLite](#sqlite)
- [PostgreSQL](#postgresql)

Others:
- [MySql](#mysql)
- [MongoDB](#mongodb)
- [Oracle](#oracle)

These systems enable users to perform [CRUD](#crud) operations while maintaining data integrity and providing tools for backup, security, and optimization.

Can be proprietary (paid, with support) or Open source (free, self-supported).


# Database Schema {#database-schema}


A [Database Schema|schema](#database-schemaschema) is the structure that defines how data is organized in a [Database](#database), used in [Data Management](#data-management). It specifies the tables, columns, relationships, and constraints within the database. The schema is used for ensuring data is stored consistently and can be queried efficiently.

1. Definition and Components: A database schema represents the <mark>structure</mark> around the data, including tables, views, fields, relationships, and various other elements like indexes and triggers. It provides a framework for organizing and understanding data.

2. Importance of <mark>Structure</mark>: Without a schema, data can be chaotic and difficult to <mark>interpret</mark>. A well-defined schema organizes data, making it <mark>manageable and meaningful.</mark>

3. Schema on Read vs. Schema on Write: 
   - Schema on Read: Structure is applied when the data is read, useful for unstructured data stores.
   - Schema on Write: Structure is enforced when data is written, typical of traditional databases.

1. Design Influences: The design of a schema impacts database behavior. For example, schemas designed with tables connected by primary keys are optimized for transactional applications, while star schemas are designed for efficient read operations in data warehouses.

2. Performance Impact: A good schema can significantly <mark>improve query performance</mark>, reducing processing <mark>time</mark> and <mark>cost</mark>, and simplifying query complexity.

3. [Data Modelling](#data-modelling): Despite being considered an old concept, data modeling remains crucial for creating effective schemas, particularly in the context of big data and analytics.

4. Iterative Process: Developing a data warehouse schema involves iterative refinement, starting with interviews to create a [conceptual data model](#conceptual-data-model), which is then tested and refined through multiple iterations before being implemented.

5. Strategic Importance: The strategic design and deployment of a database schema are vital for efficient data warehousing and analytics. Intracity specializes in this process, helping organizations define and execute their data strategies.

Related to: 
- [Types of Database Schema](#types-of-database-schema)
- [Implementing Database Schema](#implementing-database-schema)

#### Resources
[link](https://www.youtube.com/watch?v=3BZz8R7mqu0)

# Database Storage {#database-storage}


Methods and optimizations for storing, retrieving, and processing data in [database](#database) systems. 

[Columnar Storage](#columnar-storage)

[Row-based Storage](#row-based-storage)

[Vectorized Engine](#vectorized-engine)



# Database Techniques {#database-techniques}


Techniques:
- [Soft Deletion](#soft-deletion)
- [Concurrency](#concurrency) 
	- [Race Conditions](#race-conditions)
- [Querying](#querying)
	- [SQL Joins](#sql-joins)
	- [Stored Procedures](#stored-procedures)
	- Cleaning: Use **Levenshtein Distance** (if SQLite extension is available) to group similar entries.
- [Database Index|Indexing](#database-indexindexing)
	- [Query Plan](#query-plan)
	- [Vacuum](#vacuum)

# Database {#database}


Databases manage large data volumes with scalability, speed, and flexibility. Key systems include:

- [MySql](#mysql)
- [PostgreSQL](#postgresql)
- [MongoDB](#mongodb)

They facilitate efficient [CRUD](#crud) operations and transactional processing ([OLTP](#oltp)) structured by [Database Schema|schema](#database-schemaschema) that organizes data into tables and relationships.

Key Features
- **[Structured Data](#structured-data)**: Organized for efficient CRUD operations, allowing reliable access.
- **Relational Databases**: Use SQL to manage data in tables with relationships expressed through foreign keys and joins, minimizing redundancy.

Structure
- Data is organized into tables (like spreadsheets) with columns (fields) and rows (records), enabling efficient storage and retrieval.

Flexibility
- Databases have a flexible schema that adapts to evolving requirements, unlike static solutions like spreadsheets.

Related Ideas:
- [Spreadsheets vs Databases](#spreadsheets-vs-databases)
- [Database Management System (DBMS)](#database-management-system-dbms)
- [Components of the database](#components-of-the-database)
- [Relating Tables Together](#relating-tables-together)
- [Turning a flat file into a database](#turning-a-flat-file-into-a-database)
- [Database Techniques](#database-techniques)

# Databricks Vs Snowflake {#databricks-vs-snowflake}


Comparison between **[Databricks](#databricks)** and **[Snowflake](#snowflake)**:

- **Databricks** is a versatile platform that emphasizes collaborative data science and engineering through interactive notebooks, making it suitable for advanced analytics and machine learning applications.
- **Snowflake**, on the other hand, focuses on [Data Warehouse](#data-warehouse) and offers a robust SQL interface for analytics, making it a preferred choice for organizations prioritizing data storage and reporting capabilities.

| Feature                      | **Databricks**                                         | **Snowflake**                                      |
|------------------------------|-------------------------------------------------------|---------------------------------------------------|
| **Primary Functionality**     | Unified analytics platform for big data processing and machine learning. | Cloud-based data warehousing and analytics platform. |
| **Data Processing**           | Built on **Apache Spark**, optimized for large-scale data processing and machine learning workflows. | Uses its own SQL-based engine for data warehousing; excels in querying structured data. |
| **Collaboration**             | Emphasizes collaboration through **notebooks** (e.g., Jupyter `.ipynb` files) that allow for interactive data analysis and coding. | Provides features for data sharing and collaboration but lacks the notebook interface. |
| **Data Structure**            | Supports both structured and unstructured data, integrating seamlessly with data lakes (e.g., Delta Lake). | Primarily designed for structured data and semi-structured data (like JSON) stored in tables. |
| **Scalability**               | Uses clusters to scale up compute resources dynamically; suitable for big data workloads. | Offers automatic scaling of compute and storage resources, focusing on cost-effective scaling. |
| **Machine Learning Support**  | Integrated support for ML libraries (e.g., MLlib, MLflow) to build and deploy machine learning models. | Limited built-in support for machine learning, primarily used for data storage and querying. |
| **Query Language**            | Supports multiple programming languages (Python, R, Scala, SQL) within notebooks. | Primarily uses SQL for querying data, providing a familiar interface for data analysts. |
| **Deployment**                | Available on major cloud platforms (AWS, Azure, GCP); allows for more customization and flexibility in deployment. | Also cloud-native, designed for seamless deployment in the cloud, with less emphasis on infrastructure management. |
| **Use Cases**                 | Ideal for big data analytics, data engineering, and data science projects requiring complex processing. | Best suited for traditional data warehousing, business intelligence, and analytics use cases. |




# Databricks {#databricks}


### **Databricks Overview**

>[!Summary]  
Databricks is a cloud-based platform for [big data](#big-data) processing built on [Apache Spark](#apache-spark). It provides an integrated workspace for collaboration among [data engineer](#data-engineer)s, data scientists, and analysts. Databricks on Azure simplifies Spark deployment by offering auto-scaling clusters, real-time analytics, and integration with various Azure services, such as Azure [Data Lake](#data-lake) for large-scale data storage.


**Cloud Platform Compatibility**: 
  - Supports the big three cloud providers (AWS, Azure, GCP).
- **Integration with Other Technologies**: 
  - Combines capabilities of:
    - **Apache Spark**
    - **Delta Lake**
    - **MLflow**
- **Data Lakehouse Architecture**:
  - Represents a combination of a **data [Data Warehouse|warehouse](#data-warehousewarehouse)** and a **data lake**.

Core Components:
1. **Tables**: 
   - Represents files and data sources.
2. **Clusters**: 
   - Provides computing power for data processing.
3. **Notebooks**: 
   - Similar to Jupyter notebooks; support multiple programming languages and allow for productionization of code.
4. **Workspaces**: 
   - Collaborative environments for teams to work together.

Scalability
- Leverages the scalability of **[Hadoop](#hadoop)** while integrating advanced features for big data processing.

[Databricks vs Snowflake](#databricks-vs-snowflake)

# Datasets {#datasets}

This note collects notes on datasets that are good examples for exploring various concepts.
## Heart Failure Prediction Dataset
- **Link**: [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Useful for**: Exploring predictive modeling in healthcare.

## Time Series Exploration
- **Description**: There is a dataset with seasonality, bikes, which can be used to explore [Time Series](#time-series) concepts.

## Numenta Anomaly Benchmark (NAB)
- **Link**: [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB?ref=hackernoon.com)
- **Columns**: timestamp, value
- **Description**: NAB is used to evaluate and compare the performance of different anomaly detection algorithms on a diverse set of time series data. It includes real-world and artificial time series data covering domains such as finance, transportation, and environmental monitoring.

## U.S. Census Bureau's International Data Base (IDB)
- **Link**: [International Data Base (IDB)](https://www.census.gov/data-tools/demo/idb?ref=hackernoon.com)
- **Useful for**: Researchers, policymakers, and businesses studying population dynamics, forecasting future population growth, monitoring economic development, and comparing demographic and economic characteristics of different countries.

## Wikipedia Web Traffic Time Series Dataset
- **Link**: [Wikipedia Web Traffic Time Series Dataset](https://www.kaggle.com/code/muonneutrino/wikipedia-traffic-data-exploration/data?ref=hackernoon.com)
- **Useful for**: Examining the dynamics of website traffic, understanding interactions with Wikipedia, and identifying patterns and trends in online behavior. It can be used to compare traffic across languages, analyze the popularity of articles, and track the evolution of articles over time.



# Debugging Ipynb {#debugging-ipynb}

debugging jupyter cells

https://www.youtube.com/watch?v=CY6uZIoF_kQ

Sometimes dissapears:
https://stackoverflow.com/questions/72671709/vs-code-debug-cell-disappears-arbitrarily-in-jupyter-notebook-view



# Debugging {#debugging}


Debugging is the process of identifying, analyzing, and resolving bugs or defects in software while [Testing](#testing). It is a critical part of the [Software Development Life Cycle](#software-development-life-cycle), ensuring that applications function correctly and efficiently. Debugging involves several techniques and tools to pinpoint the source of errors and fix them.

In [ML_Tools](#ml_tools) see:
- [Debugging.py](#debuggingpy)
- [Testing_unittest.py](#testing_unittestpy)
- [Testing_Pytest.py](#testing_pytestpy)
#### Key Concepts in Debugging

1. [Types of Computational Bugs](#types-of-computational-bugs): Understanding the types of bugs, such as cumulative rounding errors, integer overflow, and race conditions, is essential for effective debugging.

2. **How to Manage/View Bugs**:
   
   - **Console Log/Dir**: Use console logging to output variable values and program states to the console, helping to trace the flow of execution.
    
   - **Availability in VSCode**: Visual Studio Code provides powerful debugging features like breakpoints, watch expressions, and call stacks to help developers inspect and modify code execution.
     
   - **Sample Script**: Creating a minimal script that reproduces the bug can simplify the debugging process by isolating the problem.
     
   - **Logging in Python**: Python's logging module allows developers to record events, errors, and informational messages, which can be crucial for diagnosing issues.
     
   - **Run and Debug**: Step through code execution using debugging tools to observe the program's behavior and identify where it deviates from expected results.
     
   - **Log Point/Break Point**: Set breakpoints to pause execution at specific lines of code, allowing inspection of variables and program state at that moment.

#### Solution Attempts

1. **Reproduce the Bug**: Simplifying the code to reproduce the bug helps in understanding its cause and facilitates easier sharing with others for collaborative debugging. Sharing on platforms like [StackBiz](#stackbiz) can help others contribute to the solution.

2. **Automated Testing**: Implementing automated tests ensures that code changes do not introduce new bugs and that existing functionality remains intact.

3. **Test-Driven Development (TDD)**: Writing tests before the actual code helps define expected behavior and ensures that the code meets these expectations.

4. **Static Analysis**: Tools like [TypeScript](#typescript) and ESLint analyze code for potential errors without executing it, helping to catch issues early in the development process.

#### Debugging Tools and Techniques

- **Integrated Development Environments (IDEs)**: IDEs like Visual Studio Code, IntelliJ IDEA, and Eclipse offer built-in debugging tools that streamline the debugging process.
  
- **Version Control Systems**: Tools like [Git](#git) allow developers to track changes and revert to previous versions if a bug is introduced.
  
- **Profilers**: These tools analyze program performance and help identify bottlenecks or inefficient code paths.
  
- **Memory Analyzers**: Tools like Valgrind help detect memory leaks and other memory-related issues.





# Debugging.Py {#debuggingpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Debugging.py

This script includes examples of logging, using breakpoints, and reproducing a simple bug for practice
### Key Concepts Demonstrated in the Script

1. **Logging in Python**: The script uses Python's logging module to record debug, info, error, and warning messages. This helps track the flow of execution and diagnose issues.

2. **Reproduce the Bug**: The script intentionally includes a division by zero bug to demonstrate how to identify and fix it.

3. **Breakpoints**: You can set a breakpoint in your IDE at the line where `result = divide_numbers(num1, num2)` to inspect the values of `num1` and `num2`.

4. **Automated Testing**: The script includes a simple assertion to test the `divide_numbers` function, ensuring it behaves as expected.

5. **Static Analysis**: The commented line `unused_variable = 42` can be used to simulate a static analysis warning for an unused variable.



# Decision Tree {#decision-tree}


A Decision Tree is a type of [Supervised Learning](#supervised-learning) algorithm used to predict a target variable based on input features. It involves splitting data into subsets to create a tree-like model.

Decision Tree Structure are a flowchart-like model where each internal node represents a decision based on a feature, branches represent outcomes, and leaf nodes represent final predictions.

Splits data recursively based on feature importance, forming a tree-like structure.

The decision tree algorithm calculates the [Gini impurity](#gini-impurity) for each possible split and selects the one with the lowest impurity. Use to make predictions on new data, the algorithm traverses the decision tree from the root node to a leaf node, following decision rules based on input features. Once it reaches a leaf node, it assigns the corresponding class label or prediction.

![Pasted image 20240404154526.png|500](../content/images/Pasted%20image%2020240404154526.png|500)
### Key Concepts

1. Objective: Predict a target variable using input features.
2. Splitting: Identify the best feature to split the data into subsets, aiming for homogeneous groups.
3. Impurity Calculation: Use metrics like [Gini Impurity](#gini-impurity) or [Cross Entropy](#cross-entropy) ([Gini Impurity vs Cross Entropy](#gini-impurity-vs-cross-entropy))to evaluate splits. Choose the split that minimizes impurity.
4. Purity: A node is pure if it perfectly classifies the data, requiring no further splits.
5. Leaf Node Output: Assigns the most common class label or average value in the node.
6. [Overfitting](#overfitting): Can occur if the tree is too complex. Mitigate with [pruning](#pruning) and limiting tree depth.
7. [Cross Validation](#cross-validation): Refine the model to better generalize to new data.

### Splitting Process

The splitting process in a Decision Tree involves dividing the dataset into subsets to create a tree-like structure. This process is crucial for building an effective model that can predict target variables accurately.

Splitting Criteria:
- The algorithm evaluates various features to determine the best split at each node.
- It selects the feature and split point that minimize impurity in the resulting child nodes.
- The split that most effectively reduces impurity is chosen, ensuring that each subset is as homogeneous as possible.

#### Building Process

1. **Initial Splitting**:
   - Begin at the root node and select the best feature to split the data. This selection is based on impurity measures such as Gini impurity or entropy for [classification](#classification) tasks, and variance reduction for regression tasks.
   - The goal is to create subsets that are as homogeneous as possible with respect to the target variable.

2. **Recursive Partitioning**:
   - After the initial split, each subset becomes a child node.
   - The algorithm recursively applies the splitting process to each child node.
   - Continue splitting until stopping criteria are met, such as reaching a maximum tree depth, having a minimum number of samples per node, or achieving insufficient improvement in purity.

3. **Leaf Nodes**:
   - The process continues until reaching leaf nodes, which have no further splits.
   - At each leaf node, assign a class label (for classification) or predict a continuous value (for regression) based on the majority class or average value of the samples in that node.

### Refinement

Pruning:
  - Pre-pruning: Stop tree growth early based on criteria like maximum depth or minimum impurity improvement.
  - Post-pruning: Allow the tree to grow fully, then prune back based on performance metrics.

### [Hyperparameter](#hyperparameter)

Can use [GridSeachCv](#gridseachcv) to pick the best paramaters.

| **Parameter**       | **Purpose**                      | **Effect**                  | **Example**                                      |
|---------------------|----------------------------------|-----------------------------|--------------------------------------------------|
| `criterion`         | Splitting criteria               | Impacts decision logic.     | `criterion='gini'` or `criterion='entropy'`      |
| `max_depth`         | Maximum tree depth               | Prevents overfitting.       | `max_depth=5` limits the tree depth to 5.        |
| `min_samples_split` | Min samples to split a node      | Limits tree growth.         | `min_samples_split=10` requires at least 10 samples to split a node. |
| `min_samples_leaf`  | Min samples at leaf node         | Reduces overfitting.        | `min_samples_leaf=5` ensures every leaf has at least 5 samples. |
| `max_features`      | Features considered for splitting| Adds randomness.            | `max_features='sqrt'` or `max_features=3`.       |
| `max_leaf_nodes`    | Max leaf nodes allowed           | Reduces overfitting.        | `max_leaf_nodes=20` caps the tree at 20 leaves.  |
| `class_weight`      | Adjusts for imbalanced data      | Improves fairness.          | `class_weight='balanced'` or `class_weight={0:1, 1:2}`. |
| `ccp_alpha`         | Pruning parameter                | Simplifies tree.            | `ccp_alpha=0.01` prunes weak splits based on complexity. |


### Advantages and Disadvantages of Decision Trees

Advantages:
- Simple and [interpretability|interpretable](#interpretabilityinterpretable) model.
- Minimal data preparation required.
- Transparent decision-making process.

Disadvantages:
- Prone to overfitting, especially with complex datasets.
- Sensitive to small changes in data.
- Can become complex with many features.

[Decision Tree](#decision-tree)



# Deep Learning Frameworks {#deep-learning-frameworks}


[Watch Overview Video](https://www.youtube.com/watch?v=MDP9FfsNx60)

### TensorFlow

**Focus**: 
  TensorFlow is a comprehensive open-source platform for machine learning. It provides a flexible and comprehensive ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ML, and developers easily build and deploy ML-powered applications.
  
**Integration**: 
  TensorFlow can implement a wide range of machine learning algorithms, including those available in [Sci-kit Learn](#sci-kit-learn), making it versatile for various applications.
  
**Modularity**: 
  Its modular architecture allows users to deploy computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.
  
**Parallelization**: 
  TensorFlow is optimized for high-performance numerical computation, making it suitable for large-scale machine learning tasks that require parallel processing.
  
**Use Cases**: 
  TensorFlow is widely used in both academic research and industry for tasks such as image and speech recognition, natural language processing, and more.


### Sci-kit Learn

**Focus**: 
  Sci-kit Learn is a simple and efficient tool for data mining and data analysis, built on NumPy, SciPy, and matplotlib. It is primarily used for traditional machine learning techniques such as classification, regression, clustering, and dimensionality reduction.
  
**Limitations**: 
  While excellent for classical machine learning tasks, Sci-kit Learn is not designed for deep learning or neural network architectures, which require more specialized frameworks like TensorFlow or PyTorch.

### Keras

**API Level**: 
  Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It allows for easy and fast prototyping through user-friendly, modular, and extensible code.
  
**Integration**: 
  Keras is tightly integrated with TensorFlow 2.0, providing a simplified interface for building and training deep learning models.
  
**Purpose**: 
  Designed to enable fast experimentation, Keras is ideal for beginners and researchers who need to quickly prototype and test new ideas.
  
**Performance**: 
  While Keras simplifies model building, it may not be as performant as lower-level frameworks like TensorFlow when it comes to fine-tuning and optimizing models for production.


# Deep Learning {#deep-learning}


>[!Summary]
> Deep learning is a subset of machine learning that uses neural networks to process large-scale data for tasks like image and speech recognition, natural language processing, and recommendation systems. 
> 
> A neural network consists of layers of nodes where each node performs weighted sums of its inputs, applies activation functions like ReLU or sigmoid, and produces an output. 
> 
> [Backpropagation](#backpropagation) is the primary algorithm for training neural networks by minimizing error through [Gradient Descent](#gradient-descent). Regularization techniques, such as dropout, prevent overfitting. 
> 
> Popular frameworks like [PyTorch](#pytorch) and [TensorFlow](#tensorflow) facilitate deep learning model development.

Questions:
- [What is the role of gradient-based optimization in training deep learning models. ](#what-is-the-role-of-gradient-based-optimization-in-training-deep-learning-models)
- [Explain different gradient descent algorithms, their advantages, and limitations.](#explain-different-gradient-descent-algorithms-their-advantages-and-limitations)

Areas of Deep Learning:
- [LLM](#llm)
- [Neural network|Neural Network](#neural-networkneural-network)


>[!Follow up questions]
> - How does the choice of activation function affect the performance of deep learning models across different tasks?
> - What are the trade-offs between different gradient descent algorithms (e.g., [Stochastic Gradient Descent|SGD](#stochastic-gradient-descentsgd) vs. Adam) in training neural networks? See [Optimisation techniques](#optimisation-techniques).

>[!Related Topics]
> - [Transfer Learning](#transfer-learning): Applying pre-trained models to new tasks.



# Deep Q Learning {#deep-q-learning}


Deep [Q-Learning](#q-learning) is a type of [reinforcement learning](#reinforcement-learning) algorithm that combines Q-Learning with [Neural network](#neural-network). Necessary when Q-Table grows too large.

Updates the weights in the model.

![Pasted image 20250220133838.png](../content/images/Pasted%20image%2020250220133838.png)

## Key Concepts

### Target Network

- **Purpose**: The target network is used to stabilize the training process in Deep Q-Learning.
- **When is it needed?**: It is needed when updating the Q-values to prevent oscillations and divergence during training.
- **How it works**: The target network is a copy of the main Q-network and is used to generate target Q-values. It is updated less frequently than the main network, often using a technique called a "soft update," where the target network is slowly adjusted towards the main network over time.

### Experience Replay

- **Purpose**: Experience replay is used to break the correlation between consecutive experiences, which can lead to inefficient learning and instability.
- **Issue it resolves**: When an agent learns from sequential experiences, the strong correlations between them can cause problems such as oscillations and instability in learning.
- **How it works**: 
  - Experiences (state, action, reward, next state) are stored in a memory buffer.
  - During training, random mini-batches of experiences are sampled from this buffer to update the network.
  - This random sampling helps to generate uncorrelated experiences, improving stability and efficiency.
  - It also allows the agent to reuse experiences for multiple updates, increasing data efficiency.


# Deepseek {#deepseek}


[LLM](#llm) example 

open source

optimising for preformance vs efficency.

Deepseek leading in efficency

o3 mini

[Chain of thought](#chain-of-thought)
can see it - ui choice

[Distillation](#distillation) - use gpt output and trains on it. 

[security](#security)

#drafting 

[jevon paradox](#jevon-paradox) - as cost decreases usage increases 

edge inference [Edge Machine Learning Models](#edge-machine-learning-models)

[The Genius of DeepSeek’s 57X Efficiency Boost](https://www.youtube.com/watch?v=0VLAoVGf_74)
key value caching 
impacts attention block compute scaling: linear
increased memory usage
Solution: 
multi-query attention vs mutli - head attention
Grouped-query attention
Multi-head latent attention - deepseek uses
uses compresses latent space
linear algebra 
absorbed weights at training




- **Access to Advanced AI Features Without Payment**: You can now access powerful reasoning models like DeepSeek's R1 and ChatGPT's 03 Mini for free. These models are good at complex math, programming, and step-by-step reasoning.

- **Privacy Protection**: If you are concerned about privacy, you have options to protect your data by using platforms like Perplexity, Venice AI, or Cursor to access DeepSeek models, which keeps data in the US. For full privacy, you can run DeepSeek models locally using LM Studio or oLama, but this may limit access to more powerful models due to hardware constraints.

- **Smart Choices About Switching**: Consider if DeepSeek offers clear advantages for your specific needs before changing your current AI tools or workflows. DeepSeek is beneficial for developers focused on cost minimization. If you are an everyday user already paying for ChatGPT and concerned about data storage, switching may not be necessary unless DeepSeek significantly improves your workflow.

# Deleting Rows Or Filling Them With The Mean Is Not Always Best {#deleting-rows-or-filling-them-with-the-mean-is-not-always-best}



# Demand Forecasting {#demand-forecasting}


- **Overview**: Demand response programs encourage consumers to adjust their energy usage during peak periods in response to time-based rates or other incentives. RL can optimize how these programs are implemented.
- **Applications**:
    - **Incentive Management**: [Reinforcement learning|RL](#reinforcement-learningrl) models can dynamically adjust incentives for consumers to reduce usage during peak times based on real-time grid conditions and consumer behavior.
    - **Behavioral Adaptation**: By learning from historical consumer response data, RL systems can predict how different consumers will react to incentives, allowing for more tailored and effective demand response strategies.

How can we model the effects of energy consumption patterns on demand forecasting

- **Dynamic Programming**: Useful in solving multi-stage decision problems, such as optimal scheduling of power plants.


- **Linear Programming**: Used for optimizing resource allocation in energy production and distribution, such as maximizing output while minimizing costs.
-

# Dendrograms {#dendrograms}

Dendrograms show **close** vectors is the data where taken as a vector.

Can tell which <mark>features are the most similar</mark> with [Dendrograms](#dendrograms)

![Pasted image 20240405173403.png](../content/images/Pasted%20image%2020240405173403.png)






# Design Thinking Questions {#design-thinking-questions}

- "What is the user’s need?"
- "What constraints are at play?"
- "How might we…?" — a classic starter for idea generation.

# Determining Threshold Values {#determining-threshold-values}

In [Binary Classification](#binary-classification) problems, a threshold value is used to convert predicted probabilities into discrete class labels. The choice of threshold significantly impacts the model's performance, affecting [Evaluation Metrics](#evaluation-metrics).

Important Considerations:
* [Imbalanced Datasets|Class Imbalance](#imbalanced-datasetsclass-imbalance): If the classes are imbalanced, the choice of threshold can be significantly affected. Techniques like oversampling, undersampling, or using weighted loss functions can help mitigate the impact of class imbalance.
* [Data Quality](#data-quality): The quality of the training data can also influence the choice of threshold. If the data is noisy or contains outliers, the chosen values may not be optimal.
* Choose [Evaluation Metrics](#evaluation-metrics) that are appropriate for the specific problem and the desired trade-off between different types of errors.

Here are common methods for determining the optimal threshold value:
- Receiver Operating Characteristic (ROC) Curve Analysis : [ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic)
- [Precision-Recall Curve](#precision-recall-curve) Analysis
- [Cost-Sensitive Analysis](#cost-sensitive-analysis)

# Devops {#devops}


DevOps refers to practices for collaboration and automation between [Software Development Portal](#software-development-portal) (Dev) and IT operations (Ops) teams, aiming for faster, more reliable software delivery.

**Integration**: It integrates the work of software development and operations teams by fostering a culture of collaboration and shared responsibility.

**Principles**: 
- Emerges from Agile principles.
- Emphasizes collaboration between development and operations teams.

**Approach**: 
- Utilizes continuous integration and continuous delivery ([CI-CD](#ci-cd)) to ensure frequent code changes and quick feedback loops.
- Enables rapid and reliable updates with high levels of automation and efficiency.

**Goals**: 
 - Ensures existing processes are optimized and streamlined.

 Related to:
- [DataOps](#dataops)

# Difference Between Databricks Vs. Snowflake {#difference-between-databricks-vs-snowflake}



# Difference Between Snowflake To Hadoop {#difference-between-snowflake-to-hadoop}


Snowflake and Hadoop are both [Data Management](#data-management) systems, but they serve different purposes and have distinct architectures and functionalities. 

In summary, Snowflake and Hadoop are both powerful tools for managing and analyzing data, but they are optimized for different types of workloads and use cases. Snowflake excels in <mark>cloud-based data warehousing</mark> and real-time analytics, while Hadoop is suited for <mark>large-scale data processing</mark> and storage in a distributed environment.

[Snowflake](#snowflake)

[Hadoop](#hadoop)
### **Key Differences**

1. **Deployment**:
   - **Snowflake**: Cloud-based, requires no hardware or infrastructure management by users.
   - **Hadoop**: Can be deployed on-premises or in the cloud, but typically requires more hands-on management.

2. **Ease of Use**:
   - **Snowflake**: User-friendly with a simple SQL interface, automated maintenance, and optimization.
   - **Hadoop**: Requires more technical expertise to set up, manage, and optimize.

3. **Performance and Scalability**:
   - **Snowflake**: Excels in performance for analytical queries with the ability to scale compute resources independently.
   - **Hadoop**: Scales horizontally by adding more nodes, suitable for large-scale data processing but may have higher query latency.

4. **Cost**:
   - **Snowflake**: Pay-as-you-go model based on compute and storage usage.
   - **Hadoop**: Costs depend on the infrastructure (hardware or cloud resources) and maintenance overhead.

### Example Use Case Scenarios

- **Snowflake**: A retail company wanting to perform real-time analytics and reporting on sales data would benefit from Snowflake’s high performance and ease of use for SQL-based queries and dashboards.

- **Hadoop**: A tech company needing to process and analyze massive amounts of log data for machine learning models might use Hadoop due to its ability to handle large-scale data processing and diverse data types.



# Differentation {#differentation}


# Forward Mode Automatic Differentiation

uses dual numbers

implemented in tensor flow

see also Reverse Mode Automatic Differentiation

Fast,Flexible,Exact

# Digital Transformation {#digital-transformation}



<mark>"Digital transformation starts with data centralisation"</mark>

To digitally transform your department, you'll need to approach the process in a structured and strategic way that addresses both technological and organizational changes.

### [Data Audit](#data-audit)
   - **Understand Department Processes:** Identify current workflows, key processes, and technologies being used.
   - **Identify Pain Points:** Collect feedback from employees on inefficiencies, bottlenecks, and areas for improvement.
   - <mark>**Data Collection and Analysis</mark>:** Review existing data and determine how it's being used (or underutilized) in decision-making processes.
   
### 2. **Define Clear Objectives**
   - **Set Transformation Goals:** Align transformation efforts with the broader goals of the organization. For example, improving efficiency, enhancing customer experience, or better data-driven decision-making.
   - **Quantifiable Metrics:** Define success metrics (e.g., reduced processing time, increased customer satisfaction, or data accuracy).

### 3. **Engage Stakeholders**
   - **Get Leadership Buy-In:** Ensure leadership is aligned with the transformation and committed to driving it forward.
   - **Collaborate with Employees:** Involve employees in planning to understand their needs and gain their support for changes.
   - **Map Stakeholders:** Identify key influencers, decision-makers, and implementers within the department.

### 4. **Evaluate and Select Technologies**
   - **Research Tools and Platforms:** Evaluate technologies such as cloud platforms, automation tools, analytics solutions, and collaboration software.
   - **Pilot New Technologies:** Test small-scale pilots to validate the value of proposed solutions before large-scale rollouts.
   - **Interoperability:** Ensure that new technologies can integrate smoothly with existing systems and data.

### 5. **Build a Transformation Roadmap**
   - <mark>**Prioritize Initiatives</mark>:** Focus on high-impact areas that align with the department's goals.
   - **Create a Timeline:** Develop a step-by-step implementation timeline, including milestones and deadlines.
   - **Allocate Resources:** Assign teams, budgets, and technology resources for each phase of the transformation.

### 6. **Change Management and Training**
   - **Implement Change Management Strategies:** Proactively manage resistance to change by communicating the benefits of digital transformation.
   - **Provide Training and Support:** Train staff on new technologies and provide ongoing support to ensure a smooth transition.
   - **Foster a Digital Culture:** Encourage a mindset of innovation, experimentation, and continuous improvement.

### 7. **Implement New Technologies and Processes**
   - **Roll Out in Phases:** Implement technology solutions gradually, allowing time for adjustments and feedback.
   - **Monitor and Iterate:** Regularly check the implementation process and adjust based on feedback and performance metrics.

### 8. **Measure and Optimize**
   - **Track Key Metrics:** Use the pre-defined success metrics to measure the impact of digital transformation.
   - **Continuously Improve:** Adjust and optimize workflows, tools, and processes based on performance and evolving needs.

### 9. **Ensure Long-Term Sustainability**
   - **Encourage Innovation:** Promote continuous learning and adoption of new technologies to keep the department adaptable to future changes.
   - **Monitor Industry Trends:** Stay updated on new digital trends and opportunities that can benefit the department.
   - **Review and Update Goals:** Periodically revisit your transformation goals and adjust them based on organizational needs and market shifts.

### Example Focus Areas:
   - **Automation:** Streamline repetitive processes using Robotic Process Automation (RPA) or AI.
   - **Data-Driven Decision Making:** Introduce analytics tools and dashboards for real-time insights.
   - **Cloud Adoption:** Shift to cloud-based platforms for better scalability, collaboration, and remote work capabilities.
   - **Collaboration Tools:** Implement communication and project management platforms like Microsoft Teams or Slack to enhance collaboration.

[Digital Transformation](#digital-transformation)

Businesses have data, but need to evaluate, organise, clean , and prepare before using.

[Digital Transformation](#digital-transformation) is a [Change Management|Change Program](#change-managementchange-program).

Where a business can benefit. Areas that can be improved:
- reporting and financial
- data quality / backend
- network management
- Customer services

What we want to move away from/towards
- Silos in our work
- No KPIs
- Multiple spreadsheets doing the same thing
- One-time reporting (get a standarised process in place)
- Systems that are not understood or replicatible
- Poor data quality
- Missed opportunities
- Small, reactive team to issues.

Client focues:
- What does the client want? A fuller understanding of the network. The ability to dive deeper if they wanted to. The ability to understand their bills (AccM). To be able to plan their projects better knowing the information we have?
- What can the water usage tell us about a site/all sites, that could be beneficial to the client? Stress levels on the system - predictive maintenance? Does high usage result in higher leakage.
- What are the reasons for leaks? - non error - what happened on the system that caused it?

---

Digital transformation aims to provide innovation to business processes.  
- save time
- do more, 
- save money.

Digital transformation aims to:
- Automate repetitive tasks.  
- Create systematic improvements to business sops, through strategies, methodologies, technologies.  
- How can we prepare digital assets for further automation?  

To consider when conducting:
 - Conduct a Data audit company wide, what is the format of data and where is it? is the data accessible?
 - Reporting mechanisms what can you do with the data 
 - How to handle data debt/minimise it.
 - What does leadership want as a data/product pathway.
 - How can we continuously improve the quality of our data? 
 - Need to track by design, i.e. what do we need to know to benefit in the future.   
  




  


  



# Digital Twin {#digital-twin}




>[!Summary]
> A **digital twin** is a virtual representation of a physical object, system, or process that mirrors its real-world counterpart in real-time. This digital model is used to simulate, monitor, analyze, and optimize the physical entity by continuously updating based on data collected from sensors, devices, or other inputs. The concept is widely applied in industries such as manufacturing, healthcare, [Energy](#energy), smart cities, and more to improve decision-making, predictive maintenance, and efficiency. A digital twin is a powerful tool for enhancing real-time decision-making, optimizing processes, and predicting future performance by bridging the physical and digital worlds. Its applications continue to expand across various industries, helping organizations to reduce costs, improve efficiency, and innovate faster.

>[!Example]
>Consider a **digital twin of a wind turbine**. Sensors installed on the turbine gather data on operational conditions such as wind speed, blade position, temperature, and vibration. This data is continuously transmitted to the digital twin, which mirrors the turbine’s state in real-time. The digital twin runs simulations to predict when parts of the turbine might fail due to wear and tear. Maintenance teams can use this information to schedule repairs before a breakdown occurs, minimizing downtime and improving the turbine's efficiency.

### Key Components of a Digital Twin:

1. **Physical Object or Process**:
   - The real-world entity (such as a machine, a building, a production line, or even a human body) that the digital twin replicates.

2. **Digital Model**:
   - A virtual replica of the physical entity, designed using data models, physics-based simulations, and other analytical tools. The digital model reflects the structure, behavior, and function of the physical object or process.

3. **Data Integration**:
   - The digital twin <mark>relies on real-time data from sensors,</mark> IoT devices, or historical databases connected to the physical counterpart. This data flow enables the twin to reflect current operating conditions and states.

4. **Analytics and Simulation**:
   - Advanced analytics (e.g., machine learning, artificial intelligence) and simulations are applied to the digital twin to gain insights into the performance, predict future behavior, and test scenarios that would be difficult or expensive to replicate in the real world.

5. **Feedback Loop**:
   - A digital twin allows for continuous interaction between the physical and digital worlds. Insights or predictions from the digital twin can inform changes to the physical system, and any updates in the physical system feed back into the digital twin, maintaining accuracy and alignment.

### Types of Digital Twins:

1. **Component/Asset Twin**:
   - Represents individual components or parts of a larger system (e.g., the digital twin of a jet engine or an electric motor).

2. **System or Unit Twin**:
   - Models entire systems or units, such as a production line in a factory or the electrical system of a building.

3. **Process Twin**:
   - Focuses on simulating and optimizing processes, such as a manufacturing workflow or supply chain operations.

4. **Environment Twin**:
   - Used to simulate larger, more complex systems like cities, ecosystems, or large-scale infrastructure (e.g., smart city initiatives or environmental monitoring).

### Applications of Digital Twins:

1. **Manufacturing**:
   - In smart factories, digital twins are used to simulate production processes, predict machine failures, optimize maintenance schedules, and improve product design by running real-time simulations of manufacturing conditions.

2. **Healthcare**:
   - Digital twins of patients are being developed to model individual health profiles, allowing for personalized treatment plans and predictive diagnostics. A digital twin of a human organ, for example, could simulate medical treatments before they are applied to the patient.

3. **[Energy](#energy)**:
   - In energy systems, digital twins help optimize the operation of power plants, monitor grid performance, and simulate the impacts of renewable energy integration, improving reliability and efficiency.

4. **Smart Cities**:
   - Urban planners use digital twins to model traffic flow, infrastructure usage, or environmental conditions. This allows them to simulate different scenarios and optimize city operations, reduce congestion, and improve public services.

5. **Aerospace and Automotive**:
   - Digital twins are used extensively in designing, testing, and maintaining complex systems like aircraft, satellites, and autonomous vehicles. Engineers can simulate operational conditions to identify potential problems before they occur in the physical system.

6. **Building Management**:
   - Digital twins of buildings or infrastructure monitor and control systems like HVAC, lighting, and security, improving energy efficiency and safety. They are also used for simulating how a building will perform under different conditions (e.g., weather events or occupancy changes).

### Benefits of Digital Twins:

1. **Real-time Monitoring**:
   - Provides live feedback from the physical entity, which enables organizations to make faster, more informed decisions.

2. **Predictive Maintenance**:
   - Predicts when equipment or systems are likely to fail based on real-time data and simulations, reducing downtime and maintenance costs.

3. **Optimization**:
   - Enables the continuous improvement of processes by testing scenarios in a virtual environment without disrupting real-world operations.

4. **Improved Design and Innovation**:
   - Digital twins allow engineers and designers to experiment with different configurations, materials, or processes virtually, leading to faster, cheaper, and more innovative solutions.

5. **Reduced Risk**:
   - By simulating potential failures or dangerous scenarios in the digital world, organizations can assess risk and plan mitigation strategies without putting the physical system at risk.

### Challenges:

1. **[Data Management](#data-management)**:
   - Digital twins require a large amount of real-time data to maintain accuracy. Collecting, managing, and processing this data efficiently can be complex and costly.

2. **Integration**:
   - Integrating the digital twin with physical systems, particularly in legacy environments, can be challenging due to compatibility issues and the need for IoT infrastructure.

3. **Security**:
   - Because digital twins rely on real-time data transmission, they are vulnerable to cyberattacks, which can lead to compromised systems or intellectual property theft.

4. **Scalability**:
   - Scaling digital twin models to encompass entire cities or large systems involves high computational and infrastructural requirements.






# Dimension Table {#dimension-table}

A dimension table is a key component of a [star schema](#star-schema) or snowflake schema in a data warehouse. It provides descriptive attributes (or dimensions) related to the [Facts](#facts) stored in a fact table.

They provide the context and descriptive information necessary for analyzing the quantitative data stored in fact tables (e.g., product names, customer demographics, time periods).

1. **Descriptive Attributes**: Dimension tables contain qualitative data that describe the entities involved in the business process. For example, a product dimension table might include attributes such as product name, category, brand, and manufacturer.

2. **Primary Key**: Each dimension table has a primary key that uniquely identifies each record in the table. This primary key is used as a foreign key in the [Fact Table](#fact-table) to establish relationships between the two.

3. **Hierarchies**: Dimension tables often include hierarchies that allow for data to be analyzed at different levels of granularity. For example, a time dimension might include attributes for year, quarter, month, and day, allowing users to drill down or roll up in their analysis.

4. **Smaller Size**: Compared to fact tables, dimension tables are typically smaller in size, as they contain descriptive data rather than large volumes of transactional data.

5. **Static Data**: Dimension tables usually contain relatively static data that does not change frequently, such as product details or customer information. However, they can be updated as needed to reflect changes in the business.

6. **Support for Filtering and Grouping**: Dimension tables enable users to filter and group data in reports and analyses. For example, users can analyze sales data by different dimensions such as time, geography, or product category.

Examples
  - **TimeDimension**: Contains information about the time period.
    - Columns: `DateKey`, `Year`, `Quarter`, `Month`, `Day`
  - **ProductDimension**: Contains product details.
    - Columns: `ProductKey`, `ProductName`, `ProductCategory`
  - **RegionDimension**: Contains regional information.
    - Columns: `RegionKey`, `RegionName`, `Country`





[Dimension Table](#dimension-table)
   **Tags**: #data_modeling, #data_warehouse

# Dimensional Modelling {#dimensional-modelling}


Dimensional modeling is a design technique used in [Data Warehouse](#data-warehouse)used to structure data for efficient <mark>retrieval</mark> and analysis. It is particularly well-suited for organizing data in a way that supports complex [queries](#queries) and reporting, making it easier for business users to understand and interact with the data. 

Dimensional modeling is a foundational technique in building data warehouses and is often associated with methodologies like the <mark>Kimball</mark> approach, which emphasizes the use of [Star Schema](#star-schema) and the importance of understanding business processes and user requirements.

Key Concepts in Dimensional Modeling

 - [Fact Table](#fact-table) & [Facts](#facts)
- [Dimension Table](#dimension-table)
- [Grain](#grain)

Benefits of Dimensional Modeling: [Performance Dimensions](#performance-dimensions)





[Dimensional Modelling](#dimensional-modelling)
   **Tags**: #data_modeling, #data_warehouse

# Dimensionality Reduction {#dimensionality-reduction}


Dimensionality reduction is a step in the [Preprocessing](#preprocessing) phase of machine learning that helps simplify models, enhance interpretability, and improve computational efficiency.

Its a technique used to reduce the number of input variables (features) in a dataset while retaining as much information as possible. This process is essential for several reasons:

1. **Improves Model Performance**: Reducing the number of features can help improve the performance of machine learning models by minimizing overfitting and reducing noise.

2. **Enhances Visualization**: It allows for easier [Data Visualisation](#data-visualisation) of high-dimensional data by projecting it into lower dimensions (e.g., 2D or 3D).

3. **Reduces Computational Cost**: Fewer features mean less computational power and time required for training models.

### Common Techniques
- **Principal Component Analysis ([Principal Component Analysis](#principal-component-analysis))**: A statistical method that transforms the data into a new coordinate system, where the greatest variance by any projection lies on the first coordinate <mark>(principal component/orthogonal components )</mark>, the second greatest variance on the second coordinate, and so on.

- **t-Distributed Stochastic Neighbor Embedding ([t-SNE](#t-sne))**: A technique particularly well-suited for visualizing high-dimensional data by reducing it to two or three dimensions while preserving the local structure of the data. t-SNE is a non-linear technique used for visualization and dimensionality reduction by preserving pairwise similarities between data points, making it suitable for exploring high-dimensional data.

- [Linear Discriminant Analysis](#linear-discriminant-analysis) method used for both classification and dimensionality reduction, which finds a linear combination of features that best separates two or more classes.

### [Explain the curse of dimensionality](#explain-the-curse-of-dimensionality)


# Dimensions {#dimensions}


Dimensions are the categorical buckets that can be used to segment, filter, or group—such as sales amount region, city, product, color, and distribution channel. 

Traditionally known from [OLAP (online analytical processing)|OLAP](#olap-online-analytical-processingolap)cubes with Bus Matrixes, and [Dimensional Modeling](Dimensional%20Modelling.md). 

They provide context to the [Facts](#facts).

# Directed Acyclic Graph (Dag) {#directed-acyclic-graph-dag}


DAG stands for **Directed Acyclic Graph**. 

A DAG is a graph where information must travel along with a finite set of nodes connected by vertices. There is no particular start or node and also no way for data to travel through the graph in a loop that circles back to the starting point.

It's a popular way of building data pipelines in tools like [Apache Airflow](#apache-airflow), [dagster](#dagster), [Prefect](#prefect). It clearly defines the data lineage. As well, it's made for a functional approach where you have the [idempotency](term/idempotency.md) to restart pipelines without side-effects.

![](dag.png)

# Directory Structure {#directory-structure}


[To make a file tree](https://faun.pub/create-a-repository-tree-and-print-it-to-a-file-f376f103f169)

├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── src                <- Source code for use in this project.
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py

# Distillation {#distillation}

training smaller models with larger.

[Small Language Models](#small-language-models)

![Pasted image 20250130074219.png](../content/images/Pasted%20image%2020250130074219.png)



# Distributed Computing {#distributed-computing}


**Distributed Computing** is essential for managing **massive data volumes** by distributing tasks across multiple servers or machines. This enables scalability and efficient data processing.

**[Hadoop](#hadoop)** is a framework that handles both data storage and processing across **clusters of servers**:
  - It ensures **scalability**: can easily grow as more data is added.
  - Provides **redundancy**: data is replicated across servers to prevent loss in case of failures.

Tools like [Apache Spark](#apache-spark) are built to process data in these **distributed environments**, allowing for fast, parallel processing across the cluster.

Distributed computing is central to modern data handling, driven by frameworks like Spark and Hadoop, supported by cloud infrastructure, and expanding into real-time and [edge computing](#edge-computing).
### Distributed Computing: Current State

Distributed computing enables the processing of massive datasets and computational tasks by distributing them across multiple machines. This approach increases [scalability](#scalability), parallelism, and fault tolerance, making it essential for modern data processing.
#### Key Frameworks

- **Hadoop**: An early pioneer, Hadoop introduced distributed storage via **HDFS** and data processing with **MapReduce**, allowing tasks to be split across clusters of servers.
- **Apache Spark**: A faster alternative to MapReduce, Spark uses in-memory computing for real-time, iterative tasks, improving speed and efficiency. It has become the leading tool for distributed data processing.

#### Distributed Storage

- **HDFS** and cloud storage systems like **[Amazon S3](#amazon-s3)** break data into smaller parts and distribute them across multiple servers. This setup provides high throughput, redundancy, and fault tolerance.
- **Distributed databases** such as **Cassandra** and **Bigtable** offer scalable storage for structured data, ensuring availability across nodes.

#### Real-Time Processing ([Data Streaming](#data-streaming))

- Frameworks like **Apache Flink** and **Kafka Streams** are critical for real-time data processing, enabling continuous data handling as it is generated. They are commonly used in applications requiring instant processing, such as fraud detection or live analytics.

#### Cloud-Native Computing

- **Cloud platforms** (e.g., AWS, Google Cloud, Azure) have made distributed computing accessible through services like **Amazon EMR** and **Google Dataproc**. These services simplify the deployment and management of distributed applications.
- [kubernetes](#kubernetes) has become the standard for orchestrating distributed applications, managing containers and ensuring high availability across clusters.

#### Trends and Challenges

- **Edge computing** is gaining momentum, enabling data to be processed closer to the source (e.g., IoT devices), reducing latency and bandwidth usage.
- Challenges include **fault tolerance**, **network [latency](#latency)**, and **data consistency**. Innovations in consensus algorithms and fault-tolerant storage systems are working to mitigate these issues.


# Distribution_Analysis.Py {#distribution_analysispy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Distribution_Analysis.py

The goodness-of-fit results represent the **p-values** from the **Kolmogorov-Smirnov (KS) test**, which assesses how well the data fits each distribution. Here's how to interpret these values:

1. **Higher p-value** → The distribution is a better fit.
2. **Lower p-value** → The distribution is a poor fit (likely not the correct model for the data).
3. **Threshold**: A common significance level is **0.05**.
    - If **p > 0.05**, we do **not** reject the hypothesis that the data follows this distribution.
    - If **p < 0.05**, we reject the hypothesis, meaning the data **likely does not follow** that distribution.

Example using penguins.csv column "bill_depth_mm"

Goodness-of-fit results for bill_depth_mm
Gaussian: 0.026308596409291618
T: 0.025906678848475195
Chi-squared: 1.4504381882536289e-15
Exponential: 2.8020502445188308e-14
Logistic: 0.05019989765502264

- **Gaussian (0.0263)** → **Poor fit** (p < 0.05). The data likely does not follow a normal distribution.
- **T (0.0259)** → **Poor fit** (p < 0.05). The data does not fit a t-distribution well.
- **Chi-squared (1.45e-15)** → **Very poor fit** (extremely low p-value). The data is **highly unlikely** to follow a chi-squared distribution.
- **Exponential (2.80e-14)** → **Very poor fit** (extremely low p-value). The data is **not** exponentially distributed.
- **Logistic (0.0502)** → **Acceptable fit** (p ≈ 0.05). The data could **potentially** follow a logistic distribution.

The **Logistic distribution** has the **highest p-value (0.0502)**, making it the **best candidate** among the tested distributions. However, since it's **borderline (≈0.05)**, you may want to visualize the distribution and compare the fits.

# Distributions {#distributions}


In [ML_Tools](#ml_tools) see:
- [Distribution_Analysis.py](#distribution_analysispy)
#### Discrete Distributions
These distributions have probabilities concentrated on specific values.

- [Uniform](#uniform) Distribution: All outcomes are equally likely. Example: Drawing a card from a shuffled deck. A boxplot can be meaningful if there’s variation in the distribution. Since the values are discrete, the boxplot will show the range and quartiles effectively.
- [Bernoulli](#bernoulli) Distribution: Represents two possible outcomes. Example: Coin flip (heads or tails), true/false scenarios. A bar chart or frequency plot would be better for visualizing the proportions. or rolling a dice.
- [Binomial](#binomial) Distribution: Represents the number of successes in a sequence of Bernoulli trials. Example: Number of heads in 10 coin flips,
- [Poisson](#poisson) Distribution: Models the frequency of events in a fixed interval. Example: Number of website visits per hour. A boxplot is suitable for this distribution, showing central tendencies, spread, and potential outliers.

#### Continuous Distributions
These distributions have probabilities spread over a continuous range.

- [Gaussian Distribution](#gaussian-distribution): Characterized by a bell-shaped curve, symmetric with thin tails. Example: Heights, exam scores.
- T Distribution: Similar to the normal distribution but with fatter tails, useful with limited data.
- Chi-squared Distribution: Asymmetric and non-negative, commonly used in [hypothesis testing](#hypothesis-testing).
- Exponential Distribution: Models the time between events. Example: Time between website traffic hits, radioactive decay.
- Logistic Distribution: S-shaped curve, often used in forecasting and modeling growth.
  ![Pasted image 20250308191945.png](../content/images/Pasted%20image%2020250308191945.png)
  
#### Q-Q plots (Quantile-Quantile Plots)

A Q-Q (quantile-quantile) plot is a graphical tool used to compare the distribution of a dataset against a theoretical distribution (e.g., normal, logistic, exponential). It helps assess how well a given distribution fits the data.

How Q-Q Plots Work:

1. Sort your dataset → Compute the sample quantiles (percentiles).
2. Compute the theoretical quantiles → Take the same number of points from the theoretical distribution (e.g., normal, logistic).
3. Plot sample quantiles vs. theoretical quantiles:
    - If the points lie on a straight diagonal line, the data follows the theoretical distribution.
    - If the points deviate significantly, the data does not fit that distribution.

Interpreting a Q-Q Plot:

- Straight diagonal line → Data follows the chosen distribution.
- Curved S-shape → Data has skewness.
    - Upward curve (right tail high) → Right-skewed.
    - Downward curve (left tail high) → Left-skewed.
- Heavy tails (outliers) → Points at the ends deviate from the line.
- Light tails (thin-tailed distribution) → Points at the ends fall below the line.
### Practical Applications

Feature Distribution: Understanding the distribution of numerical/ categortical feature values across samples can provide insights into data characteristics.

  - Observation: Analyze the spread and central tendency of data.
  - Decision: Determine appropriate statistical methods or transformations.

### Related Notes

In [ML_Tools](#ml_tools) see: [Feature_Distribution.py](#feature_distributionpy)

- [Violin plot](#violin-plot)
- [Boxplot](#boxplot)


# Docker Image {#docker-image}

A Docker image is a lightweight, standalone, and executable package that includes everything needed to run a piece of software, including the code, runtime, libraries, environment variables, and configuration files. Docker images are used to create Docker containers, which are instances of these images running in an isolated environment.

Docker images are used for:

1. **Consistency**: They ensure that software runs the same way regardless of where it is deployed, whether on a developer's laptop, a test server, or in production.

2. **Portability**: Docker images can be easily shared and moved across different environments, making it easier to deploy applications.

3. **Version Control**: Images can be versioned, allowing developers to track changes and roll back to previous versions if necessary.

4. **Isolation**: Each Docker container runs in its own isolated environment, which helps in avoiding conflicts between different applications or services running on the same host.

5. **Scalability**: Docker images can be used to quickly scale applications by running multiple containers from the same image.
### Example: Running a Simple Web Server (dont do unless required)

1. **Install Docker**: First, ensure Docker is installed on your machine. You can download it from the [Docker website](https://www.docker.com/products/docker-desktop).

2. **Pull a Docker Image**: Use a pre-built Docker image from Docker Hub. For this example, we'll use the official Nginx image, which is a popular web server.

   ```bash
   docker pull nginx
   ```

3. **Run a Docker Container**: Start a container using the Nginx image. This will run the web server.

   ```bash
   docker run --name my-nginx -d -p 8080:80 nginx
   ```

   - `--name my-nginx`: Names the container "my-nginx".
   - `-d`: Runs the container in detached mode (in the background).
   - `-p 8080:80`: Maps port 80 in the container to port 8080 on your host machine.

4. **Access the Web Server**: Open a web browser and go to `http://localhost:8080`. You should see the default Nginx welcome page, indicating that the web server is running inside the Docker container.

5. **List Running Containers**: Check which containers are running.

   ```bash
   docker ps
   ```

6. **Stop the Container**: When you're done, you can stop the container.

   ```bash
   docker stop my-nginx
   ```

7. **Remove the Container**: If you no longer need the container, you can remove it.

   ```bash
   docker rm my-nginx
   ```

### Key Features Demonstrated:

- **Portability**: The Nginx server runs the same way on any machine with Docker installed.
- **Isolation**: The web server runs in its own environment, separate from other applications.
- **Ease of Use**: Starting and stopping services is straightforward with simple commands.
- **Resource Efficiency**: Docker containers are lightweight compared to virtual machines.

This example gives you a taste of how Docker can be used to quickly deploy and manage applications. As you become more familiar with Docker, you can explore building your own images, managing multi-container applications with Docker Compose, and more.

# Docker {#docker}

Utilizes Docker images [Docker Image](#docker-image) to set up containers for consistent development and testing environments.

Containers can include necessary dependencies like Python and pip.

Tutorial:
- [The Only Docker Tutorial You Need To Get Started](https://www.youtube.com/watch?v=DQdB7wFEygo)

Docker Volumes - storing data

Docker Compose

Multi use containers

`docker init` command
- generated Dockerfile - image
- compose.yaml
- .dockerignore





## Tools  
- [pdoc](#pdoc) – Auto-generate Python API documentation  
- [Mermaid](#mermaid) – Create diagrams and flowcharts from text in a Markdown-like syntax  
## Templates  

### Project & Technical Meetings  

- [Technical Design Doc Template](#technical-design-doc-template) – Document system architecture, components, data flow  
- [Pull Request Template](#pull-request-template) – Checklist and summary for PRs (code, tests, reviewers, notes)  
- [Experiment Plan Template](#experiment-plan-template) – Hypothesis, variables, metrics, and analysis plan  
- [Retrospective Template](#retrospective-template) – Guide for reviewing past sprint/project cycles  
### Data & Analytics Meetings  
- [Data Request Template](#data-request-template) – Intake form for new analysis or data extract needs  
- [One Pager Template](#one-pager-template) – Summarize a project, idea, or proposal on a single page  
- [Meeting Notes Template](#meeting-notes-template) – Standard format for taking concise and structured notes  
### Cross-functional / Stakeholder Meetings  
- [One Pager Template](#one-pager-template) – Summarize a project, idea, or proposal on a single page  
- [Meeting Notes Template](#meeting-notes-template) – Standard format for taking concise and structured notes  
- [Data Request Template](#data-request-template) – Intake form for new analysis or data extract needs  
### Reporting & Strategic Meetings  
- [Postmortem Template](#postmortem-template) – Structure for documenting incidents and learnings  
- [Retrospective Template](#retrospective-template) – Guide for reviewing past sprint/project cycles  
- [One Pager Template](#one-pager-template) – Summarize a project, idea, or proposal on a single page  
### Collaborative & Feedback Sessions  
- [1-on-1 Template](#1-on-1-template) – Agenda for 1-on-1 check-ins with manager or reports  
- [Feedback Template](#feedback-template) – Structure for giving/receiving peer or performance feedback  
## Meeting Types

### Project & Technical Meetings  
- **Sprint Planning / Stand-ups (Agile):**  
  Define priorities, plan tasks, and report blockers on a daily/weekly basis  

- **Design & Architecture Reviews:**  
  Evaluate technical designs—e.g., pipeline architecture, schemas, model design  

- **Code Reviews / Pair Programming:**  
  Collaborative code sessions for learning and validation  

- **Pipeline/Model Monitoring & Debugging:**  
  Diagnose failures, resolve data/model performance issues in production  

### Data & Analytics Meetings  
- **Exploratory Data Analysis (EDA) Discussions:**  
  Share insights, anomalies, or feature engineering results  

- **Model Review & Evaluation:**  
  Present metrics, discuss trade-offs (e.g., ROC-AUC, fairness, overfitting)  

- **Data Quality & Validation:**  
  Ensure schema consistency, missing data checks, rule-based validations  

- **Experiment Review:**  
  Discuss A/B test results, statistical significance, and business impact  

### Cross-functional / Stakeholder Meetings  

- **Product or Domain Team Syncs:**  
  Discuss project goals, KPIs, data availability  

- **Ad Hoc Analysis Requests:**  
  Clarify requirements for exploratory or one-off reporting  

- **User Feedback / Data Product Reviews:**  
  Gather user input on dashboards, ML outputs, or data access tools  

- **Requirements Grooming:**  
  Translate business requirements into data/technical specifications  

### Reporting & Strategic Meetings  

- **Quarterly Business Reviews / OKR Alignment:**  
  Evaluate progress against metrics and strategic initiatives  

- **Dashboard Walkthroughs / Reporting Demos:**  
  Demonstrate key metrics, performance trends, and data tooling  

- **Metrics Definition Meetings:**  
  Align teams on KPI definitions, calculation logic, and data sources  

### Collaborative Initiatives  

- **Knowledge Sharing / Lunch & Learns:**  
  Internal sessions to demo tools, share best practices, or teach concepts  

- **Workshops (e.g., dbt, Airflow, MLOps):**  
  Hands-on learning sessions for technical upskilling  

- **Data Governance / Compliance:**  
  Ensure responsible data use, privacy adherence, and lineage tracking  

- **Hackathons / Innovation Days:**  
  Time-boxed events for experimentation and cross-team collaboration  


# Dropout {#dropout}


**Dropout** is a [Regularisation](#regularisation) technique used in [Neural network](#neural-network) training to prevent [overfitting](#overfitting). It works by randomly dropping units (neurons) during training, which helps the network to not rely too heavily on any single neuron.

Purpose
- The main goal of dropout is to improve the generalization of the model by reducing over-reliance on specific neurons. This encourages the network to learn more robust features that are useful in different contexts.

How It Works
- During each training iteration, a subset of neurons is randomly selected and ignored (dropped out). This means their contribution to the activation of downstream neurons is temporarily removed on the forward pass, and any weight updates are not applied to the neuron on the backward pass.

Implementation
```python
from tensorflow.keras.layers import Dropout
# Add a dropout layer with a dropout rate of 0.5
# The dropout rate (e.g., 0.5) specifies the fraction of neurons to drop during training. A rate of 0.5 means that half of the neurons will be dropped at each iteration.
Dropout(0.5)
```

# Duckdb In Python {#duckdb-in-python}

To use **DuckDB** in Python, you can follow these steps to install the DuckDB library and perform basic operations such as creating a database, running queries, and manipulating data. Here's a simple guide:

### Step 1: Install DuckDB

You can install DuckDB using pip. Open your terminal or command prompt and run:

```bash
pip install duckdb
```
### Example: Full Code

Here’s a complete example that incorporates all the steps:

```python
import duckdb

# Step 1: Connect to an in-memory database
conn = duckdb.connect(database=':memory:')

# Step 2: Create a table
conn.execute("""
CREATE TABLE users (
    id INTEGER,
    name VARCHAR,
    age INTEGER
)
""")

# Step 3: Insert data
conn.execute("""
INSERT INTO users VALUES
(1, 'Alice', 30),
(2, 'Bob', 25),
(3, 'Charlie', 35)
""")

# Step 4: Query data
result = conn.execute("SELECT * FROM users").fetchall()

# Print the results
for row in result:
    print(row)

# Step 5: Close the connection
conn.close()
```

### Additional Features

DuckDB also supports advanced features such as:

- **Reading from CSV and Parquet files**: You can load data directly from these formats using commands like `READ_CSV` or `READ_PARQUET`.
- **Integration with Pandas**: You can easily convert DuckDB query results to Pandas DataFrames for further analysis.

### Example of Reading from a CSV

```python
# Load data from a CSV file into a DuckDB table
conn.execute("CREATE TABLE my_data AS SELECT * FROM read_csv_auto('path/to/your/file.csv')")
```

# Duckdb Vs Sqlite {#duckdb-vs-sqlite}

Choosing between **[DuckDB](#duckdb)** and **[SQLite](#sqlite)** for data processing in [Python](#python) depends on your specific use case and requirements.

While **SQLite** is an excellent choice for lightweight applications, local data storage, and simple transactional workloads.

**DuckDB** shines in scenarios involving complex analytical queries, large datasets, and data science workflows.

If your primary focus is on data analysis and you need high performance for analytical tasks, DuckDB may be the better option. However, if you need a simple, lightweight database for small-scale applications, SQLite could be more appropriate.

### 1. **Performance for Analytical Queries**
- **DuckDB** is optimized for analytical workloads and can handle complex queries involving large datasets more efficiently than SQLite. It uses a [columnar storage](#columnar-storage) format, which is particularly beneficial for aggregation and analytical operations.
- **SQLite**, while fast for transactional workloads, may not perform as well with large-scale analytical queries.

### 2. **In-Memory Processing**
- DuckDB can operate entirely in-memory, allowing for faster data processing and query execution. This is especially useful for data science tasks where speed is critical.
- SQLite can also work in-memory, but its performance may not match that of DuckDB for analytical tasks.

### 3. **Support for Complex Data Types**
- DuckDB supports more complex data types and operations, such as nested data structures and advanced analytical functions, which can be advantageous for data analysis.
- SQLite has a more limited set of data types and may not support some advanced analytical functions.

### 4. **Integration with Data Science Tools**
- DuckDB is designed to integrate seamlessly with data science libraries like Pandas, making it easier to perform data analysis and manipulation directly within your Python workflows.
- While SQLite can also be used with Pandas, DuckDB's integration is often more straightforward for analytical tasks.

### 5. **Cloud and Big Data Compatibility**
- DuckDB is designed to work well with cloud-based data warehouses and big data environments, making it suitable for modern data workflows that involve large datasets stored in cloud storage.
- SQLite is more suited for lightweight applications and local data storage.

### 6. **Columnar Storage Format**
- DuckDB's columnar storage format allows for more efficient data compression and faster query performance on analytical workloads, as it reads only the necessary columns for a query.
- SQLite uses a row-based storage format, which can be less efficient for certain types of analytical queries.

### 7. **Ease of Use for Data Transformation**
- DuckDB simplifies data transformation processes, allowing you to perform transformations directly within the database after loading the data. This can streamline workflows and reduce the need for additional data processing steps.
- SQLite requires more manual handling for data transformations, especially when dealing with large datasets.


# Duckdb {#duckdb}

**DuckDB** is an open-source analytical database management system designed for efficient data processing and analysis. It is optimized for running complex queries on large datasets and is particularly well-suited for data science and analytics tasks. Here are some key features and characteristics of DuckDB:

Resources:
- https://duckdb.org/
- [DuckDB in python](#duckdb-in-python)
- [DuckDB vs SQLite](#duckdb-vs-sqlite) 

### Key Features

1. **In-Memory Processing**: DuckDB operates primarily in-memory, which allows for fast query execution and data manipulation.

2. [Columnar Storage](#columnar-storage) It uses a columnar storage format, which is efficient for analytical queries that often involve aggregations and scans over large datasets.

3. **SQL Support**: DuckDB supports SQL as its query language, making it accessible to users familiar with SQL syntax.

4. **Integration with Data Science Tools**: DuckDB can be easily integrated with popular data science tools and programming languages, such as Python and R, allowing for seamless data analysis workflows.

5. **Lightweight and Easy to Use**: It can be embedded in applications and does not require a separate server, making it lightweight and easy to deploy.

6. **Compatibility**: DuckDB can read from various data formats, including CSV, [Parquet](#parquet)

### Use Cases
- [Data Analysis](#data-analysis)
- [Data Transformation](#data-transformation)
- [Querying](#querying)


### Key Takeaways:

- The dummy variable trap occurs due to [multicollinearity](#multicollinearity), where <mark>one dummy variable can be perfectly predicted from others.</mark>
- Dropping one dummy variable avoids this issue and ensures that the model has a reference category against which the other categories are compared.
- This approach leads to a well-conditioned model and allows for more interpretable regression coefficients.
### Dummy Variable Trap

The dummy variable trap refers to a scenario in which there is multicollinearity in your dataset when you create dummy variables for categorical features. Specifically, it occurs when one of the dummy variables in a set of dummy variables can be perfectly predicted by a linear combination of the others.

This situation arises when you create dummy variables for a categorical feature with $n$ categories, leading to $n$ binary columns. However, if you include all $n$ dummy variables in your regression model, <mark>the model will face redundancy because knowing the values of $n-1$ dummy variables will already give you the value of the last one (since all the categories must add up to 1 for each observation)</mark>. This results in perfect multicollinearity.

### Why Do We Need to Drop One of the Dummy Variables?

In a regression model, multicollinearity can cause problems because it makes the estimation of coefficients unstable, leading to unreliable statistical inferences. Specifically, the model can't determine which of the correlated variables is truly responsible for explaining the variation in the target variable.
### Example:

Suppose you have a categorical feature `town` with three categories: `West Windsor`, `Robbinsville`, and `Princeton`. When you apply [one-hot encoding](#one-hot-encoding), you create three dummy variables:

|town|West Windsor|Robbinsville|Princeton|
|---|---|---|---|
|West Windsor|1|0|0|
|Robbinsville|0|1|0|
|Princeton|0|0|1|

Now, if you include all three dummy variables in a linear regression model, the columns `West Windsor`, `Robbinsville`, and `Princeton` will be perfectly correlated. For example, if the values of `West Windsor` and `Robbinsville` are both 0, then `Princeton` must be 1, and vice versa.

This creates multicollinearity because you can predict one dummy variable perfectly by knowing the others. Hence, you need to drop one of the dummy variables—usually, you drop one category, which becomes the reference group.

If you drop the `West Windsor` dummy column, your table would look like this:

|town|Robbinsville|Princeton|
|---|---|---|
|West Windsor|0|0|
|Robbinsville|1|0|
|Princeton|0|1|

Now, your model will use the `West Windsor` category as the baseline. The coefficients of `Robbinsville` and `Princeton` in the regression model will indicate how much higher or lower their prices are compared to `West Windsor`.

# Dagster {#dagster}


[Dagster](https://dagster.io/) is a [data orchestrator] focusing on data-aware scheduling that supports the whole development [Data Lifecycle Management](#data-lifecycle-management)  lifecycle, with integrated lineage and observability, a [declarative](#declarative) programming model, and best-in-class testability.

Key features are: 
- Manage your data assets with code
- A single pane of glass for your data platform 

# Data Asset {#data-asset}



# Data Lineage {#data-lineage}


Data lineage uncovers the [Data Lifecycle Management](#data-lifecycle-management) life cycle of data. It aims to show the complete data flow from start to finish. 

Data lineage is the process of understanding, recording, and visualizing data as it flows from data sources to consumption. 

This includes all [Data Transformation](Data%20Transformation.md) (what changed and why).

# Data Literacy {#data-literacy}


Data literacy is the ability to read, work with, analyze, and argue with data in order to extract meaningful information and make informed decisions. This skill set is crucial for employees across various levels of an organization, especially as data-driven decision-making becomes increasingly important.

Organizations should invest in data literacy training programs to empower their employees with the necessary skills to effectively engage with data. A data-literate employee can read charts, draw correct conclusions, recognize when data is being used inappropriately or misleadingly, and gain a deeper understanding of the business domain. This enables them to communicate more effectively using a common language of data, spot unexpected operational issues, identify root causes, and prevent poor decision-making due to data misinterpretation.

Examples of data literacy in action include:

* Implementing the Adoptive Framework to create a Data Literacy Program.
* Employees working with spreadsheets to understand the rationale behind data-driven decisions and advocating for alternative courses of action.
* Work teams identifying areas where data needs clarification for a project.

By nurturing a data-literate workforce, businesses can improve their ability to make informed decisions, drive innovation, and achieve better outcomes.


# Dbt {#dbt}


Data build tool is an open-source framework designed for [Data Transformation](#data-transformation) within a modern data stack. 

It enables analysts and engineers to transform, model, and manage data using [SQL](#sql) while <mark>adhering to software engineering best practices</mark> like version control, testing, and [Documentation & Meetings](#documentation--meetings). 

### Key Concepts of dbt:
1. **SQL-based Transformation**: dbt allows users to write SQL queries to define transformations and models, making it accessible for analysts who are already familiar with SQL. It doesn't handle extraction or loading of data, but focuses purely on transforming data that is already in a data warehouse.

2. **Modular and Reusable Models**: dbt encourages the creation of modular, <mark>reusable SQL "models."</mark> Each model represents a transformation, and these models can be built on top of each other. A model is essentially a SQL query stored as a `.sql` file that dbt uses to transform raw data into a refined dataset. Models are run in sequence, with dbt handling dependencies between models.

3. **Version Control and Collaboration**: dbt integrates with Git for version control, making it easy for teams to collaborate, track changes, and roll back to previous versions if needed. This promotes transparency and accountability within the data team.

4. **Testing**: dbt allows users to write and run tests to ensure data integrity and consistency. You can define tests for specific models or fields, like checking for non-null values or ensuring data uniqueness.

5. **Documentation**: dbt auto-generates documentation from your models, providing a clear overview of your data transformations, lineage, and dependencies. You can also add descriptions for models and fields to improve the clarity of your data pipelines.

6. **[Data Lineage](#data-lineage)**: dbt automatically tracks the lineage of your data by mapping dependencies between models. This makes it easy to understand how data flows through the pipeline and where any upstream or downstream issues might originate.

7. **Extensibility**: dbt has a plugin architecture that allows users to extend functionality. For example, there are adapters for popular data warehouses like Snowflake, BigQuery, Redshift, and others, making dbt highly flexible in different data stack environments.

8. **Cloud and Core Versions**: 
   - **dbt Core** is the open-source version that you run locally or in your cloud infrastructure.
   - **dbt Cloud** is a fully managed service that adds features like scheduling, logging, and a web-based IDE for dbt workflows.

### Workflow with dbt:
1. **Data Loading**: First, data is loaded into a data warehouse from various sources using ELT tools (e.g., Fivetran, Stitch).
2. **Transform with dbt**: Using dbt, you write SQL models to clean, transform, and aggregate the raw data into useful, analytical datasets.
3. **Build Data Models**: You organize your models into layers, often referred to as staging, intermediate, and final models.
4. **Testing and Documentation**: Run tests to validate data, generate lineage diagrams, and create documentation.
5. **Deploy**: Schedule or trigger dbt jobs to run in production environments, ensuring consistent and accurate data transformations.

### Example of a dbt Model:
```sql
-- models/staging_orders.sql
WITH raw_orders AS (
    SELECT * FROM {{ ref('raw_orders_data') }}
)
SELECT 
    order_id,
    customer_id,
    order_date,
    amount
FROM raw_orders
WHERE order_status = 'completed';
```
In this model:
- `ref('raw_orders_data')` is referencing another model that contains raw order data.
- The model selects and transforms only the completed orders.

### Benefits of Using dbt:
1. **Analyst Empowerment**: dbt empowers data analysts to own the transformation process using SQL, reducing dependency on data engineers for transformations.
2. **Version Control and Testing**: Built-in version control and testing improve data reliability and reduce risks of errors in production.
3. **Modularity and Scalability**: The modular nature of dbt models makes it easier to scale transformations and manage complex pipelines.
4. **Transparency and Documentation**: dbt creates clear documentation and lineage automatically, improving visibility across teams.

### Tools Integrating with dbt:
- **Data Warehouses**: Redshift, Snowflake, BigQuery, Postgres.
- **[ELT](#elt) Tools**: Stitch, Fivetran, Airbyte (for the extraction and loading phase).
- **Version Control**: GitHub, GitLab, Bitbucket (for managing dbt code).
  
### Resources:
https://www.getdbt.com/blog/what-exactly-is-dbt
[dbt](https://docs.getdbt.com/docs/introduction) 



[dbt](#dbt)
   **Tags**: #data_transformation, #data_tools

# Declarative {#declarative}


In a **declarative data pipeline**, the focus is on *what* needs to be achieved, not *how* it should be executed. You define the desired outcome or the data products, and the system takes care of the underlying execution details, such as the order in which tasks are performed. This is in contrast to an **imperative** pipeline, where the developer explicitly specifies the steps and the order in which they should be executed. Here's a breakdown of the key aspects:

### **Declarative Programming**:
- Focuses on *<mark>what</mark>* needs to be done.
- Describes the desired state or result without dictating the control flow or step-by-step process.
- In a data pipeline context, a declarative approach might involve specifying the desired data products and letting the system optimize how and when different parts of the pipeline are executed.
- Example: [SQL](#sql) is often considered declarative because you specify the result you want (e.g., the output of a query) without explicitly stating the steps for how the database engine should retrieve it.

### **Imperative Programming**:
- Focuses on *<mark>how</mark>* tasks should be done.
- Specifies the <mark>control flow</mark> explicitly, dictating the exact steps to be performed and the order of operations.
- In a data pipeline, this would involve writing scripts that detail each step in the transformation and loading process in the sequence they must be executed.
- Example: A series of Python scripts that process data in a specific sequence.

### **Advantages of Declarative Pipelines**:
1. **Easier to Debug**: Since the desired state is clearly defined, it is easier to identify discrepancies between the intended outcome and the current state. This can help pinpoint issues in the pipeline.
   
2. **Automation**: Declarative systems often enable better automation since the system has the flexibility to determine the most efficient way to achieve the defined goals.

3. **Simplicity and Intent**: Declarative approaches focus on the *<mark>intent</mark>* of the program, making it easier for others to understand what the program is supposed to do without having to dive into implementation details. 

4. **Reactivity**: The pipeline can automatically adjust when inputs or dependencies change. For example, if certain data dependencies change, the system can rerun the necessary parts of the pipeline to maintain consistency.

### **Example in Data Engineering**:

A declarative approach to data engineering would involve **Functional Data Engineering** principles. This involves treating data as immutable and focusing on defining the desired transformations and outputs in a declarative manner. Instead of writing imperative scripts for each data transformation step, you'd define the desired outputs, and the system would optimize the execution.

### **Use Cases**:
Declarative pipelines are particularly useful in [data lineage](#data-lineage), **[Data Observability](#data-observability)**, and [Data Quality](#data-quality) monitoring**. By defining *what* data products should exist and what their properties should be, it's easier to track changes and ensure the consistency and quality of data. It also makes systems more resilient to changes, as the declarative nature enables the system to adjust the execution order or method dynamically, based on current conditions.

# Dependency Manager {#dependency-manager}

[Virtual environments](#virtual-environments)

[requirements.txt](#requirementstxt)

[TOML](#toml)

[Poetry](#poetry)