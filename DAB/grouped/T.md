# T

## Table of Contents
* [T-test](#t-test)
* [TF-IDF](#tf-idf)
* [TOML](#toml)
* [TS_Anomaly_Detection](#ts_anomaly_detection)
* [TS_Anomaly_Detection.py](#ts_anomaly_detectionpy)
* [Tableau](#tableau)
* [Tags](#tags)
* [Technical Debt](#technical-debt)
* [Technical Design Doc Template](#)
* [Telecommunications](#telecommunications)
* [Tensorflow](#tensorflow)
* [Terminal commands](#terminal-commands)
* [Test Loss When Evaluating Models](#test-loss-when-evaluating-models)
* [Testing](#testing)
* [Testing_Pytest.py](#testing_pytestpy)
* [Testing_unittest.py](#testing_unittestpy)
* [Text2Cypher](#text2cypher)
* [Thinking Systems](#thinking-systems)
* [Time Series Forecasting](#time-series-forecasting)
* [Time Series Identify Trends and Patterns](#time-series-identify-trends-and-patterns)
* [Time Series](#time-series)
* [Tokenisation](#tokenisation)
* [Train-Dev-Test Sets](#train-dev-test-sets)
* [Transaction](#transaction)
* [Transfer Learning](#transfer-learning)
* [Transformed Target Regressor](#transformed-target-regressor)
* [Transformer](#transformer)
* [Transformers vs RNNs](#transformers-vs-rnns)
* [Turning a flat file into a database](#turning-a-flat-file-into-a-database)
* [TypeScript](#typescript)
* [Types of Computational Bugs](#types-of-computational-bugs)
* [Types of Database Schema](#types-of-database-schema)
* [Types of Neural Networks](#types-of-neural-networks)
* [Typical Output Formats in Neural Networks](#typical-output-formats-in-neural-networks)
* [t-SNE](#t-sne)
* [tool.bandit](#)
* [tool.ruff](#toolruff)
* [tool.uv](#tooluv)
* [topic modeling](#topic-modeling)
* [transfer_learning.py](#transfer_learningpy)



<a id="t-test"></a>
# T Test {#t-test}

The T-test is a statistical method <mark>used to determine if there is a significant difference between the means of two groups, especially when the population [standard deviation](#standard-deviation) is unknown.</mark> It is particularly useful when dealing with small sample sizes.

## Types of T-tests

1. **One-Sample T-test**: This test compares the mean of a single sample to a known value (often the population mean). It helps determine if the sample mean significantly differs from the population mean.

2. **Two-Sample T-test**: This test compares the means of two independent samples. It can be further categorized into:
   - **Two-Sample T-test with Known [Variance](#variance)**: Used when the variances of the two groups are known and assumed to be equal.
   - **Two-Sample T-test with Unknown Variance**: Used when the variances are unknown and may differ between the two groups. This version is more common in practice.

## Characteristics of the T-distribution

The T-distribution resembles the normal [Distributions|distribution](#distributionsdistribution) but has fatter tails. This characteristic accounts for the increased variability expected with smaller sample sizes. As the sample size increases, the T-distribution approaches the normal distribution.

## Assumptions

For the T-test to be valid, certain assumptions must be met:
- The data should be approximately normally distributed, especially for small sample sizes.
- The samples should be independent of each other.
- For the two-sample T-test, the variances of the two groups should be equal (for the equal variance version).

## Estimation of Standard Deviation

Since the population standard deviation is unknown, the sample standard deviation is used to estimate it. This estimation is crucial for calculating the test statistic.

## Test Statistic

The test statistic for the T-test is calculated using the formula:

$$ T = \frac{\bar{X} - \mu}{s / \sqrt{n}} $$

where:
- $\bar{X}$ = sample mean
- $\mu$ = population mean (or mean of the second sample in the two-sample test)
- $s$ = sample standard deviation
- $n$ = sample size

This formulation condenses all the data into a single variable, allowing for [hypothesis testing](#hypothesis-testing).
## Importance of the T-test

The T-test is a Uniformly Most Powerful Unbiased (UMPU) test, meaning it is optimal for detecting differences in means under the specified conditions.


<a id="tf-idf"></a>
# Tf Idf {#tf-idf}


TF-IDF is a statistical technique used in text analysis to determine the importance of a word in a document relative to a collection of documents (corpus). It balances two ideas:

- Term Frequency (TF): Captures how often a term occurs in a document.
- Inverse Document Frequency (IDF): Discounts terms that appear in many documents.

High TF-IDF scores indicate terms that are frequent in a document but rare in the corpus, making them useful for distinguishing between documents in tasks such as information retrieval, document classification, and recommendation.

TF-IDF combines local and global term [Statistics](#statistics):
- TF gives high scores to frequent terms in a document
- IDF reduces the weight of common terms across documents
- TF-IDF identifies terms that are both frequent and distinctive

### Equations

#### Term Frequency

$TF(t, d)$ measures how often a term $t$ appears in a document $d$, normalized by the total number of terms in $d$:

$$
TF(t, d) = \frac{f_{t,d}}{\sum_k f_{k,d}}
$$

Where:
- $f_{t,d}$ is the raw count of term $t$ in document $d$  
- $\sum_k f_{k,d}$ is the total number of terms in $d$ (i.e. the document length)

#### Inverse Document Frequency

IDF assigns lower weights to frequent terms:

$$
IDF(t, D) = \log \left( \frac{N}{1 + |\{d \in D : t \in d\}|} \right)
$$

Where:
- $N$ is the number of documents in the corpus $D$  
- $|\{d \in D : t \in d\}|$ is the number of documents containing term $t$  
- Adding 1 to the denominator avoids division by zero

#### TF-IDF Score

The final score is:

$$
TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)
$$

### Related Notes

- [Bag of words](#bag-of-words)
- [Tokenisation](#tokenisation)
- [Clustering](#clustering)
- [Search](#search)
- [Recommender systems](#recommender-systems)
- [nltk](#nltk)

### Exploratory Ideas
- Can track TF-IDF over time (e.g., note evolution)
- Can cluster or classify the documents using TF-IDF?
## Implementations 

### Python Script (scikit-learn version)

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Step 1: Tokenize and vectorize using Bag of Words
bow = CountVectorizer(tokenizer=normalize_document)
X_counts = bow.fit_transform(corpus)

# Step 2: Apply TF-IDF transformation
tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)

# Optional: View TF-IDF scores per document
for doc_id in range(len(corpus)):
    print(f"Document {doc_id}: {corpus[doc_id]}")
    print("TF-IDF values:")
    tfidf_vector = X_tfidf[doc_id].T.toarray()
    for term, score in zip(bow.get_feature_names_out(), tfidf_vector):
        if score > 0:
            print(f"{term.rjust(10)} : {score[0]:.4f}")
```

### Python Script (custom TF-IDF implementation)

```python
import math
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.util import bigrams, trigrams

stop_words = stopwords.words('english')
tokenizer = RegexpTokenizer(r'\w+')

def tokenize(text):
    tokens = tokenizer.tokenize(text.lower())
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
    return tokens + [' '.join(b) for b in bigrams(tokens)] + [' '.join(t) for t in trigrams(tokens)]

def tf(term, doc_tokens):
    return doc_tokens.count(term) / len(doc_tokens)

def idf(term, docs_tokens):
    doc_count = sum(1 for doc in docs_tokens if term in doc)
    return math.log(len(docs_tokens) / (1 + doc_count))

def compute_tfidf(docs):
    docs_tokens = [tokenize(doc) for doc in docs]
    all_terms = set(term for doc in docs_tokens for term in doc)
    tfidf_scores = []
    for tokens in docs_tokens:
        tfidf = {}
        for term in all_terms:
            if term in tokens:
                tfidf[term] = tf(term, tokens) * idf(term, docs_tokens)
        tfidf_scores.append(tfidf)
    return tfidf_scores
```





<a id="toml"></a>
# Toml {#toml}

A `.toml` file is a configuration file format that stands for "Tom's Obvious, Minimal Language." 

It is designed to be easy to read due to its simple syntax. TOML files are often used for configuration because they are straightforward to parse and write for both humans and machines. 

The format supports basic data types like strings, integers, floats, booleans, arrays, and tables (which are similar to dictionaries or objects in other programming languages).

```toml
title = "TOML Example"

[owner]
name = "Tom Preston-Werner"
dob = 1979-05-27T07:32:00Z

[database]
server = "192.168.1.1"
ports = [ 8001, 8001, 8002 ]
connection_max = 5000
enabled = true
```

In this example, you can see how different data types and structures are represented in a TOML file.

## What can you do with these files?

TOML files are primarily used for configuration purposes. 

1. **Application Configuration**: Many software applications use TOML files to store configuration settings. This allows users to easily modify settings without altering the code.

2. **Project Metadata**: In some programming environments, TOML files are used to define project metadata, such as dependencies, version numbers, and other project-specific information.

3. **Data Serialization**: TOML can be used to serialize data in a format that is both human-readable and easy to parse programmatically.

4. **Environment Settings**: TOML files can be used to manage environment-specific settings, such as database connections or API keys, which can vary between development, testing, and production environments.

5. **Configuration for Build Tools**: Some build tools and package managers use TOML files to define build configurations and dependencies.

## Contents of TOML file

[tool.ruff](#toolruff)

[tool.bandit](#toolbandit)

[tool.uv](#tooluv)

[Pytest](#pytest)

<a id="ts_anomaly_detection"></a>
# Ts_Anomaly_Detection {#ts_anomaly_detection}



<a id="ts_anomaly_detectionpy"></a>
# Ts_Anomaly_Detection.Py {#ts_anomaly_detectionpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries/TS_Anomaly_Detection.py

<a id="tableau"></a>
# Tableau {#tableau}


## Next Steps
- Load a [PostgreSQL](#postgresql) database and perform analytics as an example.
## Resources
- [Tableau How-To Videos](https://public.tableau.com/app/learn/how-to-videos)
- [Tutorial Link](https://public.tableau.com/app/learn/how-to-videos)
- [Example Usage Video](https://www.youtube.com/watch?v=L5PL0gg1cPQ)
## Features
- Can publish to blogs and embed dashboards online ([Data Visualisation](#data-visualisation))
- Dashboards can be shared online.
- Easier than doing visualizations in Python.



<a id="tags"></a>
# Tags {#tags}



<a id="technical-debt"></a>
# Technical Debt {#technical-debt}


Technical debt refers to the concept in software development where developers take shortcuts or make suboptimal decisions to spped up the delivery of a project. 

These shortcuts can lead to increased complexity and potential issues in the codebase, which may require additional effort to address in the future.

Just like financial debt, technical debt incurs "interest," meaning that the longer it remains unaddressed, the more costly it becomes to fix.

### How Can Businesses Reduce Technical Debt?

1. **Automate Testing and Code Quality Checks**: 
   - Implement automated tests to ensure code quality and catch issues early. Tools like RUFF, mypy, and fixit can help enforce coding standards and identify potential problems.
   - Use type checkers and automated checks for coding conventions to maintain consistency and reduce errors.

2. **Track Technical Debt**:
   - Use dashboards to monitor and visualize technical debt. This helps in identifying areas that need attention and prioritizing them accordingly.

3. **Code Refactoring and "Spa Days"**:
   - Schedule regular "spa days" for the codebase, where the focus is on cleaning and refactoring specific areas. This helps in gradually reducing technical debt without impacting ongoing development.

4. **Empower Developers**:
   - Allow developers to identify and address technical debt as they work on the codebase. They are often best positioned to recognize areas that need improvement.

5. **Prioritize and Plan**:
   - Make technical debt reduction a part of the project planning process. Prioritize tasks that address high-impact debt and allocate time for refactoring in each development cycle.

# [Project name] Design Doc

# About this doc

_Metadata about this document. Describe the scope and current status._

_This doc describes the technical approach, milestones, and work planned for the [Project name linked to Product Requirements Doc]_

|   |   |
|---|---|
|Sign off deadline|_Date_|
|Status|_Draft_|
|Author(s)|_Name 1, Name 2_|

Sign offs

- *Name 1*
    
- *Name 2*
    
- Add your name here to sign off
    

# Context

_A sentence or two on the “what.” What is being built._

_Then include the “why” (metrics we intend for). Link off to Product Requirements Doc for details._

# Non-goals

_What is out of scope for this project that we don’t want to focus on?_

- _Out of scope detail 1_
    
- _Out of scope detail 2_
    
- …
    

# Terminology

_Define any new terms that are used in the document._

# High level approach

_Explain the technical approach in a few sentences so the reader understands the system flow_

# Alternatives considered

_Bullet points of alternative approaches considered and why you’re not going with them._

- _Alternative 1_
    
- _Alternative 2_
    
- _…_
    

# Detailed design

_APIs, DB tables modified, Data models changed, and any diagrams that would help the reader understand at a high level_

# Risks

_What can go wrong with the proposed approach? How are you mitigating that?_

- _Risk 1_
    
- _Risk 2_
    
- _…_
    

# Test plan

_How will your approach be tested? Browser testing? Manual testing? Will anything be tricky to test? Adding information here also makes it easier to make a case for a longer timeline._

# Milestones

_How will the work be divided into chunks of progress?_

_Focus on the user milestones rather than technical ones. For example, having a minimal working feature behind a feature flag initially._

_I usually add an additional 1 week per 2 weeks of expected feature work._

- _Milestone 1: Date 1_
    
- _Milestone 2: Date 2_
    
- …
    
- _Rough ETA of project finish date: …_
    
- _Project retro: …_
    

# Rollout plan

_[optional] How will you gradually ramp up usage of the feature for safety?_

_A feature flag starting at 1% of users? Only testing in a specific region? What feature flags?_

# Open Questions

_Anything still being figured out that you could use some additional eyes or thoughts on._

# Parties involved

_Who is working on this? Are there any external teams that need to sign off on this as well?_

- Eng 1: [Name]
    
- Eng 2: [Name]
    
- PM: [Name]
    
- Designer: [Name]
    
- [External team name]
    

# Appendix

_Add any detailed figures you didn’t want to inline for space._

<a id="telecommunications"></a>
# Telecommunications {#telecommunications}


Network Optimization

- **Overview**: In telecommunications, RL is used to enhance network performance, optimize resource allocation, and manage traffic efficiently.
- **Applications**:
    - **Traffic Management**: RL algorithms can analyze real-time network traffic to optimize routing and minimize congestion, ensuring that data packets are transmitted through the least congested paths.
    - **Quality of Service (QoS)**: RL can be used to allocate bandwidth dynamically based on current demand and service-level agreements (SLAs), improving user experience by maintaining high QoS standards.
    - **Fault Detection and Recovery**: RL systems can learn to identify and respond to network anomalies or failures, automatically rerouting traffic or reallocating resources to maintain service continuity.

###### 8.2 Dynamic Resource Allocation

- **Overview**: Dynamic resource allocation in telecommunications involves adjusting network resources (like bandwidth and processing power) in real-time based on user demand and network conditions.
- **Applications**:
    - **Load Balancing**: RL can help in distributing network loads across multiple servers or paths, ensuring optimal use of available resources while preventing any single point from becoming overloaded.
    - **Adaptive Scheduling**: RL algorithms can manage the scheduling of data transmission and resource allocation in cellular networks, allowing for efficient handling of varying traffic patterns and user behaviors.

<a id="tensorflow"></a>
# Tensorflow {#tensorflow}


Open sourced by Google
Based on a dataflow graph

[Text summarization with TensorFlow](https://research.googleblog.com/2016/08/text-summarization-with-tensorflow.html)

Open-source library for numerical computation and large-scale [Machine Learning](#machine-learning), focusing on static dataflow graphs.

Get same code use of tensorflow example:

Basic example is

[Handwritten Digit Classification](#handwritten-digit-classification)

[Pytorch vs Tensorflow](#pytorch-vs-tensorflow)

<a id="terminal-commands"></a>
# Terminal Commands {#terminal-commands}


jupyter nbconvert K-Means_VideoGames_Raw.ipynb --to python --no-prompt



<a id="test-loss-when-evaluating-models"></a>
# Test Loss When Evaluating Models {#test-loss-when-evaluating-models}

Test loss is used for [Model Evaluation](#model-evaluation) to assess how well a model generalizes to unseen data, which is essential for evaluating its performance in real-world applications.

## Importance of Test Loss

- Test Accuracy: Indicates the percentage of correct predictions.
- <mark>Test Loss: Measures the magnitude of errors in predictions, providing complementary information to accuracy.</mark>
- Balancing Metrics: Depending on the application, you might prioritize [Accuracy](#accuracy) (e.g., in classification tasks) or loss (e.g., when evaluating prediction confidence or calibrating probabilistic models). Balancing both is crucial for most real-world problems.

Test loss is an [Evaluation Metrics](#evaluation-metrics)  that uses the [loss function](#loss-function) to measure the model's performance on new, unseen data.
## Key Considerations

Balance Between Accuracy and Error Magnitude:
  - Accuracy reflects the percentage of correct predictions but not the degree of correctness.
  - Loss can reveal situations where the model is confident but wrong, or struggling despite being correct in many cases, helping to understand prediction quality beyond simple accuracy.

 Overfitting or Underfitting Detection:
  - High accuracy but high loss may indicate overfitting, where the model memorizes patterns rather than learning the underlying structure.
  - Low accuracy and high loss suggest underfitting, meaning the model hasn't learned the data well enough.

Model Calibration:
  - In probabilistic models, test loss is crucial for understanding calibration.
  - A model that’s accurate but poorly calibrated (where predicted probabilities don't match true outcomes) will have low test accuracy but high loss.
  - For example, in classification tasks, cross-entropy loss indicates how confidently and correctly the model assigns probabilities to each class.

[Hyperparameter Tuning](#hyperparameter-tuning)
  - During hyperparameter tuning (e.g., learning rate, batch size), configurations might yield high accuracy but poor loss (or vice versa).
  - Considering both metrics provides a balanced view of performance, aiding in fine-tuning the model for both high accuracy and low error.

Model Comparison: [Model Selection](#model-selection)
  - Models with similar accuracy can have significantly different losses.
  - The model with lower test loss is generally preferred as it suggests reliability in predicting probabilities, especially in critical applications like medical diagnoses or risk assessment.

 Outlier Sensitivity: [standardised/Outliers](#standardisedoutliers)/ [standardised/Outliers|Handling Outliers](#standardisedoutliershandling-outliers)
  - Test loss can help identify model sensitivity to outliers.
  - A model might achieve high accuracy but perform poorly in terms of test loss if it incorrectly classifies a few outliers.
  - Conversely, a model with low test loss might be more stable in making predictions, even for edge cases.

<a id="testing"></a>
# Testing {#testing}


Testing in coding projects refers to the systematic process of evaluating software to ensure it meets specified requirements and functions correctly. It enhances software robustness, reduces maintenance costs, and improves user satisfaction.

Testing is crucial for:
- Identifying bugs
- Ensuring code quality
- Validating that the software behaves as expected under various conditions

Key Insights
- Testing reduces the probability of software failure, $P(\text{failure})$, by identifying defects before [Model Deployment|deployment](#model-deploymentdeployment).
- Effective testing strategies can lead to a decrease in the expected cost of errors, $E(\text{cost})$, associated with software bugs.

## Comprehensive Python Testing Strategy
Testing a [Python](#python) program effectively involves multiple levels to ensure correctness, performance, and security. Key testing types include:

1. **Unit Testing**
   - Tests individual functions and methods in isolation.
   - **Tools:** pytest, [unittest](#unittest), doctest.
   - **Best Practices:** Use meaningful test names, mock dependencies, and write focused tests.

2. **Integration Testing**
   - Verifies interactions between modules and external dependencies (databases, APIs).
   - **Best Practices:** Use in-memory databases, mock services, and reset the environment before tests.

3. **Functional Testing**
   - Ensures the application behaves as expected from a user’s perspective.
   - **Best Practices:** Simulate real-world scenarios, automate UI/API tests, and validate expected outputs.

4. **Performance Testing**
   - Measures execution speed, scalability, and resource usage.
   - **Best Practices:** Profile bottlenecks, stress-test under load, and monitor system performance.

5. **Security Testing** [Common Security Vulnerabilities in Software Development](#common-security-vulnerabilities-in-software-development)
   - Identifies vulnerabilities like SQL injection, XSS, and authentication flaws.
   - **Best Practices:** Validate inputs, enforce authentication, and use static analysis tools (e.g., Bandit).
## Related Topics
- [Continuous integration](#continuous-integration) and deployment (CI/CD)  
- Test-driven development (TDD)  
- Software quality assurance methodologies
- Performance testing and optimization techniques
- [Pytest](#pytest)
- Unit testing
- Integration testing
- System testing
- [Hypothesis testing](#hypothesis-testing)
- Test coverage
- [Types of Computational Bugs](#types-of-computational-bugs)


<a id="testing_pytestpy"></a>
# Testing_Pytest.Py {#testing_pytestpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/Testing_Pytest.py

In [ML_Tools](#ml_tools) see: [Testing_Pytest.py](#testing_pytestpy)

The `pytest` example script demonstrates several key features of the [Pytest](#pytest) testing framework:

1. **Fixtures**: The script uses a fixture named `sample_data` to provide common test data that can be reused across multiple test functions. This helps reduce code duplication and enhances test maintainability.

2. **Parametrization**: The script employs the `@pytest.mark.parametrize` decorator to run a test function with multiple sets of arguments. This allows for testing a function with various inputs without writing separate test cases for each scenario.

3. **Custom Markers**: A custom marker `@pytest.mark.slow` is used to categorize tests. This enables selective test execution based on markers, allowing you to run specific groups of tests using the `-m` option.

4. **Mocking**: The script demonstrates how to use `unittest.mock.patch` with `pytest` to replace a function with a mock. This allows for controlling the behavior of the function during the test, facilitating isolated testing of units.




<a id="testing_unittestpy"></a>
# Testing_Unittest.Py {#testing_unittestpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/Testing_unittest.py

To explore testing in Python, let's focus on some key concepts and provide a simple example using the `unittest` framework, which is a built-in module for writing and running tests.

By writing and running tests, you can ensure that your code behaves as expected and catch bugs early in the development process. 

[Pytest](#pytest)
### Key Concepts in Testing

1. **Unit Testing**: Testing individual components or functions in isolation to ensure they work as expected. Unit tests are typically small and fast.

2. **Test-Driven Development (TDD)**: A development approach where tests are written before the actual code. This helps define the expected behavior and ensures the code meets these expectations.

3. **Assertions**: Statements in tests that check if a condition is true. If an assertion fails, the test fails.

4. **Test Suites**: Collections of test cases that can be run together.

5. **Mocking**: Simulating the behavior of complex objects or systems to isolate the unit being tested.

### How to Run the Tests

1. Save the script to a file, e.g., `test_math_operations.py`.
2. Run the script using Python: `python test_math_operations.py`.
3. The `unittest` framework will automatically discover and run the test cases, reporting any failures or errors.

### Exploring Further

- **Test Coverage**: Measure how much of your code is covered by tests. Tools like `coverage.py` can help identify untested parts of your code.
- **Continuous Integration (CI)**: Automate the running of tests using CI tools like Jenkins, Travis CI, or GitHub Actions to ensure code quality in every commit.
- **Behavior-Driven Development (BDD)**: An extension of TDD that uses natural language to describe the behavior of software, often using tools like `pytest-bdd` or `behave`.

2. **Edge Cases and Error Handling**:
    
    - The `divide` function raises a `ValueError` if division by zero is attempted.
    - The `test_divide` method includes a test case to check for this exception using `assertRaises`.
3. **Mocking**:
    - The `test_mock_add` method demonstrates how to use `unittest.mock.patch` to replace the `add` function with a mock that returns a fixed value.
    - This is useful for isolating the unit under test and controlling its behavior.
4. **Comprehensive Testing**:
    
    - Each function is tested with multiple inputs, including positive, negative, and zero values, to ensure robustness.

<a id="text2cypher"></a>
# Text2Cypher {#text2cypher}

Text2Cypher is a concept that allows users to convert natural language queries into Cypher queries, which are used to interact with [GraphRAG|graph database](#graphraggraph-database) like [Neo4j](#neo4j). This functionality enables users to ask questions in a more intuitive/[interpretability|interpretable](#interpretabilityinterpretable), conversational manner, rather than needing to know the specific syntax of [Cypher](#cypher).

Allows the user to ask vague questions.
Allows for multihop queries on the graph

Overall, Text2Cypher aims to simplify the interaction with graph databases, making it accessible to users who may not be familiar with query languages.
### Key Features of Text2Cypher:

1. **Natural Language Processing**: It utilizes natural language processing (NLP) techniques to understand user queries and translate them into structured Cypher queries.

2. **Flexibility**: Users can ask vague or complex questions that may not directly relate to the underlying data structure, making it easier to retrieve information from a graph database.

3. **Traversal Queries**: Text2Cypher can generate traversal queries that navigate through the graph, allowing for multi-hop queries that explore relationships between entities.

4. **Explainability**: By converting natural language into Cypher, it helps provide a clearer understanding of how the data is structured and how the queries are executed, enhancing interpretability.



<a id="thinking-systems"></a>
# Thinking Systems {#thinking-systems}


A thinking system is a point of view that helps solve a problem. Part of [Knowledge Work](#knowledge-work). We view problems through the view of our own specialism (mathematics).  Thinking systems help with perspective.

Types of thinking systems:
- Design
- Engineering
### Design thinking

Historically we trained people to use a product.  

- users have choices, 
- be user focused, 
- empathy and contextual enquiry for the product/system  

This puts human at the center, and focus on thinking about <mark>pain points</mark> of their experience.  

Remember:
- User segments are not the same and should be handled separately.  
- It is important to understand how the user interacts with the environment/other beings  
- <mark>Ludic properties</mark> reduce design load, for example chatgpt and chat feature, or use of the iphone.

### Scientific thinking:  

Strong ideas but loosely held.
 
Examples:
- AlphaFold and protein folding. They had lots of data and where able to derive insights from that using AI.  

As problems are too complex for human minds [Scientific Method](#scientific-method) now:  
- We have data,  
- Pick an algo ,  
- Compute then gives hypothesis.  
### System Thinking

We need to design for the whole system not just the individual.  

I am stuck in traffic, versus you are the traffic.  

Emergent behaviour with lime moulds, tokyo metro network.  

Collective intelligence. 

Designing system so that individuals impact the whole.

<a id="time-series-forecasting"></a>
# Time Series Forecasting {#time-series-forecasting}


With [Time Series](#time-series) dataset we often want to predict future terms. These are methods to do so.

Resources:
[TimeSeries Forecasting](https://simrenbasra.github.io/simys-blog/2024/09/19/timeseries_part2.html)

### Statistical Methods
- [Forecasting_Baseline.py](#forecasting_baselinepy)
- [Forecasting_Exponential_Smoothing.py](#forecasting_exponential_smoothingpy)
- [Forecasting_AutoArima.py](#forecasting_autoarimapy)

### Machine Learning Methods

[XGBoost](#xgboost)

[LightGBM](#lightgbm)

<a id="time-series-identify-trends-and-patterns"></a>
# Time Series Identify Trends And Patterns {#time-series-identify-trends-and-patterns}

 Analyze long-term trends, seasonal patterns, and cyclical behaviors.
 
ARIMA or SARIMA


<a id="time-series"></a>
# Time Series {#time-series}


Time series data is a sequence of data points collected or recorded at successive points in time, typically at uniform intervals. It captures the temporal ordering of data, which is crucial for analyzing trends, patterns, and changes over time.

Time series data is widely used across various domains, including:
- Finance: Stock prices, interest rates, and economic indicators.
- Weather Forecasting: Temperature, precipitation, and wind speed data.

In [ML_Tools](#ml_tools) see: TimeSeries folder
- https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries"
## What Can You Do with Time Series Data?

With time series data, you can:

- [Time Series Forecasting](#time-series-forecasting)
- [Time Series Identify Trends and Patterns](#time-series-identify-trends-and-patterns)
- [Anomaly Detection](#anomaly-detection)

```python
# We set the 'day' column as the index to facilitate time-series operations.
df.set_index('day', inplace=True)
print(df)
```


<a id="tokenisation"></a>
# Tokenisation {#tokenisation}


**Tokenisation** is a fundamental process in natural language processing ([NLP](#nlp)) that involves breaking down text into smaller units called tokens. These tokens can be words, sentences, or <mark>subwords</mark>, depending on the level of tokenization. 

Word tokenisation
```python

from nltk.tokenize import word_tokenize #keeps punctuation
text_word_tokens_nltk = word_tokenize(text_original)
print(text_word_tokens_nltk)
```

Sentence tokenisation

```python
from nltk.tokenize import sent_tokenize
text_sentence_tokens_nltk = sent_tokenize(text_original)
print(text_sentence_tokens_nltk)
```

Basic tokenisation
```python
temp = text_original.lower()
temp = re.sub(r"[^a-zA-Z0-9]", " ", temp) # just letters and numbers
temp = re.sub(r"\[[0-9]+\]", "", temp) #remove weird stuff
temp = word_tokenize(temp) #break up text to word list
tokens_no_stopwords = [token for token in temp if token not in stopwords.words("english")] #remove common words
print(tokens_no_stopwords)
```

<a id="train-dev-test-sets"></a>
# Train Dev Test Sets {#train-dev-test-sets}

In [Model Building](#model-building) train the model using the prepared data to learn patterns and make predictions. The model is trained on your dataset, which is typically divided into three main subsets: training, development (dev), and test sets.
### Purpose of Each Set

- Training Set ([training data](#training-data)) : Used to fit the model. This is where the model learns the patterns and relationships within the data. The majority of the data is allocated here to ensure the model has enough information to learn effectively.

- Development (Dev) Set: Also known as the [validation data](#validation-data), it is used to tune the [model parameters](#model-parameters) and make decisions about model architecture. It helps in preventing overfitting by providing a separate dataset to evaluate the model's performance during training.

- Test Set: Used to evaluate the final model's performance/ [Model Evaluation](#model-evaluation). This set is not used during the training process and provides an unbiased evaluation of the model's ability to generalize to new, unseen data.


### Why do it this way

**Preventing Overfitting**: By monitoring performance on the validation set, practitioners can detect overfitting early and take corrective actions, such as adjusting model complexity or applying regularization techniques.

**Hyperparameter Tuning**: The validation set is crucial for tuning hyperparameters (e.g., learning rate, regularization strength) to optimize model performance.

Historical Suggestions
- Train-Test Sets: 70% training, 30% testing.
- Train-Dev-Test: 60% training, 20% development, 20% testing.

Modern Approach
- With larger [Datasets](#datasets), a split of 98% training, 1% development, and 1% testing is often used. This is because modern models require more data to learn effectively, and larger datasets allow for smaller proportions to be allocated to dev and test sets while still maintaining sufficient data for evaluation.

Code Example
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.3)
```

Considerations
- Data Setup: Be careful when setting up the training and test data to ensure they are <mark>representative</mark> of the problem domain.
- <mark>[Distributions](#distributions): Dev and test sets should be from the same distribution to ensure consistent [evaluation metrics](#evaluation-metrics). Avoid having subsets that are biased, such as data from the same geographical area.</mark>
- [Handling Different Distributions](#handling-different-distributions): Randomly shuffle the data before splitting to ensure that each subset is representative of the whole dataset.
- [Cross Validation](#cross-validation): Consider using cross-validation techniques to make the most of your data, especially when the dataset is small.





<a id="transaction"></a>
# Transaction {#transaction}


Transactions are used for maintaining [Data Integrity](#data-integrity) and should adhere to the [ACID Transaction](#acid-transaction).

## Transaction Operations

- Commit: Saves all changes made during the transaction.
- Rollback: Reverts the database to its previous state if an error occurs during the transaction.

## Concurrency and Transactions

- Concurrency: Allows multiple [queries](#queries) to run simultaneously, essential for high-traffic applications.
- Race Conditions: Occur when concurrent transactions access and modify shared data, potentially causing inconsistencies.

## Transaction Locks

To prevent <mark>race conditions,</mark> transactions and locking mechanisms are employed to ensure that operations occur sequentially. Locks manage access to the database during [Transaction|Transactions](#transactiontransactions):

- UNLOCK: Allows anyone to read or add data.
- SHARED: Permits reading while allowing others to access the data.
- LOCKED: Grants exclusive write access to ensure that no other transactions can interfere.

### Types of Locks
- Shared Locks: Used for read operations.
- Exclusive Locks: Used for write operations.

### [Granularity](#granularity) of Locks

[SQLite](#sqlite) locks the entire database during exclusive transactions. While finer granularity (e.g., row-level locks) is possible in other database management systems ([Database Management System (DBMS)|DBMS](#database-management-system-dbmsdbms)), SQLite's approach is simpler but can lead to contention in high-concurrency scenarios.

### Timestamping
Using timestamping can help manage access to exclusive locks, allowing for more efficient handling of concurrent transactions ([Concurrency](#concurrency)).


<a id="transfer-learning"></a>
# Transfer Learning {#transfer-learning}


Transfer learning is a  technique in machine learning that <mark>leverages knowledge gained from one setting (source domain) to improve performance on a different but related setting (target domain).</mark> 

The core idea is to train a model on a large dataset in the source domain, learning rich feature representations that capture general patterns and relationships in the data.<mark> These learned representations can then be transferred to the target domain, where they can be fine-tuned with a smaller dataset to achieve good performance on the target task.

Transfer learning makes sense when:
- It makes sense to do when there is lots of examples for basic layers and training, and there are few of the specialised data set.
- Task A and B have the same input x.
- You have a lot more data for Task A than Task B.
- Low level features from A could be helpful for learning B.
- When labelled data is scarce.

We can use pretrained models (i.e. from [Hugging Face](#hugging-face), [Keras](#keras) applications, [PyTorch](#pytorch) pretrained models, model zoo).

Examples of Transfer Learning
- **Image Recognition:** A model trained on a large dataset of labelled images (e.g., ImageNet) can learn features like edges, shapes, and textures that are useful for recognising a wide variety of objects. These features can then be transferred to a different image recognition task with a smaller dataset, such as classifying medical images or identifying different species of plants.
	- Pretraining - training on image recognition. Fine-training - retraining on radiology. 
	  
- **Natural Language Processing:** A language model trained on a massive text corpus can learn word embeddings that capture semantic relationships between words. These embeddings can then be transferred to tasks like sentiment analysis or machine translation, where they can improve performance, even with limited labelled data in the target language.

Types of Transfer Learning
- **Unsupervised Pretraining for [Supervised Learning](#supervised-learning):** The sources describe how unsupervised pretraining with models like denoising autoencoders can be used to learn useful representations that can be transferred to supervised learning tasks.
- **Cross-Domain Transfer Learning:** This involves transferring knowledge between domains with different input distributions but the same task. 
- **[Performance Drift](#performance-drift):** This is a form of transfer learning where the data distribution changes gradually over time. The model needs to adapt to these changes to maintain good performance.

Benefits of Transfer Learning
- **Improved Generalisation:** By leveraging knowledge from a larger dataset, transfer learning can help models generalise better to new data, especially when the target dataset is small.
- **Reduced Data Requirements:** Transfer learning can significantly </mark>reduce the amount of labelled data needed to train a model in the target domain.== This is particularly beneficial for tasks where labelled data is expensive or time-consuming to obtain.
- **Faster Training:** Fine-tuning a pretrained model on a smaller dataset is typically faster than training a model from scratch.
### Follow up questions

- Why might fine-tuning a pre-trained model like GPT yield better results than training from scratch
### Practical Implementation

In [ML_Tools](#ml_tools) see: [transfer_learning.py](#transfer_learningpy)

### Links
- [Video Overview](https://www.youtube.com/watch?v=yofjFQddwHE&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=19)
- [Deep Learning Video](https://www.youtube.com/watch?v=DyPW-994t7w&list=PLcWfeUsAys2nPgh-gYRlexc6xvscdvHqX&index=15)



<a id="transformed-target-regressor"></a>
# Transformed Target Regressor {#transformed-target-regressor}

[Sklearn](#sklearn)

The `TransformedTargetRegressor` is a utility class in `scikit-learn` that applies a transformation to the target values in a regression problem. This can be useful in several scenarios:

1. **Non-normal target distribution**: Many regression algorithms assume that the target variable is normally distributed. If your target variable has a skewed distribution, applying a transformation (like a log transformation) can help improve the performance of the model.
    
2. **Heteroscedasticity**: This is a situation where the variance of the error terms in a regression model is not constant. In such cases, applying a transformation to the target variable can help stabilize the variance and improve the model's performance.
    
3. **Non-linear relationships**: If the relationship between the predictors and the target variable is not linear, a transformation can help capture the non-linearity.
    

The `TransformedTargetRegressor` applies the transformation before training the model and automatically applies the inverse transformation when making predictions. This makes it easier to work with transformed target variables, as you don't have to manually apply the inverse transformation every time you want to make a prediction.

<a id="transformer"></a>
# Transformer {#transformer}


A transformer in machine learning (ML) refers to a deep learning model architecture designed to process sequential data, such as natural language processing ([NLP](#nlp)). It was introduced in the paper "[standardised/Attention Is All You Need](#standardisedattention-is-all-you-need)" and has since become a cornerstone in NLP tasks.

 Transformers excel at handling sequence-based data and are particularly known for their self-attention mechanisms [Attention mechanism](#attention-mechanism), which allow them to process long-range dependencies in data.

### Key Concepts of a Transformer

1. **Architecture Overview**:
    
    - A transformer model consists of an encoder and a decoder, although some models use only the **encoder** (like [BERT](#bert) only consists of encoders) or only the **decoder** (like GPT3). Each of these components is made up of layers that include mechanisms for attention and [Feed Forward Neural Network](#feed-forward-neural-network).
    - Encoder learns the context, decoder does the task.

1. **Self-[Attention Mechanism](#attention-mechanism)**:
    
    - The core innovation of transformers is the self-attention mechanism, which allows the model to weigh the importance of different words in a sentence relative to each other. This is crucial for understanding context and relationships in language. See [Attention mechanism].
    - **Scaled Dot-Product Attention**: For each word in a sentence, the model computes attention scores with every other word. These scores are used to create a weighted representation of the input, emphasizing relevant words and de-emphasizing less relevant ones.

1. [Multi-head attention](#multi-head-attention)
    
    - Instead of having a single attention mechanism, transformers use multiple attention heads. Each head learns different aspects of the relationships between words, allowing the model to capture various linguistic features.

1. **Positional Encoding**:
    
    - Since transformers do not inherently understand the order of words (unlike [Recurrent Neural Networks|RNNs](#recurrent-neural-networksrnns)), they use positional encoding to inject information about the position of each word in the sequence.
5. **Feed-Forward Neural Network**:
    
    - After the attention mechanism, the output is passed through a feed-forward neural network, which is applied independently to each position.
6. **Layer Normalization and Residual Connections**:
    
    - Transformers use layer normalization and residual connections to stabilize training and help with gradient flow, making it easier to train deep networks.
7. **Training and Applications**:
    
    - Transformers are trained on large corpora of text data using [Unsupervised Learning|unsupervised](#unsupervised-learningunsupervised) or semi-supervised learning techniques. They are used for a variety of NLP tasks, including translation, summarization, and question answering.

### Additional Concepts

- **Encoder-Decoder Structure**:
    
    - The encoder processes the input sequence to build a representation, while the decoder takes this representation and generates the output sequence. This setup is particularly useful for tasks like translation.

- **Parallelization**:
    
    - Unlike Recurrent Neural Networks ([Recurrent Neural Networks](#recurrent-neural-networks)), transformers do **not require sequential processing,** making them more efficient, especially when training large datasets.

Follow-up questions:
- [Transformers vs RNNs](#transformers-vs-rnns)


<a id="transformers-vs-rnns"></a>
# Transformers Vs Rnns {#transformers-vs-rnns}


[Transformer|Transformers](#transformertransformers) and Recurrent Neural Networks ([Recurrent Neural Networks](#recurrent-neural-networks)) are both deep learning architectures <mark>used for processing sequential data</mark>, but they differ significantly in structure, operation, and performance.

While RNNs have been essential for sequence modeling, transformers have become the dominant architecture in ML due to their ability to handle large-scale data and long-range dependencies more efficiently. 

RNNs still have use cases, especially for tasks where memory constraints are critical <mark>or for smaller datasets</mark>, but transformers are the go-to solution for most modern ML applications.

### Summary Table:

| Aspect                      | RNNs                               | Transformers                     |
| --------------------------- | ---------------------------------- | -------------------------------- |
| **Architecture**            | Sequential (step-by-step)          | Parallel (process all at once)   |
| **Attention**               | Implicit through hidden states     | Explicit via self-attention      |
| **Parallelization**         | Not parallelizable                 | Fully parallelizable             |
| **Handling Long Sequences** | Struggles with long dependencies   | Excellent with long dependencies |
| **Efficiency**              | Slower training                    | Faster due to parallelization    |
| **Scalability**             | Poor scalability to long sequences | Scalable but memory-intensive    |
| **Use Cases**               | Time-series, small datasets        | NLP, large datasets, vision      |
### 1. **Architecture**
- **RNNs**: 
  - RNNs <mark>process data sequentially,</mark> one time step at a time. They maintain a <mark>hidden state</mark> that is updated as the model processes each token in the sequence, making them suitable for time-dependent tasks.
  - Common variants include **[LSTM](#lstm) (Long Short-Term Memory)** and **GRU ([Gated Recurrent Units](#gated-recurrent-units))**, which are designed to capture <mark>long-term dependencies</mark> more effectively.

- **[Transformer](#transformer)**:
  - Transformers do not process data sequentially. Instead, they <mark>process the entire sequence in parallel</mark>, allowing them to <mark>model relationships between tokens regardless of their position</mark>. This is achieved through the **self-attention mechanism**.
  - Transformers include **positional encodings** to account for the order of tokens, since their architecture doesn't have an inherent understanding of sequence order.

### 2. **Processing Mechanism**
- **RNNs**:
  - RNNs depend on the previous hidden state to process the next token, which means they inherently process information sequentially.
  - The <mark>hidden state is updated at each time step</mark>, which can lead to issues like **[vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem)**, especially in long sequences, making it <mark>difficult for RNNs to capture long-range dependencies</mark>.

- **Transformers**:
  - Transformers use [Attention mechanism](#attention-mechanism) to allow each token to interact directly with every other token in the sequence. This allows transformers to capture long-range dependencies more effectively and efficiently.
  - The self-attention mechanism enables **<mark>parallelization</mark>** of the computation for all tokens in the sequence, which speeds up training and inference.

### 3. **Parallelization and Efficiency**
- **RNNs**:
  - Since RNNs must process sequences one step at a time, they <mark>cannot be easily parallelized</mark>. This makes them less efficient, especially for long sequences.
  - RNNs are also slower to train because of this sequential dependency.

- **Transformers**:
  - Transformers can process entire sequences at once, making it easier to parallelize computation, especially on GPUs. This leads to much faster training times compared to RNNs.
  - This parallelization is a major reason transformers have become the preferred model in large-scale tasks.

### 4. **Handling Long-Term Dependencies**
- **RNNs**:
  - RNNs often struggle with capturing long-term dependencies because the information must be passed through multiple time steps, which can lead to forgetting or corruption of information over long sequences.
  - <mark>LSTMs and GRUs were developed to mitigate this problem, but they are still not as effective as transformers for capturing long-range relationships.</mark>

- **Transformers**:
  - The self-attention mechanism in transformers allows the model to directly connect tokens from distant parts of the sequence. This makes transformers much better at modeling long-range dependencies.
  - Transformers can also model relationships across sequences regardless of their length, leading to better performance on tasks requiring a global understanding of the data.

### 5. **Memory and Scalability**
- **RNNs**:
  - RNNs are relatively <mark>more memory-efficient for shorter sequences</mark> but become inefficient for longer ones due to their sequential nature and the need to store hidden states at each time step.
  - They also scale poorly to long sequences or large datasets because of the need to compute one step at a time.

- **Transformers**:
  - Transformers, while faster, require more memory due to the computation of attention matrices, which scale quadratically with the sequence length. This can be a bottleneck for very long sequences or resource-constrained environments.
  - New transformer variants (e.g., **[Longformer](#longformer)** or **[Reformer](#reformer)**) have been introduced to improve memory efficiency for longer sequences.

### 6. **Use Cases**
- **RNNs**:
  - Traditionally used for [time-series data](#time-series-data), **speech recognition**, and **sequence generation** tasks.
  - They are useful when the order of data is crucial and when handling smaller datasets or shorter sequences.

### 7. **Performance**
- **RNNs**:
  - While effective in small-scale, low-latency tasks, RNNs often perform worse than transformers on complex tasks that involve large-scale data or long-range dependencies.

- **Transformers**:
  - Transformers have significantly outperformed RNNs in most tasks requiring sequential data processing, particularly in NLP. Pre-trained models like **[BERT](#bert)**, **GPT**, and **T5** are based on the transformer architecture and have set state-of-the-art results in many benchmarks.

Transformer-based models like [BERT](#bert) and GPT outperform traditional RNNs in NLP tasks for several key reasons:

1. **Parallelization**: Unlike [Recurrent Neural Networks|RNNs](#recurrent-neural-networksrnns), which process sequences sequentially (one time step at a time), transformers can process entire sequences in parallel. This significantly speeds up training and allows for more efficient use of computational resources.

2. **Self-[Attention Mechanism](#attention-mechanism)**: Transformers utilize a self-attention mechanism that enables them to weigh the importance of different words in a sentence relative to each other. This allows the model to capture long-range dependencies and relationships between words more effectively than RNNs, which often struggle with long-term dependencies due to their sequential nature.

3. **Handling Long Sequences**: RNNs, especially vanilla ones, can suffer from issues like vanishing and exploding gradients, making it difficult to learn from long sequences. Transformers, on the other hand, can directly connect tokens from distant parts of the sequence, making them much better at modeling long-range dependencies.

4. **Multi-Head Attention**: Transformers employ multi-head attention, which allows the model to focus on different parts of the input sequence simultaneously. Each attention head can learn different aspects of the relationships between words, enhancing the model's ability to understand context and meaning.

5. **Positional Encoding**: Since transformers do not inherently understand the order of words, they use positional encoding to inject information about the position of each word in the sequence. This allows them to maintain the sequential nature of language while still benefiting from parallel processing.

6. **Scalability and Performance**: Transformers have shown to be more scalable and perform better on large datasets, which is crucial for many NLP tasks. Pre-trained models like BERT and GPT have set state-of-the-art results in various benchmarks due to their architecture and training methodologies.

The combination of parallel processing, self-attention mechanisms, and the ability to handle long-range dependencies makes transformer-based models like BERT and GPT significantly more effective than traditional RNNs in NLP tasks. 

For further reading, you can refer to the note on [Transformers vs RNNs](#transformers-vs-rnns) for a detailed comparison of their architectures and performance.

#### Sources:
- [Transformer](obsidian://open?vault=content&file=Transformer)
- [Transformers vs RNNs](obsidian://open?vault=content&file=Transformers%20vs%20RNNs)
- [BERT](obsidian://open?vault=content&file=BERT)
- [LSTM](obsidian://open?vault=content&file=LSTM)
- [Attention mechanism](obsidian://open?vault=content&file=Attention%20mechanism)
- [Mathematical Reasoning in Transformers](obsidian://open?vault=content&file=Mathematical%20Reasoning%20in%20Transformers)
- [Recurrent Neural Networks](obsidian://open?vault=content&file=Recurrent%20Neural%20Networks)
- [Transfer Learning](obsidian://open?vault=content&file=Transfer%20Learning)
- [Multi-head attention](obsidian://open?vault=content&file=Multi-head%20attention)
- [BERT Pretraining of Deep Bidirectional Transformers for Language Understanding](obsidian://open?vault=content&file=BERT%20Pretraining%20of%20Deep%20Bidirectional%20Transformers%20for%20Language%20Understanding)
- [LLM](obsidian://open?vault=content&file=LLM)
- [Evaluating Language Models](obsidian://open?vault=content&file=Evaluating%20Language%20Models)
- [Bert Pretraining](obsidian://open?vault=content&file=Bert%20Pretraining)
- [Reasoning tokens](obsidian://open?vault=content&file=Reasoning%20tokens)
- [NLP](obsidian://open?vault=content&file=NLP)
- [Hugging Face](obsidian://open?vault=content&file=Hugging%20Face)
- [Questions](obsidian://open?vault=content&file=Questions)
- [Neural network](obsidian://open?vault=content&file=Neural%20network)
- [Boosting](obsidian://open?vault=content&file=Boosting)
- [Named Entity Recognition](obsidian://open?vault=content&file=Named%20Entity%20Recognition)
- [Deep Learning](obsidian://open?vault=content&file=Deep%20Learning)
- [Language Model Output Optimisation](obsidian://open?vault=content&file=Language%20Model%20Output%20Optimisation)
- [Small Language Models](obsidian://open?vault=content&file=Small%20Language%20Models)

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



<a id="typescript"></a>
# Typescript {#typescript}


Superset of JavaScript adding static typing and object-oriented features for building large-scale applications.

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

<a id="t-sne"></a>
# T Sne {#t-sne}


t-SNE (t-distributed Stochastic Neighbor Embedding) is a [Dimensionality Reduction](#dimensionality-reduction) technique used primarily for visualizing high-dimensional data. Unlike methods such as **[Principal Component Analysis|PCA](#principal-component-analysispca)** (Principal Component Analysis), which are linear, t-SNE is a **non-linear** method that excels at preserving the local structure of the data. 

### Key Characteristics of t-SNE:
- **Non-linear Mapping**: It attempts to capture non-linear relationships in the data by embedding it in a lower-dimensional space (usually 2D or 3D).
- **Local Similarities**: t-SNE preserves the local structure of the data. This means that points that are close in the high-dimensional space remain close in the lower-dimensional space.
- **Global Structure**: t-SNE may distort global structures to focus more on local relationships, which is both a strength and limitation.
  
### How t-SNE Works:
1. **Pairwise Similarities**: t-SNE first calculates pairwise similarities between data points in the high-dimensional space.
2. **Probability Distribution**: These similarities are transformed into probabilities representing how likely it is that two points are neighbors.
3. **Lower-Dimensional Mapping**: t-SNE tries to replicate this distribution of neighbors in the lower-dimensional space by iteratively adjusting the positions of the points.

### Applications:
- **Data Visualization**: t-SNE is widely used in data visualization, especially when exploring clusters or patterns in high-dimensional datasets.
- **Exploratory Data Analysis (EDA)**: It helps in finding clusters or subgroups in complex datasets, such as in genomics, image processing, or natural language processing.

### Limitations:
- **Computationally Intensive**: t-SNE can be slow and resource-heavy, particularly on large datasets.
- **Random Initialization**: Results can vary due to its sensitivity to initialization and the perplexity parameter (which controls how t-SNE balances attention between local and global data structure).
- **Difficult to Interpret**: While t-SNE is great for visualization, interpreting the precise distances and positions of points can be tricky.

### example


```python
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Applying t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting the results
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('t-SNE visualization of Iris dataset')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()

```


![Pasted image 20241015211844.png](../content/images/Pasted%20image%2020241015211844.png)

# Bandit: A Security Linter for Python

## Resources
- [Bandit Documentation](https://bandit.readthedocs.io/en/latest/)

## How to Use Bandit

Installation
To install Bandit, use pip by running the following command in your terminal:
```bash
pip install bandit
```
Running Bandit
After installation, you can run Bandit on your Python files or directories. For example, to scan a file named `example.py`, use:
```bash
bandit example.py
```
This command will analyze the file and report any security issues it finds.
### 3. Customizing Bandit
You can customize Bandit's behavior by specifying options. For example, to scan a directory and exclude certain subdirectories, use:

```bash
bandit -r example_directory -x example_directory/venv
```

- `-r` specifies the directory to scan.
- `-x` specifies directories to exclude.

### 4. Example Script
Here's a simple Python script that Bandit can analyze:

```python
import subprocess

user_input = input("Enter your name: ")
subprocess.call(["echo", user_input])
```

This script takes user input and passes it to the `echo` command using `subprocess.call()`. This can be dangerous as it might allow command injection.

To analyze the script, run:
```bash
bandit example.py
```
Bandit will generate a report highlighting potential security issues. For the script above, it might flag the use of `subprocess.call()` as a potential injection vector.

Fixing Issues
Based on Bandit's report, you can modify your code to fix vulnerabilities. For example, to mitigate the risk of command injection, you can set `shell=False`:

```python
import subprocess

user_input = input("Enter your name: ")
subprocess.call(f"echo {user_input}", shell=True)
```
Then rerun [Bandit example output](#bandit-example-output)
## Example Script for Bandit Analysis

In [ML_Tools](#ml_tools) see: [Bandit_Example_Nonfixed.py](#bandit_example_nonfixedpy)

Features Demonstrated: [Common Security Vulnerabilities in Software Development](#common-security-vulnerabilities-in-software-development)
1. **Command Injection**: The `dangerous_subprocess` function uses `subprocess.call` with `shell=True`, which can lead to command injection if user input is not properly sanitized.
2. **Hardcoded Password**: The `hardcoded_password` function contains a hardcoded password, which is a common security issue.
3. **Use of `eval`**: The `unsafe_eval` function uses `eval`, which can execute arbitrary code if the input is not controlled.
4. **Insecure Deserialization**: The `insecure_deserialization` function uses `pickle.loads`, which can be exploited if untrusted data is deserialized.

### Running Bandit on the Example Script
To analyze this script with Bandit, save it as `example.py` and run:

```bash
bandit example.py
```
Bandit will generate a report highlighting the security issues in the script, providing insights into how each feature can be potentially exploited and suggesting ways to mitigate these risks.

By following these steps, you can use Bandit to identify and address security vulnerabilities in your Python code. Remember, while Bandit is a powerful tool, it's important to complement it with good coding practices and thorough security testing.

<a id="toolruff"></a>
# Tool.Ruff {#toolruff}

Ruff is a fast [Python](#python) linter and code formatter.

It is designed to enforce coding style and catch potential errors in Python code. 

Ruff aims to be efficient and comprehensive, supporting a wide range of linting rules and style checks. It can be used to automatically format code to adhere to a specified style guide, making it a useful tool for maintaining consistent code quality across a project.

## in [TOML](#toml)

have:

```toml
[tool.ruff.format]
# Like Black, use double quotes for strings.
quote-style = "double"
# Like Black, indent with spaces, rather than tabs.
indent-style = "space"
# Like Black, respect magic trailing commas.
skip-magic-trailing-comma = false
# Like Black, automatically detect the appropriate line ending.
line-ending = "auto"
```


<a id="tooluv"></a>
# Tool.Uv {#tooluv}


Appears in [TOML](#toml) file

Link: https://github.com/astral-sh/uv

---

`uv` is a  tool for managing Python development [Virtual environments](#virtual-environments), projects, and dependencies. It offers a range of features that streamline various aspects of Python development, from installing Python itself to managing projects and dependencies. 

1. **Python Version Management**: `uv` allows you to install, list, find, pin, and uninstall Python versions. This is useful for managing multiple Python versions across different projects, ensuring compatibility and ease of switching between environments.

2. **Script Execution**: You can run standalone Python scripts and manage their dependencies directly with `uv`. This simplifies the process of executing scripts with specific dependencies without needing a full project setup.

3. **Project Management**: `uv` provides commands to create new projects, manage dependencies, sync environments, and build and publish projects. This is particularly useful for maintaining consistent environments and dependencies across development, testing, and production stages.

4. **Tool Management**: It supports running and installing tools from Python package indexes, making it easier to integrate tools like linters and formatters into your workflow.

5. **Pip Interface**: `uv` offers a pip-like interface for managing packages and environments, which can be used in legacy workflows or when more granular control is needed.

6. **Utility Commands**: It includes utility commands for managing cache, directories, and performing self-updates, which help maintain the tool's efficiency and keep it up-to-date.



<a id="topic-modeling"></a>
# Topic Modeling {#topic-modeling}



<a id="transfer_learningpy"></a>
# Transfer_Learning.Py {#transfer_learningpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/Neural_Network/transfer_learning.py

For deep learning, to do  [Transfer Learning](#transfer-learning) we take out and replace a few end layers of the network. We can then train just the last layer of weights of a neural network. 

The number of layers to remove and then added from pretrained depends on the similarity between tasks. Higher layers in networks are able to recognise higher detail components. 