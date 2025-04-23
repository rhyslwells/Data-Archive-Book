

# Imbalanced Datasets {#imbalanced-datasets}


Handling imbalanced datasets to ensure robustness of models is a common challenge in machine learning, particularly in classification tasks where one class significantly outnumbers the other(s). 

In [Classification](#classification) tasks, an imbalanced dataset can lead to a model that <mark>performs well on the majority class but poorly on the minority class</mark>. This is because the model may learn to predict the majority class more often due to its prevalence. 

For [Regression](#regression) tasks, handling outliers or data skewness might be necessary.

In [ML_Tools](#ml_tools) see:
- [Imbalanced_Datasets_SMOTE.py](#imbalanced_datasets_smotepy)
### Examples

Consider a scenario where you have an imbalanced dataset of resumes, with a majority of male resumes and a minority of female resumes. You want to build a model to predict gender based on resume features.
## Strategies to address imbalances
### Data-Level Approaches

Resampling Techniques:
  - Oversampling: Increase the number of instances in the minority class by duplicating existing samples or generating new ones using techniques like [SMOTE (Synthetic Minority Over-sampling Technique)](#smote-synthetic-minority-over-sampling-technique).
  - Undersampling: Reduce the number of instances in the majority class by randomly removing samples. This can help balance the dataset but may lead to loss of important information.
  - Data Augmentation: Apply transformations to existing data to create new samples, which is particularly useful in image data. Techniques include rotation, flipping, scaling, and cropping.

### Algorithm-Level Approaches

- [Cost-Sensitive Analysis](#cost-sensitive-analysis) / Cost-Sensitive Learning: Modify the learning algorithm to give more importance to the minority class. This can be done by assigning higher misclassification costs to the minority class during training.1. You have a perfectly balanced dataset but still experience poor classification accuracy. Why might the class separability be the issue?
- Ensemble Methods: [Bagging](#bagging) and Boosting: Use ensemble techniques like [Random Forests](#random-forests) or AdaBoost, which can be adapted to handle class imbalance by adjusting the sample weights or using balanced bootstrap samples.

### Evaluation Metrics & Others

- [Model Evaluation](#model-evaluation)/[Evaluation Metrics](#evaluation-metrics)Use Appropriate Metrics: Instead of accuracy, use metrics that are more informative for [imbalanced datasets](#imbalanced-datasets), such as precision, recall, F1-score, and the area under the ROC curve (AUC-ROC).
- [Anomaly Detection](#anomaly-detection) Models: Treat the minority class as anomalies and use [anomaly detection](#anomaly-detection) techniques to identify them.
- [Transfer Learning](#transfer-learning): Use pre-trained models that have learned features from a balanced dataset, which can be fine-tuned on the imbalanced dataset.



# Imbalanced_Datasets_Smote.Py {#imbalanced_datasets_smotepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Imbalanced_Datasets_SMOTE.py

### Demonstrating the Value of Resampling in Imbalanced Classification

This example highlights the effectiveness of resampling techniques, such as [SMOTE (Synthetic Minority Over-sampling Technique)|SMOTE](#smote-synthetic-minority-over-sampling-techniquesmote), in addressing [Imbalanced Datasets|class imbalance](#imbalanced-datasetsclass-imbalance) issues in classification tasks. By implementing the following strategies, the setup ensures a measurable improvement in model performance:

1. **Severe Imbalance and Dataset Size**:
    - Utilizing a larger dataset with a severe imbalance ratio (e.g., 99:1) makes the impact of resampling more apparent. This imbalance necessitates resampling for the model to predict the minority class accurately.

2. **Choice of Classifier**:
    - Switching from robust classifiers like [Random Forests](#random-forests) to more sensitive ones like [Logistic Regression](#logistic-regression) or Support Vector Machine ([Support Vector Machines|SVM](#support-vector-machinessvm)) highlights the benefits of resampling. These simpler models struggle with imbalance, providing a clear contrast between resampling and non-resampling scenarios.

3. **Feature Overlap**:
    - Ensuring overlap in the feature space between minority and majority classes enhances the effectiveness of synthetic resampling techniques, such as SMOTE.

4. **Focus on Minority Class Metrics**:
    - Emphasizing evaluation metrics like [recall](#recall) and F1-score for the minority class explicitly measures the model's ability to capture minority class instances, demonstrating the value of resampling in improving these metrics.

### Results:

#### Without Resampling:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.99      | 1.00   | 1.00     | 990     |
| 1     | 0.67      | 0.20   | 0.31     | 10      |
| **Accuracy** |       |        | 0.99     | 1000    |
| **Macro Avg** | 0.83      | 0.60   | 0.65     | 1000    |
| **Weighted Avg** | 0.99      | 0.99   | 0.99     | 1000    |

- The minority class recall will likely be very low (close to 0), as the classifier may predict the majority class almost exclusively.
- Overall [accuracy](#accuracy) will be high because the majority class dominates.

#### With SMOTE Resampling:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 1.00      | 0.82   | 0.90     | 990     |
| 1     | 0.04      | 0.70   | 0.07     | 10      |
| **Accuracy** |       |        | 0.82     | 1000    |
| **Macro Avg** | 0.52      | 0.76   | 0.49     | 1000    |
| **Weighted Avg** | 0.99      | 0.82   | 0.89     | 1000    |

- Minority class recall and F1-score should improve significantly, as SMOTE provides synthetic samples to balance the training set.
- Accuracy might decrease slightly due to more emphasis on minority class performance.




# Immutable Vs Mutable {#immutable-vs-mutable}

[Python](#python)

list being mutable

Side effect
```python
def get_largest_numbers(numbers, n):
numbers. sort()

return numbers[-n:]

nums [2, 3, 4, 1,34, 123, 321, 1]

print(nums)
largest = get_largest_numbers(nums, 2)
print(nums)
```


# Impact Of Multicollinearity On Model Parameters {#impact-of-multicollinearity-on-model-parameters}

See https://youtu.be/StSAJIZuqws?t=655

```R
)

# Monte Carlo Simulation: Multicollinearity & Harm
results = expand_grid(
rho = seq(0, 0.95, 0.05),
rep = 1:1000
) %>%
mutate(
sim = map(rho, function(p) {

set.seed(runif(1, 1, 10000) %>% ceiling)
R = matrix(c(1, p, p, 1), nrow = 2, ncol = 2, byrow = TRUE)
Sigma = cor2cov(R, c(1, 1))

data = MASS :: mvrnorm(n= 30, mu = c(0, 0), Sigma = Sigma) %>%
as_tibble %>%
mutate( Y=1+0.5*V1+0.5*V2+ rnorm(30) )

model = 1m(Y ~ V1 + V2, data = data)

summary(model) $coefficients %>% as_tibble
})
```



# Implementing Database Schema {#implementing-database-schema}

To manage and create a database schema in SQLite, you can use the following commands:

- To view all commands used to create a database, execute:
  ```
  .schema
  ```
- To view the schema for a specific table, use:
  ```
  .schema table
  ```
- To run a schema from a file, use:
  ```
  .read schema.sql
  ```

## Creating a Database Schema

When creating a database schema, follow these steps:

1. **Identify the Tables**: Determine which tables are necessary for your data.
2. **Define Columns**: Specify the columns for each table.
3. **Choose Data Types**: Select appropriate data types for each column.
4. **Establish Keys**: Define primary and foreign keys to maintain data integrity.
5. **Set Column Constraints**: Ensure that values adhere to specified conditions.

Note that constraints do not need to apply to primary and foreign keys. Common constraints include:

- `CHECK`: Ensures values meet certain criteria (e.g., amount must be greater than 0).
- `DEFAULT`: Sets a default value for a column.
- `NOT NULL`: Ensures a column cannot have a NULL value.
- `UNIQUE`: Ensures all values in a column are distinct.

### Example Schema Creation

Here’s an example of how to create tables for a database:

```sql
CREATE TABLE cards (
    "id" INTEGER PRIMARY KEY
);

CREATE TABLE stations (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT UNIQUE NOT NULL,
    "line" TEXT NOT NULL
);

CREATE TABLE swipes (
    "id" INTEGER PRIMARY KEY,
    "card_id" INTEGER,
    "station_id" INTEGER,
    "type" TEXT NOT NULL CHECK("type" IN ('enter', 'exit', 'deposit')),
    "datetime" NUMERIC NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "amount" NUMERIC NOT NULL CHECK("amount" != 0),
    FOREIGN KEY("card_id") REFERENCES "cards"("id"),
    FOREIGN KEY("station_id") REFERENCES "stations"("id")
);
```

## Modifying the Schema

To change an existing schema, you can use commands such as `RENAME`, `ADD COLUMN`, and `DROP COLUMN`. 

```sql
ALTER TABLE visits
RENAME TO swipes;

ALTER TABLE swipes
ADD COLUMN "swipetype" TEXT;

DROP TABLE "riders"; 
```

## Relating Entities

Done using foreign keys.

```sql
CREATE TABLE visits (
    "rider_id" INTEGER,
    "station_id" INTEGER,
    FOREIGN KEY("rider_id") REFERENCES "riders"("id"),
    FOREIGN KEY("station_id") REFERENCES "stations"("id")
);
```

Also See:
- [Many-to-Many Relationships](#many-to-many-relationships)
- [ER Diagrams](#er-diagrams)

# In Ner How Would You Handle Ambiguous Entities {#in-ner-how-would-you-handle-ambiguous-entities}

Handling ambiguous entities in Named Entity Recognition (NER) can be quite challenging. Here are some strategies that can be employed:

1. **Contextual Analysis**: Utilize the surrounding <mark>context</mark> of the ambiguous entity to determine its correct classification. For example, the word "Apple" could refer to the fruit or the company, but the context in which it appears can help disambiguate its meaning.

2. **Disambiguation Models**: Implement additional models specifically designed for entity disambiguation. These models can leverage knowledge bases or ontologies to determine the most likely entity based on context.

3. **Multi-label Classification**: Instead of forcing a single label, allow for multiple possible labels for ambiguous entities. This can be useful in cases where an entity might belong to more than one category.

4. **Training Data**: Ensure that the training dataset includes examples of ambiguous entities in various contexts. This can help the model learn to recognize and differentiate between them.

5. **User Feedback**: Incorporate user feedback mechanisms to refine the model's predictions. If users can correct or confirm entity classifications, this can improve the model over time.


# Industries Of Interest {#industries-of-interest}


Industries to investigate related to my background & interests:
- [Energy](#energy)
- [Telecommunications](#telecommunications)
- [Education and Training](#education-and-training)

Both Reinforcement Learning and Explainable AI offer exciting opportunities for mathematicians to contribute significantly. Your deep mathematical understanding allows you to tackle complex problems, develop new methodologies, and provide theoretical foundations for emerging techniques.

Exploratory Questions
- [What algorithms or models are used within the energy sector](#what-algorithms-or-models-are-used-within-the-energy-sector)
- [What algorithms or models are used within the telecommunication sector](#what-algorithms-or-models-are-used-within-the-telecommunication-sector)

### [Reinforcement learning](#reinforcement-learning)

- **Stochastic Processes**: Your background will allow you to delve into the mathematical properties of [Markov Decision Processes](#markov-decision-processes) MDPs, optimizing transition dynamics, and improving algorithms based on theoretical insights.
- **Theoretical Analysis**: You can contribute to the development of new algorithms by providing theoretical proofs of convergence and performance guarantees, applying concepts from real analysis and optimization.
- **Complexity Analysis**: Understanding the computational complexity of various RL algorithms and contributing to the design of more efficient algorithms will leverage your mathematical skills.




# Input Is Not Properly Sanitized {#input-is-not-properly-sanitized}


### Input is Not Properly Sanitized

When we say that <mark>"input is not properly sanitized,"</mark> it means that the input data from users or external sources is not being adequately checked or cleaned before being processed by the application. Proper sanitization involves validating and filtering input to ensure it is safe and expected, preventing malicious data from causing harm. Without proper sanitization, applications can be vulnerable to various attacks, such as:

- **Command Injection**: Malicious commands can be executed on the server.
- **SQL Injection**: Malicious SQL queries can be executed against a database.
- **Cross-Site Scripting (XSS)**: Malicious scripts can be injected into web pages.

Sanitization typically involves:
- Validating input against expected formats or values.
- Escaping special characters that could be interpreted as code.
- Removing or encoding potentially harmful content.

# Interpreting Logistic Regression Model Parameters {#interpreting-logistic-regression-model-parameters}

How do this in terms of odds, probabilities ,odds ratio.

[Logistic Regression](#logistic-regression)


 


# Interquartile Range (Iqr) Detection {#interquartile-range-iqr-detection}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Outliers/outliers_IQR.py

Context:  

The IQR method is a robust and widely used statistical technique for identifying outliers, especially in [univariate data](#univariate-data). It is based on the distribution of data and is less sensitive to extreme values compared to methods reliant on mean and standard deviation.

Steps:
- Compute the IQR:
    - The IQR is the range within which the central 50% of the data lies.
    - Formula:  
        $\text{IQR} = Q3 - Q1$  
        where:
        - $Q1$: The first quartile (25th percentile)
        - $Q3$: The third quartile (75th percentile).
          
- Determine the bounds:
    - Define lower and upper bounds to detect potential outliers:  
        $\text{Lower Bound} = Q1 - 1.5 \cdot \text{IQR}$  
        $\text{Upper Bound} = Q3 + 1.5 \cdot \text{IQR}$
        
- Identify anomalies:
    - Any data point outside the lower or upper bounds is flagged as an anomaly.

Applications:
- Best suited for non-Gaussian distributions.
- Commonly used in boxplots for visualizing outliers.

# Isolated Forest {#isolated-forest}



 Isolation Forest (iForest) is an [Model Ensemble](#model-ensemble)-based method used for anomaly detection. It operates by isolating data points using a series of random binary splits.
 
The key idea is that [standardised/Outliers|anomalies](#standardisedoutliersanomalies), being rare and different, are easier to isolate and thus require fewer splits. 

Mathematically, the isolation of a point is captured by the path length in a decision tree, where shorter paths indicate anomalies. The algorithm constructs multiple isolation trees, and the <mark>anomaly score of a point</mark> is determined by the average path length across all trees. 

Isolation Forest is highly efficient for large datasets and is particularly useful when the assumption is that anomalies are rare and distinct from normal instances.

**Steps:**
- Randomly select a feature and a split value between the maximum and minimum values of that feature.
- Repeat this process to create a tree structure.
- Anomalies are isolated faster than normal points, leading to shorter path lengths in the tree.
- The average path length across multiple trees is used to compute an anomaly score.


 Key Components:  
 - **Isolation Trees (iTrees)**: Binary trees where the goal is to isolate observations based on randomly chosen features and split values.  
 - **Anomaly Score**: Calculated based on the average path length across all isolation trees.  
 - **Path Length**: Anomalies tend to have shorter path lengths as they are easier to isolate.  
 - **Random Splitting**: Random feature selection and splitting result in the separation of instances, with fewer splits isolating anomalies.

Important
 - Anomalies are identified based on shorter average path lengths in the isolation forest, <mark>indicating that fewer splits are needed to isolate them.</mark>  
 - The method scales well with large datasets because it relies on randomly generated trees, avoiding complex distance or density computations.
 - Isolation Forest assumes that anomalies are few and distinct; it may perform poorly when anomalies are not easily distinguishable.  
 - The method is sensitive to the [Hyperparameter](#hyperparameter) such as the number of trees and sample size.

Follow up questions
 - How does the isolation forest compare to density-based methods like [DBSCAN](#dbscan) in terms of detecting complex anomalies?  [Anomaly Detection with Clustering](#anomaly-detection-with-clustering)
 - What impact does the choice of sample size have on the performance and accuracy of isolation forests in high-dimensional data?

Related Topics
 - [Random Forests](#random-forests) for classification and regression  
 - One-Class [Support Vector Machines|SVM](#support-vector-machinessvm) for anomaly detection


# Imperative {#imperative}


An **imperative** pipeline tells <mark>_how_ to proceed</mark> at each step in a procedural manner. 

In contrast, a **[declarative](term/declarative.md)** data pipeline does not tell the order it needs to be executed but instead <mark>allows each step/task to find the best time and way to run.</mark> 

The *how* should be taken care of by the tool, framework, or platform running on. 

For example, update an asset when upstream data has changed. 

Both approaches result in the same output. 

However, the declarative approach benefits from **leveraging compile-time query planners** and **considering runtime statistics** to choose the best way to compute and find patterns to reduce the amount of transformed data.





# In Memory Format {#in-memory-format}


The term "in-memory format" refers to the way data is stored and managed directly in a <mark>computer's RAM</mark> (Random Access Memory) rather than on disk storage like a hard drive or SSD. This approach is used to optimize performance, as accessing data in RAM is significantly faster than accessing data on disk.

In-memory formats are often used in applications that require high-speed data processing, such as real-time analytics, caching systems, and certain types of databases (e.g., in-memory databases like Redis or SAP HANA). By keeping data in memory, these systems can reduce latency and improve throughput, enabling faster data retrieval and processing.

In-memory formats may involve <mark>specific data structures</mark> or serialization methods that are optimized for quick access and manipulation in RAM. These formats are designed to make efficient use of memory resources while ensuring that data can be quickly read and written by the application.

In-memory formats are optimized to:
- hit fast instruction sets 
- be cache friendly 
- be parallelizable

Formats:
- [Apache Arrow](term/apache%20arrow.md) 
- [Apache Spark](Apache%20Spark.md)
- [NumPy](term/numpy.md)
- [Pandas](term/pandas.md)

The opposed to in-memory formats are [Data Lake File Formats](Data%20Lake%20File%20Formats) which save space, be cross-language and serve as long-term storage. 




# Incremental Synchronization {#incremental-synchronization}



# Inference Versus Prediction {#inference-versus-prediction}


**[inference](#inference)** is similar to **prediction**, <mark>but in the context of **[Generative AI](#generative-ai)**,</mark> it is more specific to the application of a pre-trained model to <mark>produce an output from new input data</mark>. 

While **[prediction](#prediction)** often refers to tasks like classification or regression, inferencing in **Gen AI** refers to generating novel outputs, such as text, images, or audio, based on learned patterns.

The key distinction is that in **generative models**, inferencing not only predicts but <mark>**creates new data**</mark> (like text or images) rather than assigning categories or predicting numerical values, as in traditional machine learning models. For example:
- In a **language model** [LLM](#llm), inferencing is generating the next word or sentence in a text.
- In a **text-to-image model**, inferencing produces an image based on a textual description.



# Inference {#inference}


Inferencing involves prediction, but the output is more generative and creative in nature.

[inference versus prediction](#inference-versus-prediction)

# Information Theory {#information-theory}



Information theory is a mathematical framework for quantifying the transmission, processing, and storage of information. 

Information theory has profound implications and applications across various domains, providing the theoretical foundation for understanding and optimizing how information is communicated and processed.

1. **Entropy**: Often referred to as Shannon entropy, it measures the average amount of uncertainty or surprise associated with random variables. In essence, it quantifies the amount of information contained in a message or dataset.

2. **Information**: In information theory, <mark>information is defined as the reduction in uncertainty</mark>. When you receive a message, the amount of information it provides is related to how much it reduces your uncertainty about the subject.

3. **Mutual Information**: This measures the amount of information that two random variables share. It quantifies the reduction in uncertainty about one variable given knowledge of the other.

4. **Channel Capacity**: This is the maximum rate at which information can be reliably transmitted over a communication channel. It is determined by the channel's bandwidth and noise characteristics.

5. **Data Compression**: Information theory provides the basis for data compression techniques, which aim to reduce the size of data without losing essential information. Lossless compression (e.g., ZIP) and lossy compression (e.g., JPEG) are two types of compression.

6. **Error Detection and Correction**: Information theory also deals with methods for detecting and correcting errors in data transmission, ensuring that information can be accurately received even in the presence of noise.

7. **Rate-Distortion Theory**: This aspect of information theory deals with the trade-offs between the fidelity of data representation and the amount of compression, which is crucial in applications like audio and video compression.



# Interoperable {#interoperable}



# Interpretability {#interpretability}


# Links
1. [Interpretability Importance](https://christophm.github.io/interpretable-ml-book/interpretability-importance.html)
2. https://christophm.github.io/interpretable-ml-book/index.html

# Interpretability

Interpretability in machine learning (ML) is about understanding the reasoning behind a model's predictions. It involves making the model's decision-making process comprehensible to humans, which is crucial for trust, debugging, and ensuring fairness and reliability. 

Importance of Interpretability
- **Trust**: Stakeholders are more likely to trust models they understand.
- **Debugging**: Easier to identify and fix issues in interpretable models.
- **Bias Detection**: Helps identify biases in data and model predictions.
- **Social Acceptance**: Models that can explain their decisions are more socially acceptable.
- **Fairness and Reliability**: Ensures models are fair and reliable, especially in high-impact areas.

Levels of Interpretability
- **Global, Holistic Model Interpretability**: Involves comprehending the entire model at once, including feature importance and interactions. This level of interpretability is challenging, especially for models with many parameters.
- **Global Model Interpretability on a Modular Level**: Focuses on understanding parts of the model (e.g., weights in linear models or splits in decision trees). While individual parameters may be interpretable, their interdependence complicates interpretation.
- **Local Interpretability for a Single Prediction**: Allows for detailed examination of why a model made a specific prediction for an individual instance. This can provide clearer insights as local predictions may exhibit simpler relationships than the global model.

Challenges in Achieving Interpretability
- Effective interpretation requires **<mark>context</mark>**; for instance, understanding the significance of weights in linear models is often conditional on other feature values.
- **Trade-offs**: Users must weigh the need for predictions against the need for understanding the rationale, particularly in contexts where decisions have significant consequences.
- **Human Learning**: Interpretability supports human curiosity, facilitating updates to mental models based on new information.
- **Safety and Bias Detection**: Essential for high-risk applications (e.g., self-driving cars) and for identifying biases in decision-making.
- **Social Acceptance**: Machines that explain their decisions tend to be more accepted.

---
# Properties of Explanations

These properties provide a framework for evaluating and comparing explanation methods in interpretable machine learning, ensuring they are effective and useful for understanding model predictions. 

### Properties of Explanation Methods

1. **Expressive Power**: Refers to the types of explanations generated (e.g., rules, decision trees, natural language).

2. **Translucency**: Measures the extent to which an explanation method examines the model's internal parameters. High translucency allows for more informative explanations, while low translucency enhances portability.

3. **Portability**: Indicates the range of models compatible with the explanation method. Methods that treat models as black boxes (e.g., surrogate models) are more portable.

4. **Algorithmic Complexity**: Reflects the computational demands of generating explanations, which is crucial when processing time is a concern.

### Properties of Explanations

1. **Accuracy**: Assesses how well the explanation predicts unseen data. High accuracy is vital if the explanation is used in place of the model.

2. **Fidelity**: Evaluates how closely the explanation matches the black box model's predictions. High fidelity is essential; otherwise, the explanation is ineffective.

3. **Consistency**: Measures how similar explanations are across models trained on the same task. High consistency is desirable when models rely on similar relationships.

4. **Stability**: Examines how consistent explanations are for similar instances. High stability is preferred to avoid erratic changes due to minor variations in input features.

5. **Comprehensibility**: Assesses how easily humans understand the explanations. This property is challenging to define but is critical for effective communication of model behavior.

6. **Certainty**: Indicates whether the explanation reflects the model's confidence in its predictions, adding value by clarifying prediction reliability.

7. **Degree of Importance**: Evaluates how well the explanation identifies the importance of features involved in a decision.

8. **Novelty**: Addresses whether the instance to be explained lies outside the training data distribution, affecting prediction accuracy.

9. **Representativeness**: Measures how many instances an explanation covers, ranging from individual predictions to broader model interpretations.

# Understanding an Explanation

Here are the key takeaways on human-friendly explanations in interpretable machine learning:

Need comprehensibility and accuracy in explanations to enhance user understanding and trust in machine learning models. 
### Importance of Human-Friendly Explanations
1. **Preference for Short Explanations**: Humans favor concise explanations (1-2 causes) that contrast current situations with hypothetical scenarios where the event did not occur.

2. **Nature of Explanations**: An explanation answers "why" questions, focusing primarily on everyday situations rather than general scientific queries.

### Characteristics of Good Explanations
1. **Contrastive Nature**: Good explanations highlight differences between predicted outcomes, aiding comprehension. For instance, explaining why a loan was rejected by comparing it to a hypothetical accepted application is more effective.

2. **Selective Focus**: People tend to prefer explanations that identify one or two key causes rather than exhaustive lists. This selective approach aligns with the "Rashomon Effect," where multiple valid explanations can exist for the same event.

3. **Social Context**: Explanations are influenced by the social context and audience. Tailoring explanations to the audience’s knowledge level enhances understanding.

4. **Emphasis on Abnormal Causes**: Humans focus on rare or abnormal causes to explain events. Including these in explanations can significantly enhance clarity.

5. **Truthfulness vs. Selectivity**: While truthfulness is important, selectivity often takes precedence. A concise, selective explanation is more impactful than a comprehensive but complex one.

6. **Consistency with Prior Beliefs**: Explanations that align with the explainee's existing beliefs are more readily accepted, highlighting the challenge of integrating complex model behaviors that contradict common intuitions.

7. **Generality**: Good explanations should be generalizable, but abnormal causes can sometimes provide more compelling insights.

### Implications for Interpretable Machine Learning
- **Design Considerations**: Create explanations that are short, contrastive, and tailored to the audience’s background.
- **Methodology**: Incorporate techniques that can produce contrastive explanations while maintaining accuracy and fidelity to the model's predictions.
- **Audience Awareness**: Understanding the audience's social context and prior beliefs is crucial for effective communication of model outcomes.

- Understand how the model makes predictions.
- Use techniques like feature importance scores or LIME to explain individual predictions.

- **How can we design machine learning models that are both accurate and interpretable?** While deep learning models often achieve high accuracy, their complexity can make them difficult to interpret. This raises questions about how to balance accuracy and interpretability. Exploring techniques for visualizing and understanding the internal representations learned by deep networks, or developing inherently interpretable models that still achieve high performance, could lead to greater trust and adoption of machine learning in critical applications like healthcare and finance.

# Interview Notepad {#interview-notepad}


Tell about a recent project of yours;; Collaborating in the image matching kaggle competition. Obviously S2DS project too.
<!--SR:!2024-04-15,4,270-->

What are some areas in this business you are interested in?;;
Technical consulting, energy projects. Any area with the room to problem solve, to apply the scientific method to business problems. Areas where data can be turned into decisions. Building technical systems too for operations (feasibility projects).

How do you approach prioritizing tasks in a data science project?;; Current project objectives, complexity, dependencies, and client impact.
<!--SR:!2024-04-15,4,270-->

How do you handle conflicts or disagreements within a data science team?;; I promote open communication, facilitate discussions, and seek win-win solutions to resolve conflicts constructively.

How do you stay updated with the latest developments and trends in data science?;; I would like to attend events, talk to peers, stay connect to colleagues online. Watch updates on youtube.

How do you ensure the quality and reliability of data used in your data science projects? ;; I implement data validation and cleaning procedures, conduct exploratory data analysis, and would collaborate with domain experts  and engineers to verify data accuracy and relevance.
<!--SR:!2024-04-15,4,277-->

Can you describe a time when you had to make a decision under tight deadlines in a data science project? ;; In S2DS the client wanted further features to be added, so we said no nicely.
<!--SR:!2024-04-15,4,270-->

What strategies do you employ to ensure alignment between data science initiatives and business objectives? ;; I collaborate closely with stakeholders to understand business goals, prioritize projects based on their strategic importance, and regularly communicate progress and results to ensure alignment and maximize impact.

How do you stay organized and manage deadlines in your data science projects? ;; I utilize project management tools, break down tasks into manageable components, set realistic timelines, and regularly reassess priorities to ensure timely delivery.
<!--SR:!2024-04-15,4,277-->

Can you discuss a challenging problem you encountered in a data science project and how you resolved it? ;; In alice in wonderland people issue,, I analyzed the root cause, consulted domain experts or literature, experimented with alternative approaches, and iteratively refined solutions until achieving satisfactory results.

What do you think is the most important thing in a team?;; Buy in, communication, also initiative.

What do you think is a no-go in a team?;; Lack of accountability/blaming people, just own your mistakes and learn from them its more productive

### Business and situational questions

Tell about a recent project of yours;; Collaborating in the image matching kaggle competition. Obviously S2DS project too.
<!--SR:!2024-04-15,4,270-->

What are some areas in this business you are interested in?;;
Technical consulting, [Energy](#energy) projects. Any area with the room to problem solve, to apply the scientific method to business problems. Areas where data can be turned into decisions. Building technical systems too for operations (feasibility projects).

How do you approach prioritizing tasks in a data science project?;; Current project objectives, complexity, dependencies, and client impact.
<!--SR:!2024-04-15,4,270-->

How do you handle conflicts or disagreements within a data science team?;; I promote open communication, facilitate discussions, and seek win-win solutions to resolve conflicts constructively.

How do you stay updated with the latest developments and trends in data science?;; I would like to attend events, talk to peers, stay connect to colleagues online. Watch updates on youtube.

How do you ensure the quality and reliability of data used in your data science projects? ;; I implement data validation and cleaning procedures, conduct exploratory data analysis, and would collaborate with domain experts  and engineers to verify data accuracy and relevance.
<!--SR:!2024-04-15,4,277-->

Can you describe a time when you had to make a decision under tight deadlines in a data science project? ;; In S2DS the client wanted further features to be added, so we said no nicely.
<!--SR:!2024-04-15,4,270-->

What strategies do you employ to ensure alignment between data science initiatives and business objectives? ;; I collaborate closely with stakeholders to understand business goals, prioritize projects based on their strategic importance, and regularly communicate progress and results to ensure alignment and maximize impact.

How do you stay organized and manage deadlines in your data science projects? ;; I utilize project management tools, break down tasks into manageable components, set realistic timelines, and regularly reassess priorities to ensure timely delivery.
<!--SR:!2024-04-15,4,277-->

Can you discuss a challenging problem you encountered in a data science project and how you resolved it? ;; In alice in wonderland people issue,, I analyzed the root cause, consulted domain experts or literature, experimented with alternative approaches, and iteratively refined solutions until achieving satisfactory results.
### Team work questions

What do you think is the most important thing in a team?;; Buy in, communication, also initiative.

What do you think is a no-go in a team?;; Lack of accountability/blaming people, just own your mistakes and learn from them its more productive
### General questions

What are some areas of the DS field you are interested in?;; NLP (the techniques machines use to understand complex concepts), time series analysis (very real, forecasting).

Why are you interested in data science?;; Problem solving aspect, with tools that are technically interesting. Work with technical minded people. I enjoy the scientific viewpoint.

How would you interact with the data science community?;; Participate in Datafest, Kaggle projects, and engage with colleagues.
<!--SR:!2024-04-15,4,277-->

	#interview_questions 

What is data normalization and why do we need it? ;; Data [Normalised Schema](#normalised-schema) is used in [Preprocessing](#preprocessing) for preprocessing as it rescales values to fit within a specific range.
   
Explain [Dimensionality Reduction](#dimensionality-reduction), where it’s used, and its benefits? ;; [Dimensionality Reduction](#dimensionality-reduction) involves reducing the number of feature variables by obtaining a set of principal variables, reducing storage space, speeding up computation, removing redundant features, and enabling data visualization to identify patterns.
   
How do you handle missing or corrupted data in a dataset? ;; Missing or corrupted data can be handled by dropping affected rows or columns, replacing them with another value, or filling them with a placeholder value using methods like isnull(), dropna(), or fillna() in Pandas.
<!--SR:!2024-04-15,4,270-->
   
How would you go about doing an exploratory data analysis (EDA)? ;; EDA involves gaining insights from data before applying predictive models, starting with high-level global insights, dropping unnecessary columns, filling missing values, and creating basic visualizations such as bar plots and scatter plots to understand feature relationships.
   
How do you know which machine learning model you should use? ;; [Model Selection](#model-selection) depends on factors such as the nature of the problem, data characteristics, and desired outcomes, often involving trial-and-error.
   
Explain your phd and its outcomes. ;; Looking for a counter examples, what are FCJ. Built algorithims to compute, then computed them
<!--SR:!2024-04-15,4,270-->


Q2. What is the difference between Type I vs Type II error? ;; Type I error occurs when the null hypothesis is true, but it is rejected. Type II error occurs when the null hypothesis is false, but it is not rejected.
<!--SR:!2024-04-12,1,234-->

Q3. What is [Linear Regression](#linear-regression)?;;Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables. 

What do the terms P-value, coefficient, R-Squared value mean? ;;P-value indicates the significance of the coefficient, coefficient represents the strength and direction of the relationship, and R-Squared value measures the proportion of variance explained by the model.
<!--SR:!2024-04-15,4,270-->

What are the assumptions required for [Linear Regression](#linear-regression)? ;; The assumptions include a linear relationship between dependent and independent variables, normally distributed and independent errors, minimal multicollinearity among explanatory variables.

What is a statistical interaction? ;; Statistical interaction occurs when the effect of one variable on a dependent variable is dependent on the value of another variable.

What is selection bias? ;; Selection bias refers to a systematic error in sampling that results in a sample that is <mark>not representative of the population</mark>, leading to incorrect conclusions about the population.
<!--SR:!2024-04-12,1,230-->

# Ipynb {#ipynb}


### Printing without code : [Documentation & Meetings](#documentation--meetings)/ [nbconvert](#nbconvert)

https://stackoverflow.com/questions/49907455/hide-code-when-exporting-jupyter-notebook-to-html

jupyter nbconvert stock_analysis.ipynb --no-input --to pdf

jupyter nbconvert --to html --no-input --no-prompt phi_analysis.ipynb

--clear -output

jupyter nbconvert --to html --TemplateExporter.exclude_input=True Querying.ipynb


