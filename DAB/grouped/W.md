# W

## Table of Contents
* [WCSS and elbow method](#wcss-and-elbow-method)
* [Weak Learners](#weak-learners)
* [Web Feature Server (WFS)](#web-feature-server-wfs)
* [Web Map Tile Service (WMTS)](#web-map-tile-service-wmts)
* [Webpages relevant](#webpages-relevant)
* [What algorithms or models are used within the energy sector](#what-algorithms-or-models-are-used-within-the-energy-sector)
* [What algorithms or models are used within the telecommunication sector](#what-algorithms-or-models-are-used-within-the-telecommunication-sector)
* [What are the best practices for evaluating the effectiveness of different prompts](#what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts)
* [What can ABM solve within the energy sector](#what-can-abm-solve-within-the-energy-sector)
* [What is the difference between odds and probability](#what-is-the-difference-between-odds-and-probability)
* [What is the role of gradient-based optimization in training deep learning models.](#what-is-the-role-of-gradient-based-optimization-in-training-deep-learning-models)
* [When and why not to us regularisation](#when-and-why-not-to-us-regularisation)
* [Why JSON is Better than Pickle for Untrusted Data](#why-json-is-better-than-pickle-for-untrusted-data)
* [Why Type 1 and Type 2 matter](#why-type-1-and-type-2-matter)
* [Why and when is feature scaling necessary](#why-and-when-is-feature-scaling-necessary)
* [Why does increasing the number of models in a ensemble not necessarily improve the accuracy](#why-does-increasing-the-number-of-models-in-a-ensemble-not-necessarily-improve-the-accuracy)
* [Why does label encoding give different predictions from one-hot encoding](#why-does-label-encoding-give-different-predictions-from-one-hot-encoding)
* [Why does the Adam Optimizer converge](#)
* [Why is named entity recognition (NER) a challenging task](#why-is-named-entity-recognition-ner-a-challenging-task)
* [Why is the Central Limit Theorem important when working with small sample sizes](#why-is-the-central-limit-theorem-important-when-working-with-small-sample-sizes)
* [Why use ER diagrams](#why-use-er-diagrams)
* [Wikipedia_API.py](#wikipedia_apipy)
* [Windows Subsystem for Linux](#windows-subsystem-for-linux)
* [Word2Vec.py](#word2vecpy)
* [Word2vec](#word2vec)
* [WordNet](#wordnet)
* [Wrapper Methods](#wrapper-methods)



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

