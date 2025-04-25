# Web Map Tile Service (Wmts) {#web-map-tile-service-wmts}

[GIS](#gis)

### Web Map Tile Service (WMTS)

**Purpose**: WMTS is designed to serve pre-rendered, cached image tiles of maps.

**Functionality**:
- **Tile-Based**: It serves map images as small, fixed-size tiles, usually in a format such as PNG or JPEG.
- **Performance**: By using cached tiles, WMTS can quickly deliver map images, making it highly efficient for applications requiring fast map rendering, like web mapping applications.
- **Scalability**: The tile-based approach allows for easy scaling and efficient handling of high load, as the same tiles can be reused for multiple requests.
- **Standardization**: It is standardized by the Open Geospatial Consortium (OGC), ensuring interoperability between different systems and software.

# Webpages Relevant {#webpages-relevant}

Using bookmarks:
#### [Time Series](#time-series)

https://aeturrell.com/blog/posts/time-series-explosion/?utm_source=substack&utm_medium=email

https://otexts.com/fpp3/?utm_source=substack&utm_medium=email#

# What Algorithms Or Models Are Used Within The Energy Sector {#what-algorithms-or-models-are-used-within-the-energy-sector}





# What Algorithms Or Models Are Used Within The Telecommunication Sector {#what-algorithms-or-models-are-used-within-the-telecommunication-sector}





# What Are The Best Practices For Evaluating The Effectiveness Of Different Prompts {#what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts}



# What Can Abm Solve Within The Energy Sector {#what-can-abm-solve-within-the-energy-sector}



[Agent-Based Modelling](#agent-based-modelling)

energy systems analysis

# What Is The Difference Between Odds And Probability {#what-is-the-difference-between-odds-and-probability}





# What Is The Role Of Gradient Based Optimization In Training Deep Learning Models. {#what-is-the-role-of-gradient-based-optimization-in-training-deep-learning-models}





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



# Why And When Is Feature Scaling Necessary {#why-and-when-is-feature-scaling-necessary}  

[Feature Scaling](#feature-scaling) is useful for models that use distances like [Support Vector Machines|SVM](#support-vector-machinessvm) and [K-means](#k-means)
### When Scaling Is Unnecessary

1. **Tree-based Algorithms:**
    - Algorithms like [Decision Tree](#decision-tree), [Random Forests](#random-forests), and Gradient Boosted Trees are invariant to feature scaling because they split data based on thresholds, not distances.
    - Example: Splits are determined by feature values, not their magnitude.

2. **Data with Uniform Scales:**
    - If all features have the same range or are already normalized (e.g., percentages), scaling may not be required.



# Why Does Increasing The Number Of Models In A Ensemble Not Necessarily Improve The Accuracy {#why-does-increasing-the-number-of-models-in-a-ensemble-not-necessarily-improve-the-accuracy}


Increasing the number of models in an ensemble ([Model Ensemble](#model-ensemble)) does not always lead to improved accuracy due to several limiting factors:

- **Convergence of Predictions**: Additional models may lead to similar predictions, resulting in minimal changes to the overall output.
- **Limited Data Representation**: If the dataset is noisy or incomplete, more models will only aggregate existing noise without capturing new patterns.
- **Diminishing Returns**: Each new model contributes less unique information, and performance is ultimately limited by the irreducible error in the data.
- **Increased Complexity**: More models increase computational costs and training times without necessarily improving accuracy.
- **Overfitting Risk**: Adding complex models can lead to overfitting, where the ensemble learns noise instead of underlying patterns.

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

Let's assume you are predicting house prices, and you're using a linear regression model where the {% raw %}`town`{% endraw %} feature is the only predictor (along with some other features like {% raw %}`area`{% endraw %}).

- **With Label Encoding**:
    - The model will interpret the encoded numeric values (0 for West Windsor, 1 for Robbinsville, and 2 for Princeton) and might incorrectly assume a relationship such as: "Princeton" (2) is somehow numerically "higher" than "West Windsor" (0), which doesn’t reflect any meaningful relationship.
    - This can lead to biased coefficients and, therefore, inaccurate predictions.        

- **With One-Hot Encoding**:
    - The model will learn the effect of each category (West Windsor, Robbinsville, and Princeton) as a separate feature, with no assumption of ordinality.
    - This often results in more accurate predictions, especially when categorical features have no inherent order.


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




# Why Is Named Entity Recognition (Ner) A Challenging Task {#why-is-named-entity-recognition-ner-a-challenging-task}

Named Entity Recognition (NER) is considered a challenging task for several reasons:      

1. **Ambiguity**: Entities can be ambiguous, meaning the same word or phrase can refer to different entities depending on the context. For example, "Washington" could refer to a city, a state, or a person. Disambiguating these entities requires a deep understanding of context.

2. **Variability in Language**: Natural language is highly variable and can include slang, idioms, and different syntactic structures. This variability makes it difficult for NER models to consistently identify entities across different texts.

3. **Named Entity Diversity**: Entities can take many forms, including names, organizations, locations, dates, and more. Each type may have different characteristics, requiring the model to adapt to various patterns.

4. **Lack of Annotated Data**: High-quality annotated datasets are crucial for training NER models. However, creating such datasets can be time-consuming and expensive, leading to limited training data for certain domains or languages.

5. **Multilingual Challenges**: NER systems often struggle with multilingual texts, where the same entity may be represented differently in different languages. This adds complexity to the recognition process.

6. **Nested Entities**: In some cases, entities can be nested within each other (e.g., "The University of California, Berkeley"). Recognizing such nested structures can be particularly challenging for NER systems.

7. **Domain-Specific Language**: Different domains (e.g., medical, legal, technical) may have specific terminologies and entities that general NER models may not recognize effectively without domain-specific training.

# Why Is The Central Limit Theorem Important When Working With Small Sample Sizes {#why-is-the-central-limit-theorem-important-when-working-with-small-sample-sizes}

The [Central Limit Theorem](#central-limit-theorem) (CLT) is particularly important for data scientists working with small sample sizes. It enables the use of various statistical methods, and helps in making valid inferences about the population from limited data.      

1. **Assumption of Normality**: The CLT states that the sampling [Distributions|distribution](#distributionsdistribution) of the sample means will approximate a normal distribution, regardless of the underlying population distribution, as long as the sample size is sufficiently large.
2.
3. This is crucial for data scientists because many statistical methods and tests (such as t-tests, ANOVA, and regression analysis) rely on the [assumption of normality](#assumption-of-normality). Even with small sample sizes, the CLT provides a foundation for making inferences about the population.

4. **Confidence Intervals and [Hypothesis Testing](#hypothesis-testing)**: The CLT enables data scientists to construct confidence intervals and perform hypothesis tests even when the sample size is small. By using the sample mean and the standard error (which is derived from the sample size), data scientists can estimate the range within which the true population mean is likely to fall, and test hypotheses about population parameters.

5. **Reduction of Variability**: The variance of the sampling distribution decreases as the sample size increases, which means that larger samples provide more reliable estimates of the population mean. For small sample sizes, the CLT helps data scientists understand the potential variability in their estimates and make more informed decisions based on their data.

6. **Practical Application**: In many real-world scenarios, obtaining large samples may not be feasible due to time, cost, or logistical constraints. The CLT allows data scientists to work with smaller samples while still applying statistical techniques that assume normality, thus broadening the scope of analysis.

7. **Robustness of Results**: The CLT provides a theoretical justification for the robustness of statistical methods. Even if the original data is not normally distributed, the means of sufficiently large samples will tend to be normally distributed, allowing for more reliable conclusions.


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



# Why Use Er Diagrams {#why-use-er-diagrams}

[Why use ER diagrams](#why-use-er-diagrams)

Cleaning a dataset before creating an [ER Diagrams](#er-diagrams) is crucial for ensuring accuracy and reliability in your database design

1. [Data Quality](#data-quality): Cleaning the dataset helps identify and rectify errors, inconsistencies, and missing values. This ensures that the data accurately represents the real-world entities and relationships you intend to model.

2. [Normalised Schema](#normalised-schema): Before creating an ER diagram, it's essential to normalize the data, which involves organizing it efficiently to reduce redundancy and dependency. Cleaning the dataset beforehand allows you to identify redundant information and eliminate it, leading to a more streamlined ER diagram.

3. Entity Identification: Through data cleaning, you can properly identify the entities within your dataset. This involves determining which attributes belong to which entity, as well as identifying any composite or derived attributes. Proper entity identification is fundamental to creating an accurate ER diagram.

4. Relationship Clarity: Cleaning the dataset helps clarify the relationships between entities. By ensuring that the data accurately reflects the relationships between different entities, you can create a more precise ER diagram that accurately represents the connections between various elements.

5. Data Consistency: [Data Cleansing](#data-cleansing) ensures consistency across the dataset, which is essential for maintaining integrity in the ER diagram. Consistent data allows for clearer identification of relationships and attributes, leading to a more effective database design.

# Wikipedia_Api.Py {#wikipedia_apipy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Utilities/Wikipedia_API.py  

# Windows Subsystem For Linux {#windows-subsystem-for-linux}

[Windows Subsystem for Linux](#windows-subsystem-for-linux) (WSL) is a compatibility layer for running Linux binary executables natively on Windows 10 and Windows 11. It allows users to run a Linux environment directly on Windows without the need for a virtual machine or dual-boot setup.

Key features of WSL include:

1. **Integration with Windows**: Users can access files from both Windows and the [Linux](#linux) environment seamlessly.
2. **Multiple Distributions**: WSL supports various Linux distributions, such as [Ubuntu](#ubuntu), Debian, and Fedora, which can be installed from the Microsoft Store.
3. **Command-Line Tools**: Users can run Linux command-line tools and applications directly in Windows, making it easier for developers to work in a familiar environment.
4. **Performance**: WSL provides near-native performance for Linux applications, making it suitable for development and testing.


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

# Wordnet {#wordnet}



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

{% raw %}```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
```
{% endraw %}#### Step 3: Prepare Your Data

Split your dataset into training and testing sets:

{% raw %}```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 
```
{% endraw %}
#### Step 4: Convert Data to DMatrix

Convert the data into DMatrix, the optimized data structure used by XGBoost:

{% raw %}```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
```
{% endraw %}
#### Step 5: Set Parameters

Define the parameters for the XGBoost model:

{% raw %}```python
params = {
    'max_depth': 6,
    'eta': 0.1,
    'objective': 'binary:logistic',  # Use 'reg:squarederror' for regression tasks        
    'eval_metric': 'logloss'
}
```
{% endraw %}
#### Step 6: Train the Model

Train the XGBoost model using the training data:
{% raw %}```python
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)
```
{% endraw %}
#### Step 7: Make Predictions and Evaluate
Make predictions on the test set and evaluate the model's performance:

{% raw %}```python
y_pred = bst.predict(dtest)
y_pred_binary = [1 if y > 0.5 else 0 for y in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy:.2f}")
```
{% endraw %}
# Notes

Set up an example of XGBoost. Plot the paramater space slices "Min_Samples_split", "Max_Depth" vs accuracy.

{% raw %}```python
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 10)
xgb_model.best_itersation
```
{% endraw %}
# Yaml {#yaml}


Stands for [YAML ain't markup language](https://github.com/yaml/yaml-spec) and is a superset of JSON

- lists begin with a hyphen
- dependent on whitespace / indentation
- better suited for configuration than [Json](#json)

YAML is a data serialization language often used to write configuration files. Depending on whom you ask, YAML stands for yet another markup language, or YAML isn’t markup language (a recursive acronym), which emphasizes that YAML is for data, not documents.

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
{% raw %}```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)  # Rescales each feature to [0, 1]
```
{% endraw %}
# Z Score {#z-score}


Z-scores standardize a value relative to a distribution by measuring how many standard deviations it is from the mean. This is useful for [standardised/Outliers|Outliers](#standardisedoutliersoutliers) and [Normalisation](#normalisation).

Definition:
The Z-score of a value $x$ is given by:
{% raw %}    $$Z = \frac{x - \bar{x}}{s}$$
{% endraw %}
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
error: error while generating page "book/book.md":
Template render error: (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\book\book.md) [Line 9041, Column 6]
  unknown block tag: endmath
    at Object._prettifyError (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\nunjucks\src\lib.js:32:11)
    at Template.render (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\nunjucks\src\environment.js:442:21)
    at Environment.renderString (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\nunjucks\src\environment.js:313:17)
    at Promise.apply (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\q\q.js:1185:26)
    at Promise.promise.promiseDispatch (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\q\q.js:808:41)
    at C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\q\q.js:1411:14
    at runSingle (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\q\q.js:137:13)    at flush (C:\Users\RhysL\Desktop\Data-Archive-Book\DAB\node_modules\q\q.js:125:13)    
    at process.processTicksAndRejections (node:internal/process/task_queues:77:11)        

C:\Users\RhysL\Desktop\Data-Archive-Book\DAB>  













