

# Ml Engineer {#ml-engineer}

ML Engineer
  - Configures and optimizes production ML models.
  - Monitors the performance and accuracy of ML models in production environments.

# Mnist {#mnist}

[Datasets](#datasets)

# Machine Learning Algorithms {#machine-learning-algorithms}


 Machine learning [Algorithms](#algorithms) are used to automate tasks, extract insights, and make more informed decisions.

Choosing the right algorithm for a specific problem involves understanding the task, the characteristics of the data, and the strengths and limitations of different algorithms.
# [Supervised Learning](#supervised-learning)

Common [Classification](#classification) algorithms include:

- [Logistic Regression](#logistic-regression)
- [Support Vector Machines](#support-vector-machines)
- [Naive Bayes](#naive-bayes)
- [Decision Tree](#decision-tree)
- [Random Forests](#random-forests)

Common [Regression](#regression) algorithms include:
- [Linear Regression](#linear-regression)
- [Support Vector Regression](#support-vector-regression)
- [Random Forest Regression](#random-forest-regression)

# [Unsupervised Learning](#unsupervised-learning)

Common [Clustering](#clustering) algorithms include:

- [K-means](#k-means)
- [Gaussian Mixture Models](#gaussian-mixture-models)
- [Clustering](#clustering)
- [Dimensionality Reduction](#dimensionality-reduction)

Common [Dimensionality Reduction](#dimensionality-reduction) algorithms include:

- [Principal Component Analysis](#principal-component-analysis)
- [Manifold Learning](#manifold-learning)
## Strengths and Limitations of Machine Learning Algorithms

##### Strengths:

Automation: Machine learning algorithms can automate complex tasks, freeing up human resources for other activities.

Adaptability: Machine learning algorithms can adapt to changing data patterns, making them suitable for dynamic environments.

[Scalability](#scalability): Machine learning algorithms can handle large datasets efficiently, making them applicable to big data problems.

Knowledge Discovery: Machine learning algorithms can help discover hidden patterns and relationships in data, leading to new insights and knowledge.

##### Limitations:

Data Dependence: The performance of machine learning algorithms heavily depends on the [Data Quality](#data-quality) and quantity of the training data.

[Overfitting](#overfitting) occurs when the model learns the training data too well and fails to generalise to new, unseen data.

[Bias and variance](#bias-and-variance): Machine learning algorithms can be biased, reflecting the biases present in the training data.

[Interpretability](#interpretability): Some machine learning algorithms, especially deep learning models, can be complex and difficult to interpret, making it challenging to understand the reasoning behind their predictions.

# Machine Learning Operations {#machine-learning-operations}


Machine Learning Operations (MLOps) is a set of practices and tools designed to streamline the entire lifecycle of machine learning models, from development to deployment and maintenance. It aims to integrate machine learning with [DevOps](#devops) principles to ensure that models are reliable, scalable, and efficient in production environments. 

1. Development: MLOps focuses on creating a seamless workflow for developing machine learning models. This includes data preprocessing, feature engineering, model building, and training. The goal is to ensure that models can be developed quickly and efficiently. See [DS & ML Portal](#ds--ml-portal).

2. Deployment: Once a model is developed and evaluated, MLOps facilitates its deployment into a production environment. This involves setting up the necessary infrastructure to serve the model and ensuring that it can handle real-world data and workloads.

3. Maintenance: MLOps emphasizes the importance of monitoring and maintaining models over time. This includes tracking model performance, detecting data drift, and retraining models as needed to ensure they remain accurate and relevant.

4. Generalization and Robustness: MLOps aims to create models that generalize well to new, unseen data, especially in dynamic environments. It also focuses on ensuring models remain robust to noisy or unexpected data inputs.

5. Collaboration and Automation: MLOps encourages collaboration between data scientists, engineers, and operations teams. It also leverages automation to streamline repetitive tasks, such as model training, evaluation, and deployment.

6. [Model Observability](#model-observability) and Retraining: Continuous monitoring of model performance is crucial in MLOps. Observability tools help track metrics and identify when a model needs retraining due to changes in data patterns or performance degradation.

# Machine Learning {#machine-learning}


Machine learning (ML) is a type of artificial intelligence (AI) that allows software applications to become more accurate at predicting outcomes without being explicitly programmed to do so. Machine learning [Machine Learning Algorithms](#machine-learning-algorithms) use historical data as input to predict new output values.





















# Maintainable Code {#maintainable-code}

[Pydantic](#pydantic) : runtine analysis

[Pyright](#pyright): static analysis

[Testing](#testing)

Want robust and reliable Python applications.

# Makefile {#makefile}


A Makefile is a special file used by the `make` build automation tool to manage the build process of a project. It defines a set of tasks to be executed, typically to compile and link a program. Here are some key functions of a Makefile:

1. **Compilation Instructions**: It specifies how to compile and link the program. This includes defining the source files, the compiler to use, and any necessary flags or options.

2. **Dependencies**: Makefiles list dependencies between files, ensuring that changes in source files trigger recompilation of only the necessary parts of the program.

3. **Automation**: It automates repetitive tasks, such as cleaning up build artifacts, running tests, or deploying software.

4. **Targets and Rules**: A Makefile consists of targets, dependencies, and rules. A target is usually a file to be generated, dependencies are files that the target depends on, and rules are the commands to create the target from the dependencies.

5. **Variables**: Makefiles can use variables to simplify and manage complex build processes, making it easier to maintain and modify.

## Example

```makefile
# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -g

# Target executable
TARGET = myprogram

# Source files
SRCS = main.cpp utils.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Rule to build the target executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

# Rule to build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean
```

### Explanation:

- **CXX and CXXFLAGS**: These variables define the compiler and the flags used during compilation.
- **TARGET**: The name of the final executable.
- **SRCS and OBJS**: Lists of source and object files. The `OBJS` variable is automatically generated by replacing `.cpp` with `.o` in the `SRCS` list.
- **all**: The default target that builds the executable.
- **$(TARGET)**: This rule specifies how to link object files into the final executable.
- **%.o: %.cpp**: A pattern rule to compile each `.cpp` file into a `.o` object file.
- **clean**: A target to remove all object files and the executable, useful for cleaning up the build directory.
- **.PHONY**: Declares `all` and `clean` as phony targets, meaning they are not actual files but just names for commands to run.

### Running it

If you run this Makefile using the `make` command in a terminal, here's what would happen:

1. **Compilation**: The `make` tool will look for a file named `Makefile` in the current directory. It will then execute the default target, which is `all` in this case.

2. **Building the Executable**: 
   - `make` will check if the target executable `myprogram` needs to be built. It does this by comparing the timestamps of the source files (`main.cpp`, `utils.cpp`) and the corresponding object files (`main.o`, `utils.o`).
   - If any of the source files are newer than their corresponding object files, or if the object files do not exist, `make` will compile the source files into object files using the rule `%.o: %.cpp`.
   - Once the object files are up-to-date, `make` will link them together to create the `myprogram` executable using the rule for `$(TARGET)`.

3. **Output**: During this process, you'll see the compilation and linking commands being executed in the terminal. If there are any errors in the source code, the compiler will output error messages.

4. **Clean Up**: If you run `make clean`, it will execute the `clean` target, which removes the object files and the executable, cleaning up the build directory.


# Manifold Learning {#manifold-learning}


**Manifold learning** is a powerful approach for high-dimensional data exploration, focusing on uncovering the lower-dimensional manifold that the data resides on. These algorithms aim to identify and map the underlying low-dimensional structure, or **manifold**, that the data is assumed to lie on, within the high-dimensional space. This is particularly useful for reducing dimensionality while preserving the intrinsic properties of the data.

Methods like **<mark>Isomap</mark>** aim to preserve the geodesic distances between points, which better represent the data's true structure than straight-line distances in high-dimensional space. This enables effective [Dimensionality Reduction](#dimensionality-reduction) for non-linear data while preserving important relationships between data points.

### Key Concepts of Manifold Learning:

1. **High-Dimensional Data**:
   - In many machine learning problems, data can have a high number of dimensions (features), making it challenging to analyze directly. However, often this high-dimensional data lies on a much simpler, lower-dimensional structure, or **manifold**, embedded in the high-dimensional space. Manifold learning seeks to find and represent this lower-dimensional structure, simplifying the analysis and visualization of complex datasets.

2. **Manifold Assumption**:
   - Manifold learning assumes that although the data may appear high-dimensional, the <mark>true degrees of freedom are much fewer.</mark> This means the data can be represented in a lower-dimensional space without losing important information about its structure.

3. **Geodesic Distances**:
   - In manifold learning, the goal is often to <mark>preserve certain distances or relationships between data points</mark>. **Isomap**, for example, is a popular manifold learning algorithm that aims to preserve **geodesic distances**—the shortest paths between points along the manifold. These distances represent the true relationships between points in the underlying lower-dimensional space, even though they may seem far apart in the high-dimensional space.

4. [Dimensionality Reduction](#dimensionality-reduction)
   - Like other dimensionality reduction techniques (such as PCA), manifold learning helps reduce the number of features in the data. However, manifold learning is particularly effective when the data is non-linear, meaning traditional linear techniques like PCA might not capture the true underlying structure. Algorithms like **Isomap**, **Locally Linear Embedding (LLE)**, and **t-SNE** are examples of manifold learning methods that handle such <mark>non-linear structures</mark>.
### Example: Isomap

- **Isomap** is a manifold learning algorithm that tries to preserve the **geodesic distances** between all pairs of data points. It computes these distances by constructing a graph in which the edges represent the shortest path along the manifold (not the straight-line distance in high-dimensional space). Then, it uses these distances to map the data to a lower-dimensional space, retaining the structure of the original data.

![Pasted image 20240127124620.png|500](../content/images/Pasted%20image%2020240127124620.png|500)




### Many-to-Many Relationships

Occurs when multiple records in one table are associated with multiple records in another table.

Need to use a **junction table** (also known as a bridge table or associative entity). This table will contain foreign keys that reference the primary keys of the two tables involved in the relationship.

### Steps to Implement Many-to-Many Relationships

1. **Identify the Entities**: Determine the two entities that will participate in the many-to-many relationship. For example, consider `students` and `courses`.

2. **Create the Junction Table**: Create a new table that will serve as the junction table. This table will hold the foreign keys from both entities. In our example, we can create a table called `enrollments`.

3. **Define Foreign Keys**: In the junction table, define foreign keys that reference the primary keys of the two related tables. This establishes the relationship between the entities.

4. **Add Additional Attributes (if necessary)**: If needed, you can also include additional attributes in the junction table that are relevant to the relationship itself. For instance, you might want to track the enrollment date.

Example Schema

```sql
-- Create the students table
CREATE TABLE students (
    "id" INTEGER PRIMARY KEY,
    "name" TEXT NOT NULL,
    "email" TEXT UNIQUE NOT NULL
);

-- Create the courses table
CREATE TABLE courses (
    "id" INTEGER PRIMARY KEY,
    "title" TEXT NOT NULL,
    "description" TEXT
);

-- Create the junction table for the many-to-many relationship
CREATE TABLE enrollments (
    "student_id" INTEGER,
    "course_id" INTEGER,
    "enrollment_date" DATE NOT NULL,
    PRIMARY KEY("student_id", "course_id"), -- Composite primary key
    FOREIGN KEY("student_id") REFERENCES "students"("id"),
    FOREIGN KEY("course_id") REFERENCES "courses"("id")
);
```

A composite primary key (`student_id`, `course_id`) ensures that each student can enroll in a course only once.

To retrieve data from a many-to-many relationship, you can use SQL JOIN statements. For example, to find all courses a specific student is enrolled in, you can run:

```sql
SELECT courses.title
FROM courses
JOIN enrollments ON courses.id = enrollments.course_id
WHERE enrollments.student_id = 1;  -- Replace 1 with the desired student ID
```



[Many-to-Many Relationships](#many-to-many-relationships)
   - Records in Table A relate to multiple records in Table B, and vice versa.
   - <mark>Requires a junction table</mark> to manage the relationships.
   - Example: Students and Courses tables with a junction table Enrollments.

# Markov Decision Processes {#markov-decision-processes}


**Markov Decision Process ([Markov Decision Processes|MDP](#markov-decision-processesmdp))** is a formal framework for decision-making where outcomes depend solely on the current state (Markov property).
\

architecture 



[Markov Decision Processes](#markov-decision-processes) 
([Markov Decision Processes|MDP](#markov-decision-processesmdp)s): The mathematical framework for modelling decision-making, characterized by states, actions, transition probabilities, and rewards. Your understanding of probability theory and stochastic processes will be crucial here.

# Markov Chain {#markov-chain}

Is a stochastic model that describes a sequence of events in which the probability of each event depends only on the state attained in the previous event.


# Master Observability Datadog {#master-observability-datadog}

what happens in prod, pre prod.

monitoring web frontend.

how is infrastructure working in prod

Datadog

agents

tagging

profile how it was working versus other dates.

dashboards

[Lambdas](#lambdas)

logging




# Mathematical Reasoning In Transformers {#mathematical-reasoning-in-transformers}


**transformer-based models** that address mathematical reasoning either through pretraining, hybrid systems, or fine-tuning on specific mathematical tasks

- **Challenges**: General-purpose transformers [Transformer|Transformer](#transformertransformer) are trained primarily on large corpora of text, which include mathematical problems but lack systematic and rigorous math-specific training. This results in limited capabilities for handling complex calculations or abstract algebraic problems.

- **Grokking in Mathematical Reasoning**: This is an area of research where models are trained on small datasets of synthetic math problems to encourage **grokking**, a phenomenon where the model suddenly achieves near-perfect performance after extended training. Researchers are interested in how transformers might be able to **"[grok](#grok)"** math concepts after seeing many examples.

math data sets: MATH dataset,Aristo

Pretrained transformers on math specific data.

[GPT-f](#gpt-f) represents a significant advancement in the use of **transformer-based models for mathematical reasoning**,






# Mathematics {#mathematics}


[Johnson–Lindenstrauss lemma](#johnsonlindenstrauss-lemma)

[Big O Notation](#big-o-notation)

[Directed Acyclic Graph (DAG)](#directed-acyclic-graph-dag)

[information theory](#information-theory)

# Maximum Likelihood Estimation {#maximum-likelihood-estimation}

Resource:
- https://www.youtube.com/watch?v=YevSE6bRhTo

Used to infer [Model Parameters](#model-parameters) from collected data for example in [Linear Regression](#linear-regression) ($\beta_0,\beta_1$).

Definition: Likelihood

Why is it a good tool for guessing parameter values?

The likelihoods plot a distribution, the max gives the most likely.

This is called the MLE.

Properties of a MLE:
- As more data comes in the [Estimator](#estimator) should approach a true value
- MLE is a <mark>consistent</mark> [Estimator](#estimator), i.e it gets closer to the true parameter value as the sample size grows.
- Asymptotical Normal
- Asymptotic Efficiency

Assumptions for MLE:
- Regularity

[parametric vs non-parametric models](#parametric-vs-non-parametric-models)

Likelihood is a function of a parameter

# Mean Squared Error {#mean-squared-error}

<mark>Measures numerical proximity.</mark>



# Melt {#melt}


In pandas, the `melt` function is used to <mark>transform ([Data Transformation](#data-transformation)) a DataFrame from a wide format to a long format</mark>. This is especially useful for data analysis and visualization tasks where long-format data is preferred or required. The wide format typically has multiple columns for different variables, whereas the long format has a single column for variable names and a single column for values. 

Related:
- [Database Techniques](#database-techniques)
- [Turning a flat file into a database](#turning-a-flat-file-into-a-database)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/reshaping.ipynb
### Key Reasons to Use `melt`:

1. [Normalisation](#normalisation)
   - Wide to Long Transformation: `melt` helps in converting data with many columns (wide format) into a more normalized form with fewer columns (long format). This is useful for many statistical and visualization libraries that prefer long-format data.
   
1. Easier Analysis and [Data Visualisation](#data-visualisation):
   - Compatibility with Plotting Libraries: Many plotting libraries like `seaborn` and `ggplot` require data in a long format for creating certain types of plots, such as [Grouped plots](#grouped-plots).

1. Simplifying Complex Data Structures:
   - Handling [Multi-level index](#multi-level-index): If a DataFrame has multiple levels of columns, `melt` can help flatten this structure, making it easier to work with.
   
3. Preparation for Aggregation:
   - Facilitating [Groupby](#groupby) Operations: Long-format data is often more suitable for these.

### Parameters of `melt`:

- `id_vars`: Columns to use as identifier variables. These columns are kept as-is in the output.
- `value_vars`: Columns to unpivot. These columns are transformed into a single column.
- `var_name`: Name to use for the `variable` column in the output.
- `value_name`: Name to use for the `value` column in the output.

### Example Usage:

Consider a DataFrame in wide format:

```python
import pandas as pd

# Sample wide format data
data = {
    'id': [1, 2, 3],
    'math_score': [88, 92, 95],
    'science_score': [85, 90, 89],
    'english_score': [78, 85, 88]
}
df_wide = pd.DataFrame(data)
print(df_wide)
```

Output:

```
   id  math_score  science_score  english_score
0   1          88             85             78
1   2          92             90             85
2   3          95             89             88
```

To convert this wide-format DataFrame into a long-format DataFrame using `melt`:

```python
# Melt the DataFrame
df_long = pd.melt(df_wide, id_vars=['id'], 
                  value_vars=['math_score', 'science_score', 'english_score'], 
                  var_name='subject', value_name='score')
print(df_long)
```

Output:

```
   id       subject  score
0   1    math_score     88
1   2    math_score     92
2   3    math_score     95
3   1  science_score     85
4   2  science_score     90
5   3  science_score     89
6   1  english_score     78
7   2  english_score     85
8   3  english_score     88
```




# Memory Caching {#memory-caching}

Memory Caching
  - Use in-memory caches to store frequently accessed data closer to the user, reducing latency.

# Memory {#memory}


Memory in large [language models](#language-models) (LLMs) involves managing context windows to enhance reasoning capabilities without the high costs associated with traditional training methods. The goal of [Memory](#memory) is to address challenges like "forgetting," where LLMs struggle to retain context across interactions.
## Key Concepts:

**Forgetting Context**:

Understanding how and why LLMs lose context, especially in multi-turn dialogues, and its impact on response accuracy. Forgetting occurs due to the limitations of fixed **context windows**, manifesting differently in single-turn (immediate forgetting) versus multi-turn interactions (progressive loss of context).

**Prioritization of Context**:
Techniques for determining which parts of the context are most relevant and need to be retained, optimizing memory usage.

**Time Length of Memory**:
Balancing how long memory should be maintained to ensure it remains useful and relevant over time.

**Dynamic Memory Management**:
Adapting memory structures in real-time to accommodate evolving knowledge and interactions.

**In-Context Memory**:
Memory tied to specific interactions, making it more relevant and easier to apply in particular scenarios.

**Multi-turn Interactions**:
Addressing context retention across multiple interactions, emphasizing the importance of maintaining coherence over extended conversations.
## Types of Memory:

**Semantic Memory**:
Focuses on the meaning and [Semantic Relationships](#semantic-relationships) between concepts, which is crucial for improving LLM reasoning and context understanding.

**Hierarchical Memory**:
Balances immediate retrieval with long-term storage of information, enabling better performance in various applications.

Supports evolving and persistent memory systems tailored to specific tasks.

# Merge {#merge}



# Metadata Handling {#metadata-handling}



### Trimming

- Description: Removing data points identified as outliers based on criteria such as being beyond a certain number of standard deviations from the mean or outside a specified percentile range.
- Implementation Example:
  ```python
  lower_quantile = df["var1"].quantile(0.01)
  upper_quantile = df["var1"].quantile(0.99)
  df_no_outliers = df[(df["var1"] >= lower_quantile) & (df["var1"] <= upper_quantile)]
  ```

### Capping or Flooring

- Description: Setting a maximum or minimum threshold beyond which data points are considered outliers and replacing them with the threshold value.

### Winsorizing 

- Description: Similar to capping and flooring, winsorizing replaces extreme values with less extreme values within a specified range, typically using percentiles.

# Metric {#metric}


### Metrics in Machine Learning

[Evaluation Metrics](#evaluation-metrics)
[Regression Metrics](#regression-metrics)
### Metrics in business

A metric, also called [KPI](term/key%20performance%20indicator%20(kpi).md) or (calculated) measure, are terms that serve as the building blocks for how business performance is both measured and defined, as knowledge of how to define an organization's KPIs. It is fundamental to have a common understanding of them. Metrics usually surface as business reports and dashboards with direct access to the entire organization.

For example, think of <mark>operational metrics</mark> that represent your company's performance and service level or financial metrics that describe its financial health. 

Calculated measures are part of metrics and apply to specific [Dimensions](Dimensions.md) traditionally mapped inside a [Bus Matrix](term/bus%20matrix.md). 




# Microsoft Access {#microsoft-access}



### Tasks

- [ ] How to update (or more so insert) into multiple related tables. How to insert existing data into a [Database](#database). SQL triggers?
- [ ] Investigate: Helper table to gather many small tables into one.
- [ ] What are typical types of databases.
- [ ] Questions: Make a form that accesses multiple tables.
### Resources
[Tutorial](https://www.youtube.com/watch?v=ubmwp8kbfPc)

[Best Practices](https://www.youtube.com/watch?v=ymc9CYnziS4)

[LINK](https://youtu.be/ymc9CYnziS4)

[TIME](https://youtu.be/ymc9CYnziS4?t=1042)
### Notes

Why use access:
	Handles lots of data better than excel. Understand relationships between sources of data.
 
Querying:
	Can do querying. Which might be hard to do in excel.
	Graphical way to make queries.

Forms:
	Access can make it easier for user interfaces 
	forms- opening other forms, drop downs, user interface
	secure fields on forms so users can only see so much 

Features:
	Has user security.
	Has control over the types of data input.
	User control feature
	user friendly. 

Issues:
	Possible limitations when scale increases. Next steps, can upscale to SQL server. 







# Mini Batch Gradient Descent {#mini-batch-gradient-descent}



# Mixture Of Experts {#mixture-of-experts}

Different parts of the network focusing on parts of the questions

Routing, distribution

activating 






# Model Building {#model-building}


The Model Building phase follows the [Preprocessing](#preprocessing) phase, where data is organized and prepared for analysis. This phase focuses on selecting and setting up the appropriate machine learning models to solve the problem at hand.
## Key Steps

Types of Models:
- Choose a model to apply based on the problem requirements and data characteristics.
- Explore different [Machine Learning Algorithms](#machine-learning-algorithms) to find the best fit for your data.
- Consider the tradeoffs between [parametric vs nonparametric models](#parametric-vs-nonparametric-models).

Setting Up a Model:
- Divide the data into [Train-Dev-Test Sets](#train-dev-test-sets) to ensure robust evaluation and tuning.
- Optimize [Model Parameters](#model-parameters) and configurations for best performance.

Model Selection:
- Evaluate the appropriateness of models in the [Model Selection](#model-selection) phase.



# Model Cascading {#model-cascading}



# Model Deployment {#model-deployment}


Deploying a machine learning model involves moving it from a development environment to a production environment where it can make predictions on new data.

## Steps for Model Deployment

Model Exporting
   - Use tools like `joblib` or `pickle` to serialize the model.
     ```python
     import joblib
     joblib.dump(model, 'linear_regression_model.pkl')
     ```

Deployment Options
   - Application Integration: Embed the model into an application for real-time predictions.
   - [API](#api) Deployment: Use frameworks like [Flask](#flask) or [FastAPI](#fastapi) to create an API endpoint for the model.
   - Automated Workflows: Integrate the model into automated data processing pipelines.
## Tools and Platforms

- [Sklearn Pipeline](#sklearn-pipeline): Streamline the deployment process by integrating [Preprocessing](#preprocessing) and model steps.
- [Gradio](#gradio): Create user-friendly interfaces for model interaction.
- [Streamlit.io](#streamlitio)

## Considerations

- [Scalability](#scalability): Ensure the deployment solution can handle the expected load.
- [Model Observability](#model-observability): Implement monitoring to track model performance and detect issues.

# Deploying using [PyCaret](#pycaret)

**AWS:**  When deploying model on AWS S3, environment variables must be configured using the command-line interface. To configure AWS environment variables, type `aws configure` in terminal. The following information is required which can be generated using the Identity and Access Management (IAM) portal of your amazon console account:

- AWS Access Key ID
- AWS Secret Key Access
- Default Region Name (can be seen under Global settings on your AWS console)
- Default output format (must be left blank)

**GCP:** To deploy a model on Google Cloud Platform ('gcp'), the project must be created using the command-line or GCP console. Once the project is created, you must create a service account and download the service account key as a JSON file to set environment variables in your local environment. Learn more about it: https://cloud.google.com/docs/authentication/production

**Azure:** To deploy a model on Microsoft Azure ('azure'), environment variables for the connection string must be set in your local environment. Go to settings of storage account on Azure portal to access the connection string required.
AZURE_STORAGE_CONNECTION_STRING (required as environment variable)
Learn more about it: https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python?toc=%2Fpython%2Fazure%2FTOC.json

# Model Ensemble {#model-ensemble}


Ensemble models in machine learning are techniques that <mark>combine the predictions of multiple individual models</mark> to improve overall performance. Ensemble methods can achieve better accuracy and robustness than any single model alone. 

Key Concepts of Ensemble Models:
1. **Diversity**: The strength of ensemble models lies in the <mark>diversity</mark> of the base models. Different models may capture different patterns or errors in the data, and combining them can lead to more accurate predictions.
2. **Combination**: Ensemble methods aggregate the predictions of individual models using <mark>techniques like averaging, voting, or weighted sums</mark> to produce a final prediction.

Main Ensemble Techniques:
- [Bagging](#bagging)
- [Boosting](#boosting)
- [Stacking](#stacking)
- [Isolated Forest](#isolated-forest)

In [ML_Tools](#ml_tools) see: [Comparing_Ensembles.py](#comparing_ensemblespy)
# Further Understanding
### Analogy:
- Ensemble methods can be likened to consulting multiple doctors for a diagnosis. Each doctor (model) may have a different opinion, but by considering all opinions, the final diagnosis (prediction) is more accurate than relying on a single doctor's opinion.

### Advantages of Ensemble Models:
- **Increased Accuracy**: By combining multiple models, ensemble methods often achieve higher accuracy than individual models.
- **Robustness**: They are less sensitive to overfitting, especially when using techniques like bagging.
- **Flexibility**: Ensemble methods can be applied to various types of base models and are not limited to a specific algorithm.

### Challenges:
- **Complexity**: Ensemble models can be more complex and computationally intensive than single models.
- **[Interpretability](#interpretability)**: The final model may be harder to interpret compared to simpler models like decision trees.

# Model Evaluation Vs Model Optimisation {#model-evaluation-vs-model-optimisation}


[Model Evaluation](#model-evaluation)focuses on assessing a model's performance, while [Model Optimisation](#model-optimisation)aims to improve that performance through various techniques. 

Iterative Process: Model evaluation and optimization are often iterative. After evaluating a model, insights gained can guide further optimization. Conversely, after optimizing a model, it needs to be re-evaluated to ensure improvements.

Feedback Loop: Evaluation provides feedback on the effectiveness of optimization efforts, helping refine the model further.



# Model Evaluation {#model-evaluation}


Assess the model's performance using various metrics to ensure it meets the desired accuracy and reliability.

Appropriate evaluation metrics are used based on the problem type (classification vs. regression), to assess how well the model predicts.

<mark>The aim is to improve accuracy but also to generalize and avoid biases</mark> and [Overfitting](#overfitting).

- **Performance Assessment**: Models are evaluated on a testing set using metrics relevant to the problem type.
- **Generalization and Bias**: Evaluation includes assessing how well the model generalizes to new data and identifying any biases.

For categorical classifiers: [Evaluation Metrics](#evaluation-metrics):  Use metrics such as accuracy, precision, recall, F1-score, and confusion matrix to evaluate performance.

For regression tasks: [Regression Metrics](#regression-metrics): Metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²) are used.

[Cross Validation](#cross-validation) is a technique used to assess the performance of a model by splitting the data into multiple subsets for training and testing to assesses performance and generalization. It helps detect [Overfitting](#overfitting), provides reliable performance estimates.

[Feature Importance](#feature-importance): After training, analyze which features have the most significant impact on the model's predictions.

# Model Interpretability {#model-interpretability}

Model [interpretability](#interpretability) tools are crucial in ensuring that machine learning models are transparent, explainable, and understandable to stakeholders, particularly in industries where decisions need to be justifiable (e.g., finance, healthcare). 

These tools are becoming standard for ensuring trustworthiness and transparency in ML models, enabling organizations to defend model predictions in regulated industries and maintain user trust.

[p values](#p-values) and [Confidence Interval](#confidence-interval): If statistical significance is needed, interpret these values to determine which features significantly contribute to the model.

[SHapley Additive exPlanations](#shapley-additive-explanations)

[Local Interpretable Model-agnostic Explanations](#local-interpretable-model-agnostic-explanations)

Counterfactual Explanations:

   - Purpose: Counterfactual explanations aim to provide insight into <mark>how small changes in the input features could lead to different outcomes</mark>, helping users understand model behavior.

   - How it works: It identifies the minimal changes needed to alter a prediction. For example, in a credit scoring model, it might show how an individual could change their features (e.g., increasing income) to get approved for a loan.

   - Use cases: Particularly useful in sensitive fields like credit scoring, hiring, and medical diagnosis, where actionable explanations are critical.

   - Advantage: Provides intuitive and actionable feedback on predictions.


Global Surrogate Models

   - Purpose: A global surrogate is an interpretable model that is trained to approximate the predictions of a black-box model.

   - How it works: It uses simpler models (like decision trees) to mimic the behavior of a complex model and provide a global, easy-to-understand representation of how the model makes decisions.

   - Use cases: Provides <mark>insight into overall model behavior</mark>, though not as accurate as local explanations for specific predictions.
- 
   - Advantage: Simplicity and clarity for non-technical stakeholders.

- Scenario: An e-commerce company uses a neural network to predict customer churn based on features like purchase history, browsing behavior, and customer support interactions.

- Surrogate Model: To explain the overall decision-making process of the complex neural network, the data science team trains a decision tree as a global surrogate model. This decision tree offers a simplified view, showing that customers with a decline in recent purchases and frequent negative support interactions are most likely to churn. 



# Model Observability {#model-observability}


Monitor the model's performance over time (in production). Similar to [Model Validation](#model-validation).

In the context of machine learning (ML), Observability refers to the ability to <mark>monitor, understand, and diagnose the performance and behaviour of ML models</mark> in production. 

It encompasses the processes, tools, and techniques that help practitioners ensure models are functioning as expected and identify when they deviate from desired outcomes. 

[Master Observability Datadog](#master-observability-datadog)
## Key aspects of observability in machine learning include:

Observability is a process in ML, and is usually achieved through logging, metrics collection, real-time monitoring, and advanced diagnostic tools integrated into the ML pipeline.

1. Monitoring Model Performance <mark>(monitoring metrics)</mark>:
   - Tracking key metrics such as [Accuracy](#accuracy), [Precision](#precision),[Recall](#recall),[F1 Score](#f1-score),[ROC (Receiver Operating Characteristic)](#roc-receiver-operating-characteristic)d other relevant KPIs over time to identify performance degradation or improvements.
   - Monitoring [Performance Drift](#performance-drift) in model inputs (features) and outputs (predictions) to detect when the model no longer performs well due to changes in data distribution ([data drift](#data-drift)) or changes in relationships between variables ([Performance Drift](#performance-drift)).

1. Error and [Isolated Forest](#isolated-forest):
   - Identifying when predictions are out of the expected range or when the model behaves abnormally, such as high error rates on specific subsets of data or excessive latency in prediction generation.
   
2. [Interpretability](#interpretability):
   - Ensuring that the internal workings of the model (e.g., feature importance, decision pathways) are visible, interpretable, and explainable to humans. This allows for easier debugging and accountability, especially in critical applications such as finance, healthcare, or autonomous systems.

2. [data lineage](#data-lineage) and Provenance:
   - Tracking the data sources, transformations, and processes that influence the model’s input data. This provides visibility into how data flows through the pipeline and helps in reproducing results or addressing data-related issues.

2. Pipeline Monitoring:
   - Observing the entire ML pipeline from data ingestion and preprocessing to model training, [validation](#validation), and deployment. This includes identifying bottlenecks, delays, and system failures that may affect the model's ability to make predictions in real-time.

2. Alerts and Automation:
   - Setting up <mark>automated alerts</mark> when certain thresholds are breached, such as a sudden drop in accuracy or an increase in response time. This allows for prompt interventions, whether retraining the model, adjusting the pipeline, or tuning hyperparameters.

## Why Observability Matters in Machine Learning:

- Ensures Reliability: Observability provides insights into how models behave under different conditions, ensuring that they remain reliable and consistent in their performance.
- Prevents Model Drift ([Performance Drift](#performance-drift)): With observability, teams can detect model drift early, enabling them to retrain or recalibrate the model before performance deteriorates.
- Improves Accountability: Particularly in high-stakes applications, having observability in place allows organizations to understand and justify the model’s decisions.
- Supports Continuous Monitoring: Observability is critical in ML systems that operate continuously in production, ensuring they are making accurate and meaningful predictions over time.

Monitor the model's performance over time. If the data distribution changes (concept drift), or the model's accuracy declines, retraining or updating the model may be necessary.

## Related to:
- [Data Observability](#data-observability)
- [Model Validation](#model-validation)


# Model Optimisation {#model-optimisation}


Model optimization is a step in the machine learning workflow aimed at enhancing a model's performance by fine-tuning its parameters and hyperparameters. The goal is to improve the model's accuracy, efficiency, and ability to generalize to new data. 
### Purpose:
- Accuracy: Improve the model's predictive performance.
- Efficiency: Ensure the model runs efficiently in terms of computation and resource usage.
- Generalization: Enhance the model's ability to perform well on unseen data, avoiding overfitting.

### Process:
0. [Model Parameters](#model-parameters) tuning

1. [Hyperparameter|Hyperparameter tuning](#hyperparameterhyperparameter-tuning)
   - Adjust hyperparameters such as learning rate, number of layers in a neural network, and regularization strength to find the optimal configuration.
   - Techniques like grid search, random search, or Bayesian optimization can be used for this purpose.

2. [Feature Engineering](#feature-engineering)
   - Involves selecting, transforming, or creating new features that can improve model performance.
   - This step can significantly impact the model's ability to learn patterns from the data.

3. [Model Evaluation](#model-evaluation)
   - Evaluate the model using appropriate metrics based on the problem type (e.g., classification or regression).
   - Metrics for classification include accuracy, precision, recall, F1-score, and confusion matrix.
   - Metrics for regression include Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).

4. [Cross Validation](#cross-validation)
   - A technique to assess the model's performance by splitting the data into multiple subsets for training and testing.
   - Helps in detecting overfitting and provides reliable performance estimates.

5. [Model Ensemble](#model-ensemble): Combining models to get better performance

# Model Parameters Tuning {#model-parameters-tuning}


To find optimal [Model Parameters](#model-parameters).
## Finding Optimal Model Parameters

1. Parameter Space Exploration:
   - It's useful to visualize slices of the parameter space by selecting two parameters at a time. This helps in understanding how different parameter values affect the model's performance.

2. [Cost Function](#cost-function)
   - The cost function is used to find the minimum error in predictions. It measures the difference between predicted and actual values, and the goal is to minimize this function to improve model accuracy.

3. [Optimisation function](#optimisation-function)
   - Ideal parameters are found using optimization functions, which adjust the model parameters to minimize the loss function. Common optimization algorithms include Gradient Descent, Adam Optimizer, and Stochastic Gradient Descent.

4. Data Splitting:
   - Split the data into training and cross-validation sets to evaluate model performance. Plot the parameter of interest on the x-axis and accuracy on the y-axis to visualize performance.

[Optimisation techniques](#optimisation-techniques)
### Example


![Pasted image 20241231142918.png](../content/images/Pasted%20image%2020241231142918.png)


To find optimal model parameters, graph the parameter against error of the model.

On the left plot, the solid lines represent the predictions from these models. A polynomial model with degree 1 produces a straight line that intersects very few data points, while the maximum degree hews very closely to every data point. 

On the right:
    - the error on the trained data (blue) decreases as the model complexity increases as expected
    - the error of the cross-validation data decreases initially as the model starts to conform to the data, but then increases as the model starts to over-fit on the training data (fails to *generalize*).

# Model Parameters {#model-parameters}

Model parameters are also called weights and biases.

These parameters are adjusted during the training process to optimize the model's performance on the given task.

See also: 
- [Model Parameters Tuning](#model-parameters-tuning)
- [Optimisation techniques](#optimisation-techniques)
### Examples

1. **[Linear Regression](#linear-regression)**: 
   - Coefficients (weights) for each feature in the input data.
   - Intercept term (bias).

2. **[Logistic Regression](#logistic-regression)**:
   - Similar to [linear regression](#linear-regression), it has coefficients for each feature and an intercept term, but it models the probability of a binary outcome.

3. **[Deep Learning|Neural Networks](#deep-learningneural-networks)**:
   - Weights: The connections between neurons in different layers.
   - Biases: Additional parameters added to the weighted sum of inputs to a neuron.

4. **[Support Vector Machines](#support-vector-machines) (SVM)**:
   - Support vectors: Data points that define the decision boundary.
   - Coefficients for the hyperplane equation.

5. **Decision Trees**: [Decision Tree](#decision-tree)
   - Splitting thresholds for each node.
   - Structure of the tree (which features are used at each split).

6. **[K-Means](#k-means) [Clustering](#clustering)**:
   - Centroids: The center points of each cluster.



# Model Selection {#model-selection}


Model selection is an integral part of building a [Machine Learning Operations](#machine-learning-operations) to ensure that the best performing model is chosen for a given task, avoiding issues like overfitting or underfitting.

This is a crucial step because the model's ability to <mark>generalize</mark> to unseen data depends on selecting the right one.

Model selection typically involves the following steps:

1. Define candidate models: These can be models of different types (e.g., decision trees, support vector machines, neural networks) or the same model type but with varying hyperparameters.
   
2. Train each model: Train all the candidate models on the training set using different algorithms or parameter settings.
   
3. Evaluate performance ([Model Evaluation](#model-evaluation)): Use a validation set or cross-validation to evaluate the performance of each model. Common evaluation metrics include accuracy, precision, recall, F1 score, and mean squared error, depending on the type of problem (classification or regression).

4. Select the best model: Based on the evaluation metrics, choose the model that performs best on the validation set. The aim is to balance bias and variance to achieve good generalization to unseen data.

5. Test on unseen data: Finally, test the selected model on a test set to ensure that it generalizes well and has not been overfitted to the validation data.

Common approaches for model selection include:
- [GridSeachCv](#gridseachcv) and [Random Search](#random-search) for hyperparameter tuning.
- [Cross Validation](#cross-validation) to ensure robustness by evaluating model performance on different subsets of the data.
- Bayesian Optimization, which can be used to efficiently search the hyperparameter space.
- Choose the best-performing model based on [Evaluation Metrics](#evaluation-metrics) and optimization results.
- [Cross Validation](#cross-validation): Evaluate the model more robustly by splitting the training data into smaller chunks and training the model multiple times.
- [Model Interpretability](#model-interpretability): Utilize tools to understand and interpret the model's predictions, ensuring transparency and trustworthiness.

# Model Validation {#model-validation}

Model Validation refers to the process of evaluating a machine learning model's performance on a separate dataset (often called the validation set) to ensure it generalizes well to new, unseen data. This step is crucial for tuning [model parameters](#model-parameters), selecting the best model, and preventing overfitting. Validation helps in assessing how well the model will perform in real-world scenarios.

[Model Observability](#model-observability), on the other hand, involves monitoring and understanding the model's performance and behavior in production over time. It includes tracking metrics, detecting [Performance Drift](#performance-drift), and ensuring the model continues to function as expected in dynamic environments.

While model validation is a step in the model development process, model observability is an ongoing practice once the model is deployed. Both are related in that they aim to ensure the model's reliability and effectiveness, but they occur at different stages of the model lifecycle. Validation is about initial performance assessment, whereas observability is about continuous monitoring and maintenance.

# Model Parameters Vs Hyperparameters {#model-parameters-vs-hyperparameters}

Model parameters and hyperparameters serve different roles:

[Model Parameters](#model-parameters)
   - These are the internal variables of the model that are learned from the training data. They define the model's structure and are adjusted during the training process to minimize the [Loss function](#loss-function).
   - Examples include:
	   - the weights and biases in a neural network,
	   - the coefficients in a linear regression model,
	   - or the support vectors in a support vector machine.
   - Model parameters are directly influenced by the data and are optimized through algorithms like [Gradient Descent](#gradient-descent).

[Hyperparameter](#hyperparameter)
   - These are external configurations set before the training process begins. They are not learned from the data but are used for controlling the learning process and the model's architecture.
   - Examples include the:
	   - [learning rate](#learning-rate), 
	   - the number of hidden layers in a [Neural network](#neural-network),
	   - the number of trees in a random forest,
	   - or the regularization parameter in a regression model.
   - Hyperparameters are typically tuned through methods like grid search or random search to find the best configuration that results in optimal model performance.

# Model Preparation {#model-preparation}


```python
model = 
model.fit(X_train, y_train)
print(model)
y_pred = model.predict(X_test)
print(accuracy_score(y_expect, y_pred))
```


# Momentum {#momentum}

Momentum is an [Model Optimisation|Optimisation](#model-optimisationoptimisation)  technique used to accelerate the [Gradient Descent](#gradient-descent) algorithm by incorporating the concept of inertia. It helps in reducing oscillations and speeding up convergence, especially in scenarios where the [cost function](#cost-function) has a complex landscape (surface). Momentum helps in dampening oscillations and achieving faster convergence. Momentum is a technique that helps accelerate gradient descent by adding a fraction of the previous update to the current update. Formula:
  $$
  v_{t+1} = \beta v_t + (1 - \beta) \nabla_{\theta} J(\theta)
  $$
  $$
  \theta_{t+1} = \theta_t - \alpha v_{t+1}
  $$
  Where:
  - $v_t$ is the velocity (the accumulated gradient).
  - $\beta$ is the momentum factor.
  - $\nabla_{\theta} J(\theta)$ is the gradient of the cost function with respect to the parameters $\theta$.
  - $\alpha$ is the learning rate.

In [ML_Tools](#ml_tools) see: [Momentum.py](#momentumpy)
## Key Features of Momentum

**Inertia Effect:** Momentum uses the past gradients to smooth out the updates, which helps in navigating the parameter space more effectively.

**Parameter Update Rule:** The update rule for momentum involves maintaining a velocity vector that accumulates the gradients. The parameters are then updated using this velocity, which is a combination of the current gradient and the previous velocity.

[Hyperparameter](#hyperparameter)
  - **[Learning Rate](#learning-rate) ($\alpha$):** Controls the size of the steps taken towards the minimum.
  - **Momentum Coefficient ($\beta$):** Determines the contribution of the previous gradients to the current update. A typical value is 0.9.




# Momentum.Py {#momentumpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Optimisation/Momentum.py

# Mongodb {#mongodb}



# Monolith Architecture {#monolith-architecture}


A monolith, in the context of [software architecture](#software-architecture), refers to a <mark>single, unified application where all components and functionalities are interconnected and interdependent</mark>. In a monolithic architecture, the entire application is typically built as a <mark>single codebase</mark>, and all functions and modules are tightly coupled.

While monolithic architectures can be simpler to develop and deploy initially, they can become cumbersome as the application grows in complexity. Many organizations eventually transition to [microservices](#microservices) or other modular architectures to improve scalability, flexibility, and maintainability. However, monoliths can still be effective for smaller applications or teams with limited resources.

When we talk about a "function call-driven" monolith, we are referring to the way in which different parts of the application interact with each other. In such a system:

1. **Tightly Coupled Components**: All components of the application are part of a single codebase and often share the same resources, such as databases and libraries.

2. **Function Calls**: Communication between different parts of the application is primarily done through direct function or method calls. This means that one part of the application can directly invoke functions or methods in another part.

3. **Single Deployment Unit**: The entire application is deployed as a single unit. Any changes to one part of the application require redeploying the whole application.

4. **Shared Memory Space**: Since all components are part of the same application, they often share the same memory space, which can simplify data sharing but also lead to issues with scalability and fault isolation.

5. **Challenges with Scalability and Flexibility**: As the application grows, a monolithic architecture can become difficult to manage, scale, and update. Changes in one part of the system can have unintended consequences elsewhere, making it challenging to innovate quickly.



# Monte Carlo Simulation {#monte-carlo-simulation}

Resources:
- https://www.youtube.com/watch?v=r7cn3WS5x9c

Algorithms that use repeated random sampling.

Monte Carlo: random

How does the randomness in data generation impact the randomness of the paramaeter calculation.

Simulation study:
1) FIGURE OUT A WAY TO APPROXIMATE A PROCESS WITH A RANDOM NUMBER GENERATOR
2) GENERATE THE DATA AND CALCULATE A VALUE OF INTEREST(I.E.MEAN, BIAS, COVERAGE)
3) REPEAT STEPS 1& 2 MANY TIMES TO LEARN ABOUT THE UNCERTAINTY IN THIS VALUE

Simulation studies:  



# Multi Agent Reinforcement Learning {#multi-agent-reinforcement-learning}






# Multi Head Attention {#multi-head-attention}


# Summary

 <mark>Aggregates different perspectives</mark>

This approach allows the model to attend to different parts of the input sequence simultaneously, capturing various aspects of the context more effectively. 

Like a hydra, it focuses on different aspects of the context. Getting a finer understanding.

Multi-head attention captures more context by dividing the input processing into multiple independent attention heads. Each head focuses on different parts of the input and captures diverse types of relationships, both local and global. This parallelism allows the model to learn multiple perspectives simultaneously, enriching its understanding of the input sequence and improving performance on complex tasks like language modeling, machine translation, and more.

Related to the [Attention mechanism](#attention-mechanism).
### **Multi-Head Attention**
In **multi-head attention**, the idea is to split the input into multiple subsets of attention heads. Instead of computing a single attention score for each token pair, multiple attention "heads" are used, with each head attending to different parts of the input. This provides several benefits:

#### a) **Diverse Attention Patterns**
Each head in multi-head attention can focus on different aspects of the input sequence, allowing the model to capture multiple relationships between tokens. For instance:
- One head may focus on syntactic relationships (like word order or structure).
- Another head may focus on semantic relationships (like the meaning or context of words).

By having multiple attention heads, the model can learn to capture **different types of context simultaneously**.

#### b) **Different Projection Spaces**
Each attention head has its own set of parameters, which project the input into a different subspace (i.e., they use different weight matrices to transform queries, keys, and values). This allows each head to learn different relationships between tokens in various representational subspaces, thus increasing the model's capacity to understand complex dependencies.

For example, one head might focus on short-range dependencies (like adjacent words), while another head could capture long-range dependencies (like relationships between words at opposite ends of the sentence).

#### c) **Improved Expressiveness**
By combining the outputs of multiple heads, the model gains a richer representation of the context. Each attention head contributes unique insights about how tokens in the sequence relate to each other, and by concatenating these different perspectives, the overall attention mechanism becomes more expressive.

### **Multi-Head Attention Process**
The process for multi-head attention involves the following steps:
1. **Linear Transformations**: The input vectors (representing words or tokens) are linearly transformed into **queries (Q)**, **keys (K)**, and **values (V)** for each head. These transformations are different for each head, allowing each head to capture different relationships in the data.
  
2. **Attention Calculation for Each Head**: For each head, scaled dot-product attention is calculated independently. The attention mechanism computes scores by comparing queries and keys, and these scores are used to weigh the values. This results in a unique context vector for each head.

3. **Concatenation**: The outputs from all the attention heads are concatenated into a single vector. This step combines the different perspectives learned by each head.

4. **Final Linear Transformation**: After concatenation, a final linear transformation is applied to combine the information from all heads into a single vector that can be used in the next layer of the model.

### **How Multi-Head Attention Captures More Context**

Multi-head attention captures more context than single-head attention for several reasons:

- <mark>**Multiple Focus Areas**:</mark> By using multiple heads, the model can simultaneously focus on different parts of the sequence. Some heads might <mark>attend to local dependencies,</mark> while others might capture more distant relationships. This gives the model a <mark>broader understanding</mark> of the entire sequence.
  
- **Handling Ambiguity**: In natural language, the meaning of words can depend heavily on context. Multi-head attention allows the model to disambiguate meanings by attending to different context clues in parallel. For instance, the word "bank" in "I went to the bank" can mean different things, and different heads can <mark>capture clues</mark> from the surrounding words to determine whether "bank" refers to a financial institution or a riverbank.

- **Diverse Representations**: Each head transforms the input into <mark>different representational subspaces</mark>, meaning the model learns diverse representations of the same input. This diversity enhances the model's ability to generalize and capture complex relationships in the data.

### **Application Example: Language Translation**
Consider translating a sentence from one language to another. The multi-head attention mechanism in the Transformer model helps capture different linguistic structures:
- One head might focus on aligning <mark>subject-verb</mark> pairs between the two languages.
- Another head might capture <mark>longer dependencies</mark>, like how nouns and pronouns refer to each other across a long sentence.
- A third head might capture <mark>grammatical structure</mark> differences between the source and target languages.

By <mark>aggregating these different perspectives</mark>, multi-head attention ensures that the model understands both the local and global context, leading to better translation quality.


# Multi Level Index {#multi-level-index}


Multi-level indexing in pandas—also called hierarchical indexing—enables you to work with higher-dimensional data in a 2D DataFrame. It's particularly useful for working with grouped or nested data structures.

Why use multi-level index:
- MultiIndex makes your data [interoperable](#interoperable)
- Enables systematic slicing and aggregation
- Logical grouping of variables

Operations like `.stack()` and `.unstack()` rely on MultiIndex to move between long and wide formats.
- In a flat DataFrame, reshaping often requires column renaming or pivoting.
- With MultiIndex, it's structured and reversible.
- Stack can be used to make a multi index from a flat dataframe.

If you need frequent slicing/aggregation across multiple levels, MultiIndex saves effort and code.

When _not_ to use it
- If your data is simple or small.
- If you're just loading, cleaning, and exporting CSVs.
- If you don’t need `.groupby(level=...)`, `.stack()`, or `.xs()` operations.

Similar to:
- SQL composite keys
- Python nested dictionaries
- [JSON](#json) hierarchical structures

Related:
- [Groupby](#groupby)
- [Pandas Stack](#pandas-stack)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/multi__level_index.ipynb

How this mimics a 3D array:
- You can think of each (Product, Store) pair as defining a "slice" of a 2D array.
- The columns (Jan, Feb) represent time-like progression (3rd axis).
- Visually, it’s like you’ve flattened a cube into a matrix while retaining the ability to slice along all original axes.

```python
Month             Jan  Feb
Product   Store            
Product A Store X  100  110
          Store Y  120  115
Product B Store X   90  105
          Store Y   95  100
```

# Multicollinearity {#multicollinearity}


When two or more regressors are in [Correlation](#correlation)

Multicollinearity refers to the <mark>instability</mark> of a model due to <mark>highly correlated independent variables.</mark>

It occurs when two or more independent variables in a regression model are highly correlated, which can make it difficult to determine the individual effect of each variable on the dependent variable.

Multicollinearity affects regression models primarily because it leads to instability in the estimated coefficients of the independent variables.

Also see:
- [Addressing Multicollinearity](#addressing-multicollinearity)
- [Impact of multicollinearity on model parameters](#impact-of-multicollinearity-on-model-parameters)

Related:
- Multicollinearity hurts your hypothesis test
	- Correlation increases bias in the estimated parameters
	- Decreases power via exploded standard errors

##### Results of Multicollinearity:

1. **Difficulty in Estimating Coefficients**: When independent variables are highly correlated, it becomes challenging to isolate the individual effect of each variable on the dependent variable. This can result in large standard errors for the coefficients, making them unreliable.
    
2. **Inflated Variance**: The presence of multicollinearity inflates the variance of the coefficient estimates, which can lead to less precise estimates. This means that small changes in the data can lead to large changes in the estimated coefficients.
    
3. **Misleading Significance Tests**: Multicollinearity can cause some variables to appear statistically insignificant when they might actually be significant. This can lead to incorrect conclusions about the importance of predictors in the model.
    
4. **Model Interpretation**: The interpretation of the coefficients becomes complicated, as the effect of one variable may be confounded with the effect of another correlated variable.
### Key Points

- Assumption: The multicollinearity assumption suggests that <mark>independent variables should not be collinear.</mark>
- Detection: Use tools like [Heatmap](#heatmap) or [Clustering](#clustering) to visualize [Correlation](#correlation) and identify multicollinearity.
- Variance Inflation Factor (VIF): High VIF values (typically greater than 10) indicate a high degree of multicollinearity. <mark>Features with high VIF should be dropped to improve model stability.</mark>


# Multinomial Naive Bayes {#multinomial-naive-bayes}



# Mysql {#mysql}

MySQL has more <mark>granularity</mark> with types than SQLite. For example, an integer could be `TINYINT`, `SMALLINT`, `MEDIUMINT`, `INT` or `BIGINT` based on the size of the number we want to store. 

The following table shows us the size and range of numbers we can store in each of the integer types.
    
!["Table of integer types in MySQL"|500](https://cs50.harvard.edu/sql/2024/notes/6/images/12.jpg)
### Tags
- **Tags**: #relational_database, #data_management

# Maintainability {#maintainability}



# Map Reduce {#map-reduce}


MapReduce is a programming model and processing technique used for processing and generating large data sets with a parallel, distributed algorithm on a cluster. [Distributed Computing](#distributed-computing)

It is a core component of the Apache Hadoop [Hadoop](#hadoop) framework, which is designed to handle vast amounts of data across many servers. The MapReduce model simplifies data processing across large clusters by breaking down the task into two main functions: **Map** and **Reduce**.

MapReduce is particularly effective for [Batch Processing](#batch-processing)  tasks where the data can be processed independently and aggregated later. However, it may not be the best choice for [real-time processing](#real-time-processing)  or tasks that require low-latency responses, where other frameworks like [Apache Spark](#apache-spark) might be more suitable.
### Key Components of MapReduce

1. **Map Function**:
   - **Purpose**: To process and transform input data into a set of intermediate key-value pairs.
   - **Functionality**: Each input data element is processed independently, and the output is a collection of key-value pairs.
   - **Example**: In a word count application, the map function reads a document and emits each word as a key with a count of one as the value.

2. **Shuffle and Sort**:
   - **Purpose**: To organize the intermediate data by keys.
   - **Functionality**: The framework sorts the output of the map function and groups all values associated with the same key together. This step is crucial for the reduce function to process data efficiently.

3. **Reduce Function**:
   - **Purpose**: To aggregate and summarize the intermediate data.
   - **Functionality**: The reduce function takes the grouped key-value pairs and processes them to produce a smaller set of output values.
   - **Example**: Continuing with the word count example, the reduce function sums up the counts for each word, resulting in the total count for each word across all documents.

### Why MapReduce is Used

- **Scalability**: MapReduce can process petabytes of data by distributing the workload across a large number of servers in a cluster.
- **Fault Tolerance**: The framework automatically handles failures by reassigning tasks to other nodes, ensuring that the processing continues without data loss.
- **Simplicity**: It abstracts the complexity of parallel processing, allowing developers to focus on the map and reduce logic without worrying about the underlying infrastructure.
- **Flexibility**: MapReduce can be used for a wide range of applications, including data mining, log analysis, and machine learning, among others.
- **Cost-Effectiveness**: By using commodity hardware and open-source software, organizations can process large data sets without significant investment in specialized hardware.



# Master Data Management {#master-data-management}


Master data management is a method to <mark>centralize</mark> master data.

It's the bridge between the business that maintain the data and know them best and the data folks, and it's a tool of choice. It helps with uniformity, accuracy, stewardship, semantic consistency, and accountability of mostly enterprise master data assets.

Master [Data Management](#data-management)(MDM) refers to the processes, technologies, and tools used to define, manage, and maintain an organization's critical data entities, such as customers, products, employees, suppliers, and locations, <mark>ensuring that this data is accurate, consistent, and up-to-date across all systems and departments.</mark> The goal of MDM is to create a single, authoritative [source of truth](#source-of-truth) for master data, which is shared and synchronized across the organization to improve decision-making, reduce duplication, and maintain data integrity.

MDM is especially important in large organizations where data is often siloed across various departments and systems, leading to <mark>inconsistencies, duplication, and errors.</mark> By centralizing the management of key data, MDM helps improve operational efficiency, regulatory compliance, and the overall effectiveness of business processes.

Key aspects of MDM include:

1. [Data Governance](#data-governance): Establishing policies, rules, and standards for how master data is managed, who is responsible for it, and how data quality is monitored.

2. **Data Integration**: Consolidating and harmonizing data from various sources (e.g., databases, applications) to create a unified, consistent view of master data.

3. [Data Quality](#data-quality): Ensuring that the data is complete, accurate, valid, and consistent across the organization.

4. **Data Stewardship**: Assigning roles and responsibilities for managing the master data and ensuring that it complies with the established governance policies.

5. **Metadata Management**: Maintaining a consistent definition of data entities, relationships, and attributes, helping stakeholders understand the meaning and usage of the data.

6. **Data Synchronization**: Ensuring that any updates or changes to master data in one system are reflected across all relevant systems.



# Mean Absolute Error {#mean-absolute-error}

