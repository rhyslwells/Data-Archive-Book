# S

## Table of Contents
* [SHapley Additive exPlanations](#shapley-additive-explanations)
* [SMOTE (Synthetic Minority Over-sampling Technique)](#smote-synthetic-minority-over-sampling-technique)
* [SMSS](#smss)
* [SQL Groupby](#sql-groupby)
* [SQL Injection](#sql-injection)
* [SQL Joins](#sql-joins)
* [SQL Window functions](#sql-window-functions)
* [SQL vs NoSQL](#sql-vs-nosql)
* [SQL](#sql)
* [SQLAlchemy vs. sqlite3](#)
* [SQLAlchemy](#sqlalchemy)
* [SQLite Studio](#sqlite-studio)
* [SQLite](#sqlite)
* [SVM_Example.py](#svm_examplepy)
* [Sarsa](#sarsa)
* [Scala](#scala)
* [Scalability](#scalability)
* [Scaling Agentic Systems](#scaling-agentic-systems)
* [Scaling Server](#scaling-server)
* [Scheduled Tasks](#scheduled-tasks)
* [Schema Evolution](#schema-evolution)
* [Scientific Method](#scientific-method)
* [Seaborn](#seaborn)
* [Search](#search)
* [Security](#security)
* [Semantic Relationships](#semantic-relationships)
* [Sentence Similarity](#sentence-similarity)
* [Sharepoint](#sharepoint)
* [Silhouette Analysis](#silhouette-analysis)
* [Similarity Search](#similarity-search)
* [Single Source of Truth](#single-source-of-truth)
* [Sklearn Pipiline](#sklearn-pipiline)
* [Sklearn](#sklearn)
* [Slowly Changing Dimension](#slowly-changing-dimension)
* [Small Language Models](#small-language-models)
* [Smart Grids](#smart-grids)
* [Snowflake Schema](#snowflake-schema)
* [Snowflake](#)
* [Soft Deletion](#soft-deletion)
* [Software Design Patterns](#software-design-patterns)
* [Software Development Life Cycle](#software-development-life-cycle)
* [SparseCategorialCrossentropy or CategoricalCrossEntropy](#sparsecategorialcrossentropy-or-categoricalcrossentropy)
* [Specificity](#specificity)
* [Spreadsheets vs Databases](#spreadsheets-vs-databases)
* [Stacking](#stacking)
* [Standard deviation](#standard-deviation)
* [Standardisation](#standardisation)
* [Star Schema](#star-schema)
* [Statistical Assumptions](#statistical-assumptions)
* [Statistical Tests](#statistical-tests)
* [Statistics](#statistics)
* [Stemming](#stemming)
* [Stochastic Gradient Descent](#stochastic-gradient-descent)
* [Stored Procedures](#stored-procedures)
* [Strongly vs Weakly typed language](#strongly-vs-weakly-typed-language)
* [Structuring and organizing data](#structuring-and-organizing-data)
* [Summarisation](#summarisation)
* [Supervised Learning](#supervised-learning)
* [Support Vector Classifier (SVC)](#support-vector-classifier-svc)
* [Support Vector Machines](#support-vector-machines)
* [Support Vector Regression](#support-vector-regression)
* [Symbolic computation](#symbolic-computation)
* [Sympy](#sympy)
* [semantic layer](#semantic-layer)
* [semi-structured data](#semi-structured-data)
* [shapefile](#shapefile)
* [sklearn datasets](#sklearn-datasets)
* [spaCy](#spacy)
* [storage layer object store](#storage-layer-object-store)
* [structured data](#structured-data)
* [syntactic relationships](#syntactic-relationships)



# Shapley Additive Explanations {#shapley-additive-explanations}


SHAP provides a unified approach to measure [Feature Importance](#feature-importance) by computing the contribution of each feature to each prediction, based on game theory.

### Key Points

- **Purpose**: SHAP provides consistent and locally accurate explanations by assigning each feature <mark>an importance value based</mark> on Shapley values from cooperative game theory.

- **How it Works**: 
  - It calculates how each feature contributes to the model's output by comparing predictions with and without the feature, across various feature value combinations.

- **Use Cases**: Suitable for complex models like neural networks, random forests, or gradient boosting machines, where internal logic is difficult to understand.

- **Advantage**: 
  - Provides global explanations (model-wide feature importance) and local explanations (individual prediction reasons).

- **Scenario**: 
  - A financial institution uses a black-box XGBoost model to predict whether a loan applicant should be approved. The model takes several factors into account, such as credit score, income, employment history, and outstanding debts.
  - **SHAP Explanation**: For a specific loan rejection case, SHAP values reveal that the applicant’s high debt-to-income ratio and short employment history contributed the most to the rejection decision. These factors had the highest negative SHAP values for this prediction, providing actionable insights to both the applicant and the loan officers.

### Example Code

To compute SHAP values, you can use the SHAP library to interpret feature importance for any machine learning model:

```python
import shap

# Explain the model's predictions using SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Plot the summary plot of feature importance
shap.summary_plot(shap_values, X_test)
```

# Smote (Synthetic Minority Over Sampling Technique) {#smote-synthetic-minority-over-sampling-technique}


SMOTE (Synthetic Minority Over-sampling Technique)

Generate synthetic samples for the minority class by interpolating between existing samples.

SMOTE: This technique generates synthetic samples for the minority class (female resumes) by creating new instances that are interpolations of existing ones.

# Smss {#smss}

microsoft sql server management.

# Sql Groupby {#sql-groupby}

The [SQL](#sql) `GROUP BY` clause is used to group rows that have the same values in specified columns into summary rows, like "total sales per region" or "average age per department." 

It is often used in conjunction with aggregate functions such as `COUNT()`, `SUM()`, `AVG()`, `MAX()`, and `MIN()` to perform calculations on each group.
### Basic Syntax

```sql
SELECT column1, aggregate_function(column2)
FROM table_name
WHERE condition
GROUP BY column1;
```

### Example Usage

Let's say you have a table called `sales` with the following columns:

- `id`: Unique identifier for each sale
- `product`: The name of the product sold
- `amount`: The sale amount
- `region`: The region where the sale occurred

#### 1. Count the Number of Sales per Product

To count how many sales were made for each product, you would use:

```sql
SELECT product, COUNT(*) AS total_sales
FROM sales
GROUP BY product;
```

#### 2. Calculate Total Sales Amount per Region

To calculate the total sales amount for each region, you would use:

```sql
SELECT region, SUM(amount) AS total_sales_amount
FROM sales
GROUP BY region;
```

### Using `HAVING` with `GROUP BY`

You can also filter the results of a `GROUP BY` [Querying|query](#queryingquery) using the `HAVING` clause. This is useful when you want to filter groups based on aggregate values.

#### Example: Filter Groups

For example, to find products with total sales greater than $1000:

```sql
SELECT product, SUM(amount) AS total_sales_amount
FROM sales
GROUP BY product
HAVING SUM(amount) > 1000;
```

### Important Points

- **Columns in SELECT**: When using `GROUP BY`, all columns in the `SELECT` statement must either be included in the `GROUP BY` clause or be used in an aggregate function.
- **Order of Execution**: The `GROUP BY` clause is processed after the `WHERE` clause but before the `ORDER BY` clause in the SQL execution order.





[SQL Groupby](#sql-groupby)
   **Tags**: #data_transformation  #querying

# Sql Injection {#sql-injection}

SQL injection is a code injection technique that targets applications using SQL databases. It occurs when a malicious user injects harmful SQL code into a query, potentially compromising the security of the database. 

### How SQL Injection Works

Consider a scenario where a website prompts users to log in with their username and password. The application might execute a query like this:

```sql
SELECT `id` FROM `users`
WHERE `user` = 'Carter' AND `password` = 'password';
```

If the user named Carter enters their credentials correctly, the query functions as intended. However, a malicious user could input a different string, such as:

```
password' OR '1' = '1
```

This alters the query to:

```sql
SELECT `id` FROM `users`
WHERE `user` = 'Carter' AND `password` = 'password' OR '1' = '1';
```

As a result, the attacker could gain unauthorized access to the database.

### Example of Vulnerable Code

The following Python function demonstrates how SQL injection can occur due to unsafe query construction:

```python
import sqlite3

def unsafe_query(user_input):
    query = "SELECT * FROM users WHERE name = '" + user_input + "'"
    conn = sqlite3.connect('example.db')
    conn.execute(query)
```

In this example, the `unsafe_query` function constructs SQL queries using string concatenation, making it vulnerable if user input is not properly sanitized.

### Preventing SQL Injection

To mitigate the risk of SQL injection, it is essential to use prepared statements or parameterized queries. For example, consider an SQL injection attack that aims to display all user accounts from the `accounts` table:

```sql
SELECT * FROM `accounts`
WHERE `id` = 1 UNION SELECT * FROM `accounts`;
```

Using a prepared statement, we can safeguard against such attacks:

```sql
PREPARE `balance_check`
FROM 'SELECT * FROM `accounts`
WHERE `id` = ?';
```

In this statement, the question mark acts as a placeholder for user input, preventing the execution of unintended SQL code.

### Executing the Prepared Statement

To execute the prepared statement and check a user’s balance, we can set a variable for the user ID:

```sql
SET @id = 1;
EXECUTE `balance_check` USING @id;
```

Here, the `SET` statement simulates obtaining the user’s ID through the application, with the `@` symbol denoting a variable in MySQL.

### Testing with Malicious Input

If we attempt to run the same statements with a malicious ID:

```sql
SET @id = '1 UNION SELECT * FROM `accounts`';
EXECUTE `balance_check` USING @id;
```

The output will still reflect the balance of the user with ID 1, without exposing any additional data. This demonstrates that prepared statements effectively prevent SQL injection attacks.

### Mitigation Strategies

- **Use Parameterized Queries**: Always use parameterized queries or prepared statements to prevent SQL injection.
- **Validate and Sanitize Inputs**: Ensure user inputs are validated and sanitized before being processed.


# Sql Joins {#sql-joins}

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/Joining.ipynb


![Pasted image 20250323083319.png|800](../content/images/Pasted%20image%2020250323083319.png|800)

# Sql Window Functions {#sql-window-functions}

SQL Window Functions are a feature in SQL that allow you to perform calculations across a set of table rows that are somehow related to the current row. 

Unlike regular aggregate functions, which return a single value for a group of rows, window functions return a value for each row in the result set while still allowing access to the individual row data.

### Key Characteristics of Window Functions

1. Non-Aggregating: Window functions do not collapse rows into a single output row. Instead, they perform calculations across a defined "window" of rows related to the current row.

2. OVER Clause: Window functions are defined using the `OVER` clause, which specifies the window of rows to be considered for the function.

3. Partitioning: You can partition the result set into groups using the `PARTITION BY` clause within the `OVER` clause. Each partition is processed independently.

4. Ordering: You can specify the order of rows within each partition using the `ORDER BY` clause within the `OVER` clause.
### Example Use Case

Suppose you have a table `sales` with columns `salesperson`, `region`, and `amount`. You can use window functions to calculate the total sales for each salesperson while still displaying individual sales records:

Initial Table: `employees`

| id | name    | department | salary |
|----|---------|------------|--------|
| 1  | Alice   | Sales      | 50000  |
| 2  | Bob     | Sales      | 60000  |
| 3  | Charlie | HR         | 55000  |
| 4  | David   | HR         | 70000  |
| 5  | Eve     | IT         | 80000  |
| 6  | Frank   | IT         | 75000  |

### Example [Querying|Queries](#queryingqueries) Using SQL Window Functions

#### 1. Using `ROW_NUMBER()`

The `ROW_NUMBER()` function assigns a unique rank to each employee within their department based on their salary.

```sql
SELECT 
    id, 
    name, 
    department, 
    salary, 
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank
FROM employees;
```

Resulting Table:

| id | name    | department | salary | rank |
|----|---------|------------|--------|------|
| 2  | Bob     | Sales      | 60000  | 1    |
| 1  | Alice   | Sales      | 50000  | 2    |
| 4  | David   | HR         | 70000  | 1    |
| 3  | Charlie | HR         | 55000  | 2    |
| 5  | Eve     | IT         | 80000  | 1    |
| 6  | Frank   | IT         | 75000  | 2    |

#### 2. Using `SUM()` similar AVG

The `SUM()` function calculates the total salary for each department, showing the same total for each employee in that department.

```sql
SELECT 
    id, 
    name, 
    department, 
    salary, 
    SUM(salary) OVER (PARTITION BY department) AS total_department_salary
FROM employees;
```

Resulting Table:

| id | name    | department | salary | total_department_salary |
|----|---------|------------|--------|-------------------------|
| 1  | Alice   | Sales      | 50000  | 110000                  |
| 2  | Bob     | Sales      | 60000  | 110000                  |
| 3  | Charlie | HR         | 55000  | 125000                  |
| 4  | David   | HR         | 70000  | 125000                  |
| 5  | Eve     | IT         | 80000  | 155000                  |
| 6  | Frank   | IT         | 75000  | 155000                  |









[SQL Window functions](#sql-window-functions)
   **Tags**: #data_analysis, #querying

# Sql Vs Nosql {#sql-vs-nosql}


[NoSQL](#nosql)

# Sql {#sql}


Structured Query Language (SQL) is the standard language for interacting with relational databases, enabling efficient data [Querying](#querying) and manipulation. It serves as a common interface for [Database](#database)s and data lakes.

Features: 
  - Declarative language for storing and querying structured data.
  - Transactional properties enhance speed and efficiency.
  
### Good Practices

Capitalization: 
  - Use uppercase for SQL keywords for better readability.
  - Use lowercase for table and column names.
  
Quotes:
  - Use double quotes for SQL identifiers (table and column names).
  - Use single quotes for string values.

### Related terms

[Database Techniques](#database-techniques)

### SQLAlchemy vs. sqlite3: Which One Should You Use?

The choice between [SQLAlchemy](#sqlalchemy) and [SQLite](#sqlite) depends on your specific needs. Here’s a comparison based on key factors:

### 1. Abstraction and Ease of Use

| Feature     | SQLAlchemy                                 | sqlite3                          |
| ----------- | ------------------------------------------ | -------------------------------- |
| Abstraction | High-level ORM (Object Relational Mapping) | Low-level, direct SQL execution  |
| Ease of Use | Pythonic API for working with databases    | Requires writing raw SQL queries |
| Best for    | Large projects, scalable applications      | Simple scripts, small projects   |

✅ Use SQLAlchemy if you want to work with database tables as Python objects (ORM).  
✅ Use sqlite3 if you are comfortable writing SQL queries directly.

### 2. Supported Databases

|Feature|SQLAlchemy|sqlite3|
|---|---|---|
|Database Support|Works with MySQL, PostgreSQL, SQLite, MSSQL, etc.|Only works with SQLite|
|Portability|Can switch databases easily|Tied to SQLite only|

✅ Use SQLAlchemy if you need flexibility to work with different databases.  
✅ Use sqlite3 if you are only working with SQLite.

### 3. Performance and Scalability

|Feature|SQLAlchemy|sqlite3|
|---|---|---|
|Performance|Slightly slower due to ORM overhead|Faster for simple queries|
|Scalability|Supports connection pooling, transactions, and large-scale applications|Best for local, single-user applications|

✅ Use SQLAlchemy for large applications with complex relationships.  
✅ Use sqlite3 if you just need a simple, fast database for local use.

### 4. Querying and Data Manipulation

|Feature|SQLAlchemy|sqlite3|
|---|---|---|
|Querying|Can use both ORM and raw SQL queries|Only supports raw SQL queries|
|Ease of Data Manipulation|Object-oriented approach (e.g., `session.add(obj)`)|SQL execution via `cursor.execute(query)`|

✅ Use SQLAlchemy if you prefer writing queries in a Pythonic way (ORM).  
✅ Use sqlite3 if you are fine with executing raw SQL statements.



### 5. Transaction Handling

|Feature|SQLAlchemy|sqlite3|
|---|---|---|
|Transaction Control|Automatic transaction management|Manual transaction handling (`conn.commit()`)|
|Rollback Support|Easier and more reliable|Must be explicitly handled|

✅ Use SQLAlchemy for better transaction control in complex applications.  
✅ Use sqlite3 if you want manual control over transactions.

### 6. Learning Curve

|Feature|SQLAlchemy|sqlite3|
|---|---|---|
|Difficulty Level|Higher due to ORM concepts|Easier to get started|

✅ Use sqlite3 if you want a simple database solution with SQL queries.  
✅ Use SQLAlchemy if you are comfortable with an ORM and want a scalable approach.

---

### When to Use SQLAlchemy?

- You are building a large, scalable application.
- You need database flexibility (MySQL, PostgreSQL, etc.).
- You prefer Pythonic ORM instead of writing raw SQL.
- You want better transaction handling and connection management.

### When to Use sqlite3?

- You need a lightweight, single-file database.
- You are working on a small project or script.
- You are comfortable writing raw SQL queries.
- You do not need an ORM or multiple database support.

### Final Recommendation

- For simple SQLite-based projects: Use `sqlite3` (faster, simpler).
- For larger applications needing scalability and maintainability: Use `SQLAlchemy`.

# Sqlalchemy {#sqlalchemy}

SQLAlchemy is a Python SQL toolkit and <mark>Object Relational Mapper</mark> (ORM) that provides tools to interact with databases in a more Pythonic way. It allows you to work with relational databases such as MySQL, PostgreSQL, SQLite, and others without writing raw [SQL](#sql) queries manually.

Related:
- [SQLAlchemy vs. sqlite3](#sqlalchemy-vs-sqlite3)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLAlchemy/sql_alchemy.ipynb

### Why Use SQLAlchemy?

- Reduces SQL complexity: Write Python code instead of SQL queries.
- Prevents [SQL Injection](#sql-injection): ORM prevents unsafe queries.
- Improves [maintainability](#maintainability): Easier to refactor code.
- Handles connection pooling: Manages database connections efficiently.
- Works with Pandas: Can load and save data directly to databases.
### Key Features of SQLAlchemy

1. Database Connectivity
    - Provides a unified interface to connect to different databases.
      
2. SQL Query Execution
    - Allows execution of raw SQL queries using [Pandas](#pandas)
      
3. ORM (Object Relational Mapping)
    - Converts database tables into Python objects (classes).
    - Eliminates the need to write SQL [Querying|Queries](#queryingqueries) manually.
    - Example:
        ```python
        from sqlalchemy.orm import declarative_base
        from sqlalchemy import Column, Integer, String
        
        Base = declarative_base()
        
        class Customer(Base):
            __tablename__ = 'customers'
            id = Column(Integer, primary_key=True)
            name = Column(String)
            phone_number = Column(String)
        ```
4. Transaction Management
    - Provides robust control over commit and rollback operations.
    - Ensures data integrity by handling failures safely.
      
5. Efficient Query Building
    - Allows writing Pythonic queries instead of raw SQL.
    - Example:
        
        ```python
        from sqlalchemy.orm import sessionmaker
        
        Session = sessionmaker(bind=engine)
        session = Session()
        
        customers = session.query(Customer).filter_by(name="John Doe").all()
        ```
        
6. Supports Multiple Databases
    - Works with [[MySql],[PostgreSQL](#postgresql),[SQLite](#sqlite), etc.
    - Easily switch databases without changing the core logic.


# Sqlite Studio {#sqlite-studio}



# Sqlite {#sqlite}


Lightweight [Database Management System (DBMS)|DBMS](#database-management-system-dbmsdbms) used in various applications (phone apps, desktop apps, websites).

Note [SQLite Studio](#sqlite-studio) exists

To get in terminal enter: 

sqlite3 database.db


Related notes:
- [Querying](#querying)
- [Concurrency](#concurrency)
- [SQL](#sql)

# Svm_Example.Py {#svm_examplepy}

https://github.com/rhyslwells/ML_Tools/blob/main\Explorations/Build/Classifiers/SVM/SVM_Example.py

## **Overview**

- **Objective**: To classify Iris flowers using SVM and explore various hyperparameters like kernel type, regularization (C), and gamma.
- **Dataset**: The Iris dataset contains information about sepal and petal dimensions for three flower species.
- To explore the effect of **soft boundaries** in SVMs, you can adjust the regularization parameter CCC. A smaller CCC allows a **softer boundary** (more margin violations), prioritizing generalization. A larger CCC enforces a **harder boundary** with fewer margin violations, but may lead to overfitting. Here's an extended version of the script to include this exploration:

### **Steps in the Script**

#### 1. **Data Loading and Preparation**

- The Iris dataset is loaded using `sklearn.datasets.load_iris`.
- A DataFrame is created with:
    - Features: Sepal and petal dimensions.
    - Target: Numerical representation of flower species.
    - Flower name: Categorical species name derived from the target.

#### 2. **Data Visualization**

- The data is visualized to explore relationships between features:
    - **Sepal Length vs. Sepal Width** for two species (Setosa vs. Versicolor).
    - **Petal Length vs. Petal Width** for the same species.
- Scatter plots are used to identify separable patterns.

#### 3. **Model Training**

- The data is split into training and testing sets (80%-20%).
- An **SVM classifier** (`sklearn.svm.SVC`) is trained on the training set.
- The model's performance is evaluated using the `.score()` method.

#### 4. **Hyperparameter Tuning**

- **Regularization (C)**:
    - Adjusting `C` controls the trade-off between achieving a large margin and minimizing classification errors.
    - Lower values of `C` allow a larger margin but can tolerate misclassified points.
    - Higher values of `C` prioritize correct classification over a larger margin.
- **Gamma**:
    - Controls the influence of individual data points. A high value means data points closer to the hyperplane have more influence.
- **Kernel**:
    - Different kernels (e.g., `linear`, `rbf`) are tested to find the best mapping of data into higher dimensions for better separation.

#### 5. **Prediction and Accuracy**

- The model is used to predict flower species for new samples.
- The accuracy of the model is reported for each combination of hyperparameters.

# Sarsa {#sarsa}


SARSA stands for State-Action-Reward-State-Action

SARSA is another value-based [Reinforcement learning](#reinforcement-learning) algorithm, differing from Q-learning in that it updates the Q-values based on the action actually taken by the policy.

**SARSA update rule:**

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]
$$

**Explanation:**

- **$Q(s_t, a_t)$**: The Q-value of the current state $s_t$ and action $a_t$.
- **$\alpha$**: The learning rate, determining how much new information overrides old information.
- **$r_{t+1}$**: The reward received after taking action $a_t$ from state $s_t$.
- **$\gamma$**: The discount factor, balancing immediate and future rewards.
- **$Q(s_{t+1}, a_{t+1})$**: The Q-value for the next state $s_{t+1}$ and the action $a_{t+1}$ actually taken according to the policy.

**Notes**:

- SARSA’s on-policy nature ensures that it learns a policy that aligns with its exploration strategy, leading to more stable behavior in environments with randomness or noise.
- The learning process may be slower compared to Q-learning, but it can be more robust in environments where the agent’s behavior is expected to align closely with the policy it follows.


# Scala {#scala}



> [!Summary]
> **Scala** is a **functional programming language** primarily used for **[big data](#big-data) processing**, particularly with frameworks like **[Apache Spark](#apache-spark)**. It is known for its **concise syntax** and its ability to integrate seamlessly with the **Java ecosystem**, running on the **JVM** (Java Virtual Machine).

While Scala has a **smaller user base** and is considered **hard to learn**, it is highly **expressive** and offers strong support for managing **distributed systems** and building large-scale data pipelines. Its robust features make it a top choice for **big data engineers**.

### **Key Features of Scala**

1. **Functional Programming**:
   - Scala is built around functional programming principles, offering key features such as:
     - **[Lambdas](#lambdas)** (anonymous functions)
     - **Pattern matching**
     - **Functions as first-class citizens**
     - **Data classes** for concise data modeling.
   
2. **Immutability**:
   - **Immutability** is a core principle in Scala. By default, data structures are **immutable**, which promotes **thread safety** and makes code easier to reason about. This feature aligns well with building reliable and scalable distributed systems.
   
3. **Advanced Language Features**:
   - **Type inference** allows the compiler to deduce types, leading to shorter and cleaner code.
   - **Higher-order types** and **meta-programming** support advanced abstractions and code expressiveness.
   - **Meta programming** allows compile-time code generation, improving type correctness and reducing runtime errors.
   
4. **Expressive Data Manipulation**:
   - Scala is renowned for its **concise and readable code** when it comes to data manipulation. Its type-safe methods provide **powerful tools** for working with data models efficiently and expressively.

5. **Type System**:
   - Scala has an **advanced type system** that enforces **strong typing** at compile time. This system helps prevent illegal states, reducing the need for runtime tests and making the code more robust and reliable.
  
6. **Library Over Framework**:
   - Scala promotes the use of **libraries over frameworks**, which provides developers with more flexibility in how they design and structure their applications.

### **Additional Notes**
- **Integration with [Java](#java)**: Scala can be seamlessly mixed with **Java**, allowing developers to use existing Java libraries and tools.
  
- **Used with Apache Spark**: Scala is the most common language used for **Apache Spark**, a leading big data processing framework.

- **Niche and Learning Curve**: While Scala’s adoption is smaller compared to languages like Java or Python, its **expressiveness** and **power** make it a popular choice for niche applications, especially in big data environments.

# Scalability {#scalability}

Scalability refers to the capability of a system, network, or process to handle a growing amount of work or its potential to accommodate growth.

### Key Benefits of Scalability:

- Performance Improvement: As demand increases, scalable systems can maintain or improve performance levels.
- Cost Efficiency: Organizations can manage costs by scaling resources according to demand, avoiding over-provisioning.
- Flexibility: Scalable systems can adapt to changing workloads and business needs, making it easier to accommodate growth.
- Reliability: Distributing workloads across multiple nodes can enhance system reliability and reduce the risk of a single point of failure.

### Vertical Scalability (Scaling Up):

This involves <mark>adding more resources to a single node</mark> or server to increase its capacity. For example, upgrading a server's CPU, adding more RAM, or increasing storage space. 

Vertical scaling can improve performance but <mark>has limitations,</mark> as there is a maximum capacity that a single machine can reach.

### Horizontal Scalability (Scaling Out):

This involves <mark>adding more nodes or servers</mark> to a system to distribute the load. For example, adding more servers to a web application to handle increased traffic. 

Horizontal scaling allows for greater flexibility and can often be more cost-effective, as it enables the use of multiple lower-cost machines rather than relying on a single powerful machine.

### Tags
- **Tags**: #data_management

# Scaling Agentic Systems {#scaling-agentic-systems}

[Agentic solutions](#agentic-solutions) propose an improvement over traditional Large Language Model ([LLM](#llm)) usage by employing networks of Small Language Models (SLMs). These systems aim to strike a balance between scalability, control, and performance, addressing specific tasks with precision while maintaining overall system adaptability.

Ideas from MLOPs talk by MaltedAI.

Agentic solutions represent a pragmatic approach to AI systems by focusing on modularity, task-specific efficiency, and the thoughtful integration of human expertise. These architectures show promise for enhancing scalability, control, and adaptability in real-world applications.
## Contrasting SLMs and LLMs

[Small Language Models|SLM](#small-language-modelsslm) (Small Language Models):
    - Intent-based conversations and decision trees.
    - Controlled systems, harder to build features but easier to execute.
    - Task-specific and efficient in offline environments.

LLMs (Large Language Models):
    - Flexible and natural in handling diverse queries.
    - Suitable for general-purpose scenarios and exploratory tasks.
    - High computational and scaling costs.

### Combined Approach:

- Use [Small Language Models|SLM](#small-language-modelsslm) for inference and LLMs for training.
- Shift the focus from making models larger to solving real-world problems effectively.
## Key Concepts in Agentic Solutions

1. Neural Decision Parser:
    - Acts as the "brain" of the system, determining the appropriate action given user input.
    - SLMs interpret user utterances to express code aligned with system intent.

1. Phased Policy:
    - Distinguishes between contextual and general-purpose questions.
    - Ensures deliberate task execution in stages for clarity and efficiency.

1. Knowledge Graphs and Interaction Models:
    - Complex graph structures enable intelligent conversations between models.
    - RAG setups leverage teacher-student frameworks for effective task distribution.

1. Distillation Networks of SLMs:
    - SMEs (Subject Matter Experts) guide teacher models that distill their expertise into student SLMs.
    - Enhances task scalability while controlling costs.

1. Scaling with Distillation:
    - Leverage teacher-student frameworks for high-quality, scalable data.
    - Allow teacher models to handle hard-to-scale aspects.

1. Knowledge Discovery:
    - Extract SME knowledge effectively while filtering irrelevant data.
    - Build high-quality datasets for task-specific applications.

## Applications of SLM Networks

1. Task-Specific Systems:
    - Offline processing, task search, and targeted QA.
    - Optimized embedding spaces for domain-specific applications.

1. Swarm Intelligence:
    - Decision-making through deliberation among SLMs.
    - Aggregates diverse inputs (HR, tech, CEO) for robust conclusions.

1. Business Process Models:
    
    - Search and page ranking systems.
    - Smaller, focused systems tailored to specific business needs.



## Designing Agentic Solutions

1. Role of SMEs:
    
    - Define tasks and input structures.
    - Guide model development with domain knowledge.
2. Data Preparation:
    
    - Comprehensive sampling of the problem space ensures generalization.
    - Data variability is critical for robust models.
3. Evaluation and Responsiveness:
    
    - Measure system performance to enable continuous improvement.
    - Focus on responsive, real-time processing.
4. Tool Integration:
    
    - Use LLMs with Python engines or computational tools like Wolfram for data analysis and complex interactions.



## Advantages of SLM Networks

- Precision: Models perform only what they are designed for.
- Efficiency: Smaller models are scalable and cost-effective.
- Focused Applications: Avoids the complexity of embedding spaces for entire businesses.


## Future Directions



# Scaling Server {#scaling-server}

Scaling Server
  - Horizontal Scaling: Adding more servers, preferred for scalability.
  - Vertical Scaling: Adding more resources (memory, CPU) to existing servers.

# Scheduled Tasks {#scheduled-tasks}



Similar to [Cron jobs ](#cron-jobs)in [Unix](#unix)
### **Using `schtasks` (Command-Line)**

Windows provides `schtasks`, a command-line tool to schedule tasks.

#### **Example Commands**

- **Run a Python script every 5 minutes:**
    
    cmd
    
    CopyEdit
    
    `schtasks /create /tn "MyPythonScript" /tr "C:\Python\python.exe C:\path\to\script.py" /sc minute /mo 5 /f`
    
- **Run a batch file daily at 3 AM:**
    
    cmd
    
    CopyEdit
    
    `schtasks /create /tn "DailyBackup" /tr "C:\path\to\backup.bat" /sc daily /st 03:00 /f`
    
- **Delete a scheduled task:**
    
    cmd
    
    CopyEdit
    
    `schtasks /delete /tn "DailyBackup" /f`

# Schema Evolution {#schema-evolution}


[Database Schema|Schema](#database-schemaschema) Evolution means adding new columns without breaking anything or even enlarging some types. 

You can even rename or reorder columns, although that might break backward compatibilities. Still, we can change one table, and the table format takes care of switching it on all distributed files. Best of all does not require rewrite of your table and underlying files.

### How is schema evolution done in practice

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Writing/Schema_Evolution.ipynb

See also:
- [ACID Transaction](#acid-transaction)




# Scientific Method {#scientific-method}


### Step 1: Start with Data

- **Collect Data**: Gather all relevant data sources that might be useful for your analysis.
- **Understand Data**: Familiarize yourself with the data types, structures, and any existing metadata.
- **Clean Data**: Perform data cleaning to handle missing values, outliers, and inconsistencies.

### Step 2: Develop Intuitions

- **Explore Data**: Use exploratory data analysis ([EDA](#eda)) techniques to visualize and summarize the data.
- **Identify Patterns**: Look for trends, correlations, and anomalies that might inform your understanding.
- **Ask Preliminary Questions**: Consider what initial questions the data might help answer.
### Step 3: Formulate Your Question

- [Problem Definition](#problem-definition): Clearly articulate the problem you are trying to solve.
- **Set Objectives**: Determine what you aim to achieve with your analysis.
- **Consider Stakeholders**: Ensure the question aligns with business goals and stakeholder interests.

### Step 4: Validate the Question

- **Test Feasibility**: Use the current data to assess whether the question is answerable.
- **Iterate**: Refine the question based on initial findings and feedback.
- **Formulate Hypothesis**: Develop a testable hypothesis that can guide your analysis.

### Step 5: Create a Testing Framework

- **Design Experiments**: Plan how you will test your hypothesis, including control and experimental groups if applicable.
- **Select Methods**: Choose appropriate statistical or machine learning methods for analysis.
- **Prepare Tools**: Set up the necessary tools and environments for running experiments.

### Step 6: Analyze Results

- **Run Experiments**: Execute your tests and collect results.
- **Interpret Data**: Use quantitative metrics to analyze the outcomes.
- **Draw Insights**: Identify key insights and patterns that answer your question.

### Step 7: Assess Impact

- **Define Success Metrics**: Determine how you will measure success (e.g., accuracy, ROI, user engagement).
- **Evaluate Impact**: Assess the potential impact of your solution on the business.
- **Communicate Findings**: Present your results and recommendations to stakeholders.

### Additional Considerations

- **Iterative Process**: Be prepared to revisit and refine each step as new insights emerge.
- **Documentation**: Keep thorough documentation of your process, findings, and decisions.

# Seaborn {#seaborn}


https://seaborn.pydata.org/tutorial.html

Related:
- [Data Visualisation](#data-visualisation)

# Search {#search}



# Security {#security}


[Common Security Vulnerabilities in Software Development](#common-security-vulnerabilities-in-software-development)



# Semantic Relationships {#semantic-relationships}


Semantic relationships refer to the connections and associations between words and concepts based on their meanings. 

Understanding these relationships can enhance various natural language processing tasks, such as information retrieval, text analysis, and sentiment analysis.

### Leveraging Lexical Resources like [WordNet](#wordnet)

One of the key resources for exploring semantic relationships is **WordNet**, a lexical database that groups words into sets of cognitive synonyms called **synsets**. These synsets are linked together in a hierarchy based on semantic relations, including:

- **Hypernymy**: Represents an "is-a" relationship (e.g., "dog" is a hypernym of "beagle").
- **Hyponymy**: Represents a more specific type (e.g., "beagle" is a hyponym of "dog").

You can use WordNet to find synonyms or related concepts for important words (those with high [TF-IDF](#tf-idf) scores) in your documents. If different documents contain synonyms or words related in the WordNet hierarchy, this may indicate a semantic relationship between them, even if the exact words differ.

WordNet also provides measures of semantic similarity between synsets based on their paths in the hypernym hierarchy. These measures can be explored to quantify the semantic relatedness of key terms in your documents. The Natural Language Toolkit ([NLTK](#nltk)) offers an interface to access WordNet.

### Sentiment Analysis with SentiWordNet

Another valuable resource is **SentiWordNet**, which extends WordNet by assigning sentiment scores (positive, negative, objective) to different senses of words. While your primary goal may be to explore semantic relationships, analyzing the sentiment expressed in your documents based on important words can provide an additional layer of understanding. 

Documents discussing similar topics might also share similar sentiments, strengthening the case for a semantic link. NLTK provides access to SentiWordNet, allowing you to incorporate sentiment analysis into your exploration of semantic relationships.


# Sentence Similarity {#sentence-similarity}

Sentence similarity refers to the degree to which two sentences are alike in meaning. It is a crucial concept in natural language processing ([NLP](#nlp)) tasks such as information retrieval, text summarization, and paraphrase detection. Measuring sentence similarity involves comparing the semantic content of sentences to determine how closely they relate to each other.

There are several methods to measure sentence similarity:

1. **Lexical Similarity**: This involves comparing the words in the sentences directly. Common techniques include:
   - **Jaccard Similarity**: Measures the overlap of words between two sentences.
   - **Cosine Similarity**: Represents sentences as vectors (e.g., using TF-IDF) and measures the cosine of the angle between them.

2. **Syntactic Similarity**: This considers the structure of the sentences, using techniques like:
   - **Parse Trees**: Comparing the syntactic trees of sentences to see how similar their structures are.

3. **Semantic Similarity**: This goes beyond surface-level word matching to understand the meaning of sentences:
   - **Word Embeddings**: Using models like Word2Vec, GloVe, or FastText to represent words in a continuous vector space, then averaging these vectors to represent sentences.
   - **Sentence Embeddings**: Using models like Universal Sentence Encoder, BERT, or Sentence-BERT to directly obtain embeddings for entire sentences, which can then be compared using cosine similarity or other distance metrics.

4. **Neural Network Models**: Advanced models like BERT, RoBERTa, or GPT can be fine-tuned on specific tasks to directly predict similarity scores between sentence pairs.

5. **Hybrid Approaches**: Combining multiple methods to leverage both lexical and semantic information for a more robust similarity measure.

Each method has its strengths and weaknesses, and the choice of method often depends on the specific requirements of the task and the available computational resources.

# Sharepoint {#sharepoint}


SharePoint is a web-based collaboration platform developed by [Microsoft](#microsoft). It is primarily used for creating intranet sites, document management, and team collaboration,  providing a centralized platform for managing content and communication.

SharePoint integrates with Microsoft Office and is highly customizable, making it a versatile tool for organizations of all sizes.

### Key Features

1. **Intranet Sites**:
   - Create internal websites for team collaboration and communication.
   - Share news, updates, and resources within an organization.

2. **Document Repository**:
   - Store, organize, and manage documents in a centralized location.
   - Version control and access permissions ensure document integrity and security.

3. **Lists and Libraries**:
   - Create lists to manage data and tasks.
   - Libraries for storing and organizing documents and other files.

4. **Team and Communication Sites**:
   - Team Sites: Facilitate collaboration within specific teams or projects.
   - Communication Sites: Share information broadly across an organization.

5. **Integration with Microsoft Teams**:
   - SharePoint sites can be integrated with Microsoft Teams channels, providing a seamless collaboration experience.

### Use Cases

- **Document Management**: Centralize document storage and enable easy sharing and collaboration.
- **Project Management**: Use lists and libraries to track project tasks and resources.
- **Internal Communication**: Share company news, updates, and announcements through intranet sites.



# Silhouette Analysis {#silhouette-analysis}

[Sklearn link](https://scikit-learn.org/1.5/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)

Silhouette analysis is a technique used to evaluate the quality of clustering results. It provides a measure of how similar an object is to its own cluster compared to other clusters. This analysis helps in determining the appropriateness of the number of clusters and the consistency within clusters.

Overall, silhouette analysis is a valuable tool for assessing the quality of [clustering](#clustering) results and making informed decisions about the number of clusters and the clustering algorithm's effectiveness.

### Key Concepts:

 **Silhouette Score:** For each data point, the silhouette score is calculated using the following formula:
 
  $s(i) = \frac{b(i)  a(i)}{\max(a(i), b(i))}$
  
  where:
   $a(i)$ is the average distance between the data point $i$ and all other points in the same cluster.
   $b(i)$ is the average distance between the data point $i$ and all points in the nearest cluster (the cluster with the smallest average distance to $i$).

 **Interpretation of Silhouette Scores:**
   A silhouette score ranges from -1 to 1.
   A score close to 1 indicates that the data point is <mark>wellclustered</mark> and far from neighboring clusters.
   A score close to 0 indicates that the data point is on or very close to the decision boundary between two neighboring clusters.
   A negative score indicates that the data point might have been assigned to the wrong cluster.

### Silhouette Plot:

 A silhouette plot displays the silhouette scores of all data points in a dataset. It provides a visual representation of how well each data point lies within its cluster.
 
 The plot is divided into regions, each corresponding to a cluster, and the width of each region represents the average silhouette score of the points in that cluster.

Good
 
![Pasted image 20241231172403.png](../content/images/Pasted%20image%2020241231172403.png)

Bad

![Pasted image 20241231172459.png](../content/images/Pasted%20image%2020241231172459.png)



### Applications:

 **Determining the Optimal Number of Clusters:** By calculating the average silhouette score for different numbers of clusters, one can identify the number of clusters that results in the highest average silhouette score, indicating the best clustering structure.
 
 **Cluster Validation:** Silhouette analysis helps in validating the consistency within clusters and identifying potential misclassifications.



# Similarity Search {#similarity-search}



# Single Source Of Truth {#single-source-of-truth}

Sending data from across an enterprise into a centralized system such as a:

- [Database](#database)
- [Data Warehouse](#data-warehouse)
- [Data Lakehouse](#data-lakehouse)
- [Data Lakehouse](#data-lakehouse)
- [master data management](#master-data-management)

results in a single unified location for accessing and analyzing all the information that is flowing through an organization.



[Single Source of Truth](#single-source-of-truth)
   **Tags**: #data_management, #data_storage

# Sklearn Pipiline {#sklearn-pipiline}


```python
# Naivebayesfor email prediction
from sklearn.pipeline import Pipeline
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])
clf.fit(X_train, y_train)
clf.score(X_test,y_test)
clf.predict(user_input)
```

# Sklearn {#sklearn}


# Terms of interest:

Also called Scikit-learn.

#### train_test_split

X and y are separate things (y is the target variable/column) and X is multiple is columns used to get y.


Given any pandas df use .to_numpy to convert first.

classifier score? 

-0.018 bad
0.72 good 

import #data_cleaning (puts all values between -1 and 1)

skileanr.pipeline allows you to combine steps.

to save model use pickle with 

```
with open(out_file,"wb") as out:
pickle.dump(pipe,out)
```

## [p-values in linear regression in sklearn](#p-values-in-linear-regression-in-sklearn) 

[Sklearn](#sklearn)

# Slowly Changing Dimension {#slowly-changing-dimension}


A Slowly Changing Dimension (SCD) is **a dimension that stores and manages both current and historical data over time in a [Data Warehouse](Data%20Warehouse.md)**.

It is considered and implemented as one of the most critical [ETL](#etl) tasks in tracking the history of dimension records.

## How do you track slowly changing dimensions in a [database](#database)

Take a customer dimension in a retail database.

Consider a retail company that tracks customer information, including attributes such as name, address, and membership status. Over time, customers may change their addresses or upgrade their membership levels. 
#### Implementation of SCD

- The original record for John Doe is retained with an end date to indicate that it is no longer current.
- A new record is created for the updated information, allowing the company to maintain a history of changes over time.
- This approach allows analysts to query the data and understand customer behavior and trends over time, which is essential for reporting and decision-making.

1. **Current Data**: The current state of the customer dimension might look like this:

| Customer ID | Name       | Address               | Membership Status |
|-------------|------------|-----------------------|--------------------|
| 1           | John Doe  | 123 Elm St, City A    | Gold               |
| 2           | Jane Smith | 456 Oak St, City B    | Silver             

1. **Change Occurs**: If John Doe moves to a new address and upgrades his membership to Platinum, the company needs to track this change.

2. **Historical Data**: Using the SCD approach, the [dimension table](#dimension-table) might be updated as follows:

| Customer ID | Name       | Address               | Membership Status | Effective Date | End Date   |
|-------------|------------|-----------------------|--------------------|----------------|------------|
| 1           | John Doe  | 123 Elm St, City A    | Gold               | 2022-01-01     | 2023-10-01 |
| 1           | John Doe  | 789 Pine St, City A   | Platinum           | 2023-10-01     | NULL       |
| 2           | Jane Smith | 456 Oak St, City B    | Silver             | 2022-01-01     | NULL       |


# Small Language Models {#small-language-models}


[LLM|LLMs](#llmllms) dominate many general-purpose NLP tasks, small [Language Models](#language-models) have their own place in specialized tasks, where they excel due to computational efficiency, [interpretability](#interpretability), and task-specific fine-tuning. 

SLMs remain highly relevant for [Edge Machine Learning Models](#edge-machine-learning-models) and edge computing, <mark>domain-specific tasks</mark>, and applications requiring [interpretability](#interpretability), making them a crucial tool in the NLP landscape.

### Use Cases for Small Language Models (SLMs)

- [Contrastive Decoding](#contrastive-decoding): Improve the quality of generated content by filtering out low-quality outputs, by having a SLM guide and critique a LLM or other way ([inference](#inference))
	- Mitigate hallucinations
	- Augmented Reasoning
- [Distillation](#distillation): Transfer the knowledge from a larger model to a smaller one, retaining performance but reducing computational requirements (see [BERT](#bert) Teacher model).
- [Data Synthesis:](#data-synthesis) Generate or augment datasets in scenarios with limited data.
- [Model Cascading](#model-cascading): Use a combination of smaller models and larger models in a cascading architecture, where simpler tasks are handled by SLMs and more complex ones by LLMs. Model cascading and routing allow SMs to handle simpler tasks, reducing computational overhead. Or the other way first do a general search with a LLM then refine to domain specific small model which is more [interpretability|interpretable](#interpretabilityinterpretable) and specific.
- Domain specific & Limited Data Availability: SMs, however, can be <mark>effectively fine-tuned</mark> on smaller, <mark>domain-specific datasets</mark> and outperform general LLMs in tasks with limited data availability.
- [RAG](#rag) (Retrieval Augmented Generation): Lightweight <mark>retrievers</mark> (SMs) can support LLMs in finding relevant external information.

### Advantages of SLMs

- Require less computational power and are faster in [inference](#inference) compared to LLMs.
- [Interpretability](#interpretability)
- Accessible for those without resources in power and data





# Smart Grids {#smart-grids}


## Smart Grids

Want adaptive grid that can handle the volatility of energy coming on or off. This occurs more often due to the variety of sources i.e wind.

Help with carbon commitment  

- **Overview**: Smart grids utilize advanced technology and data analytics to improve the efficiency and reliability of electricity distribution. [RL](#rl) can optimize the operation and management of these grids.
- **Applications**:
    - **Demand Forecasting**: RL algorithms predict electricity demand based on historical data and real-time inputs. They adjust energy production and distribution to match forecasted demand.
    - **Load Balancing**: RL can manage the distribution of electricity by dynamically balancing load across different sources, minimizing energy loss and enhancing stability.
    - **Renewable Energy Integration**: RL helps in integrating renewable energy sources (e.g., solar, wind) into the grid by optimizing the usage of these variable resources and managing their unpredictability.

# Snowflake Schema {#snowflake-schema}

Snowflake Schema
   - Description: A more [Normalised Schema](#normalised-schema) normalized form of a star schema where dimension tables are further broken down into additional tables.
   - Advantages: Reduces data redundancy and can save storage space, but may be more complex to query.
   - A variation of the [Star Schema](#star-schema), the snowflake schema normalizes dimension tables into multiple related tables. This can reduce data redundancy and improve data integrity but may complicate queries due to the additional joins required.

### **Snowflake**

1. **Architecture**:
   - **Cloud-Native**: Snowflake is a fully managed, cloud-native data warehousing service. It operates entirely on cloud platforms like AWS, Azure, and Google [Cloud](#cloud).
   - **Separation of Storage and Compute**: Snowflake separates storage from compute, allowing for independent scaling of each. This means you can scale up compute resources without affecting storage capacity and vice versa.
   - **Multi-Cluster Shared Data Architecture**: Snowflake uses a multi-cluster architecture to handle concurrent workloads, ensuring high performance and minimal contention.

2. **Data Storage**:
   - **Structured Data**: Primarily designed for structured data and optimized for SQL queries and analytics.
   - **Semi-Structured Data**: Also supports semi-structured data like JSON, Avro, and Parquet through its native capabilities.

3. **Management**:
   - **Fully Managed Service**: Snowflake handles infrastructure management, optimization, and maintenance tasks automatically, requiring minimal administrative effort from users.

4. **Performance**:
   - **High Performance**: Optimized for fast query performance, particularly for complex analytical queries. It uses advanced optimizations like automatic clustering and caching.

5. **Use Cases**:
   - **[Data Warehouse](#data-warehouse): Ideal for enterprise data warehousing, business intelligence, and analytics.
   - **Data Lake**: Can function as a data lake with support for semi-structured data.

# Soft Deletion {#soft-deletion}


Soft deletion is a technique used in databases to <mark>mark records as deleted without physically removing them from the database</mark>. 

This approach is particularly useful in scenarios where [data integrity](#data-integrity) and synchronization are important, such as during [Incremental Synchronization](#incremental-synchronization).

When using [incremental synchronization](#incremental-synchronization) modes, fully deleted records from a source system are not replicated. To handle this, a field can be added to each record to indicate whether it should be treated as deleted. This allows the system to maintain a complete history of records while still functioning as if certain records are removed.

## Implementation

A common way to implement soft deletion is by adding a boolean flag, such as `is_deleted`, to the record [Database Schema|schema](#database-schemaschema). Here’s how it works:

1. Flagging Records:
   - When a record is "deleted," the `is_deleted` flag is set to `true`.
  
1. Querying Data:
   - All [Querying|queries](#queryingqueries) must be designed to exclude records where `is_deleted` is `true`. For example:
     ```sql
     SELECT  FROM table_name WHERE is_deleted = false;
     ```

1. Background Jobs:
   - Periodically, background jobs can be executed to permanently remove records marked as deleted, if necessary, or to archive them for future reference.

## Benefits

- Data Integrity: Maintains a complete history of records, which can be useful for auditing and recovery.
- Ease of Recovery: Records can be easily restored by simply resetting the `is_deleted` flag.
- Synchronization: Facilitates incremental synchronization by ensuring that deleted records are still present in the database.

## Considerations

- Query Complexity: Requires careful query design to ensure that deleted records are consistently excluded.
- Storage: Over time, soft-deleted records can accumulate, potentially leading to increased storage requirements.

## Example of Soft Deletion

Let's say we have a table named `users` that stores user information. We will add a boolean column called `is_deleted` to indicate whether a user is "deleted."

In this example, we demonstrated how to implement soft deletion using a boolean flag in the `users` table. This approach allows for easy recovery of deleted records and maintains [data integrity](#data-integrity) while facilitating incremental synchronization.

#### Step 1: Modify the Table Structure

First, we need to alter the `users` table to add the `is_deleted` column:

```sql
ALTER TABLE users ADD COLUMN is_deleted BOOLEAN DEFAULT false;
```

#### Step 2: Soft Delete a User

When a user wants to delete their account, instead of removing the record from the database, we update the `is_deleted` flag:

```sql
UPDATE users SET is_deleted = true WHERE user_id = 123;
```

#### Step 3: Querying Active Users

To retrieve a list of active users (those who are not deleted), we write our queries to exclude soft-deleted records:

```sql
SELECT  FROM users WHERE is_deleted = false;
```

#### Step 4: Restoring a Soft Deleted User

If a user decides to restore their account, we can simply set the `is_deleted` flag back to `false`:

```sql
UPDATE users SET is_deleted = false WHERE user_id = 123;
```

#### Step 5: Permanently Deleting Soft Deleted Users

If we want to permanently remove users who have been soft deleted for a certain period, we can run a background job to delete those records:

```sql
DELETE FROM users WHERE is_deleted = true AND deletion_date < NOW() - INTERVAL '30 days';
```


# Software Design Patterns {#software-design-patterns}


[10 Design Patterns Explained in 10 Minutes](https://www.youtube.com/watch?v=tv-_1er1mWI)

# Software Design Patterns

## What Are Software Design Patterns?

Software design patterns provide reusable solutions to common software design problems. They help standardize approaches, making code easier to understand, maintain, and extend. The influential book _Design Patterns_ by the "Gang of Four" categorizes design patterns into three types:

- **Creational Patterns**: Handle object creation mechanisms.
- **Structural Patterns**: Define how objects and components relate.
- **Behavioral Patterns**: Govern object communication and workflows.

Using design patterns effectively can improve code quality, but excessive or incorrect use may introduce unnecessary complexity.

## Key Software Design Patterns

### Singleton Pattern

The **Singleton pattern** ensures that only one instance of a class exists and provides a global point of access.

Use Cases: [Database](#database) connections, Logging services, Global configurations
#### Example (JavaScript):

```javascript
class Singleton {
  constructor() {
    if (!Singleton.instance) {
      Singleton.instance = this;
    }
    return Singleton.instance;
  }
}
const instance1 = new Singleton();
const instance2 = new Singleton();
console.log(instance1 === instance2); // true
```

### Prototype Pattern

The **Prototype pattern** allows new objects to be created by cloning an existing object rather than instantiating a new class.

Use Cases: Performance optimization, Object cloning

#### Example (JavaScript):

```javascript
const carPrototype = {
  start: function () {
    console.log("Engine started!");
  }
};
const myCar = Object.create(carPrototype);
myCar.start(); // Engine started!
```

### Builder Pattern

The **Builder pattern** simplifies object creation when multiple configuration options exist.

Use Cases: Constructing complex objects, UI component creation

#### Example (JavaScript):

```javascript
class CarBuilder {
  constructor() {
    this.car = {};
  }
  setColor(color) {
    this.car.color = color;
    return this;
  }
  setWheels(wheels) {
    this.car.wheels = wheels;
    return this;
  }
  build() {
    return this.car;
  }
}
const myCar = new CarBuilder().setColor("red").setWheels(4).build();
console.log(myCar);
```

### Factory Pattern

The **Factory pattern** encapsulates object creation logic, making code more modular and easier to extend.

Use Cases: Dependency injection, Platform-specific object creation

#### Example (JavaScript):

```javascript
class CarFactory {
  static createCar(type) {
    const carTypes = {
      sedan: { type: "sedan", doors: 4 },
      coupe: { type: "coupe", doors: 2 },
    };
    return carTypes[type] || null;
  }
}
const myCar = CarFactory.createCar("sedan");
console.log(myCar);
```

### Facade Pattern

The **Facade pattern** provides a simplified interface to a complex system.

Use Cases: Simplifying APIs, Reducing dependencies

#### Example (JavaScript):

```javascript
class Computer {
  start() { console.log("Computer starting..."); }
  shutdown() { console.log("Computer shutting down..."); }
}
class ComputerFacade {
  constructor() {
    this.computer = new Computer();
  }
  turnOn() {
    this.computer.start();
  }
  turnOff() {
    this.computer.shutdown();
  }
}
const myComputer = new ComputerFacade();
myComputer.turnOn();
```

### Proxy Pattern

The **Proxy pattern** acts as an intermediary to control access to an object.

Use Cases: Lazy loading, [Security](#security) proxies

#### Example (JavaScript):

```javascript
class RealImage {
  constructor(filename) {
    this.filename = filename;
  }
  display() {
    console.log("Displaying " + this.filename);
  }
}
class ProxyImage {
  constructor(filename) {
    this.realImage = null;
    this.filename = filename;
  }
  display() {
    if (!this.realImage) {
      this.realImage = new RealImage(this.filename);
    }
    this.realImage.display();
  }
}
const image = new ProxyImage("test.jpg");
image.display();
```

### Iterator Pattern

The **Iterator pattern** provides a way to access elements of a collection sequentially without exposing its internal structure.

Use Cases: Collection traversal, Data processing

#### Example (JavaScript):

```javascript
class Iterator {
  constructor(items) {
    this.items = items;
    this.index = 0;
  }
  next() {
    return this.index < this.items.length ? { value: this.items[this.index++], done: false } : { done: true };
  }
}
const iterator = new Iterator(["a", "b", "c"]);
console.log(iterator.next());
console.log(iterator.next());
console.log(iterator.next());
```

### Observer Pattern

The **Observer pattern** enables a one-to-many dependency between objects, ensuring changes to one object are reflected in its dependents.

Use Cases: Event handling, Reactive programming

#### Example (JavaScript):

```javascript
class Subject {
  constructor() {
    this.observers = [];
  }
  subscribe(observer) {
    this.observers.push(observer);
  }
  notify(data) {
    this.observers.forEach(observer => observer.update(data));
  }
}
class Observer {
  update(data) {
    console.log("Received update: " + data);
  }
}
const subject = new Subject();
const observer1 = new Observer();
subject.subscribe(observer1);
subject.notify("Hello World");
```

### Mediator Pattern

The **Mediator pattern** centralizes communication between objects to reduce dependencies.

Use Cases: Chat applications, Workflow coordination

#### Example (JavaScript):

```javascript
class Mediator {
  constructor() {
    this.participants = [];
  }
  register(participant) {
    this.participants.push(participant);
  }
  send(message, sender) {
    this.participants.forEach(participant => {
      if (participant !== sender) {
        participant.receive(message);
      }
    });
  }
}
class Participant {
  constructor(name, mediator) {
    this.name = name;
    this.mediator = mediator;
    mediator.register(this);
  }
  send(message) {
    this.mediator.send(message, this);
  }
  receive(message) {
    console.log(this.name + " received: " + message);
  }
}
const mediator = new Mediator();
const p1 = new Participant("Alice", mediator);
p1.send("Hello");
```


# Software Development Life Cycle {#software-development-life-cycle}


A structured approach like the Software Development Life Cycle (SDLC) ensures <mark>systematic progression through various stages of development</mark>. SDLC remains relevant today by outlining the essential stages a product must undergo to achieve success.
## SDLC Stages

The SDLC comprises several phases, each critical to the overall development process:

1. **Planning and Analysis**
    - **Purpose**: Collect business and user requirements, perform cost and time estimation, and conduct scoping activities.
    - **Activities**: Define what the final product must do and how it should work. This phase may include a separate requirements analysis stage.

2. **Designing**
    - **Purpose**: Prepare the product's architecture and design.
    - **Activities**: A software architect sets the high-level structure of the future system, selecting technology and drafting user experience and visual design.

3. **Development**
    - **Purpose**: Transform the design into actual code.
    - **Activities**: Programmers write code according to the design specifications, revealing some aspects of the final product to stakeholders.

4. **Testing**
    - **Purpose**: Ensure the quality and functionality of the product.
    - **Activities**: Testers and QA professionals review the code and usability, identifying bugs and errors for correction before deployment.

5. **Deployment**
    - **Purpose**: Release the product to users.
    - **Activities**: Launch the product, making it available for use. Often combined with the maintenance phase.

6. **Maintenance**
    - **Purpose**: Provide ongoing support and updates.
    - **Activities**: Gather user feedback, fix issues, and add new features as needed.

Additionally, some projects include a **Prototyping** stage between planning and designing to validate ideas with minimal effort and cost.

## SDLC Methodologies

### Waterfall

The Waterfall model, strictly follows the SDLC stages sequentially. <mark>Each phase must be completed before the next begins, making it straightforward and easy to manage</mark>. However, its lack of flexibility and late testing phase often lead to delays and budget overruns when changes are needed. Out of favour. Opposite of flexible, needs roll backs.

- detail documentation

### Agile

- Produce only essential documentation. Collaborative effort. 
- Focuses on features users will like.

In response to the rigid Waterfall model, Agile emerged in the early 2000s with a <mark>focus on flexibility and continuous delivery.</mark> Agile methodologies like Scrum, Lean, and Extreme Programming (XP) integrate testing throughout the development process and welcome changes even in late stages.

See agile manifesto : Working software over documentation

- **Scrum**: Breaks development into short cycles called <mark>sprints</mark>, each lasting 1-4 weeks. At the end of each sprint<mark>, a functional product increment is presented to stakeholders,</mark> allowing for quick adaptation based on feedback.
- **Lean**: Aims to <mark>eliminate waste</mark> through a build-measure-learn feedback loop, continuously refining the product. Steps: build, measure, learn.
- **Extreme Programming (XP)**: Emphasizes technical excellence with practices such as test-driven development, code refactoring, and pair programming.
- Kanban: management method for efficiency. 

### [DevOps](#devops)

# Sparsecategorialcrossentropy Or Categoricalcrossentropy {#sparsecategorialcrossentropy-or-categoricalcrossentropy}

To understand the differences and use cases for `SparseCategoricalCrossentropy` and `CategoricalCrossentropy` in [TensorFlow](#tensorflow), let's break down each one:

### CategoricalCrossentropy

- **Use Case**: This [loss function](#loss-function) is used when you have one-hot encoded labels. [One-hot encoding](#one-hot-encoding) means that each label is represented as a vector with a length equal to the number of classes, where the correct class is marked with a 1 and all other classes are marked with 0s.
- **Example**: If you have three classes, a label might look like `[0, 1, 0]` for class 2.
- **Functionality**: It calculates the [cross entropy](#cross-entropy) loss between the true labels and the predicted probabilities.

### SparseCategoricalCrossentropy

- **Use Case**: This loss function is used when your labels are integers instead of one-hot encoded vectors. Each label is represented by a single integer corresponding to the correct class.
- **Example**: If you have three classes, a label might simply be `1` for class 2.
- **Functionality**: It also calculates the cross-entropy loss but expects the labels to be in integer form, which can be more memory efficient.

### Key Differences

- **Input Format**: The main difference is the format of the labels. `CategoricalCrossentropy` requires one-hot encoded labels, while `SparseCategoricalCrossentropy` works with integer labels.
- **Efficiency**: `SparseCategoricalCrossentropy` can be more efficient in terms of memory and computation, especially when dealing with a large number of classes.

### When to Use Which

- Use `CategoricalCrossentropy` if your labels are already one-hot encoded or if you prefer to work with one-hot encoded labels for any specific reason.
- Use `SparseCategoricalCrossentropy` if your labels are integers, which is often the case when labels are directly loaded from datasets.



# Specificity {#specificity}


**Specificity**, also known as the true negative rate, measures the proportion of actual negatives that are correctly identified by the model. It indicates how well the model is at identifying negative instances. Formula:

$$\text{Specificity} = \frac{TN}{TN + FP}$$
Importance
- Specificity is crucial in scenarios where it is important to correctly identify negative instances, such as in medical testing where a false positive could lead to unnecessary treatment.



# Spreadsheets Vs Databases {#spreadsheets-vs-databases}

Compared to spreadsheets, databases offer:

1. [Scalability](#scalability): Databases are designed to handle large volumes of data, making them suitable for applications with millions or even billions of records. In contrast, spreadsheets can become unwieldy and slow when dealing with large datasets.

2. Update Frequency: Databases support real-time updates and continuous operations, allowing for dynamic [data management](#data-management). Spreadsheets, on the other hand, are more static and may require manual updates, which can lead to outdated information.

3. Speed: Databases are optimized for quick access, retrieval, and manipulation of data. They can efficiently handle multiple concurrent users, making them ideal for environments where data is frequently accessed and modified. Spreadsheets can lag in performance under similar conditions.

### Tags
- **Tags**: #data_management, #data_storage

# Stacking {#stacking}

What is [Stacking](#stacking)?;; is an [Model Ensemble](#model-ensemble) combines predictions of multiple base models <mark>by training a meta-model</mark> on the outputs of the base models.



# Standard Deviation {#standard-deviation}

Standard deviation is a statistical measure that quantifies the amount of variation or dispersion in a set of data values. It indicates how much individual data points deviate from the mean (average) of the dataset.

## Formula

For a dataset with $n$ observations $X_1, X_2, \ldots, X_n$, the standard deviation $\sigma$ is calculated using the formula:

$$
\sigma = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (X_i - \mu)^2}
$$

Where:
- $\sigma$ = standard deviation
- $n$ = number of observations
- $X_i$ = each individual observation
- $\mu$ = mean of the dataset, calculated as:  
$$
\mu = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

## Why Standard Deviation is Preferred Over [Variance](#variance)

1. **Same Units as Data**  
   Standard deviation is expressed in the same units as the original data, making it more [interpretability|interpretable](#interpretabilityinterpretable).  
   - **Example:** If you measure height in centimeters, the standard deviation will also be in centimeters.  
   - **Contrast:** Variance is expressed in squared units (e.g., square centimeters), which can be less intuitive to understand.

2. **Direct Interpretation**  
   Standard deviation provides a direct measure of the average distance of data points from the mean.  
   - A **small standard deviation** indicates that the data points are close to the mean.  
   - A **large standard deviation** suggests that the data points are more spread out.

3. **Normal [Distributions|Distribution](#distributionsdistribution) Context**  
   In the context of a normal distribution, standard deviation helps in understanding the spread of data:  
   - Approximately **68%** of the data falls within **one standard deviation** of the mean.  
   - About **95%** falls within **two standard deviations**.  
   - About **99.7%** falls within **three standard deviations** (known as the empirical rule).  
   This property is particularly useful for identifying [standardised/Outliers](#standardisedoutliers).

4. **Ease of Communication**  
   Standard deviation is more intuitive and easier to communicate to a broader audience, including those without a strong statistical background. Its direct relation to the data makes it a preferred choice for explaining variability.


# Standardisation {#standardisation}

Standardisation is a [Preprocessing|data preprocessing](#preprocessingdata-preprocessing) technique used to [Data Transformation](#data-transformation) features so that they have a mean of 0 and a standard deviation of 1. Centers data with zero mean and unit variance, suitable for algorithms sensitive to variance.

Definition: Standardisation involves rescaling the features of your data so that they have a mean of 0 and a standard deviation of 1. This is achieved by subtracting the mean of the feature from each data point and then dividing by the standard deviation.

Purpose: 
- Useful for algorithms that assume the data is normally distributed.
- Uniformity: It helps in bringing all features to the same scale.

### Use Case

- Centred Data Assumption: Standardisation is beneficial when the model assumes that the data is centred around zero. This is common in algorithms such as linear regression, logistic regression, and [principal component analysis](#principal-component-analysis) (PCA), and distance-based algorithms like [K-nearest neighbours|KNN](#k-nearest-neighboursknn) and [Gradient Descent](#gradient-descent) descent optimization.
  
- Improved Performance: It can improve the performance and convergence speed of machine learning algorithms by ensuring that each feature contributes equally to the result.
### Formula

The formula for standardisation is:

$$z = \frac{x - \mu}{\sigma}$$

Where:
- $x$ is the original data point.
- $\mu$ is the mean of the feature.
- $\sigma$ is the standard deviation of the feature.

By applying this transformation, the data becomes more suitable for training models that rely on the assumption of [Gaussian Distribution](#gaussian-distribution).

<mark>Rescales the data to have a mean of 0 and a standard deviation of 1</mark> (unit variance). This method is particularly useful when the data follows a Gaussian distribution.

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_standardized = scaler.fit_transform(df)  # Rescales each feature to have mean 0 and std deviation 1
```

# Star Schema {#star-schema}

Star Schema
   - This schema consists of a central fact table surrounded by dimension tables. The fact table contains quantitative data, while the dimension tables provide descriptive attributes. The star schema is easy to understand and query, making it popular for OLAP applications.



[Star Schema](#star-schema)
   - Description: A simple and widely used form of dimensional modeling where a central fact table is connected to multiple dimension tables.
   - Advantages: Easy to understand and query, with straightforward joins between fact and dimension tables.

# Statistical Assumptions {#statistical-assumptions}

Statistical assumptions are essential conditions that must be met for various statistical methods and models to produce valid results. Necessary for robustness and reliability of statistical analyses. If any assumptions are violated, it may be necessary to employ alternative statistical methods or to transform the data accordingly.

Purpose of Statistical Assumptions:
- [Data Analysis](#data-analysis): Ensures that the chosen statistical methods are appropriate for the data.
- [Interpretability](#interpretability): Facilitates accurate interpretation of results and conclusions drawn from analyses.

#### Key Assumptions:

- [Assumption of Normality](#assumption-of-normality): This assumption posits that the data follows a normal distribution. Many [Statistical Tests](#statistical-tests), such as t-tests and ANOVA, rely on this assumption to validate their results. If the data is not normally distributed, alternative methods or transformations may be necessary. Heavy tailed distributions can violate this.

- Homoscedasticity: This refers to the assumption that the variance of the residuals (errors) remains constant across all levels of the independent variable(s). Violations of this assumption can lead to inefficient estimates and impact [hypothesis testing](#hypothesis-testing).

- <mark>Independence</mark>: This assumption states that the observations in the dataset should be independent of one another. Dependence among observations can result in biased estimates and incorrect conclusions.

- Normality of Residuals: In regression analysis, it is assumed that the residuals (the differences between observed and predicted values) are normally distributed. This is critical for conducting hypothesis tests on regression coefficients.

#### Broader Categories of Assumptions:

Model Assumptions: These are overarching assumptions that apply to specific statistical models. For example:
  - Linear Regression: Assumes a <mark>linear relationship</mark> between the independent and dependent variables.
  - [Logistic Regression](#logistic-regression): Assumes a binary outcome for the dependent variable.

Distribution Assumptions: Different statistical tests make specific assumptions about the distribution of the data:
  - Parametric Tests: Assume that the data follows a certain distribution (e.g., normal).
  - Non-parametric Tests: Do not require such distributional assumptions and can be applied to data that does not meet these criteria.
#### Additional Considerations:

Testing Assumptions: It is important to test these assumptions before conducting statistical analyses. Common methods include:
  - Visual Inspection: Using plots (e.g., Q-Q plots, residual plots) to visually assess normality and homoscedasticity.
  - [Statistical Tests](#statistical-tests): Employing tests like the Shapiro-Wilk test for normality or Levene's test for homoscedasticity.

Consequences of Violating Assumptions: Understanding the implications of assumption violations is crucial. For example, violating the assumption of normality can lead to:
  - Increased Type I or Type II error rates.
  - Misleading confidence intervals and p-values.

Transformations and Alternatives: When assumptions are violated, consider:
  - Data Transformations: Such as log, square root, or Box-Cox transformations to meet assumptions.
  - Alternative Methods: Using robust statistical techniques that are less sensitive to assumption violations, such as bootstrapping or non-parametric tests.
  - Contextual Relevance: The relevance of specific assumptions may vary depending on the context of the analysis and the nature of the data. Always consider the specific characteristics of the dataset when evaluating assumptions.

# Statistical Tests {#statistical-tests}

Statistical tests are methods used to determine if there is a significant difference between groups or if a relationship exists between variables. Each test has its specific [Statistical Assumptions](#statistical-assumptions) and applications.
## Types of Statistical Tests

[Z-Test](#z-test)

[T-test](#t-test)

[Chi-Squared Test](#chi-squared-test)

[Proportion Test](#proportion-test)

## Test Statistics

For each statistical test, a test statistic is calculated. This statistic measures the degree of deviation from the null hypothesis ([Hypothesis testing](#hypothesis-testing)). The [estimator](#estimator) is centered by the population mean, and then it is divided by the population standard deviation, a process known as [Standardisation](#standardisation).

# Statistics {#statistics}


Statistics want to understand the world. The world is made of probabilities, we model probabilities with functions, and we model functions with parameters.

"Observe data and construct models, infer and refine hypotheses "

Portal for all statistics notes:

Statistical theorems:
- Asymptotic Theorem: Law of large numbers: Sample mean approaches the population mean.
	- finite mean assumption

[Statistical Assumptions](#statistical-assumptions)

[Type 1 error and Power](#type-1-error-and-power)

[Distributions](#distributions)

[Statistical Tests](#statistical-tests)

[Monte Carlo Simulation](#monte-carlo-simulation)

[Logistic Regression](#logistic-regression): model how change and covariance influence the odds of an event

[Proportional Hazard Model](#proportional-hazard-model): time to an event

[Hypothesis testing](#hypothesis-testing)
[p values](#p-values)
[Confidence Interval](#confidence-interval)

[Central Limit Theorem](#central-limit-theorem)

[Correlation vs Causation](#correlation-vs-causation)

[Markov chain](#markov-chain)

[parametric vs non-parametric tests](#parametric-vs-non-parametric-tests)

[Multicollinearity](#multicollinearity)

[univariate vs multivariate](#univariate-vs-multivariate)

[R](#r)
[tidyverse](#tidyverse): visualisation in R

[Over parameterised models](#over-parameterised-models)

[Casual Inference](#casual-inference)

[Bootstrap](#bootstrap)

[Adaptive decision analysis](#adaptive-decision-analysis): interrupting the experiment in the middle

Estimation Problems: using data to estimate model parameters
- [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
- [Expectation Maximisation Algorithm](#expectation-maximisation-algorithm)

[Likelihood ratio](#likelihood-ratio): [Type 1 error and Power](#type-1-error-and-power) UMP test. used to maximise power. [T-test](#t-test) is a consequence of this.

Dashboards/Animations: Shiny, gganimate

[Estimator](#estimator)



# Stemming {#stemming}

Shorting words to the key term.

# Stochastic Gradient Descent {#stochastic-gradient-descent}


[Gradient Descent](#gradient-descent)

# Stored Procedures {#stored-procedures}

Stored procedures are a way to automate SQL statements, allowing them to be executed repeatedly without rewriting the code.

**Demonstration with the Boston MFA Database**  

We will use the Boston MFA database to illustrate stored procedures. Previously, we implemented a soft-delete feature for the `collections` table using views. Now, we will create a stored procedure to achieve similar functionality.

1. **Select the Database**:
   ```sql
   USE `mfa`;
   ```

2. **Add a Deleted Column**:  
   The `deleted` column needs to be added to the `collections` table to track soft deletions.
   ```sql
   ALTER TABLE `collections` 
   ADD COLUMN `deleted` TINYINT DEFAULT 0;
   ```
   The `TINYINT` type is appropriate since the column will only hold values of 0 or 1, with a default of 0 to retain all existing collections.

3. **Change the Delimiter**:  
   Before creating a stored procedure, change the delimiter to allow multiple statements.
   ```sql
   delimiter //
   ```

4. **Create the Stored Procedure**:  
   Define the stored procedure to select current collections that are not marked as deleted.
   ```sql
   CREATE PROCEDURE `current_collection`()
   BEGIN
       SELECT `title`, `accession_number`, `acquired` 
       FROM `collections` 
       WHERE `deleted` = 0;
   END//
   ```

5. **Reset the Delimiter**:  
   After creating the procedure, reset the delimiter back to the default.
   ```sql
   delimiter ;
   ```

6. **Call the Stored Procedure**:  
   Execute the procedure to see the current collections.
   ```sql
   CALL current_collection();
   ```

7. **Soft Delete an Item**:  
   If we soft-delete an item, such as “Farmers working at dawn,” and call the procedure again, the deleted row will not appear in the output.
   ```sql
   UPDATE `collections` 
   SET `deleted` = 1 
   WHERE `title` = 'Farmers working at dawn';
   ```

### Parameters in Stored Procedures

Stored procedures can accept parameters. For example, we can create a procedure to handle the sale of artwork.

1. **Create the Transactions Table**:
   ```sql
   CREATE TABLE `transactions` (
       `id` INT AUTO_INCREMENT,
       `title` VARCHAR(64) NOT NULL,
       `action` ENUM('bought', 'sold') NOT NULL,
       PRIMARY KEY(`id`)
   );
   ```

2. **Create the Sell Procedure**:  
   This procedure updates the `collections` table and logs the transaction.
   ```sql
   delimiter //
   CREATE PROCEDURE `sell`(IN `sold_id` INT)
   BEGIN
       UPDATE `collections` SET `deleted` = 1 
       WHERE `id` = `sold_id`;
       INSERT INTO `transactions` (`title`, `action`)
       VALUES ((SELECT `title` FROM `collections` WHERE `id` = `sold_id`), 'sold');
   END//
   delimiter ;
   ```

3. **Call the Sell Procedure**:  
   To sell a specific item, call the procedure with the item's ID.
   ```sql
   CALL `sell`(2);
   ```

### Considerations

- **Multiple Calls**:  
  If the `sell` procedure is called with the same ID multiple times, it may lead to multiple entries in the `transactions` table. Logic can be added to prevent this.

- **Programming Constructs**:  
  Stored procedures can be enhanced with programming constructs available in MySQL, allowing for more complex logic.


# Strongly Vs Weakly Typed Language {#strongly-vs-weakly-typed-language}


A **strongly typed** programming language is one where <mark>types</mark> are strictly enforced. This means that once a variable is assigned a type, it cannot be implicitly converted to another type without an explicit conversion. The goal is to <mark>minimize errors related to incorrect type handling,</mark> as the compiler or interpreter will detect type mismatches early in the development process.

### Characteristics of a Strongly Typed Language:
1. **Type Enforcement**: The language does not allow operations between incompatible types (e.g., trying to add a string to an integer).
2. **Explicit Conversions**: If you need to change the type of a variable, you must explicitly convert it (casting). The compiler or interpreter won't do it automatically.
3. **Compile-time/Runtime Type Checking**: The language performs thorough checks either at compile time (for compiled languages) or at runtime (for interpreted languages) to ensure type safety.

### Example: 

In [Java](#java), which is a strongly typed language:
```java
int number = 5;
String text = "Hello";

// The following line will result in an error since you cannot add an integer to a string directly:
text = text + number;  // Type mismatch error

// You must explicitly convert the integer to a string to perform this operation:
text = text + Integer.toString(number);  // Correct
```
### Benefits:
- **Error Prevention**: Type mismatches are caught early, reducing runtime errors.
- **Code Clarity**: Since types are explicitly defined, it’s easier to understand what kind of data is being handled.
- **Efficiency**: Some strongly typed languages can optimize code better due to the predictability of data types.

### Contrast with Weakly Typed Languages:

Weakly typed languages, like [JavaScript](#javascript), allow implicit type conversions, leading to more flexibility but also potential runtime errors due to unexpected conversions:
```javascript
let number = 5;
let text = "Hello";

// JavaScript allows this, and it implicitly converts the number to a string:
text = text + number;  // No error, result is "Hello5"
```


# Structuring And Organizing Data {#structuring-and-organizing-data}

Structuring and organizing data.

In [DE_Tools](#de_tools) see:
	- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/multi_index.ipynb
	- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/reshaping.ipynb

Related terms:
- [Multi-level index](#multi-level-index)

# Summarisation {#summarisation}


## Summarization in NLP

Summarization in natural language processing (NLP) is the process of condensing a text document into a shorter version while retaining its main ideas and key information. There are two primary forms of summarization:

The unsupervised summarization process involves <mark>splitting text, tokenizing sentences, assigning scores based on importance, and selecting top sentences</mark>. Effective scoring methods include calculating sentence <mark>similarity</mark> and analyzing <mark>word frequencies</mark> to ensure that the summary captures the essence of the original text.

[Extraction](#extraction):
- This method involves selecting specific words or sentences directly from the original text to create a summary. It focuses on identifying and pulling out the most important parts of the text without altering the original wording.
[Abstraction](#abstraction):
- The abstraction method generates a summary that may include new words and phrases not present in the original text. This approach is more complex as it requires understanding the content and rephrasing it, often using techniques like paraphrasing.

### Unsupervised Summarization Process

The basic idea behind unsupervised summarization involves the following steps:

1. **Split Text into Sentences**: The text is divided into individual sentences for analysis.
  
2. **Tokenize Sentences**: Each sentence is tokenized into separate words, allowing for detailed examination of word usage.

3. **Assign Scores to Sentences**: Sentences are evaluated based on their importance, which is a crucial step in the summarization process.

4. **Select Top Sentences**: The highest-scoring sentences are selected and displayed in their original order to form the summary.

### Methods for Assigning Scores

The main point of summarization is effectively assigning scores to sentences. Here are some common methods for doing this:

- **<mark>Similarity</mark> Calculation**: Calculate the similarity between each pair of sentences and select those that are most similar to the majority of sentences. This helps identify sentences that capture the central themes of the text.

- **Word Frequencies**: Analyze word frequencies to identify the most common words in the text. Sentences that contain a higher number of these frequent words are then selected for the summary.


# Supervised Learning {#supervised-learning}


Supervised learning is a type of machine learning where an algorithm learns from **<mark>labeled data</mark>** to make predictions or decisions. 

In supervised learning, the training data consists of input-output pairs, where each input (features) is associated with a known output (label or target). 

The algorithm's goal is to learn a mapping from the input to the output so that it can predict the output for new, unseen data.
### Examples of Supervised [Machine Learning Algorithms](#machine-learning-algorithms):
- [K-nearest neighbours](#k-nearest-neighbours) (KNN)
- [Naive Bayes](#naive-bayes)
- [Decision Tree](#decision-tree)
- [Linear Regression](#linear-regression)
- [Support Vector Machines](#support-vector-machines) (SVM)
### Key Characteristics of Supervised Learning:
1. **Labeled Data**: The training dataset contains both the input data and the corresponding correct outputs (labels).
2. **Training Phase**: The model is trained using this labeled data to minimize errors in predicting the output.
3. **Prediction**: After training, the model can predict the output (label) for new data based on what it learned.
### Types of Supervised Learning Algorithms:

**Supervised learning algorithms** learn from <mark>labeled data,</mark> where each example is associated with a target label. 

Labelled data can look like the following:

| Email Content                              | Label      |
|--------------------------------------------|------------|
| "Congratulations! You won a free iPhone."  | Spam       |
| "Meeting scheduled for 2 PM tomorrow."     | Not Spam   |
| "Special offer: Buy 1 get 1 free!"         | Spam       |
| "Please find the attached report."         | Not Spam   |
Labelling can be expensive and time consuming.

- **[Classification](#classification)**: Predicts discrete labels (e.g., categories).
- **[Regression](#regression)**: Predicts continuous values.
### Example:
In a **house price prediction** task:
- The input features could be the size of the house, number of rooms, and location.
- The output (label) would be the price of the house.

The model is trained on a dataset where the house prices are known (labeled data), and it learns to predict the price for new houses.
#### Code implementation

That is there is a y_train, then uses to get y_pred (from X_test) and compare against y_test.


![Pasted image 20241012152141.png](../content/images/Pasted%20image%2020241012152141.png)

# Support Vector Classifier (Svc) {#support-vector-classifier-svc}


## Overview

**Support Vector Classifiers (SVCs)** are a fundamental component of [Support Vector Machines|SVM](#support-vector-machinessvm)s, designed to find the optimal hyperplane that separates data into distinct classes. The primary objective of an SVC is to maximize the margin between different classes, ensuring that the separation is as clear as possible.

## Key Concepts

- **Hyperplane**: A decision boundary that separates different classes in the feature space. In a two-dimensional space, this is a line; in higher dimensions, it becomes a plane or hyperplane.
- **Support Vectors**: The data points that are closest to the hyperplane. These points are critical as they define the position and orientation of the hyperplane.
- **Margin**: The distance between the hyperplane and the nearest data point from either class. SVC aims to maximize this margin to improve classification robustness.

## SVC vs. SVM

- **SVC**: Primarily focuses on placing a hyperplane between datasets for separation. It is effective when the data is linearly separable.
- **SVM**: Extends the concept of SVC by using kernel functions to handle cases where data is **not linearly separable** in its original space. Kernels transform the data into a higher-dimensional space where a linear separation is possible.

# Support Vector Machines {#support-vector-machines}


Support Vector Machines (SVM) are a type of [supervised learning](app://obsidian.md/supervised%20learning) algorithm primarily used for [classification](#classification) tasks, though they can also be adapted for regression. The main idea is to find an optimal hyperplane that divides data into different classes by maximizing the margin between them. The support vectors are the data points closest to the hyperplane, influencing its position and orientation.

Key Features
- Hyperplane: Finds a hyperplane that maximizes the margin between classes.
- High-Dimensional Spaces: Robust in high-dimensional spaces, such as image and text classification.

Advantages
- Highly effective for high-dimensional data (datasets with many features).
- Useful for classification tasks where a clear margin of separation exists between classes.

Disadvantages
- Can be computationally expensive for large datasets and sensitive to the choice of hyperparameters.
- Performance is highly dependent on the [Kernelling](#kernelling) choice, requiring careful tuning.

## How SVM Works

In [ML_Tools](#ml_tools) see: [SVM_Example.py](#svm_examplepy)

1. Initial Space: Start in the low-dimensional space, where the data may not be linearly separable.
2. Kernel Function: Use a [Kernelling](app://obsidian.md/Kernelling) function to move the data into a higher dimension where separation is clearer.
3. Hyperplane Placement: Place hyperplanes (decision boundaries) between the data clusters to classify the data.

## Margins

- Outliers and Soft Margins: SVM allows for some miscalculations or errors in classification to handle outliers. This is part of the [Bias and variance](app://obsidian.md/Bias%20and%20variance) tradeoff, where the model is allowed to make a few mistakes to improve generalization.
- Soft Margins: Allow some data points to be within the margin or even on the wrong side of the hyperplane, enabling SVM to handle imperfect separations.

![Pasted image 20240128193726.png|700](../content/images/Pasted%20image%2020240128193726.png|700)

![Pasted image 20240128193838.png|700](../content/images/Pasted%20image%2020240128193838.png|700)






# Support Vector Regression {#support-vector-regression}

Support Vector Regression use similar principles to [Support Vector Machines|SVM](#support-vector-machinessvm)s but for predicting continuous variables.



# Symbolic Computation {#symbolic-computation}


[Mathematical Reasoning in Transformers](#mathematical-reasoning-in-transformers)
### Summary of Wolfram Alpha’s Approach:
1. **Uses symbolic computation** with precise algorithms.
2. **Leverages predefined mathematical rules** for various domains.
3. **Provides step-by-step solutions** to explain problem-solving processes.
4. **Handles natural language inputs** and translates them into mathematical expressions.
5. **Produces both exact and numerical solutions**, depending on the problem type.
6. **Visualizes results** with graphs and interactive displays.
7. **Accesses a curated knowledge base** for real-world data integration.

Example: Wolfram alpha

Wolfram Alpha is designed specifically for symbolic computation and uses rule-based algorithms to perform exact and precise mathematical operations. Here's an overview of how Wolfram Alpha processes math problems:

### 1. **Symbolic Computation Engine**
   - Wolfram Alpha is powered by **[Mathematica](#mathematica)**, a robust computational engine that specializes in **symbolic computation**. Unlike neural network models, which rely on statistical pattern recognition, symbolic computation manipulates symbols and formulas directly according to established mathematical rules. This allows Wolfram Alpha to handle a wide variety of mathematical problems with precision, from basic arithmetic to complex calculus and algebraic manipulations.
### 2. **Predefined Mathematical Rules and Algorithms**
   - Wolfram Alpha uses **predefined algorithms and mathematical rules** that are coded into the system.
   - Each mathematical operation is treated according to formal rules, so Wolfram Alpha can handle exact symbolic results (like expressing results in terms of radicals or π) or provide numerical approximations when needed.

### 3. **Step-by-Step Solutions**



### 5. **Natural Language Processing (NLP)**
   - Wolfram Alpha uses **NLP** to interpret queries that are entered in natural language. For example, a user might type “solve x^2 + 2x + 1 = 0” or simply “solve quadratic equation,” and Wolfram Alpha translates this into formal mathematical expressions to process through its symbolic engine.
   - This NLP capability allows users to input problems in a variety of ways, making it more accessible to non-expert users.

### 6. **Exact vs. Numerical Solutions**
   - Wolfram Alpha can handle both <mark>**exact symbolic solutions**</mark> (such as expressing a result in terms of fractions, square roots, or constants like π) and <mark>**numerical approximations**</mark>. For example, for equations that have no simple symbolic solutions, it can provide highly accurate numerical answers using **numerical methods** such as Newton’s method or Monte Carlo simulations.

### 7. **Knowledge-Based System**
   - Wolfram Alpha is connected to a vast database of curated knowledge, not only in mathematics but across many disciplines. This allows it to draw on data, formulas, and algorithms to solve not only pure math problems but also applied math problems, such as in physics, engineering, and economics.

### 8. **Graphing and Visualization**

### 9. **Error Handling and Interpretation**
   - Wolfram Alpha handles user input carefully, identifying potential errors or ambiguities. For example, if an equation is underdetermined (too few equations for the number of variables), it may provide a parametric solution. If input is ambiguous, it often offers multiple possible interpretations or asks the user to clarify.



# Sympy {#sympy}



# Semantic Layer {#semantic-layer}


A [Semantic Layer](semantic%20layer.md) is much more flexible and makes the most sense on top of [transformed data](Data%20Transformation.md) in a [Data Warehouse](Data%20Warehouse.md).

A semantic layer in the context of a data warehouse is an abstraction layer that sits between the raw data stored in the warehouse and the end users who need to access and analyze that data.

Its primary purpose is to simplify complex data structures and present them in a more user-friendly and business-oriented way. This allows users to interact with the data without needing to understand the underlying complexities of the database schema or query languages.

Bridging the gap between complex data systems and business users, enabling more effective and efficient data-driven decision-making.

Avoid extensive reshuffles or reprocesses of large amounts of data. 

Think of [OLAP](standardised/OLAP%20(online%20analytical%20processing).md) cubes where you can dice-and-slice ad-hoc on significant amounts of data without storing them ahead of time

### Key Features of a Semantic Layer

1. Business-Friendly Terminology:
   - Translates technical database terms into business-friendly language that is easier for non-technical users to understand.
   - For example, instead of using column names like `cust_id` or `prod_sku`, the semantic layer might present them as "Customer ID" or "Product SKU."

2. Data Abstraction:
   - Hides the complexity of the underlying data model, such as joins, table structures, and data transformations.
   - Users can focus on business concepts rather than technical details.

3. Consistent [Metric](#metric) and Calculations:
   - Provides a centralized definition of key metrics and calculations, ensuring consistency across reports and analyses.
   - For example, a metric like "Total Revenue" would be consistently calculated and presented, regardless of who is querying the data.

4. Security and Access Control:
   - Implements security rules and access controls to ensure that users only see data they are authorized to access.
   - This can include row-level security, column-level security, and user-specific data views.

5. Enhanced Query Performance:
   - Optimizes queries by pre-aggregating data or using materialized views, reducing the load on the data warehouse and improving response times for users.

### Benefits of a Semantic Layer

- Ease of Use: Makes it easier for business users to access and analyze data without needing deep technical knowledge.
- Faster Insights: Users can quickly generate reports and dashboards using familiar business terms and concepts.
- Consistency: Ensures that all users are working with the same definitions and calculations, reducing discrepancies in reporting.
- Scalability: Supports a wide range of analytical tools and applications, allowing organizations to scale their data analytics capabilities.

### Implementation

A semantic layer can be implemented using various tools and technologies, such as:

- Business Intelligence (BI) Tools: Many BI platforms, like [Tableau](#tableau), [PowerBI](#powerbi), and Looker, offer built-in semantic layer capabilities.
- [Data Virtualization](#data-virtualization) Tools: Tools like Denodo or Dremio provide semantic layer functionality by creating virtual views of data.
- Custom Solutions: Organizations can build custom semantic layers using middleware or data modeling tools.





# Semi Structured Data {#semi-structured-data}



Semi-structured data is data that lacks a rigid structure and that does not conform directly to a data model, but that has tags, metadata, or elements that describe the data. 

Examples of semi-structured data are JSON or [XML](#xml) files. 

<mark>Semi-structured data often contains enough information that it can be relatively easily converted into </mark> [structured data](#structured-data). 

[JSON](#json) data embedded inside of a string, is an example of semi-structured data. The string contains all the information required to understand the structure of the data, but is still for the moment just a string -- it hasn't been structured yet.

|          | **data**                        |
| -------- | ------------------------------- |
| Record 1 | \"{'id': 1, 'name': 'Mary X'}\" |
| Record 2 | \"{'id': 2, 'name': 'John D'}\" |


It is often relatively straightforward to convert semi-structured data into structured data. Converting semi-structured data into structured data is often done during the [Data Transformation](Data%20Transformation.md) stage in an [ETL](ETL.md) or [ELT](term/elt.md) process.  





# Shapefile {#shapefile}


A shapefile is a popular geospatial vector data format for geographic information system (GIS) software. It is widely used for storing the location, shape, and attributes of geographic features. Developed by Esri, shapefiles are commonly used in the GIS community for exchanging and managing geospatial data.

A shapefile is a widely used [GIS](#gis) vector data format consisting of multiple files that store both spatial geometry and attribute data. Its ease of use and broad compatibility have made it a standard format for geospatial data exchange and analysis in the GIS community.
### Components of a Shapefile

A shapefile is not a single file, but rather a set of several files that work together. The primary components include:

1. **.shp file**: This file contains the geometry data (points, lines, polygons) that represents the spatial features.

2. **.shx file**: This is an index file that allows for quick access to the geometry data in the .shp file.

3. **.dbf file**: This file stores attribute data in tabular format, linked to the spatial data in the .shp file. It uses the dBASE format to hold the attributes of each shape, such as names, categories, or other descriptive information.

In addition to these mandatory files, a shapefile can also include several optional files that provide additional information:

4. **.prj file**: Contains the coordinate system and projection information for the spatial data. This file is crucial for ensuring that the data is displayed correctly in GIS software.

5. **.cpg file**: Defines the character encoding to be used for the .dbf file, ensuring that text attributes are interpreted correctly.

6. **.qix file**: An optional spatial index file that can improve the performance of spatial queries on the shapefile.

### Characteristics of Shapefiles

- **Geometry Types**: Shapefiles can store different types of geometric data, including points, lines, and polygons. However, each shapefile can contain only one type of geometry.
- **Attribute Data**: The .dbf file allows shapefiles to store descriptive data about each spatial feature, which can be used for analysis and mapping.
- **Limitations**: Shapefiles have some limitations, such as a maximum file size of 2 GB for each component file, lack of support for advanced geometric types (like curves), and potential data redundancy and inefficiencies.

### Usage of Shapefiles

Shapefiles are extensively used in GIS for various purposes, including:

- **Mapping**: Displaying geographic features on maps for visualization.
- **Spatial Analysis**: Performing spatial queries, analysis, and geoprocessing tasks.
- **Data Exchange**: Sharing geospatial data between different GIS software and systems.

### Example Scenario

Consider a city planning department that wants to map all the parks within the city. They might use a shapefile to store the polygon geometries representing park boundaries along with attributes such as park names, areas, and facilities available. This shapefile can then be loaded into GIS software to create maps, analyze park distributions, and manage urban planning tasks.




# Sklearn Datasets {#sklearn-datasets}

make a dataframe by 

```python
ds = datasets.load_dataset()
df = pd.DataFrame(ds.data,columns=ds.feature_names)
df.head()
#add target column
df['target'] = ds.target


# Spacy {#spacy}



# Storage Layer Object Store {#storage-layer-object-store}


A storage layer or object storage are services from the three big [Cloud Providers](#cloud-providers), 

AWS S3,[S3 bucket](#s3-bucket)
Azure Blob Storage,
and Google Cloud Storage. 

The web user interface is easy to use. **Its features are very basic, where, in fact, these object stores store distributed files exceptionally well.** They are also highly configurable, with solid security and reliability built-in.


# Structured Data {#structured-data}


Structured data refers to data that has been formatted into a well-defined schema ([Database Schema](#database-schema)). An example would be data that is stored with precisely defined columns in a relational [Database](#database) or excel spreadsheet. Examples of <mark>structured fields could be age, name, phone number, credit card numbers or address.</mark> Storing data in a structured format allows it to be easily understood and queried by machines and with tools such as SQL.

## Example of structure data

Below is an example of structured data as it would appear in a database:

|         |  **age**| **name**| **phone**| 
|---------|-----|------|-----|
|Record 1| 29 | Bob | 123-456 |
|Record 2| 30 | Sue | 789-123 | 

It may seem that all data is structured, but this is not always the case -- data can be unstructured, or semi-structured. The differences are best understood by example, and are discussed in the following sections. 

## Structured data vs. unstructured data

Structured data can be contrasted with [unstructured data](unstructured%20data.md), which does not conform to a data model and has no easily identifiable structure. Unstructured data cannot be easily used by programs, and is difficult to analyze. Examples of unstructured data could be the contents of an email, contents of a word document, data from social media, photos, videos, survey results, etc.   

An simple example of unstructured data is a string that contains interesting information inside of it, but that has not been formatted into a well defined schema. An example is given below:

|               |  **UnstructuredString**|
|---------| -----------|
|Record 1| "Bob is 29" |
|Record 2| "Mary just turned 30"|

## Structuring of unstructured data

Converting unstructured data into structured data can be done during the [Data Transformation](Data%20Transformation.md) stage in an [ETL](ETL.md) or [ELT](term/elt.md) process.  

For example, in order to efficiently make use of the unstructured data given in the previous example, it may desirable to transform it into structured data such as the following:

|               |  **name** | **age** |
|---------| -----------|---- |
|Record 1| "Bob" | 29 |
|Record 2| "Mary"| 30 |

Storing the data in a structured manner makes it much more efficient to query the data. For example, after structuring the data it is possible to easily and efficiently execute the following query on the structured data:
  
``` SQL
SELECT * FROM X where Age=29
```

A query such as this would be expensive and/or more difficult to execute on unstructured data.

## Structured data vs. semi-structured data

Structured data can also be contrasted with [semi-structured data](term/semi-structured%20data.md), which lacks a rigid structure and does not conform directly to a data model. However, semi-structured data has tags and elements that describe the data. 

Examples of semi-structured data are [JSON](#json) or [XML](#xml) files. Semi-structured data often contains enough information that it can be relatively easily converted into structured data. 

<mark>[structured data](term/structured%20data.md) refers to data that has been formatted into a well-defined schema</mark>. An example would be data that is stored with precisely defined columns in a relational database or excel spreadsheet. Examples of structured fields could be age, name, phone number, credit card numbers or address.

# Syntactic Relationships {#syntactic-relationships}

Syntactic relationships refer to the structural connections between words or phrases in a sentence, focusing on grammar and the arrangement of words. They determine how words combine to form phrases, clauses, and sentences, following the rules of syntax.

[Semantic relationships](#semantic-relationships), on the other hand, deal with the meaning and interpretation of words and phrases. They focus on how words relate to each other in terms of meaning, such as synonyms, antonyms, and hierarchical relationships like hypernyms and hyponyms.

The key difference is that syntactic relationships are concerned with the form and structure of language, while semantic relationships are concerned with meaning and interpretation.