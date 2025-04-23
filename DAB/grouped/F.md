

# Faiss {#faiss}


FAISS (Facebook AI [Similarity Search](#similarity-search)) is a library developed by Facebook AI Research that enables efficient similarity search and [clustering](#clustering) of dense vectors. It is especially well-suited for applications involving high-dimensional vector data, such as [NLP](#nlp)

Related terms:
- [Vector Embedding](#vector-embedding)
### Overview

FAISS is optimized for:
- **Fast retrieval** from large collections of vectors (millions or more).
- **Approximate nearest neighbor (ANN)** search, which trades off accuracy for speed.
- **Exact search**, depending on the chosen index type.
- **GPU acceleration** for very large-scale search tasks.

### Core Concept

At its core, FAISS takes a large number of **high-dimensional vectors** (e.g., sentence or document embeddings), and enables fast **similarity search** to retrieve the most similar vectors to a given [Querying|query](#queryingquery).

For example, in an NLP [Memory|context](#memorycontext):
- Documents or notes are embedded into vector space using a model like SBERT.
- These embeddings are stored in a FAISS index.
- Given a query, its embedding is computed, and FAISS returns the nearest neighbors (i.e., most semantically similar notes).
### Index Types

FAISS offers different types of indices depending on use case:
- `IndexFlatL2`: exact search using L2 (Euclidean) distance.
- `IndexIVFFlat`: approximate search using inverted files.
- `IndexHNSW`: Hierarchical Navigable Small World graph-based index (good for high recall).
- `IndexPQ`: product quantization for memory-efficient indexing.




# Fabric {#fabric}


Fabric is a unified analytics platform that operates in the cloud, eliminating the need for local installations. It provides an integrated environment for data analysis, similar to how [Microsoft](#microsoft) Office serves as a suite for productivity tasks.

## Key Features

- Unified Platform: Combines various data tools and services into a single platform, streamlining data analysis and management.
- Cloud-Based: Operates entirely in the cloud, ensuring accessibility and scalability without the need for local installations.
- Integrated Environment: Offers a cohesive environment for data analysis, integrating tools like [Data Factory](#data-factory), [Synapse](#synapse), and [PowerBI](#powerbi).

## Components

- Data Factory: Facilitates data integration and transformation.
- Synapse: Acts as a [Relational Database](#relational-database) and [Data Warehouse](#data-warehouse), supporting [T-SQL](#t-sql) for data management.
- [Data Lake](#data-lake): Fabric provides open data storage solutions, allowing for efficient data management and analysis.
- OneLake: Offers workspaces for different departments, enabling data sharing and referencing without duplication.

## Technologies

- Programming Languages: Supports [Scala](#scala) and [PySpark](#pyspark) for data processing within the [Data Lake](#data-lake).
- PowerBI Integration: Enhances data visualization and querying capabilities within Fabric, offering faster insights.

## Advantages

- Data as Fuel: Recognizes data as the essential component powering AI and analytics.
- No ETL Required: Mirrors data sources, eliminating the need for [ETL](#etl) processes when source data is edited.
- Cross-Workspace Shortcuts: Allows departments to reference data across workspaces without creating copies.
- Copilot with PowerBI: Integrates AI-driven insights and automation within PowerBI for enhanced data analysis.

# Fact Table {#fact-table}

A fact table is a central component of a star [Database Schema|schema](#database-schemaschema) or snowflake schema in a [data warehouse](#data-warehouse), it stores [Facts](#facts).

Essential for [data analysis](#data-analysis) in a data warehouse, providing the numerical data that can be analyzed in conjunction with the descriptive data stored in dimension tables.

1. **Measures**: Fact tables contain measurable, quantitative data known as "facts" or "measures." Examples include sales revenue, quantity sold, profit, or any other numeric data that can be aggregated.

2. **Foreign Keys**: Fact tables include foreign keys that reference [Dimension Table](#dimension-table)s. These keys link the fact table to related dimensions, allowing for contextual analysis. For example, a sales fact table might include foreign keys for dimensions such as time, product, and customer.

3. **Granularity**: The [granularity](#granularity) of a fact table refers to the level of detail stored in the table. For example, a sales fact table might store data at the transaction level (each sale) or at a higher level (daily sales totals).

4. **Large Size**: Fact tables can become quite large, as they often store a significant amount of transactional data over time. This is in contrast to dimension tables, which are usually smaller and contain descriptive attributes.

5. **Aggregation**: Fact tables are often used for aggregation and analysis, allowing users to perform operations such as summing, averaging, or counting the measures.

Example:
  - Columns: `DateKey`, `ProductKey`, `RegionKey`, `SalesAmount`, `UnitsSold`



[Fact Table](#fact-table)
   **Tags**: #data_modeling, #data_warehouse

# Factor Analysis {#factor-analysis}


Factor Analysis (FA) is a statistical method used for:
- [dimensionality reduction](#dimensionality-reduction),
- [EDA](#eda)
- or latent variable detection

It identifies underlying relationships between observed variables by modeling them as linear combinations of a smaller number of <mark>unobserved latent factors</mark>.

In simpler terms, it helps reduce a large number of variables into fewer factors while retaining the core information and structure of the data. It assumes that observed variables are influenced by some common latent factors and unique errors.

### Key Features of Factor Analysis:

1. Latent Factors: These are unobserved variables that capture the shared variance among observed variables.
2. Variance Decomposition: FA splits the total variance of observed variables into:
    - Common variance: Shared by latent factors.
    - Unique variance: Specific to each observed variable.

In [ML_Tools](#ml_tools) see: [Factor_Analysis.py](#factor_analysispy)

### Next Steps:

1. Would you like to visualize the factors to understand how the data clusters in the new latent space?
2. Should we explore the relationships between the factors and target classes (e.g., species in the Iris dataset)?

# Factor_Analysis.Py {#factor_analysispy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Preprocess/Factor_Analysis.py
### **1. Factor Loadings Table**

This table shows how strongly each feature (e.g., `sepal length (cm)`) is correlated with the two extracted factors (Factor 1 and Factor 2).

- **Rows**: Represent the extracted factors (`Factor 1` and `Factor 2`).
- **Columns**: Represent the original features of the Iris dataset.
- **Values**: Represent the "loading" or contribution of each feature to the factor. Higher absolute values indicate a stronger relationship between a feature and a factor.

#### Interpretation:

- **Factor 1 (Row 0)**:
    - Strongly influenced by `petal length (cm)` (loading = 1.757902) and `petal width (cm)` (loading = 0.731005).
    - Moderately influenced by `sepal length (cm)` (loading = 0.727461).
    - Weak negative contribution from `sepal width (cm)` (loading = -0.180852).
    - This factor might represent the size of petals and sepals.
      
- **Factor 2 (Row 1)**:
    - Weak contributions from all features, with slightly negative contributions from `sepal length (cm)` and `sepal width (cm)`.
    - This factor might capture a subtle relationship or orthogonal variance not well-defined by the dataset.

---

### **2. Explained Variance**

The explained variance values indicate how much of the dataset's total variance is captured by each factor.
- **Factor 1 (0.9988)**: This factor explains ~99.88% of the variance among the features.
- **Factor 2 (0.9039)**: This factor explains ~90.39% of the variance among the features.

#### Combined Variance:
The two factors together capture a large portion of the total variance in the dataset. This suggests that most of the information in the dataset can be reduced to two latent factors, simplifying its structure while retaining the core relationships.

### **Overall Interpretation**

1. **Dimensionality Reduction**: The dataset with four features can effectively be reduced to two latent factors while retaining most of its variance.
2. **Factor 1 Dominates**: Factor 1 has strong contributions from `petal length`, `petal width`, and `sepal length`. This factor likely represents size-related characteristics.
3. **Factor 2 is Subtle**: Factor 2 shows weaker relationships with the features, potentially capturing noise or orthogonal variance.

### Next Steps:

1. Would you like to visualize the factors to understand how the data clusters in the new latent space?
2. Should we explore the relationships between the factors and target classes (e.g., species in the Iris dataset)?

### Breakdown of Extensions:

1. **Visualization**:  
    After performing factor analysis, we visualize how the data clusters in the new latent space (Factor 1 vs. Factor 2). The points are colored based on the species (target classes), which helps us see if the factors capture any clustering patterns related to species.
    
    - **Plot Details**:
        - The x-axis represents **Factor 1**.
        - The y-axis represents **Factor 2**.
        - Each species is plotted in different colors to visualize possible separations.
2. **Exploring Factor-Target Relationships**:  
    We compute the **average factor values** for each species. This shows how the latent factors (Factor 1 and Factor 2) relate to the different species in the dataset.
    
    - **Interpretation**:
        - If any species tends to cluster around specific values of Factor 1 and Factor 2, it suggests that the extracted factors capture some species-specific variance.

### Next Steps:

- The plot should give a clear idea of whether the latent factors allow for a meaningful separation of species.
- The summary table of average factor values will help understand how the factors relate to the target variable.

# Facts {#facts}

Facts are quantitative data points that are typically stored in the [Fact Table](#fact-table).

They represent measurable events or metrics, such as sales revenue or quantities sold.

# Fastapi {#fastapi}

**FastAPI** is a modern web framework for building APIs with Python. It is designed to be fast and easy to use, leveraging Python's type hints to provide features like:

1. Automatic generation of OpenAPI documentation.
2. Input data validation based on Python's type annotations.
3. Asynchronous request handling with native support for `asyncio`.
4. High performance, as it is built on Starlette and [Pydantic](#pydantic).
### Key Features

- **Automatic validation:** Based on type hints and Pydantic models.
- **Interactive API docs:** Automatically generated Swagger UI and ReDoc.
- **Asynchronous support:** Full support for async functions.
- **Dependency injection:** Built-in support for dependencies.

### How to Run

<mark>In [ML_Tools](#ml_tools) see: [FastAPI_Example.py](#fastapi_examplepy)</mark> <- see this

1. Save the script as `main.py`.
2. Install FastAPI and Uvicorn:
    `pip install fastapi uvicorn`
    
3. Run the server:  
	```cmd
	ren FastAPI_Example.py main.py # possibly change to
	uvicorn FastAPI_Example:app --reload
	```
    
4. Open the browser and navigate to:
    - **API documentation (Swagger UI):** `http://127.0.0.1:8000/docs`
    - **ReDoc documentation:** `http://127.0.0.1:8000/redoc`

# Fastapi_Example.Py {#fastapi_examplepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Deployment/FastAPI_Example.py
### Explanation of New Features

1. **Path and Query Parameter Metadata**: Added descriptions and constraints for better validation and autogenerated documentation.
2. **Nested Models**: Demonstrated hierarchical data validation with the `User` model that includes a list of `Item` instances.
3. **Partial Updates**: Introduced a `PATCH` endpoint to allow partial updates of fields using the `Body` method.
4. **Static Data**: Provided a status endpoint that returns static information about the API.
5. **Returning Key-Value Data**: Added a summary endpoint to showcase mock data.

# Main

### **What the Script Does**

1. **Starts the FastAPI Application**  
    When you run the script, it starts a web server using Uvicorn. This makes your FastAPI app accessible via the browser or API clients like Postman.
    
2. **Default Endpoint (`GET /`)**  
    The browser sends a `GET` request to the root path (`http://127.0.0.1:8000/`).  
    The root route (`@app.get("/")`) is defined in the script to return:
    
    ```python
    {"message": "Welcome to the expanded FastAPI example!"}
    ```
    
    This is why you see that message in the browser or the API response.
    
3. **404 for `/favicon.ico`**  
    The browser automatically tries to load a favicon (a small icon displayed in the browser tab). Since no favicon is defined in the script, a `404 Not Found` is returned, which is normal behavior.
    

---

### **What the Script Offers Beyond the Root Endpoint**

The script defines several API endpoints that are not automatically accessed unless you explicitly call them. Here’s a summary of the key routes:

|**Endpoint**|**HTTP Method**|**Description**|
|---|---|---|
|`/`|`GET`|Returns a welcome message.|
|`/items/{item_id}`|`GET`|Fetches an item by its `item_id`, with an optional query parameter.|
|`/search/`|`GET`|Searches items using `limit` and `offset` query parameters for pagination.|
|`/items/`|`POST`|Creates a new item using the `Item` model for validation.|
|`/items/{item_id}`|`PUT`|Updates an item by its `item_id` with a new `Item` object.|
|`/items/{item_id}`|`DELETE`|Deletes an item by its `item_id`.|
|`/users/`|`POST`|Creates a new user with optional nested items using the `User` model.|
|`/items/{item_id}`|`PATCH`|Partially updates fields (like `price` or `on_offer`) of an item by its ID.|
|`/status/`|`GET`|Returns static data, such as the API's status and version.|
|`/summary/`|`GET`|Returns a dictionary with some mock data, such as total items and users.|

---

### **Testing the Script**

You need to explicitly visit or call other endpoints to explore more features of the script. For example:

1. <mark>**Accessing the Swagger UI** Go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.</mark>  
    This interactive interface shows all the available endpoints and lets you test them directly.
    
2. **Calling Specific Endpoints** You can call the endpoints via:
    - **Browser** (e.g., `http://127.0.0.1:8000/items/123?q=test`).
    - **API Clients** like Postman or Curl.

---

### **Why You See Only the Welcome Message**

You’ve accessed only the root endpoint (`/`). The other features of the script (like creating, updating, or searching for items) require you to call the respective endpoints explicitly.

# Calling endpoints

Here's how to call the respective endpoints of the script using various tools like **browser**, **Curl**, or **Postman**. Each example demonstrates an endpoint and its purpose.
### **1. Welcome Endpoint (`GET /`)**

- **Purpose:** Displays a welcome message.
- **Call:**
    - **Browser:** Open `http://127.0.0.1:8000/`.
    - **Curl:**
        ```bash
        curl -X GET http://127.0.0.1:8000/
        ```
    - **Response:**
        ```json
        {"message": "Welcome to the expanded FastAPI example!"}
        ```
### **2. Retrieve an Item (`GET /items/{item_id}`)**

- **Purpose:** Fetches item details by its `item_id` with an optional query parameter `q`.
- **Example:** Retrieve item `3` with query `test`.
- **Call:**
    - **Browser:** Open `http://127.0.0.1:8000/items/3?q=test`.
    - **Curl:**
        
        ```bash
        curl -X GET "http://127.0.0.1:8000/items/3?q=test"
        ```
    - **Response:**
        ```json
        {"item_id": 3, "query": "test"}
        ```

---

### **3. Search Items (`GET /search/`)**

- **Purpose:** Uses `limit` and `offset` query parameters for pagination.
- **Example:** Limit results to 5, skip the first 2.
- **Call:**
    - **Browser:** Open `http://127.0.0.1:8000/search/?limit=5&offset=2`.
    - **Curl:**
        ```bash
        curl -X GET "http://127.0.0.1:8000/search/?limit=5&offset=2"
        ```     
    - **Response:**
        ```json
        {"limit": 5, "offset": 2}
        ```
### **4. Create an Item (`POST /items/`)**

- **Purpose:** Adds a new item.
- **Example:** Create an item named "Laptop" priced at 999.99.
- **Call:**
    - **Curl:**
        ```bash
curl -X POST "http://127.0.0.1:8000/items/" -H "Content-Type: application/json" -d "{\"name\": \"Laptop\", \"price\": 999.99, \"description\": \"High-end laptop\", \"on_offer\": true}"
        ```
    - **Response:**
        ```json
        {
          "message": "Item created successfully",
          "item": {
            "name": "Laptop",
            "price": 999.99,
            "description": "High-end laptop",
            "on_offer": true
          }
        }
        ```
### **5. Update an Item (`PUT /items/{item_id}`)**

- **Purpose:** Updates item details by its `item_id`.
- **Example:** Update item `3` with new data.
- **Call:**
    - **Curl:**
        
        ```bash
        curl -X PUT "http://127.0.0.1:8000/items/3" \
        -H "Content-Type: application/json" \
        -d '{"name": "Smartphone", "price": 499.99, "description": "Updated phone", "on_offer": false}'
        ```

    - **Response:**
        
        ```json
        {
          "item_id": 3,
          "updated_item": {
            "name": "Smartphone",
            "price": 499.99,
            "description": "Updated phone",
            "on_offer": false
          }
        }
        ```

### **6. Delete an Item (`DELETE /items/{item_id}`)**

- **Purpose:** Deletes an item by its `item_id`.
- **Example:** Delete item `2`.
- **Call:**
    - **Curl:**
        
        ```bash
        curl -X DELETE http://127.0.0.1:8000/items/2
        ```
        
    - **Response:**
        
        ```json
        {"message": "Item 2 deleted successfully"}
        ```
        

---

### **7. Create a User (`POST /users/`)**

- **Purpose:** Creates a user, optionally with nested items.
- **Example:** Create a user with items.
- **Call:**
    - **Curl:**
        ```bash
		curl -X POST "http://127.0.0.1:8000/users/" -H "Content-Type: application/json" -d "{\"username\": \"john_doe\", \"email\": \"john@example.com\", \"full_name\": \"John Doe\", \"items\": [{\"name\": \"Tablet\", \"price\": 299.99, \"description\": \"Portable tablet\"}]}"
        ```
        
    - **Response:**
        
        ```json
        {
          "message": "User created successfully",
          "user": {
            "username": "john_doe",
            "email": "john@example.com",
            "full_name": "John Doe",
            "items": [
              {
                "name": "Tablet",
                "price": 299.99,
                "description": "Portable tablet",
                "on_offer": null
              }
            ]
          }
        }
        ```
### **8. Partially Update an Item (`PATCH /items/{item_id}`)**

- **Purpose:** Partially updates an item using optional fields.
- **Example:** Update the price of item `5`.
- **Call:**
    - **Curl:**
        
        ```bash
        curl -X PATCH "http://127.0.0.1:8000/items/5" \
        -H "Content-Type: application/json" \
        -d '{"price": 79.99}'
        ```
        
    - **Response:**
        
        ```json
        {
          "item_id": 5,
          "updates": {
            "price": 79.99
          }
        }
        ```
### **9. Check API Status (`GET /status/`)**

- **Purpose:** Returns static information like API status.
- **Call:**
    - **Browser:** Open `http://127.0.0.1:8000/status/`.
    - **Curl:**
        
        ```bash
        curl -X GET http://127.0.0.1:8000/status/
        ```
        
    - **Response:**
        
        ```json
        {"status": "API is running", "version": "1.0.0"}
        ```
        

### **10. Retrieve a Summary (`GET /summary/`)**

- **Purpose:** Returns a mock summary of items and users.
- **Call:**
    - **Browser:** Open `http://127.0.0.1:8000/summary/`.
    - **Curl:**
        
        ```bash
        curl -X GET http://127.0.0.1:8000/summary/
        ```
        
    - **Response:**
        
        ```json
        {"total_items": 42, "total_users": 5, "recent_activity": "Item purchase"}
        ```
        

---

### **Testing Tips**

- Use **Swagger UI** at `http://127.0.0.1:8000/docs` for interactive testing of all endpoints.
- Use **Postman** for testing more complex requests with nested data.

# New 

### **Explanation of Changes**:

- **`created_items`**: Items that have been created using the `/items/` POST endpoint.
- **`updated_items`**: Items that have been updated using the `/items/{item_id}` PUT endpoint.
- **`deleted_items`**: Items that have been deleted using the `/items/{item_id}` DELETE endpoint.
- **Summary Endpoint**:
    - Returns counts of items created (`created_items_count`), updated (`updated_items_count`), and deleted (`deleted_items_count`).

---
Here's the formatted content for use in `cmd`:

<mark>TEST LATER</mark>
### **Testing with `curl`**:

1. **Create an item**:
    
    ```bash
    curl -X POST "http://127.0.0.1:8000/items/" -H "Content-Type: application/json" -d '{"name": "Tablet", "price": 299.99, "description": "Portable tablet"}'
    ```
    
2. **Update an item**:
    
    ```bash
    curl -X PUT "http://127.0.0.1:8000/items/1" \
    -H "Content-Type: application/json" \
    -d '{"name": "Tablet", "price": 250.00, "description": "Updated portable tablet"}'
    ```
    
3. **Delete an item**:
    
    ```bash
    curl -X DELETE "http://127.0.0.1:8000/items/1"
    ```
    
4. **Get the summary**:
    
    ```bash
    curl -X GET "http://127.0.0.1:8000/summary/"
    ```
    

### **Expected Response**:

```json
{
  "total_items": 0,
  "total_users": 5,
  "recent_activity": "Item purchase",
  "created_items_count": 1,
  "updated_items_count": 1,
  "deleted_items_count": 1
}
```

You can copy and paste these commands directly into `cmd` to test the API.
### **Note**:

Make sure your FastAPI server is running on `http://127.0.0.1:8000` before executing these commands.

# Feature Engineering {#feature-engineering}


Its the term given to the iterative process of building good features for a better model. Its the process that makes relevant features (using formulas and relations between others). 

We use it when we have a refined and optimised model.

What does it involve
- Create new features from existing ones (e.g., ratios, interactions).
- Transform features to better capture non-linear relationships.
- [Dimensionality Reduction](#dimensionality-reduction) if necessary.

The main techniques of feature engineering:
- are selection (picking subset), 
- learning (picking the best), 
- extraction and combination(combining).

Example:
Predicting house prices. Raw features might be square footage, number of bedrooms, and location. Feature engineering could involve: Combining square footage and bedrooms into a "living space" feature.

**Example**:
- Decompose datetime information into separate features for date and time to capture their respective predictive powers.


![C1_W2_Lab07_FeatureEngLecture.png](../content/images/C1_W2_Lab07_FeatureEngLecture.png)

# Feature Evaluation {#feature-evaluation}



# Note

Garbage in garbae out. It is the features that 
# What is involved:

Want to assess the  **usefulness** of chosen features
### **Measuring feature importance:** 

Quantifying the contribution of each feature to the model's predictions. This can be done through various methods like statistical tests, model-specific importance scores, or permutation importance.
### **Analysing feature relationships:** 

Investigating correlations and redundancy,

### **Assessing feature impact on model performance:** 


# Example:



# When are we done:

- **Reaching a stable and satisfactory model performance
- **No further room for improvement
- **Understanding the model's decision-making process



# Feature Extraction {#feature-extraction}


### Summary:

In machine learning, **Feature extraction** is the process of transforming raw data into a set of useful features that can be effectively used by algorithms. It involves identifying and selecting key attributes or characteristics of the data that are most relevant to the problem at hand. Feature extraction helps improve both the performance and efficiency of machine learning models.

Feature extraction simplifies complex data by transforming it into a smaller set of <mark>informative features.</mark> Techniques like [Dimensionality Reduction](#dimensionality-reduction) help retain the most significant information, improving model performance and [interpretability](#interpretability). It is conceptually similar to the way the [Attention mechanism](#attention-mechanism) in [LLM](#llm)s captures relationships between concepts, or how [Activation atlases](#activation-atlases) visualize key patterns in deep learning models.

### Key Concepts of Feature Extraction:

1. **Similarity and Relations**:
   - Feature extraction can be compared to how the [Attention mechanism|Attention mechanism](#attention-mechanismattention-mechanism) works in large language models (LLMs), which identify relationships and similarities between concepts. For example, the analogy "King - Queen ~ Man - Woman" highlights how certain features (gender, royalty) can be extracted to understand the underlying relationship between words.
   
   Similarly, in feature extraction, relationships between data points can be used to capture important aspects of the data, such as patterns or correlations, and transform them into features that a model can learn from.

2. [Dimensionality Reduction](#dimensionality-reduction):
   - One of the key techniques in feature extraction is **[Dimensionality Reduction](#dimensionality-reduction)**, which is used to reduce the number of features while still preserving the important information in the data. This involves compressing the data into a smaller set of features that capture most of its variance. By doing this, you improve the efficiency of the machine learning model and make the analysis more interpretable. Allowing to focus on the most important features while <mark>reducing noise</mark> and redundancy.

3. **Visual Feature Extraction**:
   - In the case of complex data like images, techniques such as [Activation atlases](#activation-atlases) can be used to visualize and understand the features that are being extracted and activated within a neural network. These atlases show how different neurons in a neural network respond to specific features or patterns within the data, giving insights into what the model "sees" as important.




# Feature Importance {#feature-importance}



Feature importance refers to <mark>techniques that assign scores to input features</mark> (predictors) in a machine learning model to <mark>indicate their relative impact on the model's predictions.</mark>

Feature importance is typically assessed <mark>after</mark> [Model Building](#model-building). It involves analyzing the trained model to determine the impact of each feature on the predictions.

Feature importance helps in:

- improving model [interpretability](#interpretability), 
- identifying key predictors, 
- and possibly performing [Feature Selection](#feature-selection) to reduce dimensionality, and refining performance

The <mark>outcome</mark> is a ranking or scoring of features based on their importance.

By understanding which features contribute the most to the predictions, you can focus on the most relevant information in your data and potentially reduce model complexity without sacrificing performance.
### Types of Feature Importance Methods

1. Model-Specific Methods:
    - Tree-based models: Models like Random Forests, Gradient Boosted Trees, and Decision Trees have built-in mechanisms for calculating feature importance. They do so based on the decrease in impurity (e.g., [Gini Impurity](#gini-impurity) in classification tasks or variance in regression tasks) or based on the reduction in error when the feature is used for splitting.
    - Linear models: In models like linear regression or logistic regression, feature importance can be derived from the absolute values of the model coefficients, assuming features are standardized.
   
2. Model-Agnostic Methods:
    - Permutation importance: This method measures the importance of a feature by randomly shuffling its values and observing the impact on the model's performance. The larger the decrease in performance, the more important the feature is.
    - [SHapley Additive exPlanations](#shapley-additive-explanations)
    - [Local Interpretable Model-agnostic Explanations](#local-interpretable-model-agnostic-explanations)
### Code snippets for conducting Feature Importance

[SHapley Additive exPlanations](#shapley-additive-explanations)

[Local Interpretable Model-agnostic Explanations](#local-interpretable-model-agnostic-explanations)

Tree-based algorithms like [Random Forests](#random-forests) or [XGBoost](#xgboost) automatically calculate feature importance. 

In Python, for example, after training a Random Forest model, you can access the feature importance scores using:

```python
from sklearn.ensemble import RandomForestClassifier

# Train a RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Get feature importance scores
importances = model.feature_importances_
```

This method uses the decrease in node impurity as a measure of feature importance.




# Feature Scaling {#feature-scaling}


Used in preparing data for machine learning models. 

Feature Scaling is a [preprocessing](#preprocessing) step in machine learning that involves adjusting the range and distribution of feature values. 

This ensures that all features contribute equally to the model's performance, especially when they are measured on different scales, which is particularly important for distance-based algorithms, [Principal Component Analysis](#principal-component-analysis), and optimization techniques like [Gradient Descent](#gradient-descent). 

By using methods like normalization and standardization, you can enhance the performance and accuracy of your models.

See sklearn.preprocessing

Examples of algorithms not affected by feature scaling are [Naive Bayes](#naive-bayes), [Decision Tree](#decision-tree), and [Linear Discriminant Analysis](#linear-discriminant-analysis).
### Why Use Feature Scaling?
Feature scaling is important for several reasons:

1. Distance-Based Algorithms: Algorithms like k-nearest neighbors (KNN) rely on distance measures (e.g., Euclidean distance). If features are on different scales, those with larger magnitudes will disproportionately influence the distance calculations. Scaling ensures that all features weigh equally.

2. Principal Component Analysis (PCA): PCA aims to identify the directions (principal components) that maximize variance in the data. If features have high magnitudes, they will dominate the variance calculation, skewing the results. Scaling helps to mitigate this issue.

3. Gradient Descent Optimization: In optimization algorithms like gradient descent, features with larger ranges can cause inefficient convergence. Scaling ensures that all features are on a similar scale, allowing for faster and more stable convergence to the optimal solution.

### Common Scaling Methods
[Normalisation](#normalisation)

[Standardisation](#standardisation)

Min-Max Scaling:  Scales features to a fixed range (e.g., $[0, 1]$), preserving relative distances.
### Example of Scaling
Here’s how you can scale a DataFrame using the `scale` function from `sklearn`:

```python
from sklearn import preprocessing
df_scaled = preprocessing.scale(df)  # Scales each variable (column) with respect to itself
```

This returns an array where each feature is standardized.

**Note:**

- Scaling is done when one feature is at a significantly different scale.
- For each data point, subtract the mean and divide by the range (max-min).

![Pasted image 20241224083928.png](../content/images/Pasted%20image%2020241224083928.png)

# Feature Selection Vs Feature Importance {#feature-selection-vs-feature-importance}


### Summary

- [Feature Selection](#feature-selection) is about choosing which features to include in the model <mark>before training</mark>, aiming to improve model performance and efficiency.
- [Feature Importance](#feature-importance) is about understanding the role and impact of each feature <mark>after the model has been trained,</mark> providing insights into the model's decision-making process.

Use for [interpretability](#interpretability) of the model, but they are applied at different stages and serve different purposes.





# Feature Selection {#feature-selection}


Purpose: The primary goal of feature selection is to identify and retain the most relevant features for model training while <mark>removing irrelevant or redundant ones</mark>. This helps in simplifying the model, reducing overfitting, and improving computational efficiency.

Process: Feature selection is typically performed before model training. It involves evaluating features based on certain criteria or algorithms to decide which features to keep or discard.

Through an iterative process, feature selection experiments with different methods, adjusts parameters, and evaluates model performance until an optimal set of features is found.

See [Feature Selection vs Feature Importance](#feature-selection-vs-feature-importance)
### Methods

The choice of feature selection method depends on factors like the size of your dataset, the number of features, and the complexity of your model. It's often a balance between computational cost and performance improvement.

- [Filter Methods](#filter-methods): Select features based on statistical properties, independent of any machine learning algorithm. (Separate stage to training)
- [Wrapper Methods](#wrapper-methods): Involve training multiple models with different subsets of features and selecting the subset that yields the best performance. (Separate stage to training)
- [Embedded Methods](#embedded-methods): Perform feature selection as part of the model training process. (Part of training)

After selecting features, it's essential to evaluate your model's performance ([Model Evaluation](#model-evaluation)) with the chosen subset. Sometimes, feature selection can inadvertently remove important information.

### Detecting Noisy or Redundant Features

- [Correlation](#correlation) Analysis: Use a [Heatmap](#heatmap) or [Clustering](#clustering). Features with low correlation to the target or high correlation with other features may be candidates for removal.

- [Dimensionality Reduction](#dimensionality-reduction) Techniques: Techniques like [Principal Component Analysis](#principal-component-analysis) or Singular Value Decomposition ([SVD](#svd)) can transform the features into a lower-dimensional space while preserving as much variance as possible. Features with low contribution to the principal components can be considered for removal.

- Visualizations: Plotting pairwise scatter plots or [Heatmap](#heatmap) of feature [Correlation](#correlation) can provide visual insights into redundant features. Clusters of highly correlated features or scatter plots showing no discernible pattern with the target variable can indicate noisy or redundant features.
### Investigating Features

1. Variance Thresholding: Check the [Variance](#variance) & [Distributions](#distributions) of each feature. Features with very low variance (close to zero) contribute little information and may be considered noisy. Removing such features can help simplify the model without sacrificing much predictive power.

1. Univariate Feature Selection: Use statistical tests like chi-square for categorical variables or [ANOVA](#anova) for numerical variables to assess the relationship between each feature and the target variable. Features with low test scores or high p-values may be less relevant and can be pruned.

# Feature Selection And Creation {#feature-selection-and-creation}

[Feature Selection](#feature-selection)

[Feature Engineering](#feature-engineering)

After the data is ready.

Which features have the best value, which play the biggest role.

Combining features to simplify the

How to select features.
- Correlation between each two (poor)
- Stepwise regression
- Lasso and ridge regression

When selecting features we ask:
- Can we control it/select it?
- Can we control it easily what do we gain from it
- is it a sensible variable?

# Feature_Distribution.Py {#feature_distributionpy}



# Feed Forward Neural Network {#feed-forward-neural-network}


A **Feedforward Neural Network (FFNN)** is the simplest type of [Neural network](#neural-network). In this model, connections between neurons do not form a cycle, allowing data to flow in one direction—from the input layer, through the hidden layers, to the output layer—without any loops or backward connections. This straightforward design is primarily used for [supervised learning](#supervised-learning) tasks.

### Structure
- Information flows in one direction: input → hidden layers → output.
- During **[forward propagation](#forward-propagation)**, input data is passed through the network, processed by each layer, and an output is produced.
- Unlike [recurrent neural networks](#recurrent-neural-networks) (RNNs), FFNNs do not share information or weights between layers, meaning the model does not maintain memory of past inputs.

### Learning
- FFNNs learn by adjusting weights and biases during training to minimize the [Loss function](#loss-function).

### Limitations
- **Shallow vs. Deep Networks:** Simple feedforward networks with only a few hidden layers (shallow networks) may struggle to learn complex, hierarchical representations of data. Deeper networks (deep feedforward networks) with many layers can model more complex patterns but require more data and computational resources to train.
- **[Overfitting](#overfitting):** FFNNs can overfit on the training data, especially if they have many parameters and not enough regularization (e.g., dropout, [Ridge|L2](#ridgel2) regularization).
- **No Temporal Understanding:** Unlike [Recurrent Neural Networks](#recurrent-neural-networks) or transformers, FFNNs cannot model sequential dependencies in data. They are better suited for static, non-sequential tasks.

# Feedback Template {#feedback-template}

- Praise: I really appreciate your work on this
    
    - _add here_
        
- FYI: It's really not a big deal, but I'm letting you know just in case.
    
    - _add here_
        
- Suggestion: I’m fairly confident this would help, but I can live without it
    
    - _add here_
        
- Recommendation: This could be holding you back
    
    - _add here_
        
- Plea: It’s almost at the breaking point if it’s not already there.
    
    - _add here_

# Filter Method {#filter-method}



# Firebase {#firebase}

Googles version of [AWS](#aws)

[Setup basics](https://www.youtube.com/watch?v=XC4Y1KLNLzI&list=WL&index=6)

Project idea: Set up a basic emailer app.

# Fishbone Diagram {#fishbone-diagram}

Fishbone diagram
[Documentation & Meetings](#documentation--meetings)
Root cause analysis: [Documentation & Meetings](#documentation--meetings)
- 5 Y's
- Fishbone diagram: start at issue at head
- ![Pasted image 20250312162034.png](../content/images/Pasted%20image%2020250312162034.png)
- People and ownership: Who is entering the data: the source data

# Fitting Weights And Biases Of A Neural Network {#fitting-weights-and-biases-of-a-neural-network}

For a neural network model, fitting weights and biases involves optimizing these [Model Parameters](#model-parameters) so the model learns to map input features ($X$) to target outputs ($y$) effectively. This is achieved through the training process, which minimizes the error between predictions and actual values.

Best Practices
- Use appropriate weight initializations like He or Xavier.
- Choose a suitable [loss function](#loss-function) for the task.
- Optimize using advanced optimizers like Adam or RMSprop.
- Experiment with batch sizes, epochs, and learning rates.
- Apply regularization (L2, [Dropout](#dropout)) to prevent overfitting.
- Monitor validation metrics and use early stopping.

In [ML_Tools](#ml_tools) see: Neural_Net_Weights_Biases.py

## Initialization of Weights and Biases

Initializing all weights randomly. The weights are assigned randomly by initializing them very close to 0. It gives better accuracy to the model since every neuron performs different computations.

Proper initialization is critical for training to converge efficiently. Poor initialization can lead to slow convergence or getting stuck in local minima. By starting with well-chosen initial values, the network can learn more effectively and avoid issues like vanishing or exploding gradients.

Weights:
- Use small random values (e.g., drawn from Gaussian or uniform [distributions](#distributions)) to break symmetry and ensure that neurons learn different features.
- Initialization techniques like He initialization (for ReLU activations) or Xavier initialization (for sigmoid/tanh activations) are commonly used because they help maintain the scale of gradients across layers, promoting stable and faster convergence.

```python
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import HeNormal

# Example of He initialization for ReLU activation
Dense(25, activation="relu", kernel_initializer=HeNormal())
```
Biases:
- Start with zeros (`0`) to ensure symmetry-breaking during optimization. This allows the network to learn offsets for the activations without introducing bias in the initial learning phase.

## [Forward Propagation](#forward-propagation)

During forward propagation, the network computes activations using the current weights and biases, and passes these activations to subsequent layers to generate predictions. This step is crucial as it determines how well the network can map inputs to outputs based on its current parameters.

## Loss Function

The loss function quantifies the difference between predicted outputs and true labels. It serves as the objective function that the network aims to minimize during training. Choosing the right loss function is essential as it directly impacts the learning process and the network's ability to generalize.

- Binary Cross-Entropy: For [Binary Classification](#binary-classification).
- Categorical Cross-Entropy: For multi-class classification.
- Mean Squared Error (MSE): For regression tasks.

Example:
```python
from tensorflow.keras.losses import BinaryCrossentropy
loss_fn = BinaryCrossentropy()
```
## [Backpropagation](#backpropagation)

Backpropagation computes the gradients of the loss function with respect to weights and biases using the chain rule. This process is fundamental for learning, as it provides the necessary information to update the parameters in a way that reduces the loss.

## [Gradient Descent](#gradient-descent) Optimization

Gradients from backpropagation are used to update weights and biases iteratively. Optimization algorithms like Adam, RMSprop, and [Stochastic Gradient Descent|SGD](#stochastic-gradient-descentsgd) with momentum are crucial as they determine the efficiency and speed of convergence, especially in large datasets and complex models.

Example:
```python
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001)
```

## Batch Training

Weights and biases are updated after processing a batch of data. Batch training helps in stabilizing the learning process and can lead to faster convergence compared to updating after each sample. The choice of batch size and number of epochs affects the trade-off between computational efficiency and the quality of the learned model.

## [Regularisation](#regularisation) Techniques

Prevent overfitting by penalizing large weights. Regularization is essential for improving the generalization of the model, ensuring it performs well on unseen data.

[Ridge](#ridge)
[Dropout](#dropout)
## Learning Rate Tuning

Learning rate impacts convergence. It is a [hyperparameter](#hyperparameter) that determines the step size during optimization. A poorly chosen learning rate can lead to divergence or slow convergence.

Techniques:
- [Learning Rate](#learning-rate) Scheduling: Reduce learning rate as training progresses to fine-tune the learning process.
- Adaptive Learning Rates: Optimizers [Optimisation techniques](#optimisation-techniques)

#software 

web app framework for writing web pages

uses decorators

![Pasted image 20240922202938.png](../content/images/Pasted%20image%2020240922202938.png)

# [Flask](#flask)
## Flask app example

https://www.youtube.com/watch?v=wBCEDCiQh3Q&list=PLcWfeUsAys2my8yUlOa6jEWB1-QbkNSUl

You can run a flask app in google colab and then share it publicly with ngrok.

flask app saved on github.

# Folder Tree Diagram {#folder-tree-diagram}


## Links 

Simple method
https://www.digitalcitizen.life/how-export-directory-tree-folder-windows/

More general

https://superuser.com/questions/272699/how-do-i-draw-a-tree-file-structure

[Treeviz](randelshofer.ch/treeviz/)
[Graphviz](https://graphviz.org/)


tree /a /f >output.doc



# Forecasting_Autoarima.Py {#forecasting_autoarimapy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries/Forecasting_AutoArima.py
## Tools and Resources

- AutoARIMA: Automatically selects the best [ARIMA](#arima) [model parameters](#model-parameters). Available in the statsforecast library by [Nixtla](https://www.linkedin.com/company/nixtlainc/).
- Implementation Example: See `TS_AutoArima.py` in [ML_Tools](#ml_tools) for practical implementation.
## Performance Insights

- [Evaluation Metrics](#evaluation-metrics): Marginal differences in evaluation metrics across ARIMA models may occur due to the volatile nature of the data.


# Forecasting_Baseline.Py {#forecasting_baselinepy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries/Forecasting_Baseline.py

Baseline methods are essential for establishing a performance benchmark. They provide insights into the data's underlying patterns and help in assessing the effectiveness of more sophisticated forecasting models. By comparing advanced models against these baselines, you can determine if the added complexity is justified by improved accuracy.

**Methods Implemented:**
    - **Mean Forecasting:** Uses the average of all past values as the forecast for future periods.
    - **Naive Forecasting:** The last observed value is used as the forecast for all future periods.
    - **Seasonal Naive Forecasting:** Uses the value from the previous seasonal period to forecast the future.
    - **Drift Method:** Predicts future values based on the trend between the first and last observations in the training data.

# Forecasting_Exponential_Smoothing.Py {#forecasting_exponential_smoothingpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/TimeSeries/Forecasting_Exponential_Smoothing.py

Exponential smoothing models are a set of [Time Series Forecasting](#time-series-forecasting) techniques that apply weighted averages of past observations, with the weights decaying exponentially over time. These methods are useful for capturing different components of time series data, such as level, trend, and seasonality.

However, their effectiveness depends on the nature of the data. For [Datasets](#datasets) with simple patterns, these models can be quite effective, but for more complex series, alternative methods may be necessary.

## Methods Implemented

Simple Exponential Smoothing (SES):
   - Description: Suitable for forecasting data without trends or seasonality. It applies a constant smoothing factor to past observations.
   - Use Case: Best for stationary series where the data fluctuates around a constant mean.

Double Exponential Smoothing (Holt's Linear Trend):
   - Description: Extends SES by adding a trend component, allowing it to model data with a linear trend.
   - Use Case: Ideal for series with a consistent upward or downward trend but no seasonality.

 Triple Exponential Smoothing (Holt-Winters):
   - Description: Incorporates both trend and seasonal components, making it suitable for data with both linear trends and seasonal patterns.
   - Use Case: Effective for series with regular seasonal fluctuations.

Advanced Alternatives: For complex datasets like stock prices, advanced models such as [Forecasting_AutoArima.py](#forecasting_autoarimapy) may be more appropriate to capture the intricacies of the data.






# Foreign Key {#foreign-key}

A foreign key is a field in one table that uniquely identifies a row in another table, linking to the primary key of that table.

For example, `DepartmentID` in the <mark>Employees</mark> table links to `DepartmentID` in the Departments table. 

Foreign keys establish relationships between tables and maintain referential integrity by ensuring valid connections between records.

**Departments Table**

| DepartmentID | DepartmentName      |
|--------------|----------------------|
| 1            | Human Resources       |
| 2            | IT                   |
| 3            | Marketing            |

**Employees Table**

| EmployeeID | EmployeeName | DepartmentID |
|------------|--------------|---------------|
| 101        | Alice        | 1             |
| 102        | Bob          | 2             |
| 103        | Charlie      | 1             |
| 104        | Dana         | 3             |

# Forward Propagation {#forward-propagation}



>[!Summary]  
> Forward propagation is the process by which input data moves through a neural network, layer by layer, to produce an output. During this process, each layer’s weights and biases are applied to the input data, and an activation function is used to transform the data at each layer. 
> 
> Mathematically, for each layer, the input $x$ is transformed into an output $y$ through the equation $y = f(Wx + b)$, where $W$ represents the weights, $b$ is the bias, and $f$ is the activation function (e.g., ReLU, sigmoid). The output from one layer becomes the input to the next, and this continues until the final layer produces the predicted output. 
> 
> This process does not involve learning; it only <mark>computes the prediction based on current weights.</mark>

>[!Breakdown]  
> Key Components:  
> - Input data: Initial values fed into the network.  
> - Weights ($W$) and biases ($b$): Parameters adjusted during training.  
> - Activation function: Non-linear transformation, e.g., ReLU or sigmoid.  
> - Output: Prediction made by the network.

>[!important]  
> - Forward propagation calculates predictions <mark>by applying current model parameters</mark> to inputs.  
> - It is the first step before backpropagation, where the error is used to adjust weights.

>[!attention]  
> - Forward propagation does not involve <mark>learning or updating weights.</mark>  
> - The accuracy of forward propagation depends entirely on the current values of weights and biases.

>[!Example]  
> In a simple neural network with one hidden layer, forward propagation can be described as:  
> $$ z_1 = W_1 x + b_1 $$  
> $$ a_1 = \text{ReLU}(z_1) $$  
> $$ z_2 = W_2 a_1 + b_2 $$  
> $$ y = \text{sigmoid}(z_2) $$  
> Here, $x$ is the input, and $y$ is the output prediction.

>[!Follow up questions]  
> - How does the choice of [activation function](#activation-function) impact the forward propagation process?  
> - In deep networks, how can [vanishing and exploding gradients problem](#vanishing-and-exploding-gradients-problem) during forward propagation affect training?

>[!Related Topics]  
> - [Backpropagation](#backpropagation) in neural networks  
> - Activation functions in deep learning

# Fuzzywuzzy {#fuzzywuzzy}

Tool used for correcting spelling with pandas.

[Data Cleansing](#data-cleansing)



# Filter Methods {#filter-methods}



For [Feature Selection](#feature-selection)

1. **Pearson [Correlation](#correlation) Coefficient**:
   - Measures the linear correlation between two continuous variables.
   - Features with low correlation with the target variable are considered less relevant.
   - Features with high correlation among themselves might be redundant.

2. **Mutual Information**:
   - Measures the amount of information obtained about one variable through another variable.
   - High mutual information indicates strong dependency between features and the target variable.
   - Can handle both continuous and categorical variables.
   - used to rank or score features based on their relevance to the target variable.
   - joint probability distribution
   - information theory,

3. **[ANOVA](#anova) (Analysis of Variance)**:
   - Assesses the differences in means among groups of a categorical variable.
   - Calculates the F-statistic and p-value to determine if there are significant differences in the means of the target variable across different levels of a categorical feature.
   - Useful for selecting features with significant impact on the target variable in classification tasks.

4. **Chi-Squared Test**:
   - Tests the independence between two [categorical](#categorical) variables.
   - Calculates the chi-squared statistic and p-value to determine if the observed frequency distribution differs from the expected distribution.
   - Helpful for [Feature Selection](#feature-selection) in classification tasks with categorical variables.


# Functional Programming {#functional-programming}



Functional Programming is a style of building functions that threaten computation as a mathematical function that avoids changing state and mutable data. It is a declarative programming paradigm, which means programming expressive and [declarative](term/declarative.md) as opposed to imperative. It's getting more popular with the rise of [Functional Data Engineering](term/functional%20data%20engineering.md).

See also [Programming Languages](programming%20languages.md).