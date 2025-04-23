# K

## Table of Contents
* [K-means](#k-means)
* [K-nearest neighbours](#k-nearest-neighbours)
* [K_Means.py](#k_meanspy)
* [Kaggle Abalone regression example](#kaggle-abalone-regression-example)
* [Kernelling](#kernelling)
* [Key Differences of Web Feature Server (WFS) and Web Feature Server (WFS)](#)
* [Kmeans vs GMM](#)
* [Knowledge Graph](#knowledge-graph)
* [Knowledge Graphs with Obsidian](#knowledge-graphs-with-obsidian)
* [Knowledge Work](#knowledge-work)
* [Knowledge graph vs RAG setup](#knowledge-graph-vs-rag-setup)
* [kubernetes](#kubernetes)



# K Means {#k-means}


K-means clustering is an [Unsupervised Learning](#unsupervised-learning) algorithm that partitions data into (k) clusters. Each data point is assigned to the cluster with the nearest centroid.

The algorithm partitions a dataset into k clusters by assigning data points to the closest cluster mean. The means are updated iteratively until convergence is achieved.

In [ML_Tools](#ml_tools) see: [K_Means.py](#k_meanspy)
## Key Features

- Unsupervised Learning: K-means organizes unlabeled data into meaningful groups without prior knowledge of the categories.
- [Hyperparameter](#hyperparameter) k: The number of clusters must be specified beforehand. The optimal number of clusters can be determined using [WCSS and elbow method](#wcss-and-elbow-method).

Algorithm Process:
  1. Randomly choose k initial centroids.
  2. Assign each data point to the nearest centroid.
  3. <mark>Recalculate</mark> the centroids based on the current cluster assignments.
  4. Repeat steps 2 and 3 until convergence (i.e., centroids no longer change significantly).

Visualization: Scatterplots can be used to visualize clusters and their centroids.

Adaptability: K-means can be updated with new data and allows for comparison of changes in centroids over time.

The initial centroids can effect the end results. 

To correct this the algo is run multiple times with varying starting positions.

The centroids are updated after each iteration.

![Pasted image 20241230200255.png](../content/images/Pasted%20image%2020241230200255.png)


## Limitations

- Sensitivity to Initialization: The algorithm is sensitive to the initial placement of centroids, which can affect the final clustering outcome.
- Predefined Number of Clusters: The number of clusters k must be specified in advance, which may not always be straightforward.
## Resources
- [Statquest Video on K-means](https://www.youtube.com/watch?v=4b5d3muPQmA)



# K Nearest Neighbours {#k-nearest-neighbours}


K-nearest Neighbors is a non-parametric method used for both [classification](#classification) and [regression](#regression) tasks. It classifies a sample by a majority vote of its neighbors, assigning the sample to the class most common among its \(k\) nearest neighbors, where \(k\) is a small positive integer.

### How It Works

- **Classification:** When a new data point needs classification, KNN identifies its \(k\) nearest neighbors in the training data based on feature similarity. The class label most common among these neighbors is assigned to the new data point.
- **Regression:** For regression tasks, KNN predicts the average value of the \(k\) nearest neighbors.

### Applications

  - [Recommender systems](#recommender-systems)
  - Pattern recognition

### Key Points

- **Non-parametric:** KNN does not make any assumptions about the underlying data distribution.
- **Supervised Learning:** Despite the note's mention of unsupervised learning, KNN is actually a supervised learning algorithm because it requires labeled training data.

### Use Cases

- KNN is useful for tasks where the decision boundary is irregular and not easily captured by parametric models ([parametric vs non-parametric models](#parametric-vs-non-parametric-models)). It is simple to implement and understand but can be computationally expensive with large datasets.


### Understanding K-nearest Neighbors

KNN is a straightforward algorithm that relies on the proximity of data points to make predictions. It is particularly effective in scenarios where the relationship between features and the target variable is complex and non-linear.

- **Choice of \(k\):** The value of \(k\) is crucial. A small \(k\) can be sensitive to noise, while a large \(k\) can smooth out the decision boundary too much.

- **Distance Metric:** The choice of distance metric (e.g., Euclidean, Manhattan) affects how neighbors are determined and can impact the algorithm's performance.

KNN is best suited for smaller datasets due to its computational intensity, as it requires calculating the distance between the new data point and all existing data points.

[K-nearest neighbours](#k-nearest-neighbours)
   - Classifies a new data point based on the majority class of its k nearest neighbors.
   - Simple but computationally expensive for large datasets.

# K_Means.Py {#k_meanspy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/Clustering/KMeans/K_Means.py
## Key Concepts Used in the Script

1. **Data Loading**:
   - The script reads data from a CSV file (`penguins.csv`) and uses a sample dataset with random features for demonstration purposes.

2. **Data Preprocessing**:
   - **Standardization**: Features are standardized using `sklearn.preprocessing.scale` and `StandardScaler` to ensure that all features contribute equally to the clustering process.

3. **Feature Selection**:
   - Specific features, such as `bill_length_mm` and `bill_depth_mm`, are selected for clustering.

4. **K-Means Clustering**:
   - The core clustering algorithm is applied with `n_clusters=3`.
   - Outputs include cluster centroids and labels for each data point.

5. **Visualization**:
   - Scatter plots are used to display the clustering results, highlighting the cluster centroids.

6. **Evaluation of Optimal Clusters**:
   - **Elbow Method**: This method iterates through different numbers of clusters to determine the optimal number based on the within-cluster sum of squares (WCSS).

7. **Cluster Assignment**:
   - Labels are assigned to data points, and the results are visualized to show the clustering outcome.

8. **Exploratory Analysis**:
   - The script examines the impact of different numbers of clusters using an example function (`scatter_elbow`).

# Kaggle Abalone Regression Example {#kaggle-abalone-regression-example}

Task: For each model as we tune the hyperparameters what happens to the (RMSLE) metric (scatter metric against hyperparameter).  

Using **Root Mean Squared Logarithmic Error** RMSLE to evaluate.

Practice with model and feature engineering ideas, create visualizations

- [ ] create eda for blog post
- [ ] create model training optuna for blog post



----
Questions 


[Hyperparameter](#hyperparameter)
[Hyperparameter](#hyperparameter) tuning can be done with [Hyperparameter](#hyperparameter)

[Cross Validation](#cross-validation)
Both (sklearn)[`StratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html) and [`RepeatedStratifiedKFold`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedStratifiedKFold.html) can be very effective when used on classification problems with a severe class imbalance. They both _stratify_ the sampling by the class label; that is, they split the dataset in such a way that preserves approximately the same class distribution (i.e., the same percentage of samples of each class) in each subset/fold as in the original dataset. However, a single run of `StratifiedKFold` might result in a noisy estimate of the model's performance, as different splits of the data might result in very different results. That is where `RepeatedStratifiedKFold` comes into play.

`RepeatedStratifiedKFold` allows improving the estimated performance of a machine learning model, by simply repeating the [cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) procedure multiple times (according to the `n_repeats` value), and reporting the _mean_ result across all folds from all runs. This _mean_ result is expected to be a more accurate estimate of the model's performance


# Kernelling {#kernelling}


[Kernelling](#kernelling) is a technique where the [Support Vector Machines|SVM](#support-vector-machinessvm) uses a kernel function to map the dataset into a higher-dimensional space, making it easier to identify separable clusters that may not be apparent in the original low-dimensional space.

 Kernel Trick:
   - When the data cannot be separated by a straight line or plane in its original (low-dimensional) space, SVM uses a technique called kernelling to project the data into a higher dimension where it becomes easier to separate.
   - The Kernel Trick allows the transformation of data into a higher dimension without explicitly computing the transformation. There are different types of kernels, with common examples being:
     - Polynomial kernel
     - Radial Basis Function (RBF) or exponential kernel



### Key Differences of Web Feature Server (WFS) and Web Feature Server (WFS)

1. **Data Type**:
   - **[Web Map Tile Service (WMTS)](#web-map-tile-service-wmts)**: Serves image tiles (raster data).
   - **[Web Feature Server (WFS)](#web-feature-server-wfs)**: Serves geographic features (vector data).

2. **Use Case**:
   - **WMTS**: Ideal for applications needing fast rendering of static maps, such as online map viewers.
   - **WFS**: Suitable for applications requiring access to and manipulation of raw geographic data, such as spatial analysis and GIS applications.

3. **Performance**:
   - **WMTS**: High performance due to pre-rendered and cached tiles, optimized for rapid delivery.
   - **WFS**: Performance depends on the complexity of the data and queries, typically slower than WMTS due to on-the-fly data retrieval and processing.

4. **Interactivity**:
   - **WMTS**: Limited interactivity, primarily for viewing maps.
   - **WFS**: High interactivity, supporting complex queries and data manipulation.
### Example Scenarios

- **WMTS**: A web application displaying a world map with zoom and pan functionality. The map is made up of pre-rendered image tiles that load quickly as the user navigates.
- **WFS**: An environmental monitoring system that allows users to query and retrieve data about specific geographic features, such as the location and attributes of water bodies, for analysis and reporting.

In summary, WMTS is focused on efficiently serving map images for fast visualization, while WFS provides access to detailed, manipulable geographic feature data for more in-depth spatial analysis and querying.

### **Key Differences Between [k-Means](#k-means) and GMM**

#### Cluster Shape

- **k-Means**: Assumes clusters are spherical and equidistant from their centroids.
- **GMM**: Models clusters as Gaussian distributions, allowing for **different shapes** (e.g., ellipses) by incorporating mean and covariance matrices. GMM can model clusters of varying **shapes and sizes** by adjusting the [Covariance Structures](#covariance-structures) (e.g., full, diagonal, spherical).

#### Probability-Based Assignments

- **k-Means**: Assigns each point deterministically to the nearest cluster centroid.
- **GMM**: Provides a **probability [Distributions|distribution](#distributionsdistribution)** for cluster membership, making it a soft clustering method.
- GMM handles overlapping clusters effectively by assigning **probabilities** to data points for each cluster, instead of enforcing hard boundaries like k-means.

#### Flexibility

- **k-Means**: Performs well when clusters are **spherical** and well-separated.
- **GMM**: Handles **overlapping clusters** and clusters with **different shapes**, leveraging its covariance modeling capability.

# Knowledge Graph {#knowledge-graph}




>[!Summary]  
> Knowledge graphs (KGs) enable large language models (LLMs) to generate more accurate, trustworthy AI outputs. Neo4j is leader in this space and make use of KG through  such as generative AI techniques like GraphRAG.
> - Knowledge graphs are critical for managing complex data relationships and making strategic AI-driven decisions.  
> - The combination of KGs and LLMs improves AI accuracy, diversity of viewpoints, and [explainability](#explainability).  
> A **knowledge graph** is a structured representation of knowledge that captures entities (e.g., people, places, concepts) and the relationships between them. Knowledge graphs are often used to represent and store factual information in a way that machines can easily query and understand. They use a **graph structure** where: **Nodes** represent entities (like "Company A" or "Person B"). **Edges** represent relationships between those entities (like "works at" or "founded").

![Pasted image 20240921154214.png|600](../content/images/Pasted%20image%2020240921154214.png|600)

>[!important]  
> KGs act as a control for Large Language Models (LLMs) by enabling knowledge-based reasoning based on the connections in the data 
> 

>[!attention]  
> - No significant limitations or concerns were highlighted, but the implementation of KGs may require technical expertise and resources.  

>[!Follow up questions]  
> - [ ] How does the integration of knowledge graphs and LLMs improve explainability in AI-generated responses?  

>[!Link]
>https://neo4j.com/blog/genai-knowledge-graph-deep-understanding/

## Key characteristics of knowledge graphs:
- **Structured data**: Information is represented in a highly structured form (triples: subject, predicate, object) that allows efficient querying and reasoning.
- **Semantic relationships**: Entities are connected by meaningful relationships, often using ontologies and taxonomies to organize the knowledge.
- **Reasoning and inference**: Some knowledge graphs can support reasoning capabilities, where new information can be inferred based on the existing relationships and rules (e.g., if "Person A works at Company B," and "Company B is in Industry C," it can infer that "Person A works in Industry C").

## **Example of a Knowledge Graph**: 
- **Nodes**: `Barack Obama`, `United States`, `President`.
- **Edges**: `Barack Obama --> President of --> United States`.

# Knowledge Graphs With Obsidian {#knowledge-graphs-with-obsidian}


>[!Summary]  
> Llama Index is a python package that can be used to create Knowledge graphs (KGs).
>There exists a method to integrate with Obsidian.
> This is a tutorial on how to set this up a [RAG](#rag) system with obsidian.
> Visualisation is done using the `pyvis` library.  
> Requires API to [LLM](#llm).



>[!important]  
> - RAG improves [LLM](#llm) performance by utilizing external databases.  
> - Llama Index facilitates the transformation of Obsidian notes into a structured Knowledge Graph.  
> - The tutorial includes steps for setup, dependencies, and visualization.  

>[!attention]  
> - [LLM](#llm)s can still produce errors known as "hallucinations."  
> - Requires familiarity with Python and Jupyter Notebook.  

>[!Code set up]


>[!Example]  
> The tutorial provides a code snippet for querying the Knowledge Graph about the assumptions of the Black-Scholes model, yielding detailed contextually relevant insights.  

>[!Follow up questions]  
> - [ ] How can RAG be applied to other data sources beyond Obsidian?  
> - [ ] What specific challenges might arise when integrating large datasets into a Knowledge Graph?  
> - [ ] Can you also set up with PDF's within Obsidian?

>[!Link]  
> https://medium.com/@haiyangli_38602/make-knowledge-graph-rag-with-llamaindex-from-own-obsidian-notes-b20a350fa354
[Knowledge Graph](#knowledge-graph)



# Knowledge Work {#knowledge-work}


Knowledge work refers to tasks that primarily involve handling or using information and require cognitive skills rather than manual labor. It is characterized by problem-solving, critical thinking, and the application of specialized knowledge.
## Key Characteristics

- Problem Solving: Knowledge work often involves identifying, analyzing, and solving complex problems. This requires creativity, analytical skills, and the ability to synthesize information from various sources.

- Use of the [Scientific Method](#scientific-method): Many knowledge work tasks, especially in research and development, rely on the scientific method. This involves forming hypotheses, conducting experiments, analyzing data, and drawing conclusions.

- Information Management: Knowledge workers must efficiently gather, process, and apply information to make informed decisions.

## Examples of Knowledge Work

- Research and development
- Software development
- Data analysis
- Strategic planning
- Writing and content creation

# Knowledge Graph Vs Rag Setup {#knowledge-graph-vs-rag-setup}


### Comparison: Knowledge Graph vs. RAG Setup

- <mark>**Knowledge Graphs** are structured representations of entities and their relationships, designed primarily for querying, reasoning, and storing factual information.</mark>
- <mark>**RAG setups** enhance generative models by retrieving external knowledge (from unstructured or semi-structured data) and integrating it into the generation process.</mark>

While not the same, these two concepts can be used together to build systems that combine structured knowledge retrieval with the natural language generation capabilities of RAG models.

While **knowledge graphs** and **RAG** are distinct, they can be integrated to improve certain systems:
- <mark>A **RAG model** could use a **knowledge graph** as the retrieval source.</mark> Instead of retrieving unstructured text documents, the RAG model could retrieve structured, factual triples from a knowledge graph and incorporate this into the generation process. This would improve the accuracy of fact-based questions and answers.

A [Knowledge Graph](#knowledge-graph) and a **Retrieval-Augmented Generation ([RAG](#rag))** setup are related but distinct concepts, particularly in how they handle knowledge representation and retrieval. While they can complement each other in certain applications, they serve different purposes and operate in different ways.

| Aspect                      | Knowledge Graph                                                                         | RAG Setup                                                                             |
| --------------------------- | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **Purpose**                 | Stores and organizes knowledge for querying and reasoning                               | Combines retrieval of external information with <mark>text generation</mark>                   |
| **Data Structure**          | <mark>Highly structured</mark> (graph with nodes and edges)                                      | Unstructured or semi-structured (documents, text snippets)                            |
| **Retrieval Mechanism**     | Queries are made through graph traversal or SPARQL-like languages                       | Information is retrieved via search mechanisms (e.g., dense embeddings)               |
| **Usage**                   | Often used for querying factual data, answering structured queries, [Semantic Relationships](#semantic-relationships) | Used to enhance the factual accuracy of generative models by retrieving external data |
| **Reasoning and Inference** | Capable of logical reasoning based on relationships                                     | Does not perform reasoning; it retrieves and integrates relevant text                 |
| **Scalability**             | Requires careful design to manage large, complex graphs                                 | Can handle large text corpora, but retrieval quality affects the final generation     |
| **Generative Capabilities** | Not generative (focused on querying existing knowledge)                                 | [Generative](#generative) (synthesizes and generates natural language responses)                 |






# Kubernetes {#kubernetes}


It’s a platform that allows you to run and orchestrate container workloads. [**Kubernetes**](https://stackoverflow.blog/2020/05/29/why-kubernetes-getting-so-popular/) **has become the de-facto standard** for your cloud-native apps to (auto-) [scale-out](https://stackoverflow.com/a/11715598/5246670) and deploy your open-source zoo fast, cloud-provider-independent. No lock-in here. Kubernetes is the **move from infrastructure as code** towards **infrastructure as data**, specifically as [YAML](term/yaml.md). With Kubernetes, developers can quickly write applications that run across multiple operating environments. Costs can be reduced by scaling down.