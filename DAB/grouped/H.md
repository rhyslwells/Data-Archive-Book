# H

## Table of Contents
* [Hadoop](#hadoop)
* [Handling Different Distributions](#handling-different-distributions)
* [Handling_Missing_Data.ipynb](#handling_missing_dataipynb)
* [Handling_Missing_Data_Basic.ipynb](#handling_missing_data_basicipynb)
* [Handwritten Digit Classification](#handwritten-digit-classification)
* [Hash](#hash)
* [Heatmap](#heatmap)
* [Heatmaps_Dendrograms.py](#heatmaps_dendrogramspy)
* [Hierarchical Clustering](#hierarchical-clustering)
* [High cross validation accuracy is not directly proportional to performance on unseen test data](#high-cross-validation-accuracy-is-not-directly-proportional-to-performance-on-unseen-test-data)
* [Honkit](#honkit)
* [Hosting](#hosting)
* [How LLMs store facts](#how-llms-store-facts)
* [How businesses use Gen AI](#how-businesses-use-gen-ai)
* [How do we evaluate of LLM Outputs](#how-do-we-evaluate-of-llm-outputs)
* [How is reinforcement learning being combined with deep learning](#how-is-reinforcement-learning-being-combined-with-deep-learning)
* [How is schema evolution done in practice with SQL](#how-is-schema-evolution-done-in-practice-with-sql)
* [How to do git commit messages properly](#)
* [How to model to improve demand forecasting](#how-to-model-to-improve-demand-forecasting)
* [How to normalise a merged table](#how-to-normalise-a-merged-table)
* [How to reduce the need for Gen AI responses](#how-to-reduce-the-need-for-gen-ai-responses)
* [How to search within a graph](#)
* [How to use Sklearn Pipeline](#how-to-use-sklearn-pipeline)
* [How would you decide between using TF-IDF and Word2Vec for text vectorization](#how-would-you-decide-between-using-tf-idf-and-word2vec-for-text-vectorization)
* [Hugging Face](#hugging-face)
* [Hyperparameter Tuning](#hyperparameter-tuning)
* [Hyperparameter](#hyperparameter)
* [Hypothesis testing](#hypothesis-testing)
* [heterogeneous features](#heterogeneous-features)
* [how do you do the data selection](#how-do-you-do-the-data-selection)



<a id="hadoop"></a>
# Hadoop {#hadoop}


   Hadoop provides the backbone for distributed storage and computation. It uses HDFS (Hadoop Distributed File System) to split large datasets across clusters of servers, while MapReduce enables parallel processing. It’s well-suited for [Batch Processing](#batch-processing)asks, though newer tools like [Apache Spark|Spark](#apache-sparkspark) often outperform Hadoop in terms of speed and ease of use.

1. **Architecture**:
   - **Open-Source Framework**: Hadoop is an open-source framework for distributed storage and processing of large datasets using clusters of commodity hardware.
   - **Distributed File System**: The Hadoop Distributed File System (HDFS) stores data across multiple machines, providing high throughput access to data.
   - **MapReduce**: Originally designed for [Batch Processing](#batch-processing) using the MapReduce programming model, though newer frameworks like Apache Spark are often used now.

2. **Data Storage**:
   - **Unstructured, Semi-Structured, and Structured Data**: Hadoop can handle a wide variety of data formats, including unstructured, semi-structured, and structured data.
   - **Scalable Storage**: HDFS can store vast amounts of data by adding more nodes to the cluster.

3. **Management**:
   - **Complex Management**: Requires more administrative effort to manage and maintain the infrastructure, including handling failures, load balancing, and tuning.

4. **Performance**:
   - **[Batch Processing](#batch-processing)**: Hadoop is optimized for batch processing of large datasets, though it can be less efficient for real-time processing compared to other systems.
   - **Latency**: Higher latency for query processing compared to Snowflake, particularly for complex analytical queries.

5. **Use Cases**:
   - **[Big Data](#big-data) Processing**: Ideal for large-scale data processing tasks, including ETL (Extract, Transform, Load), data mining, and large-scale machine learning.
   - **[Data Lake](#data-lake)**: Commonly used as a data lake to store vast amounts of raw data.



<a id="handling-different-distributions"></a>
# Handling Different Distributions {#handling-different-distributions}

Handling different [distributions](#distributions) is needed for developing robust, fair, and accurate machine learning models that can adapt to a wide range of data environments.

## Importance of Handling Different Distributions

1. [Model Robustness](#model-robustness): Ensures models generalize well to new, unseen data.
2. Bias Mitigation: Prevents bias in predictions by accommodating diverse data types.
3. Improved [Accuracy](#accuracy): Fine-tunes models for better accuracy across varied [Datasets](#datasets).
4. Maintains model effectiveness across different data sources.
5. Decision Making: Informs [Preprocessing](#preprocessing), [model selection](#model-selection), and evaluation strategies.

## Resources

Video: [Training and Testing on Different Distributions](https://www.youtube.com/watch?v=sfk5h0yC67o&list=PLkDaE6sCZn6E7jZ9sN_xHwSHOdjUxUW_b&index=16)

## Example Scenario

High-resolution photos (many) vs. amateur photos (small number) exhibit different distributions.

## Strategy for Handling Distributions

Code Example: See `Handling_Different_Distributions.py` in [ML_Tools](#ml_tools)

In this script:
- **Data Generation:** Creates two mock datasets with different distributions.
- **Data Splitting:** Combines and splits the data into train, dev, and test sets.
- **Model Tuning:** Uses `GridSearchCV` to find the best hyperparameters for a RandomForest model.
- **Model Training and Evaluation:** Trains the model on the training set and evaluates it on the dev and test sets.
- **Visualization:** Uses `matplotlib` to plot the distribution of a feature from both datasets and the model's accuracy on the dev and test sets.

### Follow up questions

How best to combine the datasets?
How should we shuffle and split based on the distributions?
How do we pick the dev set?

1. **Combining Datasets:**
    - The script combines two datasets (`dataset1` and `dataset2`) that may have different distributions. This step ensures that the model is exposed to a variety of data during training.
    
1. **Random Shuffling and Splitting:**
    - By shuffling and splitting the combined dataset into train, dev, and test sets, the script ensures that each set contains a mix of data from both distributions. This helps the model learn from the diversity in the data.

1. **Model Tuning with Diverse Data:**
    - The model tuning process uses the dev set, which contains data from both distributions. This helps in finding hyperparameters that work well across different data characteristics.

## Related Topics
- [Preprocessing](#preprocessing)

<a id="handling_missing_dataipynb"></a>
# Handling_Missing_Data.Ipynb {#handling_missing_dataipynb}

https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Investigating/Cleaning/Handling_Missing_Data.ipynb

<a id="handling_missing_data_basicipynb"></a>
# Handling_Missing_Data_Basic.Ipynb {#handling_missing_data_basicipynb}

https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Investigating/Cleaning/Handling_Missing_Data_Basic.ipynb

<a id="handwritten-digit-classification"></a>
# Handwritten Digit Classification {#handwritten-digit-classification}

![Pasted image 20241006124356.png|800](../content/images/Pasted%20image%2020241006124356.png|800)

<a id="hash"></a>
# Hash {#hash}

A hash is a fixed-size string of characters that is generated from input data of any size using a hash function. 

Hashes are used to ensure [data integrity](#data-integrity) by providing a unique representation of the data, making it easy to detect any changes or alterations.

### Key Characteristics of Hashes:

1. **Deterministic**: The same input will always produce the same hash output.
2. **Fixed Size**: Regardless of the size of the input data, the output hash will always be of a fixed length (e.g., SHA-256 produces a 256-bit hash).
3. **Fast Computation**: Hash functions are designed to compute the hash value quickly.
4. **Pre-image Resistance**: It should be computationally infeasible to reverse-engineer the original input from its hash.
5. **Collision Resistance**: It should be difficult to find two different inputs that produce the same hash output.

### How Hashes are Used in Data Integrity:

6. **Data Verification**: When data is stored or transmitted, a hash of the data is generated and stored or sent along with it. When the data is later accessed, the hash is recalculated and compared to the original hash. If they match, the data is considered intact; if not, it indicates potential corruption or tampering.

7. **Digital Signatures**: Hashes are often used in digital signatures to ensure the authenticity and integrity of a message or document.

8. **Password Storage**: Instead of storing passwords in plain text, systems often store the hash of the password. When a user logs in, the system hashes the entered password and compares it to the stored hash.

### Example of Hashing:

For example, if we take the string "Hello, World!" and apply a hash function like SHA-256, it will produce a unique hash value:

- Input: "Hello, World!"
- Hash (SHA-256): `a591a6d40bf420404a011733cfb7b190d62c65bf0bcda190f4b6c3f0f3c3b8a`

If the input data changes even slightly (e.g., "Hello, World"), the hash will be completely different, making it easy to detect any alterations.



<a id="heatmap"></a>
# Heatmap {#heatmap}


### Description

A **heatmap** is a two-dimensional graphical representation of data where individual values are represented by colors. It is particularly useful for visualizing numerical data organized in a table-like format. 

A heatmap is a graphical representation of data where individual values are represented by colors. It is useful for visualizing numerical data and analyzing the correlation between features.

A heatmap is a  visualization tool for analyzing the [Correlation](#correlation) between features in a dataset. In the context of correlation analysis, a heatmap can display the correlation coefficients between different features in a dataset.

By using a heatmap, you can easily identify [Multicollinearity](#multicollinearity) and make informed decisions about which features to retain or remove, ultimately enhancing the performance and interpretability of your machine learning models.
### Correlation Coefficients
The correlation coefficients range from -1 to 1:
- **-1**: Indicates a perfect negative correlation; if one attribute is present, the other is almost certainly absent.
- **0**: Indicates no correlation; there is no dependence between the attributes.
- **1**: Indicates a perfect positive correlation; if one attribute is present, the other is also certainly present.
### Implementation in Python

In [ML_Tools](#ml_tools) see: [Heatmaps_Dendrograms.py](#heatmaps_dendrogramspy)



<a id="heatmaps_dendrogramspy"></a>
# Heatmaps_Dendrograms.Py {#heatmaps_dendrogramspy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations\Preprocess\Correlation\Heatmaps_Dendrograms.py

See:
 - [Heatmap](#heatmap)
 - [Dendrograms](#dendrograms)


<a id="hierarchical-clustering"></a>
# Hierarchical Clustering {#hierarchical-clustering}

Hierarchical clustering builds a treelike structure of clusters, with similar clusters merged together at higher levels.

Hierarchical clustering builds a tree-like structure of clusters, with similar clusters merged together at higher levels.

<a id="high-cross-validation-accuracy-is-not-directly-proportional-to-performance-on-unseen-test-data"></a>
# High Cross Validation Accuracy Is Not Directly Proportional To Performance On Unseen Test Data {#high-cross-validation-accuracy-is-not-directly-proportional-to-performance-on-unseen-test-data}

Reasons a Model with High [Cross Validation](#cross-validation) Accuracy May Perform Poorly on Unseen Test Data

[Data Leakage](#data-leakage): 
  - Information from test folds leaks into training, inflating CV accuracy.
  - Solution: Apply [preprocessing](#preprocessing) independently within each CV fold.

Overfitting: 
  - Model captures noise in training data, leading to high CV but low test accuracy.
  - Solution: Use simpler models, regularization, and evaluate test performance during [hyperparameter tuning](#hyperparameter-tuning).

Insufficient Cross-Validation Folds: 
  - Too few folds lead to high variance in performance estimates.
  - Solution: Use more folds (e.g., 5- or 10-fold CV) for reliable estimates.

Over-Optimized Hyperparameters: 
  - Excessive tuning results in models that fail to generalize.
  - Solution: Reserve a separate validation set for tuning and use nested cross-validation.

Small Dataset Size: 
  - Small datasets may lead to unreliable accuracy estimates.
  - Solution: Use bootstrapping or collect more data if possible.

Inappropriate Performance Metric: 
  - CV accuracy may not align with the true objective (e.g., [imbalanced datasets](#imbalanced-datasets)).
  - Solution: Choose appropriate [Evaluation Metrics](#evaluation-metrics)  based on the problem context.

### Practical Recommendations
- Evaluate the model on a completely independent test set after cross-validation.
- Check for [Distributions|distribution](#distributionsdistribution) differences between training and test data.
- Avoid data leakage by ensuring strict separation of preprocessing in CV folds.


<a id="honkit"></a>
# Honkit {#honkit}


https://honkit.netlify.app/examples

https://flaviocopes.com/how-to-create-ebooks-markdown/#:~:text=honkit%20works%20great.,and%20let%20CloudFlare%20distribute%20it.

https://github.com/rhyslwells/Note_Compiler

<a id="hosting"></a>
# Hosting {#hosting}



### Using [Plotly](#plotly)

Once you have a simple Dash application.
#### What You Can Do With It

### Free Public Hosting Options

#### a) Render (https://render.com)
1. Create a free account.
2. Create a new "Web Service."
3. Connect your GitHub repo (with `app.py` and `requirements.txt`).
4. Set the start command: `python app.py`.

**Pros**: 
- ✅ Simple and free, supports Dash

**Cons**: 
- ❌ Cold start delay on the free tier

#### b) Railway (https://railway.app)
1. Sign up for a free plan.
2. Link your GitHub project.
3. Define your start command and Python version.

**Pros**: 
- ✅ Very fast to set up

**Cons**: 
- ❌ May require a credit card to unlock some features


<a id="how-llms-store-facts"></a>
# How Llms Store Facts {#how-llms-store-facts}


[How might LLMs store facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8&list=PLZx_FHIHR8AwKD9csfl6Sl_pgCXX19eer&index=6)

Not solved

How do [Multilayer Perceptrons](#multilayer-perceptrons) store facts?

Different directions encode information in [Vector Embedding](#vector-embedding) space.

MLP's are blocks of vectors, these are acted on my the context matrix 

[Johnson–Lindenstrauss lemma](#johnsonlindenstrauss-lemma)

Sparse Autoencoder - used in [interpretability](#interpretability) of [LLM](#llm) responses

See [Anthropic](#anthropic) posts
- https://transformer-circuits.pub/2022/toy_model/index.html#adversarial
- https://transformer-circuits.pub/2023/monosemantic-features

<a id="how-businesses-use-gen-ai"></a>
# How Businesses Use Gen Ai {#how-businesses-use-gen-ai}


 
Businesses leverage generative AI to transform various operations, using models like OpenAI, Gemini (Google Cloud), Anthropic, and Meta models. These models provide services through cloud providers, making them accessible via APIs. Key use cases include:

1. **Content Creation**: Generative AI can produce text, images, code, and even videos, enhancing marketing, design, and communication efforts.
2. **Customer Support**: AI chatbots and assistants automate customer interactions, reducing response times and improving service quality.
3. **Data Analysis & Insights**: Models help businesses analyze large datasets, enabling predictive analytics and trend forecasting.
4. **Customization**: Personalization of products and services, such as tailored recommendations or [transactional journeys](#transactional-journeys)/customer experiences, is powered by generative AI.
5. **Multi-Model Access**: Enterprises use AI gateways to integrate multiple generative models, allowing them to choose the best model for specific tasks based on performance or cost efficiency.

Cloud providers like **Google Cloud (Gemini)** or **Microsoft Azure (OpenAI)** offer easy integration of these models into business workflows through APIs, streamlining deployment for large-scale applications

## AI Gateway?

An AI Gateway is a middleware platform that simplifies and secures interactions between AI models and applications. In this context, businesses use AI gateways to streamline the integration, management, and deployment of generative AI models like those provided by OpenAI, Google (Gemini), and Anthropic. AI gateways provide the following key benefits:

1. **Model Access and Management**: They centralize access to multiple AI models via APIs, making it easier for businesses to switch between or utilize multiple AI models for different tasks.
2. **Security and Governance**: AI gateways add layers of security, enabling compliance with regulations and protecting proprietary data when using external AI services [1] . [2]
3. **Performance Optimization**: By handling the AI model interactions efficiently, gateways can reduce latency and improve [model performance](#model-performance) in business applications [3]
## 🌐 Sources
1. [konghq.com - What is an AI Gateway? Concepts and Examples](https://konghq.com/blog/enterprise/what-is-an-ai-gateway)
2. [ibm.com - How an AI Gateway provides leaders with greater control](https://www.ibm.com/blog/announcement/how-an-ai-gateway-provides-greater-control-and-visibility-into-ai-services/)
3. [traefik.io - AI Gateway: What Is It? How Is It Different From API Gateway?](https://traefik.io/glossary/ai-gateway/)

<a id="how-do-we-evaluate-of-llm-outputs"></a>
# How Do We Evaluate Of Llm Outputs {#how-do-we-evaluate-of-llm-outputs}

Methods for assessing the quality and relevance of LLM-generated outputs, critical for improving model performance.

The evaluation of [LLM](#llm) outputs involves various methodologies to assess their quality and relevance. 

### Important
 - Evaluating LLM outputs requires both quantitative metrics ([LLM Evaluation Metrics](#llm-evaluation-metrics)) and qualitative assessments (human judgment).
 - The iterative feedback loop from evaluations informs model improvements and prompt engineering strategies.

### Follow up questions
 - How does the inclusion of diverse datasets impact the robustness of LLM evaluations
 - [What are the best practices for evaluating the effectiveness of different prompts](#what-are-the-best-practices-for-evaluating-the-effectiveness-of-different-prompts)
### Related Topics
 - [Prompt engineering](#prompt-engineering) in natural language processing  



<a id="how-is-reinforcement-learning-being-combined-with-deep-learning"></a>
# How Is Reinforcement Learning Being Combined With Deep Learning {#how-is-reinforcement-learning-being-combined-with-deep-learning}


The sources touch upon reinforcement learning as an area beyond the scope of their discussion. However, the combination of [Reinforcement learning](#reinforcement-learning)  with [Deep Learning](#deep-learning) has shown remarkable results in recent years, particularly in areas like game playing and robotics. 

Exploring the potential of this combination in other domains and developing new algorithms that effectively integrate deep learning representations with reinforcement learning principles could lead to significant advancements in artificial intelligence.



<a id="how-is-schema-evolution-done-in-practice-with-sql"></a>
# How Is Schema Evolution Done In Practice With Sql {#how-is-schema-evolution-done-in-practice-with-sql}





## Structure of a goof [Git](#git) Commit Message

1. **Subject Line**
   - Keep it short (50 characters or less).
   - Use the imperative mood (e.g., "Fix bug" instead of "Fixed bug").
   - Capitalize the first letter.
   - Do not end with a period.

2. **Body (Optional)**
   - Separate from the subject line with a blank line.
   - Explain the "what" and "why" of the changes, not the "how".
   - Wrap text at 72 characters.

3. **Footer (Optional)**
   - Include references to issues or pull requests (e.g., "Closes #123").
   - Add any additional notes or metadata.

### **Good Examples**:

1. **Fix a Bug**  
   ```
   Fix incorrect login redirection
   The login redirection was leading to an unauthorized page after successful login. 
   This fix ensures users are redirected to their dashboard upon successful authentication.
   ```
   
2. **Add a New Feature**  
   ```
   Add search functionality to the user dashboard
   Introduced a search bar in the user dashboard, allowing users to quickly find relevant information within their profile.
   ```
   
3. **Update Documentation**  
   ```
   Update README to include new API endpoints
   Added details about the newly added API endpoints for user registration and password recovery in the README file.
   ```
   
4. **Refactor Code**  
   ```
   Refactor data fetching logic in the dashboard
   The data fetching logic was consolidated into a reusable service to improve maintainability and reduce duplication.
   ```
   
5. **Add Unit Tests**  
   ```
   Add unit tests for the authentication service
   Implemented unit tests for the login and registration methods to ensure robust coverage of authentication functionality.
   ```

### **Bad Examples**:

1. **Too Vague**  
   ```
   Update
   ```
   *This commit message doesn’t provide any meaningful context.*

2. **Incomplete Explanation**  
   ```
   Fix bug
   ```
   *This doesn’t explain the *what* or *why*, making it unclear to someone reviewing the code.*

3. **Too General**  
   ```
   Changed stuff
   ```
   *“Changed stuff” is not informative and doesn’t provide clear insight into what was actually modified.*

4. **No Context**  
   ```
   Fixed issue
   ```
   *No context is provided about what the issue was, making it difficult for others to understand the change.*

5. **Overly Short**  
   ```
   Remove unused variable.
   ```
   *The message could be expanded to include more context about why the variable was removed and what impact it had.*

---

### **Tips Expanded**:

- **Be Descriptive**: Commit messages should give a clear understanding of the changes. Describe what was changed, *why* it was changed, and *how* (if necessary).
- **Focus on One Change**: Avoid including unrelated changes in a single commit. Each commit should represent one logical unit of work.
- **Use Active Voice**: Avoid passive voice. Focus on the subject doing something (`Add`, `Fix`, `Implement`).
- **Wrap Text Appropriately**: Ensure lines don’t exceed 72 characters for better readability in Git log and discussions.
- **Be Concise**: Subject lines should be short (preferably under 50 characters), clear, and to the point without unnecessary detail.

---

### **Common Pitfalls** to Avoid:

- **Too Vague**: Commit messages like "Update" or "Fix bug" provide no actionable information.
- **Too Long**: Descriptive, yes, but avoid overly lengthy messages that become difficult to scan quickly.
- **No Context**: A good commit message should allow anyone reviewing it to understand the change without needing additional context.

<a id="how-to-model-to-improve-demand-forecasting"></a>
# How To Model To Improve Demand Forecasting {#how-to-model-to-improve-demand-forecasting}





<a id="how-to-normalise-a-merged-table"></a>
# How To Normalise A Merged Table {#how-to-normalise-a-merged-table}

See: [Normalised Schema](#normalised-schema)

Splitting out tables.

Resource:
[Database Normalization for Beginners | How to Normalize Data w/ Power Query (full tutorial!)](https://www.youtube.com/watch?v=rcrsqyFtJ_4)







<a id="how-to-reduce-the-need-for-gen-ai-responses"></a>
# How To Reduce The Need For Gen Ai Responses {#how-to-reduce-the-need-for-gen-ai-responses}



Reducing the need for frequent [Generative AI](#generative-ai) (Gen AI) responses can be done by leveraging techniques such as [caching](#caching) and setting up predefined [transactional journeys](#transactional-journeys). Here's a breakdown:

1. **Caching AI Responses**: Caching allows storing frequently requested AI responses and reusing them. This reduces the number of queries to the AI model, thus lowering both response time and cost. For example, common queries like "How do I reset my password?" can be cached for quick reuse without engaging the AI model each time (1).

2. **Predefined Transactional Journeys**: For repetitive tasks (e.g., "I want to close my account"), predefined <mark>workflows</mark> or "journeys" can be set up. These automate processes without requiring AI interaction. This is ideal for tasks like bill payments, account management, or order cancellations, where responses can be scripted or handled by traditional logic, bypassing AI.

### Examples of User Journeys:
- **Account Closure**: Guiding users through the steps to close an account without involving AI.
- **Password Reset**: Automating the reset process with predefined steps.
- **Order Tracking**: Providing real-time updates using existing tracking systems.
## 🌐 Sources
1. [medium.com - How Cache Helps in Generative AI Response and Cost Optimization](https://medium.com/@punya8147_26846/how-cache-helps-in-generative-ai-response-and-cost-optimization-9a6c9be058bb)
2. [medium.com - Slash Your AI Costs by 80%](https://medium.com/gptalk/slash-ai-costs-by-80-the-game-changing-power-of-prompt-caching-d44bcaa2e772)
3. [botpress.com - How to Optimize AI Spend Cost in Botpress](https://botpress.com/blog/how-to-optimize-ai-spend-cost-in-botpress)

### Vector Search with Graph Context

[Vector Embedding](#vector-embedding) plays a crucial role in enhancing search capabilities:

**Comparison of Vector-Only vs. Graph-RAG**: 
  - Vector-only searches may lack context, while Graph-RAG utilizes graph traversal to provide richer, multi-step context.
  - This leads to more complex and informative responses.

**Contextual Prompts**: 
  - Context is used to answer prompts (in JSON format). With graph traversal, this context involves more steps, allowing for more elaborate retrieval queries.

[Text2Cypher](#text2cypher)

[How to search within a graph](#how-to-search-within-a-graph)
#### Node Embedding

Useful in [GraphRAG](#graphrag) is understanding the relationships of nodes in a [Knowledge Graph](#knowledge-graph) using node embeddings.

![Pasted image 20241004074458.png](../content/images/Pasted%20image%2020241004074458.png)

<a id="how-to-use-sklearn-pipeline"></a>
# How To Use Sklearn Pipeline {#how-to-use-sklearn-pipeline}




<a id="how-would-you-decide-between-using-tf-idf-and-word2vec-for-text-vectorization"></a>
# How Would You Decide Between Using Tf Idf And Word2Vec For Text Vectorization {#how-would-you-decide-between-using-tf-idf-and-word2vec-for-text-vectorization}



<a id="hugging-face"></a>
# Hugging Face {#hugging-face}


Hugging Face is open-source platform known for its contributions to natural language processing (NLP) and machine learning. 

It provides a comprehensive library called [Transformer](#transformer), which includes pre-trained models for tasks such as text classification, translation, summarization, and question answering. 

Hugging Face is widely used for:

- **Access to Pre-trained Models**: Offers a vast collection of state-of-the-art models that can be easily fine-tuned for specific NLP tasks.
- **Ease of Use**: Simplifies the implementation of complex NLP models with user-friendly APIs.
- **Community and Collaboration**: Hosts a vibrant community where researchers and developers share models and datasets, fostering collaboration and innovation in AI.

### [Transformer](#transformer) library

Resources:
- [Getting Started With Hugging Face in 15 Minutes | Transformers, Pipeline, Tokenizer, Models](chat.openai.com/c/4e1f8036-5b3e-4372-9e0c-e8a7c0c15dfd)



<a id="hyperparameter-tuning"></a>
# Hyperparameter Tuning {#hyperparameter-tuning}


Objective:
- Tune the model’s hyperparameters to improve performance. For example, in regularized linear regression, the main hyperparameter to tune is the regularization strength (e.g., `alpha` in Ridge or Lasso).
- Use [Cross Validation](#cross-validation) to evaluate the model’s performance with different hyperparameters.

Optimization Techniques:
- [GridSeachCv](#gridseachcv): Exhaustively searches through a specified subset of hyperparameters.
- Random Search: Randomly samples from the hyperparameter space, often more efficient than grid search.
- [standardised/Optuna](#standardisedoptuna)
- [Regularisation](#regularisation): Often part of the hyperparameter tuning process, especially in models prone to overfitting.

Key Considerations
- Balance Between Exploration and Exploitation: Ensure a good balance between exploring the hyperparameter space and exploiting known good configurations.
- [Cross Validation](#cross-validation): Use cross-validation to ensure that the hyperparameter tuning process is robust and not overfitting to a particular train-test split.

Order matters: See [Interpretable Decision Trees](#interpretable-decision-trees). Example when tuning hyperparameters for [Random Forests](#random-forests) try the following:

1. High Train Accuracy, Low Test Accuracy (Overfitting)
- Objective: Reduce model complexity to prevent overfitting.
- Parameters to Adjust:
	- `max_depth`: Limit the depth of each tree.
	- `min_samples_split`: Increase the minimum number of samples required to split a node.
	- `max_features`: Reduce the number of features considered for splitting.
	- `n_estimators`: Decrease the number of trees in the forest.

2. Low Train Accuracy, Low Test Accuracy (Underfitting)
- Objective: Increase model complexity to improve learning capacity.
- Parameters to Adjust:
	- `n_estimators`: Increase the number of trees.
	- `max_depth`: Allow deeper trees.
	- `min_samples_split`: Decrease the minimum number of samples required to split a node.

3. Moderate Train Accuracy, Moderate Test Accuracy (Balanced but Low Performance)
- Objective: Fine-tune the model for better performance.
- Parameters to Adjust:
	- `max_features`: Experiment with different numbers of features.
	- `max_depth`: Adjust the depth of trees.
	- `n_estimators`: Fine-tune the number of trees.
	- `min_samples_split`: Adjust the minimum samples for splitting.

4. High Train Accuracy, High Test Accuracy (Optimal)
- Objective: Make minor adjustments for incremental improvements.
- Parameters to Adjust:
	- `n_estimators`: Slightly adjust the number of trees.
	- `max_features`: Fine-tune the number of features.
	- `min_samples_split`: Make small adjustments to the minimum samples for splitting.

### Links

See [ML_Tools](#ml_tools): [Hyperparameter_tuning_GridSearchCV.py](#hyperparameter_tuning_gridsearchcvpy)

Hyperparameter_tuning_RF.py
Video link: https://youtu.be/jUxhUgkKAjE?list=PLtqF5YXg7GLltQSLKSTnwCcHqTZASedbO&t=765


<a id="hyperparameter"></a>
# Hyperparameter {#hyperparameter}


Hyperparameters are parameters set before training that control the learning process, such as:
- the number of nodes in a [Neural network](#neural-network) 
- or k in [K-nearest neighbours|KNN](#k-nearest-neighboursknn).

The best ones are found with [Hyperparameter Tuning](#hyperparameter-tuning).

Also see:
- [Model Parameters](#model-parameters)
- [Model parameters vs hyperparameters](#model-parameters-vs-hyperparameters)




<a id="hypothesis-testing"></a>
# Hypothesis Testing {#hypothesis-testing}


Used to draw inferences about population parameters based on sample data. The process involves the formulation of two competing hypotheses: the null hypothesis ($H_0$) and the alternative hypothesis ($H_1$). 

### Key Concepts
- Null Hypothesis ($H_0$): The hypothesis that there is no effect or no difference, which we seek to test.
- Alternative Hypothesis ($H_1$): The hypothesis that indicates the presence of an effect or a difference.
- P-value: A measure that helps determine the strength of the evidence against $H_0$. A small p-value (typically $< 0.05$) suggests that we reject $H_0$, indicating that the observed effect is statistically significant.
### Decision Making
- Accepting $H_0$: This means there is insufficient evidence to support the alternative hypothesis, suggesting that any observed effect could be due to random chance.
- Rejecting $H_0$: This indicates that there is enough statistical evidence to conclude that the status quo does not represent the truth.
### Limitations
Hypothesis testing is subject to Type I errors (false positives) and Type II errors (false negatives). A small p-value does not guarantee practical significance or causation, and results can be influenced by sample variability.

### Example
An example of hypothesis testing is conducting a t-test to compare the means of two groups. The null hypothesis states that the means are equal ($H_0: \mu_1 = \mu_2$), while the alternative hypothesis states they are not equal ($H_1: \mu_1 \neq \mu_2$).

### Important Notes
- Hypothesis testing relies on the formulation of $H_0$ and $H_1$, and the decision to accept or reject $H_0$ is based on the [p values](#p-values).
- A small p-value indicates statistical significance but does not imply practical relevance or causation.

### Follow-up Questions and Answers

##### In hypothesis testing, why might a very small p-value still lead to incorrect conclusions?  

A very small p-value might lead to incorrect conclusions due to several factors, including:
   - Sample Size: With large sample sizes, even trivial effects can yield small p-values, leading to the rejection of $H_0$ for effects that are not practically significant.
   - Multiple Comparisons: Conducting multiple tests increases the risk of Type I errors, where we incorrectly reject $H_0$.
   - Misinterpretation: A small p-value does not imply that the effect is large or important; it merely <mark>indicates that the observed data is unlikely under $H_0$.</mark>

##### How does the inclusion of effect size metrics improve the interpretation of hypothesis testing results?  

Including <mark>effect size metrics</mark> provides a quantitative measure of the magnitude of the observed effect, allowing researchers to assess the practical significance of their findings. While p-values indicate whether an effect exists, <mark>effect sizes help determine how meaningful that effect is</mark> in real-world terms.

##### What are the implications of multiple testing on the validity of p-values in hypothesis testing?  

Multiple testing increases the likelihood of encountering false positives (Type I errors). When multiple hypotheses are tested simultaneously, the probability of incorrectly rejecting at least one true null hypothesis rises. This necessitates adjustments to p-values (e.g., Bonferroni correction) to maintain the overall error rate.

### Related Topics
- Bayesian statistics and its approach to hypothesis testing
- The role of confidence intervals in statistical [inference](#inference)
- [Statistics](#statistics)
- [Testing](#testing)


<a id="heterogeneous-features"></a>
# Heterogeneous Features {#heterogeneous-features}


## Description

In machine learning, heterogeneous features refer to a situation where the input data contains a variety of different types of features. Let's break it down:

### 1. **Features:**
   - Features are the individual measurable properties or characteristics of the data used for making predictions in a machine learning model.
   - For example, in a dataset about houses, features could include the number of bedrooms, square footage, location, and whether it has a garden.

### 2. **Homogeneous vs. Heterogeneous:**
   - **Homogeneous Features:** In some datasets, all features are of the same type, such as numerical or categorical. For instance, a dataset containing only numerical features like age, income, and temperature is homogeneous.
   - **Heterogeneous Features:** In contrast, heterogeneous features refer to datasets where features are of different types. This means the dataset may contain a mix of numerical, categorical, text, image, or other types of data.

### 3. **Examples of Heterogeneous Features:**
   - **Numerical Features:** Represented by continuous values like age, income, or temperature.
   - **Categorical Features:** Represented by discrete values such as gender, city, or type of car.
   - **Text Features:** Textual data like product descriptions, customer reviews, or email content.
   - **Image Features:** Visual data represented by pixels in an image, used in tasks like image recognition or object detection.

### 4. **Challenges and Considerations:**
   - Handling heterogeneous features requires specialized techniques in [Preprocessing](#preprocessing) and model building.
   - Different types of features may need different preprocessing steps, such as encoding categorical variables, scaling numerical features, or extracting features from text or images.
   - Models need to be capable of handling diverse data types, either through feature engineering or using algorithms specifically designed for heterogeneous data.

### 5. **Applications:**
   - Heterogeneous features are common in many real-world applications, such as e-commerce (combining text descriptions with numerical features), healthcare (integrating medical records with images or text), and social media analysis (analyzing text, images, and user profiles).

### 6. **Resources for Further Learning:**
   - Feature Engineering for Machine Learning: [https://www.datacamp.com/community/tutorials/feature-engineering-kaggle](https://www.datacamp.com/community/tutorials/feature-engineering-kaggle)
   - Handling Text Data in Machine Learning: [https://towardsdatascience.com/handling-text-data-in-machine-learning-projects-b52bbc9531d7](https://towardsdatascience.com/handling-text-data-in-machine-learning-projects-b52bbc9531d7)
   - Image Feature Extraction Techniques: [https://towardsdatascience.com/image-feature-extraction-techniques-91e8625616f1](https://towardsdatascience.com/image-feature-extraction-techniques-91e8625616f1)

Understanding how to work with heterogeneous features is essential for building effective machine learning models that can handle diverse types of data and extract meaningful insights from them.


<a id="how-do-you-do-the-data-selection"></a>
# How Do You Do The Data Selection {#how-do-you-do-the-data-selection}

When you sample a dataset, [how do you do the data selection](#how-do-you-do-the-data-selection)? [Data Selection](#data-selection)
A: By randomly sampling, by time period (use a feature)..