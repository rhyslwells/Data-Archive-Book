# G

## Table of Contents
* [GIS](#gis)
* [GRU](#gru)
* [GSheets](#gsheets)
* [Gaussian Distribution](#gaussian-distribution)
* [Gaussian Mixture Models](#gaussian-mixture-models)
* [Gaussian Model](#)
* [Gaussian_Mixture_Model_Implementation.py](#gaussian_mixture_model_implementationpy)
* [General Linear Regression](#general-linear-regression)
* [Generative AI From Theory to Practice](#)
* [Generative AI](#generative-ai)
* [Generative Adversarial Networks](#generative-adversarial-networks)
* [Get data](#get-data)
* [Gini Impurity vs Cross Entropy](#gini-impurity-vs-cross-entropy)
* [Gini Impurity](#gini-impurity)
* [Git](#git)
* [Gitlab](#gitlab)
* [Google Cloud Platform](#google-cloud-platform)
* [Google My Maps Data Extraction](#google-my-maps-data-extraction)
* [Gradient Boosting Regressor](#gradient-boosting-regressor)
* [Gradient Boosting](#gradient-boosting)
* [Gradient Descent](#gradient-descent)
* [Gradio](#gradio)
* [Grain](#grain)
* [Grammar method](#grammar-method)
* [Graph Analysis Plugin](#graph-analysis-plugin)
* [Graph Neural Network](#graph-neural-network)
* [Graph Theory Community](#graph-theory-community)
* [Graph Theory](#graph-theory)
* [GraphRAG](#graphrag)
* [Grep](#grep)
* [GridSeachCv](#gridseachcv)
* [Groupby vs Crosstab](#groupby-vs-crosstab)
* [Groupby](#groupby)
* [Grouped plots](#grouped-plots)
* [Guardrails](#guardrails)
* [gitlab-ci.yml](#gitlab-ciyml)
* [granularity](#granularity)



<a id="gis"></a>
# Gis {#gis}

Geographic information system.

File formats: 

The Web Map Tile Service (WMTS) and Web Feature Server (WFS) are both specifications used in the field of Geographic Information Systems (GIS) to serve different types of geographic data over the web. The primary differences between them lie in the type of data they serve and how they serve it.

	[Web Map Tile Service (WMTS)](#web-map-tile-service-wmts)
	[Web Feature Server (WFS)](#web-feature-server-wfs)
	[Key Differences of Web Feature Server (WFS) and Web Feature Server (WFS)](#key-differences-of-web-feature-server-wfs-and-web-feature-server-wfs)

[shapefile](#shapefile)

There are free GIS softwares




<a id="gru"></a>
# Gru {#gru}



<a id="gsheets"></a>
# Gsheets {#gsheets}


Useful functions:
- [QUERY GSheets](#query-gsheets)
- ARRAYFORMULA
- Indirect

Accessing google sheets from a script:
https://www.youtube.com/watch?v=zCEJurLGFRk

<a id="gaussian-distribution"></a>
# Gaussian Distribution {#gaussian-distribution}


Common assumption for a [Distributions](#distributions).

<a id="gaussian-mixture-models"></a>
# Gaussian Mixture Models {#gaussian-mixture-models}



Gaussian Mixture Models (GMMs) represent data as a mixture of multiple Gaussian [distributions](#distributions), with each cluster corresponding to a different Gaussian component. GMMs are more effective than [K-means](#k-means) because they consider the distributions of the data rather than relying solely on distance metrics.

Soft [Clustering](#clustering) technique.

In [ML_Tools](#ml_tools) see: [Gaussian_Mixture_Model_Implementation.py](#gaussian_mixture_model_implementationpy)

[Kmeans vs GMM](#kmeans-vs-gmm)

GMMs can have difference [Covariance Structures](#covariance-structures)
## Key Concepts

- **Gaussian Components**: Each Gaussian distribution is characterized by its mean and [Covariance](#covariance).
- **Likelihood**: The likelihood of a data point belonging to a cluster is given by the formula:
  $$
  P(X | C_k) = \pi_k \cdot \mathcal{N}(X | \mu_k, \Sigma_k)
  $$
  where $P(X | C_k)$ is the probability of data point $X$ given cluster $C_k$, $\pi_k$ is the prior probability of cluster $C_k$, and $\mathcal{N}$ is the Gaussian distribution.
- **Expectation-Maximization (EM) Algorithm**: GMMs utilize the EM algorithm to iteratively optimize the parameters of the Gaussian components.

## Advantages of GMMs

- **Complex Data Distributions**: GMMs can capture complex data distributions, unlike [K-means](#k-means), which only considers distance metrics.
- **Probabilistic Framework**: GMMs provide a probabilistic framework for clustering, allowing for soft assignments of data points to clusters.
- **Modeling Elliptical Clusters**: The use of covariance matrices enables GMMs to model elliptical clusters, enhancing clustering performance.

## Applications

- **[Anomaly Detection](#anomaly-detection)**: GMMs are widely used in various applications, including anomaly detection.

## Important Considerations

- **Covariance Types**: The choice of covariance types (full, tied, diagonal, spherical) can significantly impact the performance of GMMs.

## Follow-up Questions

- How do GMMs compare to other clustering algorithms in terms of scalability and computational efficiency?
- What are the implications of choosing different covariance types in GMMs?

![Pasted image 20250126135722.png|500](../content/images/Pasted%20image%2020250126135722.png|500)

### Gaussian Model 

(Univariate)

- **Formula:**  
    $p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2 \sigma^2}\right)$
- **Steps:**
    - Estimate $\mu$ and $\sigma^2$ from the data.
    - Compute the probability density for each data point.
    - Points with low probabilities (below a threshold $\epsilon$) are considered anomalies.

![Pasted image 20241230202826.png|500](../content/images/Pasted%20image%2020241230202826.png|500)

### **5. Multivariate Gaussian Distribution**

- **Steps:**
    - Extend the Gaussian model to include covariance across features.
    - Fit the multivariate Gaussian model:  
        $p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)$
        - $\mu$: Mean vector
        - $\Sigma$: Covariance matrix
    - Threshold low-probability examples to identify anomalies.

<a id="gaussian_mixture_model_implementationpy"></a>
# Gaussian_Mixture_Model_Implementation.Py {#gaussian_mixture_model_implementationpy}

https://github.com/rhyslwells/ML_Tools/blob/main/Explorations/Build/Clustering/Gaussian_Mixture_Model_Implementation.py

Follow-Up Questions

- How do GMMs compare to other clustering algorithms in terms of scalability and computational efficiency?
- What are the implications of choosing different covariance types in GMMs?


<a id="general-linear-regression"></a>
# General Linear Regression {#general-linear-regression}


[Linear Regression](#linear-regression)

[t-test](#t-test) - to compare means between two populations.

[ANOVA](#anova) - tests 




### Objective:

 How do LLMs work and operate.  
 Enabling [LLM](#llm)'s at scale:
 Explore recent AI and Generative AI language models 


### Steps

Math on words: Turn words into coordinates.
Statistics on words: Given context what is the probability of whats next.
Vectors on words. Cosine similarity
How train: Use [Markov chain](#markov-chain) for prediction of the next [Tokenisation](#tokenisation)


Tokeniser: map from token to number

1. Pre-training: tokenise input using [NLP](#nlp) techqinues
2. [LLM](#llm) looks at context: nearby tokens, in order to predict

different implmentationg for differnet languages. Differnet tokenisers or translating after.

Journey to scale:

1. Demos, POC (plan to scale): understand limitations
2. Beyond experiments and before production: 
3. Enterprise level: translate terms so they can use governess techniques.

Building:

![Pasted image 20240524130607.png](../content/images/Pasted%20image%2020240524130607.png)

### [Software Development Life Cycle](#software-development-life-cycle)

For GenAI: Building an applicaiton with GenAi features

1. Plan: use case: prompts : archtecture: cloud or on site
2. Build: vector database
3. Test: Quality and responsible ai. 

### [call summarisation](#call-summarisation)

take transcript - > summariser -> summarise

Source: human labeled transcripts to check summariser. 

![Pasted image 20240524131311.png|500](../content/images/Pasted%20image%2020240524131311.png|500)

[Ngrams](#ngrams) analysis - when specific words realy matter


### [RAG](#rag)

Use relvant data to make response better:

![Pasted image 20240524131603.png](../content/images/Pasted%20image%2020240524131603.png)

## [GAN](#gan)

For image models.

Examples: midjourney,stable diffusion,dall-e 3

image model techniques:
- text to image
- image to image
## Notes: 

Use [LLM](#llm)'s to get short info, then cluster.
Going round training data : called a Epochs












<a id="generative-ai"></a>
# Generative Ai {#generative-ai}



<a id="generative-adversarial-networks"></a>
# Generative Adversarial Networks {#generative-adversarial-networks}


   Composed of two neural networks, a generator, and a discriminator, that compete against each other. GANs are used for tasks like generating realistic images or videos.

<a id="get-data"></a>
# Get Data {#get-data}


# What is involved:

`df = pd.read_csv('Categorical.csv')`

- Gather relevant data from appropriate sources, addressing any quality or privacy concerns.

```python
## Get textbook data using for example:

import re
def read_file(filename):
    with open(filename, "r", encoding='UTF-8') as file:
        contents = file.read().replace('\n\n',' ').replace('[edit]', '').replace('\ufeff', '').replace('\n', ' ').replace('\u3000', ' ')
    return contents
text = read_file('Data various/Monte_Cristo.txt')

text_start = [m.start() for m in re.finditer('VOLUME ONE', text)]
text_end = [m.start() for m in re.finditer('End of Project Gutenberg', text)]
text = text[text_start[1]:text_end[0]]
```


How would you approach a colleague who is hesitant to share their data?
?
- explain the purpose and benefits
- ensure confidentiality (GDPR) with data masking.
- and finding common ground to address any concerns or objections.
- build trust.
- make agreements of terms of use/ownership/document the data accessing process.

How would you go about obtaining the necessary permissions for a dataset?
?
- establishing clear communication channels within the organsisation.
- obtaining necessary approvals
- emphasizing the value of collaboration.

How would you gather sensitive data?;; Get consent. Ensure anonyminaty (follow regularions)

How to you ensure data is unbiased and representative.
?
- Stratified sampling, (group then randomly sample).
- Examine the data sources.

<a id="gini-impurity-vs-cross-entropy"></a>
# Gini Impurity Vs Cross Entropy {#gini-impurity-vs-cross-entropy}

When working with decision trees, both [Gini Impurity](#gini-impurity) and [Cross Entropy](#cross-entropy) are metrics used to evaluate the quality of a split. They help determine how well a feature separates the classes in a dataset.

### Gini Impurity

- **Definition**: Gini impurity measures the probability of incorrectly classifying a randomly chosen element if it was randomly labeled according to the distribution of labels in the node.
- **Computation**: Generally faster to compute than cross-entropy because it does not involve logarithms.
- **Use Case**: Often used in the CART (Classification and Regression Trees) algorithm. It is a good default choice for classification tasks due to its simplicity and efficiency.

### Cross Entropy (More refined than impurity)

- **Definition**: Cross-entropy measures the amount of information needed to encode the class distribution of the node. It quantifies the expected amount of information required to classify a new instance.
- **Computation**: Involves logarithmic calculations, which can be computationally more intensive than Gini impurity.
- **Use Case**: Often used in algorithms like ID3 and C4.5. It can be more informative in cases where the class [Distributions](#distributions) is skewed or when you need a more nuanced measure of impurity.

### Choosing Between Gini Impurity and Cross Entropy

- **Performance**: In practice, both metrics often lead to similar results in terms of the structure and performance of the decision tree. The choice between them may not significantly affect the final model.
- **Efficiency**: If computational efficiency is a concern, Gini impurity might be preferred due to its simpler calculation.
- **Interpretability**: Cross-entropy provides a more information-theoretic perspective, which might be preferred if you are interested in the information gain aspect of the splits.

<a id="gini-impurity"></a>
# Gini Impurity {#gini-impurity}

Gini impurity is a metric used in decision trees to measure the degree or probability of misclassification in a dataset. It is associated with the leaves of a [Decision Tree](#decision-tree) and helps determine the best split at each node.
## Calculation

- Mathematical Formula: Gini impurity is calculated as the probability of incorrectly classifying a randomly chosen element if it were randomly labelled according to the distribution of labels in the subset.
- Formula: 

  $$ \text{Gini Impurity} = 1 - \sum_{i=1}^{n} p_i^2 $$
  where $p_i$ is the probability of an element being classified into a particular class.

## Usage

- Decision Trees: Gini impurity is commonly used in decision trees to evaluate splits. A lower Gini impurity indicates a better split, as it means the data is more homogeneously classified.
- [Classification](#classification) Tasks: It is particularly useful in classification tasks where the goal is to minimize misclassification.

## Relationship to Other Metrics

- Gini impurity is one of several [Regression Metrics](#regression-metrics) used to evaluate the performance of decision trees, alongside others like entropy.

## Example

Suppose you have a dataset with a binary classification problem, where the target variable can be either "Yes" or "No". You have a node in your decision tree with the following distribution of classes:

- 10 samples labeled "Yes"
- 5 samples labeled "No"

### Gini Impurity Calculation

The formula for Gini impurity is:

$$ \text{Gini impurity} = 1 - \sum (p_i^2) $$

where$p_i$ is the proportion of class$i$ in the node.

#### Step-by-Step Calculation

1. **Calculate the proportion of each class**:
   - Total samples = 10 (Yes) + 5 (No) = 15
   - Proportion of "Yes" =$\frac{10}{15} = 0.67$
   - Proportion of "No" =$\frac{5}{15} = 0.33$

2. **Calculate the squared proportions**:
   - $(0.67)^2 = 0.4489$
   - $(0.33)^2 = 0.1089$

3. **Sum the squared proportions**:
   - Sum =$0.4489 + 0.1089 = 0.5578$

4. **Calculate the Gini impurity**:
   - Gini impurity =$1 - 0.5578 = 0.4422$


A Gini impurity of 0.4422 indicates the level of impurity in this node. A Gini impurity of 0 would mean the node is pure (all samples belong to one class), while a higher value indicates more impurity or mixed classes.

This calculation helps in deciding whether to split the node further or not. The goal is to choose splits that <mark>minimize the Gini impurity</mark>, leading to more homogeneous branches.

<a id="git"></a>
# Git {#git}



tags:
  - software

Do git bash here.

git status

git add . (adds all)

git status

git commit -m ""

git push

## Notes

https://www.youtube.com/watch?v=xnR0dlOqNVE

[Git Fork vs. Git Clone](https://www.youtube.com/watch?v=6YQxkxw8nhE)

[How to do git commit messages properly](#how-to-do-git-commit-messages-properly)

## Examples


# Git: Common Issues and Fixes

Git can be frustrating, especially when things go wrong. This guide provides practical solutions to common Git mistakes, explained in simple terms.

https://ohshitgit.com/

### how to remove something from a git history, if i forgot to add it to the gitignore, but now have

2. Remove the file from the Git index
This tells Git to stop tracking the file.

git rm --cached path/to/file
For a folder:
git rm -r --cached path/to/folder

3. Commit this change
This saves the removal from the index.

bash
Copy code
git commit -m "Stop tracking path/to/file and add to .gitignore"


## Undoing Mistakes

### I messed up badly! Can I go back in time?

Yes! Use Git’s reflog to find a previous state:

```bash
git reflog
# Find the index of the state before things broke
git reset HEAD@{index}
```

_This is useful for recovering deleted commits, undoing bad merges, or rolling back to a working state._

## Commit Fixes

### I committed but forgot a small change!

```bash
# Make the change
git add .
git commit --amend --no-edit
```

⚠ Warning: Never amend a commit that has already been pushed!

### I need to change the last commit message!

```bash
git commit --amend
```

This will open an editor where you can modify the commit message.



## 🔀 Branching Issues

### I committed to `master` but wanted a new branch!

```bash
# Create a new branch from the current state
git branch new-branch
# Remove the commit from master
git reset HEAD~ --hard
git checkout new-branch
```

⚠ Warning: If you’ve already pushed the commit, additional steps are needed.

### I committed to the wrong branch!

```bash
# Undo the last commit but keep the changes
git reset HEAD~ --soft
git stash
git checkout correct-branch
git stash pop
git add .
git commit -m "Moved commit to correct branch"
```

Alternative:

```bash
git checkout correct-branch
git cherry-pick master  # Moves last commit to correct branch
git checkout master
git reset HEAD~ --hard  # Removes the commit from master
```



## 🔍 Diff and Reset

### I ran `git diff`, but it showed nothing!

If your changes are staged, use:

```bash
git diff --staged
```

This shows differences between the last commit and staged files.

### I need to undo a commit from 5 commits ago!

```bash
git log  # Find the commit hash
git revert [commit-hash]
```

This creates a new commit that undoes the changes.



## 🗑️ Undoing Changes

### I need to undo changes to a file!

```bash
git log  # Find a commit before the changes
git checkout [commit-hash] -- path/to/file
git commit -m "Reverted file to previous version"
```

### I want to reset my repo to match the remote!

⚠ _Destructive action—this cannot be undone!_

```bash
git fetch origin
git checkout master
git reset --hard origin/master
git clean -d --force  # Removes untracked files
```



## 🤯 Last Resort

If everything is completely broken, nuke the repo and reclone:

```bash
cd ..
sudo rm -r repo-folder
git clone https://github.com/user/repo.git
cd repo-folder
```



<a id="gitlab"></a>
# Gitlab {#gitlab}

[GitLab CI CD Tutorial for Beginners Crash Course](https://www.youtube.com/watch?v=qP8kir2GUgo)

  - Provides managed runners to execute [CI-CD](#ci-cd) pipelines.
  - Integrates with version control systems to automate the CI/CD process.




<a id="google-cloud-platform"></a>
# Google Cloud Platform {#google-cloud-platform}

Google Cloud Platform is a suite of cloud computing services offered by Google. It provides a range of services including computing, storage, and application development that run on Google hardware.

Resources:
 [Introduction to Google Cloud](https://www.youtube.com/watch?v=IeMYQqJeK4)
### Compute Engine

 Description: GCP's Infrastructure as a Service (IaaS) offering, allowing users to run virtual machines on Google's infrastructure.
 
 Features:
   Custom Machine Types: Create VMs with custom configurations.
   Preemptible VMs: Costeffective, shortlived instances for batch jobs and faulttolerant workloads.
   Sustained Use Discounts: Automatic discounts for prolonged usage.
   Persecond Billing: Charges calculated per second for cost savings.
   
 Use Cases: Suitable for web hosting, data processing, and largescale applications.
 Integration: Works seamlessly with other GCP services like Google [Kubernetes](#kubernetes) Engine, Cloud Storage, and [BigQuery](#bigquery).

### Bigtable
 A scalable [NoSQL](#nosql) database service for large analytical and operational workloads.

### App Engine
 A platform for building scalable web applications and mobile backends.

### [BigQuery](#bigquery)
 A fullymanaged, serverless data warehouse for largescale data analytics.

### Cloud Storage
 Object storage service for storing and accessing data on Google's infrastructure.

### Cloud [SQL](#sql)
 Managed relational database service for [MySQL](#mysql), PostgreSQL, and SQL Server.

### CI/CD
 Tools and services for continuous integration and continuous delivery.

### [standardised/Firebase](#standardisedfirebase)
 A platform for building mobile and web applications with realtime databases, authentication, and more.

## Notes:
 Consider setting up a personal GCP example for handson experience.
 Explore the generic repository for additional resources and examples.


<a id="google-my-maps-data-extraction"></a>
# Google My Maps Data Extraction {#google-my-maps-data-extraction}


### Summary:

This guide covers the key workflows and tools for managing and processing location data in Google Sheets and Google My Maps. Suppose we have marks on Google My Maps. In order to extract the location of markers to a google sheet.

1. **Export Marker Data** from Google My Maps as KML/CSV.
2. Convert KML to CSV
3. **Use Apps Script in Google Sheets** to extract data like addresses or postal codes from coordinates.

### **Extract Data from Google My Maps**  
   - **Export Custom Markers:**
     1. Open Google My Maps.
     2. Use the menu (three dots) to select **Export to KML**.
     3. The exported file will contain marker names, descriptions.

### Extract KML data to google sheets
   
- Rename KML file to XML.
- Open XML in excel.
- Extract marker and coordinate data.
- Paste data into google sheets.

### **Extracting Information from Coordinates in Google Sheets**  

Use [**Google Apps Script**](#google-apps-script) to extract additional information like addresses or postal codes from geographic coordinates.

- **Get Address from Coordinates**:
  ```javascript
  function getAddress(lat, lng) {
    var response = Maps.newGeocoder().reverseGeocode(lat, lng);
    var result = response.results[0];
    if (result) {
      return result.formatted_address;
    } else {
      return 'No address found';
    }
  }
  ```
- **Get Postal Code from Coordinates**:
  ```javascript
  function getPostalCode(lat, lng) {
    var response = Maps.newGeocoder().reverseGeocode(lat, lng);
    var result = response.results[0];
    if (result) {
      for (var i = 0; i < result.address_components.length; i++) {
        var component = result.address_components[i];
        if (component.types.indexOf('postal_code') !== -1) {
          return component.long_name;
        }
      }
      return 'Postal code not found';
    } else {
      return 'No results found';
    }
  }
  ```
- Use `getAddress()` with `getPostalCode()`.
- =getPostalCode(56.033139, -3.4182519)




<a id="gradient-boosting-regressor"></a>
# Gradient Boosting Regressor {#gradient-boosting-regressor}




https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor

[Boosting](#boosting)

The `GradientBoostingRegressor` from the `sklearn.ensemble` module is a model used for regression tasks. It builds an [Model Ensemble](#model-ensemble) of [Decision Tree](#decision-tree) in a sequential manner, where each tree tries to correct the errors made <mark>by the previous ones</mark>. Here’s a breakdown of the key parameters:

1. **loss**: Specifies the loss function to optimize. Default is `'squared_error'`, which is the least-squares loss function. Other options like `'absolute_error'` can be used for robustness against outliers.

2. **learning_rate**: Controls the contribution of each tree to the final prediction. A smaller value (e.g., 0.01) makes the model learn more slowly, but it can lead to better generalization. Default is 0.1.

3. **n_estimators**: The number of boosting stages (i.e., trees). More trees can improve performance but also increase the risk of overfitting. Default is 100.

4. **subsample**: The fraction of samples to be used for fitting each tree. Setting this to a value less than 1.0 can help reduce overfitting, at the cost of a slight increase in bias. Default is 1.0 (use all samples).

5. **criterion**: The function used to measure the quality of a split. `'friedman_mse'` is the default, which is an improved version of mean squared error for decision trees. Other options include `'mse'` and `'mae'`.

6. **max_depth**: The maximum depth of the individual trees. This parameter controls the complexity of each tree. Default is 3, which typically works well for most tasks.

7. **min_samples_split**: The minimum number of samples required to split an internal node. Default is 2, meaning any node can be split as long as there are at least 2 samples.

8. **min_samples_leaf**: The minimum number of samples required to be at a leaf node. This helps control overfitting by requiring more data points at each leaf. Default is 1.

9. **alpha**: The quantile used for the loss function in cases of robust regression. This is useful when dealing with data that includes outliers. Default is 0.9.

10. **validation_fraction**: The fraction of training data to set aside for validation to monitor performance during training. Default is 0.1.

11. **n_iter_no_change**: The number of iterations with no improvement on the validation score to wait before stopping the training early. Default is `None`, meaning no early stopping.

12. **ccp_alpha**: Complexity parameter used for pruning the trees. A larger value leads to more pruning (simplifying the model), which can help prevent overfitting.


<a id="gradient-boosting"></a>
# Gradient Boosting {#gradient-boosting}


Gradient Boosting is a technique used for building predictive models [Model Building](#model-building), particularly in tasks like regression and classification. It combines the concepts of [Boosting](#boosting) and [Gradient Descent](#gradient-descent) to create strong models by sequentially combining multiple [Weak Learners](#weak-learners) ([Decision Tree](#decision-tree). 

Key Idea: Instead of fitting a single strong model, Gradient Boosting builds multiple weak learners sequentially. Each new model focuses on <mark>correcting the mistakes made by the previous ones</mark> by fitting to the residuals (differences between observed and predicted values).

Gradient Boosting builds an ensemble of [Weak Learners](#weak-learners) (usually [Decision Tree](#decision-tree)) sequentially. Each new model focuses on the errors of the previous ones, aiming to minimize the residual errors.

Final Prediction: The final prediction is made by aggregating the predictions of all the weak models, usually through a weighted sum.

High Performance: Known for its high performance and efficiency in terms of speed and memory usage.

[Watch Video Explanation](https://www.youtube.com/watch?v=3CC4N4z3GJc)

[Model Ensemble](#model-ensemble)
### Key Components

- [Weak Learners](#weak-learners): Typically decision trees used in the ensemble.
- [Loss Function](#loss-function): Measures how well the model fits the data.
- [learning rate](#learning-rate): Controls the contribution of each weak learner to the final model.

### Examples

- [LightGBM](#lightgbm)
- [XGBoost](#xgboost)
- [CatBoost](#catboost)
### Benefits

- Predictive Accuracy: Often outperforms other [Machine Learning Algorithms](#machine-learning-algorithms).
- Feature Handling: Effectively manages [heterogeneous features](#heterogeneous-features) and automatically selects relevant ones.
- [Overfitting](#overfitting): Less prone to overfitting compared to other complex models.

<a id="gradient-descent"></a>
# Gradient Descent {#gradient-descent}


Gradient descent is an [Optimisation function](#optimisation-function) used to minimize errors in a model by adjusting its parameters iteratively. It works by moving in the direction of the steepest decrease of the [Loss function](#loss-function).

Uses the difference quotient.

The step size is important between derivatives (small then slow) (if large then might miss minimum).

With Stochastic method we can don't need to the entire data set again, we can just add the new information to get improvement.

Gradient descent uses the entire data set.

Used to find the min/max of [Cost Function](#cost-function).

Given any point on the cost function surfaces. Then ask, "In what direction should I go to make the biggest change downhill or up hill, i.e. gradient descent"

![Obsidian_EPIqLAto5w.png|500](../content/images/Obsidian_EPIqLAto5w.png|500)

![Obsidian_FEGflF5RKQ.png|500](../content/images/Obsidian_FEGflF5RKQ.png|500)

How do you implement Gradient descent? You update the (direction) parameter by the small step by the [learning rate](#learning-rate).

![Obsidian_M4mzGSAx7d.png|500](../content/images/Obsidian_M4mzGSAx7d.png|500)

### [Stochastic Gradient Descent](#stochastic-gradient-descent)
Stochastic uses random entries to get derivative instead of the full dataset
Why do we use [Stochastic Gradient Descent](#stochastic-gradient-descent)?;; To find the derivative of discrete data so we can determine a straight line with the Least Square Error (LSE).
What is [Stochastic Gradient Descent](#stochastic-gradient-descent)?;; updates the model parameters based on the gradient of a single randomly chosen data point. 
### [Batch gradient descent](#batch-gradient-descent)
What is [Batch gradient descent](#batch-gradient-descent)?;; computes the gradient of the entire dataset,
### [Mini-batch gradient descent](#mini-batch-gradient-descent)
Stochastic Mini-batched descent is the fastest way (groups then does randomly).
What is [Mini-batch gradient descent](#mini-batch-gradient-descent)?;; Is a compromise of [Batch gradient descent](#batch-gradient-descent) and [Stochastic Gradient Descent](#stochastic-gradient-descent).

**What is the difference between batch gradient descent and stochastic gradient descent?**;; Batch gradient descent computes the gradient of the cost function using the entire training dataset in each iteration, while stochastic gradient descent updates the model's parameters based on the gradient of the cost function with respect to one training example at a time. Mini-batch gradient descent is a compromise, using a subset of the training data in each iteration.
# [Gradient Descent](#gradient-descent)

Gradient descent is commonly used in:
- **Deep Learning**: Frameworks like TensorFlow and PyTorch use variations of gradient descent for training.
- **Custom Implementations**: If you write logistic regression from scratch, gradient descent is a straightforward optimization method.

### **How Gradient Descent Works**

Gradient Descent, a common [Optimisation techniques](#optimisation-techniques), iteratively updates the [Model Parameters](#model-parameters) by computing the gradient of the [loss function](#loss-function) with respect to the parameters. The update formula is:

$\theta = \theta - \alpha \nabla_{\theta} \text{Cost}(\theta)$

Where:
- $\theta$ are the parameters (intercept and coefficients).
- $\alpha$ is the learning rate (step size for updates).
- $\nabla_{\theta} \text{Cost}(\theta)$ is the gradient of the [cost function](#cost-function) with respect to the parameters $\theta$.

#### Process:

1. Calculate the gradient of the loss function.
2. Adjust the parameters in the direction of the negative gradient (to reduce loss).
3. Repeat until either:
    - The loss function converges (minimal change between updates), or
    - The maximum number of iterations is reached.

[Gradient Descent](#gradient-descent)

[Cost Function](#cost-function) value versus number of iterations of [Gradient Descent](#gradient-descent) should decrease

Can use contour plots to show [Gradient Descent#](#gradient-descent) moving towards minima.
![Pasted image 20241224082847.png](../content/images/Pasted%20image%2020241224082847.png)

<a id="gradio"></a>
# Gradio {#gradio}

Gradio is an open-source platform that simplifies the process of <mark>creating user interfaces</mark> for machine learning models. 

It allows users to quickly build interactive demos and applications for their models without extensive front-end development knowledge. 

Main uses:

- **Interactive Interfaces**: Gradio provides a simple way to create web-based interfaces where users can interact with machine learning models by uploading files, entering text, or adjusting sliders.
- **Rapid Prototyping**: It enables quick prototyping and sharing of machine learning models, making it easier to demonstrate model capabilities to stakeholders or gather user feedback.
- **Ease of Integration**: Gradio can be easily integrated with popular machine learning frameworks like TensorFlow, PyTorch, and Hugging Face Transformers, allowing seamless deployment of models.

### Related content

[Video Link](https://www.youtube.com/watch?v=eE7CamOE-PA&list=PLcWfeUsAys2my8yUlOa6jEWB1-QbkNSUl&index=2)
https://www.gradio.app/

[Overview](#overview)

<a id="grain"></a>
# Grain {#grain}

Grain
   - Definition: The level of detail or [granularity](#granularity) of the data stored in the fact table.
   - Importance: Defining the grain is crucial as it determines what each record in the fact table represents (e.g., individual transactions, daily summaries).

<a id="grammar-method"></a>
# Grammar Method {#grammar-method}

can understand the Grammar as a method for acceptable sentences.

<a id="graph-analysis-plugin"></a>
# Graph Analysis Plugin {#graph-analysis-plugin}



<a id="graph-neural-network"></a>
# Graph Neural Network {#graph-neural-network}


Resources:
- [How Graph Neural Networks Are Transforming Industries](https://www.youtube.com/watch?v=9QH6jnwqrAk&list=PLcWfeUsAys2kC31F4_ED1JXlkdmu6tlrm&index=6)

Use cases:
- [Recommender systems](#recommender-systems) i.e. Uber, Pinterest (PinSage)
- Traffic Prediction - Deepmind in google maps
- Weather forecasting - GraphCast - Deepmind
- Data Mining - Relational Deep Learning
- Material Science - Deepmind - GNome - Density function theory.
- Drug Discovery - MIT - antibiotic activities

<a id="graph-theory-community"></a>
# Graph Theory Community {#graph-theory-community}

In graph theory, a community (also known as a cluster or module) is a group of nodes that are more densely connected to each other than to the rest of the network.

#graph_analysis #clustering 

https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html

### Intuition
Communities often represent:
- Functional units in biological networks (e.g., protein complexes)
- Groups of friends or followers in social networks
- Topical clusters in knowledge graphs or citation networks

They capture meso-scale structure—between the local (node/edge) and global (graph-level) scale.

### Formal Definition
There is no single universal definition, but communities typically exhibit:

- High intra-community density: lots of edges within the group
- Low inter-community density: few edges connecting to other groups

Mathematically, a common goal is to maximize modularity, a measure that quantifies the density of links inside communities compared to links between them.

### Community Detection Algorithms
Some widely used algorithms:

| Algorithm | Description |
|-|-|
| Louvain | Fast and widely used; optimizes modularity |
| Girvan–Newman | Based on removing high-betweenness edges |
| Label Propagation | Propagates labels through the network |
| Leiden | Improved version of Louvain for better quality and performance |


<a id="graph-theory"></a>
# Graph Theory {#graph-theory}


[Graph Theory Community](#graph-theory-community)

[Page Rank](#page-rank)

[PyGraphviz](#pygraphviz)

[networkx](#networkx)

[Plotly](#plotly) for graphs
https://plotly.com/python/network-graphs/

<a id="graphrag"></a>
# Graphrag {#graphrag}


[GraphRAG](#graphrag) is a [RAG](#rag) framework that utilizes [Knowledge Graph](#knowledge-graph)s to enhance information retrieval and processing. A significant aspect of this framework is the use of large language models (LLMs) for [Named Entity Recognition](#named-entity-recognition) (NER) within [Neo4j](#neo4j).

[Graph Neural Network](#graph-neural-network)
### Related Terms
- [How to search within a graph](#how-to-search-within-a-graph)
- **[Text2Cypher](#text2cypher)**: This feature allows users to interact with the graph in a user-friendly manner, converting natural language queries into Cypher queries.
- How to move datasets into a graph database.
- Graphrag patterns.
- The role of [interpretability](#interpretability) in understanding graph-based retrieval.

### Implementation

I discovered an insightful LinkedIn post discussing the potential of knowledge graphs:
This specific graph is called a "Lexical Graph with Extracted Entities".
[LinkedIn Post](https://www.linkedin.com/posts/rani-baghezza-69b154b8_thats-why-im-bullish-on-knowledge-graphs-activity-7287474722039033857-BXyN?utm_source=share&utm_medium=member_desktop)
In [ML_Tools](#ml_tools) see: [Wikipedia_API.py](#wikipedia_apipy)
### Resources

- [GraphRAG Site](https://graphrag.com/concepts/intro-to-graphrag/)
- [Neo4j: Building Better GenAI: Your Intro to RAG & Graphs](https://www.youtube.com/watch?v=OuyTENdRcNs)


<a id="grep"></a>
# Grep {#grep}


![grep.png](../content/images/grep.png)



<a id="gridseachcv"></a>
# Gridseachcv {#gridseachcv}

Used [GridSeachCv](#gridseachcv) to search through the [Hyperparameter](#hyperparameter) space

```python
rf_regressor = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# Model Training with best hyperparameters
rf_regressor = RandomForestRegressor(**best_params, random_state=42)
rf_regressor.fit(X_train, y_train)
```

Given a parameter grid of [Hyperparameter](#hyperparameter), a model, then you model it on the hypers, then gives you the best hypers, that gives the highest cross validation performance.

![Pasted image 20240128194244.png|500](../content/images/Pasted%20image%2020240128194244.png|500)

<a id="groupby-vs-crosstab"></a>
# Groupby Vs Crosstab {#groupby-vs-crosstab}

In pandas, [Groupby](#groupby) and [Crosstab](#crosstab) serve related but distinct purposes for data <mark>aggregation</mark> and summarization.

- groupby is more flexible for aggregation and transformations,
- whereas `crosstab` is specifically designed for creating frequency tables and exploring the relationship between categorical variables.


In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/reshaping.ipynb
### Key Differences

1. **Purpose**:
   - `groupby`: Used for performing aggregate functions (sum, mean, count, etc.) on grouped data.
   - `crosstab`: Used for generating frequency tables or contingency tables.

2. **Output**:
   - `groupby`: Returns a DataFrame with aggregated values.
   - `crosstab`: Returns a DataFrame with counts or specified aggregation functions applied across two or more columns.

3. **Usage**:
   - `groupby`: Can be used with multiple aggregation functions and complex groupings.
   - `crosstab`: Typically used for counting occurrences and exploring the relationship between two categorical variables.





<a id="groupby"></a>
# Groupby {#groupby}


Groupby is a versatile method in pandas used to group data based on one or more columns, and then perform aggregate functions on the grouped data. 

Related:
- [Groupby vs Crosstab](#groupby-vs-crosstab)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Investigating/Transformation/group_by.ipynb
### Implementation
```python
# Sample DataFrame
df = pd.DataFrame({'Category': ['A', 'B', 'A', 'B', 'A'],'Values': [10, 20, 30, 40, 50]})
# Group by 'Category' and calculate the sum of 'Values'
grouped = df.groupby('Category').sum()
print(grouped)
```
Output:
```
          Values
Category        
A              90
B              60
```


![Pasted image 20250323081619.png](../content/images/Pasted%20image%2020250323081619.png)


<a id="grouped-plots"></a>
# Grouped Plots {#grouped-plots}

Related:
- [Data Visualisation](#data-visualisation)
- pairplots

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Load example dataset
tips = sns.load_dataset("tips")

# Facet Grid Example
g = sns.FacetGrid(tips, col="sex", row="time")
g.map_dataframe(sns.histplot, x="total_bill", bins=20)

plt.show()
```

![Pasted image 20250402212849.png](../content/images/Pasted%20image%2020250402212849.png)

<a id="guardrails"></a>
# Guardrails {#guardrails}


Controlling a [Generative AI](#generative-ai) in business through the use of [Guardrails](#guardrails) ensures that the AI remains aligned with specific business goals and avoids unintended or harmful outputs. Guardrails are essential for maintaining security, compliance, and reliability in AI systems. Here's an outline based on your notes:

### 1. Input Guardrails

   - Prompt Injection Control: [Prompting](#prompting) To prevent users from prompting the AI in ways that could result in harmful or inappropriate responses, filtering or validating inputs can be essential. This reduces the risk of the model being "jailbroken" (i.e., forced to generate outputs outside its intended use case).
   - Topic Restriction: Limit the AI’s inputs to specific business-relevant topics. For instance, if the AI is designed for customer support, it should ignore inputs about unrelated topics (e.g., entertainment or politics).
   - User Authentication: Depending on business needs, certain input guardrails can restrict access to specific features or sensitive information based on user credentials or roles.

### 2. Output Guardrails

   - Content Moderation: Post-processing can be applied to outputs to ensure they align with business values, compliance regulations, or safety standards. For example, any harmful or offensive language can be filtered out.
   - Pre-defined Boundaries: Limit the AI’s responses to fall within specific domains. For instance, when the AI is asked questions outside its scope, it can respond with a predefined message, such as "I am not programmed to handle that topic."
   - Compliance and Ethical Constraints: Outputs can be regulated to ensure the model adheres to legal, ethical, and regulatory constraints, which is especially important in industries like finance or healthcare.

### 3. Jailbreaking Concerns

   - Jailbreaking occurs when a user manipulates the system to bypass these guardrails, leading to undesirable outputs. This depends on the business context—some may tolerate more flexible AI behavior, while others, like legal or healthcare firms, need strict controls.

### 4. Business-Specific Use Cases

   - Tailor the AI to address specific business needs. For example, a generative AI for a legal firm should stick to legal advice and documentation, whereas a customer service chatbot should handle predefined topics like returns and product support.
   - [Data Observability|monitoring](#data-observabilitymonitoring) / Monitoring and Logging: Keep track of input and output interactions to ensure that the AI’s performance remains within its intended boundaries.

<a id="gitlab-ciyml"></a>
# Gitlab Ci.Yml {#gitlab-ciyml}

The purpose of a `gitlab-ci.yml` file is to define and configure the **GitLab CI/CD pipeline** for automating tasks such as building, testing, and deploying your code. It is the core configuration file that GitLab uses to orchestrate and execute CI/CD workflows in a repository.

### Key Purposes:

1. **Automation of Workflows:**
    - Automates repetitive tasks like running tests, building applications, linting code, and deploying updates.
      
2. **Pipeline Definition:**
    - Specifies the **stages** (e.g., `build`, `test`, `deploy`) and their sequence.
    - Defines the **jobs** within each stage and their respective commands.
      
3. **Consistency and Reliability:**
    - Ensures consistent execution of tasks across environments, reducing errors caused by manual intervention.
      
4. **Integration with GitLab:**
    - Automatically triggers pipelines in response to events such as code pushes, merge requests, or scheduled runs.
      
5. **Environment Management:**
    - Manages deployments to various environments (e.g., development, staging, production) with variables, conditions, and manual approvals.
      
6. **Feedback and Reporting:**
    - Provides immediate feedback on the status of tasks (e.g., whether tests passed) directly in the GitLab interface.
    - Supports artifact generation and uploads (e.g., logs, reports, or compiled binaries).

### Benefits:

- Improves development velocity by automating workflows.
- Increases code quality through consistent testing and linting.
- Simplifies deployments to various environments.
- Enables team collaboration with clear and visible pipeline progress.

### Example 

```yaml
# Define the stages of the pipeline in the order they will be executed
stages:
  - build    # The stage where the application is built
  - test     # The stage where tests are executed
  - deploy   # The stage where the application is deployed

# Job to build the project
build_job:
  stage: build           # Assign this job to the 'build' stage
  script:                # Commands to execute during this job
    - echo "Building the project" # Example build command (replace with actual build steps)
  artifacts:             # Files or directories to save for use in subsequent jobs
    paths:
      - build/           # Save the 'build' directory as an artifact for later stages

# Job to test the project
test_job:
  stage: test            # Assign this job to the 'test' stage
  script:                # Commands to execute during this job
    - echo "Running tests" # Example test command (replace with actual test steps)

# Job to deploy the project
deploy_job:
  stage: deploy          # Assign this job to the 'deploy' stage
  script:                # Commands to execute during this job
    - echo "Deploying the application" # Example deployment command (replace with actual deployment steps)
  only:                  # Specify when this job should run
    - main               # Only run this job for commits to the 'main' branch

```

<a id="granularity"></a>
# Granularity {#granularity}


Definition of Grain in [Dimensional Modelling](#dimensional-modelling)
   - The grain of a [Fact Table](#fact-table) defines what a single row in the table represents. It is the level of detail captured by the fact table.
   - Declaring the grain is essential because it sets the foundation for the entire dimensional model. It determines how detailed the data will be.

Importance of Grain Declaration:
   - The grain must be established before selecting [Dimensions](#dimensions) and [Facts](#facts) because all dimensions and facts must align with the grain.
   - This alignment ensures consistency across the data model, which is critical for the performance and usability of [business intelligence](#business-intelligence) applications.

Balancing Granularity:
   - In the transformation layer, you need to decide the level of aggregation. For instance, you might aggregate hourly data into daily data to save storage space.
   - Adding dimensions increases the number of rows exponentially, so it's important to carefully choose which dimensions to include.

Semantic Layer:
   - A [semantic layer](#semantic-layer) sits on top of transformed data in a data warehouse, providing flexibility and enabling ad-hoc analysis without needing to store every possible data representation.
   - This is akin to [OLAP](#olap) cubes, where you can perform complex queries (slice-and-dice) on large datasets without pre-storing all combinations.

## Choosing the level of granularity

Granularity, or grain, refers to the <mark>level of detail</mark> represented by a single row in a fact table within a data warehouse. 

The choice of granularity depends on the business requirements and the types of analyses you want to support. Finer granularity (e.g., transaction-level) provides more detailed insights but requires more storage and processing power. Coarser granularity (e.g., monthly product-level) reduces storage needs and can improve query performance but may limit the depth of analysis.

By clearly defining the grain, you ensure that all dimensions and facts in the data model are consistent and aligned with the intended analytical use cases.

### Example: Retail Sales Data

Imagine you are designing a data warehouse for a retail company that tracks sales transactions. You need to decide the granularity of the sales fact table. Here are a few possible options:

1. Transaction-Level Granularity:
   - Grain: Each row represents a single sales transaction.
   - Example: A row might include details such as transaction ID, date and time of sale, store location, product sold, quantity, and total sale amount.
   - Use Case: This level of granularity is useful for detailed analysis, such as examining individual customer purchases or identifying specific transaction patterns.

2. Daily Store-Level Granularity:
   - Grain: Each row represents the total sales for a specific store on a specific day.
   - Example: A row might include the store ID, date, total sales amount, and total number of transactions for that day.
   - Use Case: This granularity is suitable for analyzing daily sales trends across different stores, comparing store performance, or identifying peak sales days.

3. Monthly Product-Level Granularity:
   - Grain: Each row represents the total sales for a specific product across all stores for a specific month.
   - Example: A row might include the product ID, month, total sales amount, and total units sold.
   - Use Case: This level is ideal for tracking product performance over time, identifying best-selling products, or planning inventory and supply chain logistics.

