# N

## Table of Contents
* [NLP](#nlp)
* [Naive Bayes](#naive-bayes)
* [Named Entity Recognition](#named-entity-recognition)
* [Network Design](#network-design)
* [Neural Network Classification](#neural-network-classification)
* [Neural Scaling Laws](#neural-scaling-laws)
* [Neural network in Practice](#neural-network-in-practice)
* [Neural network](#neural-network)
* [Ngrams](#ngrams)
* [NoSQL](#nosql)
* [Node.JS](#nodejs)
* [Non-parametric tests](#non-parametric-tests)
* [Normalisation of Text](#normalisation-of-text)
* [Normalisation of data](#normalisation-of-data)
* [Normalisation vs Standardisation](#normalisation-vs-standardisation)
* [Normalisation](#normalisation)
* [Normalised Schema](#normalised-schema)
* [NotebookLM](#notebooklm)
* [nbconvert](#nbconvert)
* [neo4j](#neo4j)
* [nltk](#nltk)
* [npy Files A NumPy Array storage](#npy-files-a-numpy-array-storage)



<a id="nlp"></a>
# Nlp {#nlp}


Natural Language Processing (NLP) involves the interaction between computers and humans using natural language. It encompasses various techniques and models to process and analyze large amounts of natural language data.

## Key Concepts

### [Preprocessing](#preprocessing)
- **[Normalisation of Text](#normalisation-of-text)**: The process of converting text into a standard format, which may include lowercasing, removing punctuation, and stemming or [lemmatization](#lemmatization).
- **[Part of speech tagging](#part-of-speech-tagging)**: Assigning a specific part-of-speech category (such as noun, verb, adjective, etc.) to each word in a text.

### Models
- **[Bag of words](#bag-of-words)**: Represents text data by counting the occurrence of each word in a document, ignoring grammar and word order. It takes key terms of a text in normalized **unordered** form.
- **[TF-IDF](#tf-idf)**: Stands for Term Frequency-Inverse Document Frequency. It improves on Bag of Words by considering the importance of a word in a document relative to its frequency across multiple documents.
- **Vectorization**: Converting text into numerical vectors. Techniques like Bag of Words, TF-IDF, or [standardised/Vector Embedding](#standardisedvector-embedding) (e.g., Word2Vec, GloVe) are used to represent text data numerically.

### Analysis
- **[One Hot Encoding](#one-hot-encoding)**: Converts categorical data into a binary vector representation, indicating the presence or absence of a word from a list in the given text.

### Methods
- **[Ngrams](#ngrams)**: Creates tokens from groupings of words, not just single words. Useful for capturing context and meaning in text data.
- **[Grammar method](#grammar-method)**: Involves analyzing the grammatical structure of sentences to extract meaning and relationships between words.

### Actions
- **[Summarisation](#summarisation)**: The process of distilling the most important information from a text to produce a concise version.

## Tools and Libraries

### General Imports

```python
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
```

- **[nltk](#nltk)**: A leading platform for building Python programs to work with human language data. It provides easy-to-use interfaces to over 50 corpora and lexical resources.
  - **punkt**: An unsupervised trainable model for tokenizing text into sentences and words.
  - **stopwords**: Commonly used words (such as "the", "is", "in") that are often removed from text data because they do not carry significant meaning.
  - **wordnet**: A lexical database for the English language that groups words into sets of synonyms and provides short definitions and usage examples.
  - **re**: Regular expressions for pattern matching and text manipulation.


<a id="naive-bayes"></a>
# Naive Bayes {#naive-bayes}



Can values for X,y be categroical ? [Encoding Categorical Variables](#encoding-categorical-variables)

BernoulliNB()

Why Naive Bayes?;;Order doesn't matter, features are independent. Treated it as a [Bag of words](#bag-of-words). Which <mark>simplifies</mark>  the above equation.

Want to use this in classifiers for ML
Want to understand: [Multinomial Naive bayes](#multinomial-naive-bayes) classifer 
There is also: [Gaussian Naive Bayes](#gaussian-naive-bayes)

#### Issues

To avoid having 0 probability sometimes they add <mark>counts</mark> $\alpha$ to do this.

#### Links:

https://youtu.be/PPeaRc-r1OI?t=169

## Formula
$$P(A|B)=P(A) \times \frac{P(B|A)}{P(B)}$$
Think of the line as "given".
## Examples

### Example 1

![Pasted image 20240116184108.png|500](../content/images/Pasted%20image%2020240116184108.png|500)

### [Example](https://www.youtube.com/watch?v=yRl8Yq0M3TY) 2

In the formula above P(A) is P(+),  P(B)=P(NEW)

P(B|A) = P(A=0|+)*... *P(C=0|+)

P(A=0,B=1,C=0) is the same for both + and - class so remove.

![Pasted image 20240118111554.png|500](../content/images/Pasted%20image%2020240118111554.png|500)





### Example Car accidents

What's the probability of car having an accident given that driver is driving in summer, there is no rain, it's a night and it's an urban area?

#### Mock data:

| Season | Weather | Daytime | Area | Did Accident Occur? |
|------|------|------|------|------|
| Summer | No-Raining | Night | Urban | No |
| Summer | No-Raining | Day | Urban | No |
| Summer | Raining | Night | Rural | No |
| Summer | Raining | Night | Urban | Yes |
| Summer | Raining | Day | Urban | No |
| Summer | Raining | Night | Rural | No |
| Winter | Raining | Night | Urban | Yes |
| Winter | Raining | Night | Urban | Yes |
| Winter | Raining | Night | Rural | Yes |
| Winter | No-Raining | Night | Rural | No |
| Winter | No-Raining | Night | Urban | No |
| Winter | No-Raining | Day | Urban | Yes |
| Spring | No-Raining | Night | Rural | Yes |
| Spring | No-Raining | Day | Rural | Yes |
| Spring | Raining | Night | Urban | No |
| Spring | Raining | Day | No | No |
| Spring | No-Raining | Night | Urban | No |
| Autumn | Raining | Night | Urban | Yes |
| Autumn | Raining | Day | Rural | Yes |
| Autumn | No-Raining | Night | Urban | No |
| Autumn | No-Raining | Day | Rural | No |
| Autumn | No-Raining | Day | Urban | No |
| Autumn | Raining | Day | Yes | No |
| Autumn | Raining | Night | Yes | No |
| Autumn | No-Raining | Night | No | No |

To handle data like this it is possible to calculate frequencies for each case:

#### 0. Accident probability
$P(Accident) = \frac{9}{25} = 0.36$

$P(No-Accident) = \frac{16}{25} = 0.64$

#### 1. Season probability

Frequency table:

| Season | Accident | No Accident | |
|------|------|------|------|
| Spring | 2/9 | 3/16 | 5/25 |
| Summer | 1/9 | 5/16 | 6/25 |
| Autumn | 2/9 | 6/16 | 8/25 |
| Winter | 4/9 | 2/16 | 6/25 |
| |9/25|16/25| |

Probabilities based on table:
 
$P(Spring) = \frac{5}{25} = 0.20$

$P(Summer) = \frac{6}{25} = 0.24$

$P(Autumn) = \frac{8}{25} = 0.32$

$P(Winter) = \frac{6}{25} = 0.24$

$P(Spring | Accident) = \frac{2}{9} = 0.22$

$P(Summer | Accident) = \frac{1}{9} = 0.11$

$P(Autumn | Accident) = \frac{2}{9} = 0.22$

$P(Winter | Accident) = \frac{4}{9} = 0.44$

#### 2. Weather probability

Frequency table:

| | Accident | No Accident | |
|------|------|------|------|
| Raining | 6/9 | 7/16 | 13/25 |
| No-Raining | 3/9 | 9/16 | 12/25 |
| | 9/25 | 16/25 | |

Probabilities based on table:

$P(Raining) = \frac{13}{25} = 0.52$

$P(No-Raining) = \frac{12}{25} = 0.48$

$P(Raining|Accident) = \frac{6}{9} = 0.667$

$P(No-Raining|Accident) = \frac{12}{25} = 0.333$


#### 3. Daytime probability

Frequency table:

| | Accident | No Accident | |
|------|------|------|------|
| Day | 3/9 | 6/16 | 9/25 |
| Night | 6/9 | 10/16 | 16/25 |
| | 9/25 | 16/25 | |

Probabilities based on table:

$P(Day) = \frac{9}{25} = 0.36$

$P(Night) = \frac{16}{25} = 0.64$

$P(Day|Accident) = \frac{3}{9} = 0.333$

$P(Night|Accident) = \frac{6}{9} = 0.667$

#### 4. Area probability

Frequency table:

| | Accident | No Accident | |
|------|------|------|------|
| Urban Area | 5/9 | 8/16 | 13/25 |
| Rural Area | 4/9 | 8/16 | 12/25 |
| | 9/25 | 16/25 | |

Probabilities based on table:

$P(Urban) = \frac{13}{25} = 0.52$

$P(Rural) = \frac{12}{25} = 0.48$

$P(Urban|Accident) = \frac{5}{9} = 0.556$

$P(Rural|Accident) = \frac{4}{9} = 0.444$

#### Assemble:

Calculating probablity of car accident occuring in summer, when there is no rain and during night, in urban area.

Where B equals to:
- Season: Summer
- Weather: No-Raining
- Daytime: Night
- Area: Urban

Where A equals to:
- Accident

Using Naive Bayes:

$P(A|B) = P(Accident | Season = Summer, Weather = No-Raining, Daytime = Night, Area = Urban)$

$P(A|B) = \frac{P(Summer|Accident)P(No-Raining|Accident)P(Night|Accident)P(Urban|Accident)P(Accident)}{P(Summer)P(No-Raining)P(Night)P(Urban)}$

$P(A|B) = \frac{\frac{1}{9}\frac{6}{9}\frac{6}{9}\frac{5}{9}\frac{9}{25}}{\frac{6}{25}\frac{12}{25}\frac{16}{25}\frac{13}{25}} = \frac{0.111\cdot0.667\cdot0.667\cdot0.556\cdot0.36}{0.24\cdot0.48\cdot0.64\cdot0.52} = \frac{0.0099}{0.038} = 0.26$

---

$P(A)=P(Accident)$

$P(B)=P(Summer)P(No-Raining)P(Night)P(Urban)$

$P(B|A)=P(Summer|Accident)P(No-Raining|Accident)P(Night|Accident)P(Urban|Accident)$

**What is the Bayes theorem?**;; The formula is P(A|B) = P(B|A) * P(A) / P(B).

**What are the main advantages of Naive Bayes, and when is it commonly used?**;; simplicity, quick implementation, and scalability, used in text classification.
<!--SR:!2024-04-07,3,250-->

**When using Naive Bayes with numerical variables, what condition is assumed on the data?;; Naive Bayes assumes that numerical variables follow a normal distribution.

**How does Naive Bayes perform with categorical variables?** makes no assumptions about the data distribution.

**What is Naive Bayes, and why is it called "naive"?**;; Algo which uses Bayes theorem, used for classification problems. It is "naive" because it assumes that predictor variables are independent, which may not be the case in reality. The algorithm calculates the probability of an item belonging to each possible class and chooses the class with the highest probability as the output.



Naive Bayes
Naive Bayes classifiers are based on Bayes' theorem and assume that the features are conditionally independent given the class label.

[Naive Bayes](#naive-bayes)
   - A probabilistic classifier based on Bayes' theorem.
   - Simple and fast, especially effective for text classification.

<a id="named-entity-recognition"></a>
# Named Entity Recognition {#named-entity-recognition}


Named Entity Recognition (NER) is a subtask of [NLP|Natural Language Processing](#nlpnatural-language-processing) (NLP) that involves identifying and classifying key entities in text into predefined categories such as names, organizations, locations.

The process typically employs algorithms like Conditional Random Fields (CRFs) or deep learning models such as Bi-directional [LSTM](#lstm) (Long Short-Term Memory) networks.

Mathematically, NER can be framed as a sequence labeling problem where the goal is to assign a label $y_i$ to each token $x_i$ in a sentence. The model learns from annotated datasets, optimizing parameters to maximize the likelihood $P(y|x)$ using techniques like [backpropagation](#backpropagation).

NER has significant implications in information extraction, search engines, and automated customer support systems.

### Important
 - NER transforms unstructured text into [structured data](#structured-data) for analysis.
 - The choice of model significantly impacts the accuracy of entity recognition.

### Example
 An example of NER is identifying "Apple Inc." as an organization in the sentence: "Apple Inc. released a new product."

### Follow up questions
 - [How does the choice of training data affect the performance of NER models](#how-does-the-choice-of-training-data-affect-the-performance-of-ner-models)
 - [What are the challenges of NER in multilingual contexts](#what-are-the-challenges-of-ner-in-multilingual-contexts)
 - [Why is named entity recognition (NER) a challenging task](#why-is-named-entity-recognition-ner-a-challenging-task)
 - [In NER how would you handle ambiguous entities](#in-ner-how-would-you-handle-ambiguous-entities)

### Related Topics
 - Text classification in [NLP](#nlp)  
 - Information extraction techniques  

<a id="network-design"></a>
# Network Design {#network-design}



**Mixed-Integer Programming**: Handles problems where some variables must be integers, commonly used in optimizing network design and capacity planning.

How to systems interact.



<a id="neural-network-classification"></a>
# Neural Network Classification {#neural-network-classification}


Choosing Thresholds/Clusters in [Neural network](#neural-network) [Classification](#classification)

When working with [Deep Learning|neural networks](#deep-learningneural-networks), the output is often a probability distribution across different classes. To make a final classification decision, we need to convert these probabilities into discrete class labels. This is typically done by comparing the probabilities against a threshold or by clustering them.

## Threshold-Based Classification

In threshold-based classification, we set a specific probability value as the threshold. If the probability of a class exceeds this threshold, the input is classified as belonging to that class. Otherwise, it's classified as belonging to another class or as "unknown."

[Choosing a Threshold](#choosing-a-threshold)

## Clustering-Based Classification

In [clustering](#clustering)-based classification, we group the probability distributions into clusters. Each cluster represents a class. This approach is useful when the class boundaries are not well-defined or when there are multiple overlapping classes.

[Choosing the Number of Clusters](#choosing-the-number-of-clusters)

## Additional Considerations:

- [Imbalanced Datasets](#imbalanced-datasets): If the classes are imbalanced, the choice of threshold or number of clusters can be significantly affected. Techniques like oversampling, undersampling, or using weighted loss functions can help mitigate the impact of class imbalance.
- [Data Quality](#data-quality): The quality of the training data can also influence the choice of threshold or number of clusters. If the data is noisy or contains outliers, the chosen values may not be optimal.
- [Evaluation Metrics](#evaluation-metrics): Choose evaluation metrics that are appropriate for the specific problem and the desired trade-off between different types of errors. 

<a id="neural-scaling-laws"></a>
# Neural Scaling Laws {#neural-scaling-laws}


Even scaled model cannot cross tthe [compute efficent frontier](#compute-efficent-frontier)

![Pasted image 20241017072233.png|500](../content/images/Pasted%20image%2020241017072233.png|500)

[validation loss](#validation-loss)

Neural scaling laws. That is error rates scale with compute,model size and dataset size, independant of model aritechture. Can we drive to 0?
 
Same laws apear in video and image models.

LLMs are auto progressive models.

Theorical results guiding experiemental - saving compute time.

#### How [LLM|LLMs](#llmllms) work

![Pasted image 20241017072732.png|500](../content/images/Pasted%20image%2020241017072732.png|500)

During training we know next value, hence we have a [Loss function](#loss-function) to help learning.

L1 - loss functions

[Cross Entropy](#cross-entropy)-loss function (uses negative log of probability).  Why is cross enropy used over L1?

unabigious next words. [Entropy of natural language](#entropy-of-natural-language) due to this will LLMs cannot drive [Cross entropy loss](#cross-entropy-loss) to zero.

### Manifolds?

Example [MNIST](#mnist) data set images of number, has high dimesnional dataset space.

![Pasted image 20241017073743.png|500](../content/images/Pasted%20image%2020241017073743.png|500)

[16:03](https://www.youtube.com/watch?t=963&v=5eqRuVp65eY)
Simlar concepts group together.

density of manifold 
average distance between point. or size of neightbourhoods s

![Pasted image 20241017074030.png|500](../content/images/Pasted%20image%2020241017074030.png|500)

$S=L D^{-1/d}$

### Manifold hypothesis (data points in high dim space) and scaling laws 

Knowing the manifold will help scaling. This is called

[Resolution limited scaling](#resolution-limited-scaling)

$$LOSS < D^{-4/d}$$

[Cross entropy loss](#cross-entropy-loss) should scale wrt manifold.

[19:24](https://www.youtube.com/watch?t=1164&v=5eqRuVp65eY)

[intrinsic dimension of natural language](#intrinsic-dimension-of-natural-language)










<a id="neural-network-in-practice"></a>
# Neural Network In Practice {#neural-network-in-practice}


This guide provides practical insights into building and using [Neural network](#neural-network).

Refer to [ML_Tools](#ml_tools) for more details: `Neural_Net_Build.py`

## Softmax Placement at the End

Numerical stability is crucial. One way to enhance stability is by grouping the softmax function with the loss function rather than placing it at the output layer.

### Building the Model

**Final Dense Layer**: 
  - Use a 'linear' activation function, which means no activation is applied. This setup allows the model to output raw logits.
  
**Model Compilation**: 
  - When compiling the model, specify `from_logits=True` in the loss function to indicate that the outputs are raw logits.
  ```python
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  ```
**Target Form**: 
  - The target format remains unchanged. For instance, with SparseCategoricalCrossentropy, the target is the expected class label (e.g., digits 0-9 in the MNIST dataset).
  - See [SparseCategorialCrossentropy or CategoricalCrossEntropy](#sparsecategorialcrossentropy-or-categoricalcrossentropy)

**Output Probabilities**: 
  - Since the model outputs logits, apply a softmax function to convert these logits into probabilities if needed for interpretation or further processing.

## TensorFlow History Loss (Cost)

Monitoring the cost, or loss, during training is essential for understanding how well the model is learning.

**Monitoring Progress**: 
  - Track the progress of gradient descent by observing the cost, referred to as `loss` in TensorFlow. Ideally, the loss should decrease as training progresses.
**Loss Display**: 
  - The loss is displayed at each epoch during the execution of `model.fit`, providing real-time feedback on training performance.
**History Variable**: 
  - The `.fit` method returns a `history` object containing various metrics, including the loss. This object is stored in the `history` variable for further analysis.

The `history` object can capture additional metrics such as accuracy, validation loss, and other performance indicators, depending on what was specified during model compilation and fitting. This information is valuable for evaluating the model's performance [Model Evaluation](#model-evaluation).



<a id="neural-network"></a>
# Neural Network {#neural-network}


A [Neural network|Neural Network](#neural-networkneural-network) is a computational model inspired by biological neural networks in the human brain. It consists of layers of interconnected nodes (neurons) that process and transmit information. Neural networks are fundamental to [Deep Learning](#deep-learning).

Resources:  
- [Keras Guide](https://keras.io/guides/sequential_model/)

Also see:  
- [Types of Neural Networks](#types-of-neural-networks)  
- [Neural network in Practice](#neural-network-in-practice)
### Key Components

The number of starting nodes depends on the input parameter, similar for output. The width and depth of the net are called [Hyperparameter](#hyperparameter).

Neurons (Nodes):  
- The basic units of a neural network. Each neuron receives input, processes it, and passes it to the next layer. A neuron’s output is typically computed using a mathematical function known as the activation function.

Layers:  
- Neural networks are structured in layers of neurons:
  - Input Layer: Receives the raw input data (features) that are fed into the network.
  - Hidden Layers: Process the inputs received from the previous layer. There can be multiple hidden layers, making a neural network "deep." These layers transform the data to learn complex relationships and patterns.
  - Output Layer: Produces the final prediction or result, such as a class label in classification tasks or a continuous value in regression.

Weights and Biases:  [Model Parameters](#model-parameters)
- There are weights and biases at each layer. The shape of the weights is determined by the number of units in the previous layer and the number of units in the current layer.
- Each connection between neurons has a weight that determines how much influence one neuron has on another. Weights are adjusted during the learning process to minimize prediction errors.
- Biases allow the network to shift the output of the [activation function](#activation-function) and help it better fit the data.

Training:  See [Fitting weights and biases of a neural network](#fitting-weights-and-biases-of-a-neural-network)

Optimization:  See [Optimisation techniques](#optimisation-techniques)
- The optimization process (often gradient descent) updates the network's weights to minimize the loss function, ensuring the model improves with training and generalizes well to new, unseen data.

Inputs:
- We need [Normalisation](#normalisation) of values (inputs) here for feeding the network to have balanced weights at the nodes.

### Context

Example of Neural Network:  
A neural network can be used for a task like image classification. For instance:
- The input layer receives the pixel values of the image.
- Hidden layers transform these pixel values through a series of mathematical operations, learning important features such as edges, shapes, and textures.
- The output layer classifies the image into one of the predefined categories (e.g., "cat" or "dog").

Pros:  
- Flexibility: Can model complex, non-linear relationships.
- Adaptability: Can be applied to a wide range of problems like image recognition, speech processing, and game playing.
- Automatic Feature Extraction: Neural networks, especially CNNs, can automatically learn important features from raw data without manual intervention.

Cons:  
- Data-hungry: Neural networks typically require large datasets to perform well.
- Computationally Intensive: Training deep networks can require substantial computational resources.
- Black Box Nature: The internal decision-making process is often difficult to interpret, although research into interpretability is addressing this.



<a id="ngrams"></a>
# Ngrams {#ngrams}


N-grams are used in NLP that allow for the analysis of text data by breaking it down into smaller, manageable sequences. 

An **N-gram** is a contiguous sequence of *n* items (or tokens) from a given sample of text or speech. In the context of natural language processing ([NLP](#nlp)) and text analysis, these items are typically words or characters. 

N-grams are used to analyze and <mark>model the structure of language</mark>, and they can help in various tasks such as [text classification](#text-classification).
### Types of N-grams
- **Unigram**: An N-gram where *n = 1*. It represents individual words or tokens. For example, in the sentence "I love AI", the unigrams are ["I", "love", "AI"].

- **Bigram**: An N-gram where *n = 2*. It represents pairs of consecutive words. For the same sentence, the bigrams would be ["I love", "love AI"].

- **Higher-order N-grams**: These can go beyond three words, such as 4-grams (quadgrams) or 5-grams, and so on.
### Code implementations:

This can be does through kwargs in CountVectorizer.

<a id="nosql"></a>
# Nosql {#nosql}


(Not Only SQL):** <mark>Non-relational</mark> database management systems offering flexibility and scalability for unstructured or document-based data.

 **NoSQL Databases**: Accommodate unstructured data and can be represented through graph theory or document-based structures, allowing for flexible data models.

<a id="nodejs"></a>
# Node.Js {#nodejs}



<a id="non-parametric-tests"></a>
# Non Parametric Tests {#non-parametric-tests}



<a id="normalisation-of-text"></a>
# Normalisation Of Text {#normalisation-of-text}


[Preprocessing](#preprocessing) in NLP tasks is called Normalization involves reducing words to their base or root form, converting them to lowercase, and removing stop words.
## Processes

What are some steps involved in the pre-processing of a text?.  These include  making the text lower case, removal of punctuation, tokenize the text (split up the words in a sentence), remove stop words as they convey grammar rather than meaning, word stemming (reduce words to their stems).

[Tokenisation](#tokenisation): Used to separate words or sentences.

[Stemming](#stemming): returns part of a words that doesnt change ie breaks, breakthrough gives break. Use 

```python
from nltk.stem.porter import PorterStemmer
temp=text #decomposed
porter_stemmer = PorterStemmer()
stemmed_tokens = [porter_stemmer.stem(token) for token in temp]

print(stemmed_tokens)
```

[lemmatization](#lemmatization):  reducing word to it's base form e.g. words "is", "was", "were" will turn into "be".

```python
from nltk.stem.wordnet import WordNetLemmatizer
temp=text #decomposed
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [wordnet_lemmatizer.lemmatize(token, pos="v") for token in temp]
print(lemmatized_tokens)
```
## Code

main normaliser
```python
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords

def normalize_document(document, stemmer=porter_stemmer, lemmatizer=wodnet_lemmatizer):
    """Noramlizes data by performing following steps:
        1. Changing each word in corpus to lowercase.
        2. Removing special characters and interpunction.
        3. Dividing text into tokens.
        4. Removing english stopwords.
        5. Stemming words.
        6. Lemmatizing words.
    """
    
    temp = document.lower()
    temp = re.sub(r"[^a-zA-Z0-9]", " ", temp)
    temp = word_tokenize(temp)
    temp = [t for t in temp if t not in stopwords.words("english")]
    temp = [porter_stemmer.stem(token) for token in temp]
    temp = [lemmatizer.lemmatize(token) for token in temp]
    return temp
```



<a id="normalisation-of-data"></a>
# Normalisation Of Data {#normalisation-of-data}

Normalization is the process of structuring data from the source into a format appropriate for consumption in the destination. 

For example, when writing data from a nested, dynamically typed source like a [JSON](#json) [API](#api) to a relational destination like [PostgreSQL](#postgresql), normalization is the process that un-nests JSON from the source into a relational table format that uses the appropriate column types in the destination.



<a id="normalisation-vs-standardisation"></a>
# Normalisation Vs Standardisation {#normalisation-vs-standardisation}

Key Differences:

[Normalisation](#normalisation) changes the range of the data, while standardisation changes the data distribution.

Normalisation is preferred when the data does not follow a Gaussian distribution, whereas [standardisation](#standardisation) is used when the data is normally distributed.

![Pasted image 20241219071120.png](../content/images/Pasted%20image%2020241219071120.png)





<a id="normalisation"></a>
# Normalisation {#normalisation}


Standardizing data distributions for consistency. 

In ML:
- [Z-Normalisation](#z-normalisation)
- [Standardisation](#standardisation)
- [Normalisation vs Standardisation](#normalisation-vs-standardisation)
- [Batch Normalisation](#batch-normalisation)

In [Data Engineering](#data-engineering):
- [Normalisation of data](#normalisation-of-data)
- [Normalised Schema](#normalised-schema)
- [How to normalise a merged table](#how-to-normalise-a-merged-table)

In [NLP](#nlp):
- [Normalisation of Text](#normalisation-of-text)


```python
# --- 15. GroupBy with Transformation (Using transform to align with original dataframe)
df['Value_transformed'] = df.groupby('Category')['Value'].transform(lambda x: x - x.mean())
# get the mean value for each category
print(df.groupby('Category')['Value'].mean())
print("\nTransformed Values with mean subtracted (transform()):")
print(df.sort_values('Category'))
```



<a id="normalised-schema"></a>
# Normalised Schema {#normalised-schema}


In a normalized schema, data is organized into multiple related tables to minimize redundancy and dependency, and improve data integrity.

This approach is often used in transactional databases (OLTP) to ensure data integrity. However, it can lead to complex queries and slower performance for analytical queries.

Normalization involves organizing the columns (attributes) and tables (relations) in a database to ensure proper enforcement of dependencies through database integrity constraints. 

This is achieved by applying formal rules during the creation of a new database design or <mark>decomposition</mark> (improvement of an existing database design) process.

1.  **First Normal Form (1NF)**:
    - Eliminate duplicate data by ensuring each attribute contains only atomic values and each table has a unique primary key.
2.  **Second Normal Form (2NF)**:
    - Meet all requirements of 1NF and <mark>remove partial dependencies</mark> by ensuring that <mark>every non-prime attribute (attribute not part of any candidate key) entirely depends on the primary key.</mark>
3.  **Third Normal Form (3NF)**:
    - Meet all requirements of 2NF and remove <mark>transitive dependencies</mark> by ensuring that no non-prime attribute is transitively dependent on the primary key.

[See here for an example](https://youtu.be/rcrsqyFtJ_4?t=885)

[How to normalise a merged table](#how-to-normalise-a-merged-table)
## Denormalization

**Denormalization**, on the other hand, is the process of intentionally introducing redundancy into a database design by combining tables or adding redundant data, aiming to improve query performance or simplify the database structure. Denormalization is the **opposite of normalization**. Please consider the trade-offs between data integrity and query performance. This technique is used with [Dimensional Modeling](Dimensional%20Modelling.md) in [OLAP](standardised/OLAP%20(online%20analytical%20processing).md) cubes, for example.

# Related to:

[Normalisation of data](#normalisation-of-data)


<a id="notebooklm"></a>
# Notebooklm {#notebooklm}

https://www.youtube.com/watch?v=EOmgC3-hznM

key topics 

chat interface takes into account resources.

save to note- to dave. 

how to select and folders - from obsidian [Data Archive](#data-archive) for this ? A getter of some kind

can convert muiltple notes into a single note.

Can add website as source.

project context - similar projects notes

Focus knowledge retrieval
- get info from sources (folders)

FAQ 

Note: #portal  can help with file extraction rem utils function (for [NotebookLM](#notebooklm))

<a id="nbconvert"></a>
# Nbconvert {#nbconvert}



<a id="neo4j"></a>
# Neo4J {#neo4j}



<a id="nltk"></a>
# Nltk {#nltk}



<a id="npy-files-a-numpy-array-storage"></a>
# Npy Files A Numpy Array Storage {#npy-files-a-numpy-array-storage}


A .npy file is a binary file format specifically designed to store a single NumPy array. NumPy, or Numerical Python, is a powerful library in Python used for numerical computing and data analysis.

### Why Use .npy Files?

* Efficiency: Storing data in binary format is generally more efficient than storing it in text-based formats like CSV or JSON. This means faster read/write operations and less disk space usage.
* Preserves Data Structure: .npy files maintain the exact structure and data type of the NumPy array, ensuring that the data can be loaded back into memory without any loss of information.
* Simple Format: The .npy format is relatively straightforward, making it easy to read and write using NumPy's built-in functions.

### How to Create and Load .npy Files

Creating an .npy File:

```python
#1. Import NumPy:
   import numpy as np

#2. Create a NumPy Array:
   my_array = np.array([[1, 2, 3], [4, 5, 6]])

#3. Save the Array to a .npy File:
   np.save('my_array.npy', my_array)

#Loading a .npy File:
#1. Load the Array from the .npy File:

   loaded_array = np.load('my_azray.npy')
```
