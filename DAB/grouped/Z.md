

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
```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)  # Rescales each feature to [0, 1]
```

# Z Score {#z-score}


Z-scores standardize a value relative to a distribution by measuring how many standard deviations it is from the mean. This is useful for [standardised/Outliers|Outliers](#standardisedoutliersoutliers) and [Normalisation](#normalisation).

Definition:  
The Z-score of a value $x$ is given by:
    $$Z = \frac{x - \bar{x}}{s}$$
    
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