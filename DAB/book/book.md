# Preface {#preface}

Remember the Data Archive is non‑linear — use internal links for navigation.
If you need a specific page see the **Table of Contents** to jump to a topic.


# Accuracy {#accuracy}


## Definition

- Accuracy Score is the proportion of correct predictions out of all predictions made. In other words, it is the percentage of correct predictions.
- Accuracy can have issues with [Imbalanced Datasets](#imbalanced-datasets)where there is more of one class than another.

## Formula

- The formula for accuracy is:
  $$\text{Accuracy} = \frac{TN + TP}{\text{Total}}$$
In the context of [Classification](#classification) problems, particularly binary classification, TN and TP are components of the confusion matrix:

- TP (True Positive): The number of instances that are correctly predicted as the positive class. For example, if the model predicts a positive outcome and it is indeed positive, it counts as a true positive.
- TN (True Negative): The number of instances that are correctly predicted as the negative class. For example, if the model predicts a negative outcome and it is indeed negative, it counts as a true negative.

The [Confusion Matrix](#confusion-matrix) also includes:

- FP (False Positive): The number of instances that are incorrectly predicted as the positive class. This is also known as a "Type I error."
- FN (False Negative): The number of instances that are incorrectly predicted as the negative class. This is also known as a "Type II error."

These metrics are used to evaluate the performance of a classification model, providing insights into not just accuracy but also precision, recall, and other performance measures.
## Exploring Accuracy in Python

To explore accuracy in Python, you can use libraries such as `scikit-learn`, which provides the `accuracy_score` function. This function compares the predicted labels with the true labels and calculates the accuracy.

### Example Usage

```python
from sklearn.metrics import accuracy_score
# Assuming pred and y_test are defined
accuracy = accuracy_score(y_test, pred)
print("Prediction accuracy: {:.2f}%".format(accuracy  100.0))
```

- Make sure to replace `pred` and `y_test` with your actual prediction and test data variables.

# Acid Transaction {#acid-transaction}


An ACID [Transaction](#transaction) ensures that either all changes are successfully committed or rolled back, preventing the database from ending up in an inconsistent state. This guarantees the integrity of the data throughout the transaction process.

### Key Properties of ACID Transactions

1. Atomicity: This property ensures that transactions are treated as a single, indivisible unit. If any part of the transaction fails, the entire transaction is rolled back, and none of the changes are applied. Users do not see intermediate states of the transaction.

2. Consistency: Transactions must leave the database in a valid state, adhering to all defined constraints. If a transaction violates a constraint, it is rolled back to maintain the database's stable state.

3. Isolation: This property ensures that concurrent transactions do not interfere with each other. Each transaction operates independently, and the results of one transaction are not visible to others until it is committed.

4. Durability: Once a transaction has been committed, the changes are permanent, even in the event of a system failure. The data remains intact and recoverable.

