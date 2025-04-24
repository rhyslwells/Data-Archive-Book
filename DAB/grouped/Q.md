# Q

## Table of Contents
* [Q-Learning](#q-learning)
* [QUERY GSheets](#query-gsheets)
* [Quartz](#quartz)
* [Query Optimisation](#query-optimisation)
* [Query Plan](#query-plan)
* [Querying](#querying)
* [QuickSort](#quicksort)



<a id="q-learning"></a>
# Q Learning {#q-learning}


Q-learning is a value-based, model-free [Reinforcement learning](#reinforcement-learning) algorithm where the agent learns the optimal [policy](#policy) by updating Q-values based on the rewards received. It is particularly useful in discrete environments like grids.

Uses a Q-Table which is populated by Q-values which are the maximum expected future reward for the given state and action. We improve the Q-Table in an iterative approach

Resources:
- [Q-Learning Explained - Reinforcement Learning Tutorial](https://www.youtube.com/watch?v=kEGAMppyWkQ&list=PLcWfeUsAys2my8yUlOa6jEWB1-QbkNSUl&index=9)

**Q-learning update rule:**

The left hand side gets updated ([Bellman Equations](#bellman-equations))
$$
Q_{new}(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_{t+1} + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

**Explanation:**

- **$Q(s_t, a_t)$**: The Q-value of the current state $s_t$ and action $a_t$.
- **$\alpha$**: The learning rate, determining how much new information overrides old information.
- **$r_{t+1}$**: The reward received after taking action $a_t$ from state $s_t$.
- **$\gamma$**: The discount factor, balancing immediate and future rewards.
- **$\max_{a'} Q(s_{t+1}, a')$**: The maximum Q-value for the next state $s_{t+1}$ across all possible actions $a'$.

![Pasted image 20250220133556.png](../content/images/Pasted%20image%2020250220133556.png)

**Notes**:

- Q-learning is well-suited for environments where the state and action spaces are discrete and manageable in size.
- The algorithm is designed to converge to the optimal policy, even in non-deterministic environments, as long as each state-action pair is sufficiently explored.
- [Exploration vs. Exploitation](#exploration-vs-exploitation)



<a id="query-gsheets"></a>
# Query Gsheets {#query-gsheets}

In [standardised/GSheets](#standardisedgsheets) I want to use query, but I also want to remove certain rows based on a range of keys , can I do this ?
### **1. Use `FILTER` Inside `QUERY` (ArrayFormula Workaround)**

Since `QUERY` does not support dynamic `NOT IN`, you can first filter out the excluded keys using `FILTER`, then pass the result to `QUERY`:

```excel
=QUERY(FILTER(A1:D, ISNA(MATCH(A1:A, X1:X10, 0))), "SELECT Col1, Col2, Col3, Col4", 1)
```

- `FILTER(A1:D, ISNA(MATCH(A1:A, X1:X10, 0)))`: Removes rows where column A matches any value in `X1:X10`.
- `QUERY(..., "SELECT Col1, Col2, Col3, Col4", 1)`: Runs a query on the filtered data.


<a id="quartz"></a>
# Quartz {#quartz}


[Vim](#vim): telescope? Search preview feature?

https://www.youtube.com/watch?v=v5LGaczJaf0

How does quartz work of a software level:
- Transforming text. Think [jinja template](#jinja-template). 
- Manipulating markdown notes
- There is a diagram showing how markdown goes to html.
- [JavaScript](#javascript) for static site generators already existed.

<a id="query-optimisation"></a>
# Query Optimisation {#query-optimisation}


 [Querying](#querying) can be optimised for time, <mark>space efficiency</mark>, and concurrency of queries.
 
Optimizing SQL [Querying](#querying):
- Timing queries
- [Database Index|Indexing](#database-indexindexing)
- Managing [Transaction](#transaction)
- and vacuuming, and handling concurrency with transactions and locks ensures efficient and reliable database performance.

### Timing [Queries](#queries):

- Use `.timer on` to measure query execution time and identify slow queries.

### [Database Index|Index](#database-indexindex) Search

Creating an index on specific columns can speed up searches:
- A covering index includes all the columns required by a query, eliminating the need to access the table data.
- Partial indexes cover a subset of rows, saving space while maintaining query performance for frequently accessed data i.e more likely to search movies that are recent.

Trade-offs when using indexes
- Indexes improve query speed but <mark>consume additional space</mark> and can slow down data insertion and updates.

### To remove redundancy use [Transaction|Transactions](#transactiontransactions)

### Vacuum

SQLite's "VACUUM" command reclaims unused space after data deletion, reducing database size.



[Query Optimisation](#query-optimisation)
   **Tags**: #performance_tuning, #querying

<a id="query-plan"></a>
# Query Plan {#query-plan}

What is expected to happen to the query plan if there is [Database Index|Indexing](#database-indexindexing)?



<a id="querying"></a>
# Querying {#querying}


Querying is the process of asking questions of data. Querying makes use of keys primary and foreign within tables.

Useful Links
- [CS50 SQL Course](https://cs50.harvard.edu/sql/2024/weeks/0/)

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Querying/Querying.ipynb

SQL Commands and Examples
- SELECT
- LIMIT
- ORDER
- WHERE
- NOT
- LIKE
- WITH
- INSERT, UPDATE, or DELETE 

You can have parameterised queries so that you can pass in variables to it:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/SQLite/Querying/Parameterised_Queries.ipynb

Related terms:
- [SQL Joins](#sql-joins)
- [SQL Injection](#sql-injection):=Why we should not use f-strings in queries

<a id="quicksort"></a>
# Quicksort {#quicksort}


## [Recursive Algorithm](#recursive-algorithm)

[Quicksort Algorithm in Five Lines of Code! - Computerphile](https://www.youtube.com/watch?v=OKc2hAmMOY4)

Fast algorithm (compared to say Insertion Sort)

1) Pick pivot value
2) Divide remaining numbers into two parts
3) <mark>sort sublists in some way</mark> <- apply alog again
4) merge

Recursion stops when nothing to pick for pivot value.


```python
def quick_sort(arr, depth=0):
    indent = "  " * depth  # Indentation for better readability in recursion
    print(f"{indent}Sorting: {arr}")
    
    if len(arr) <= 1:
        print(f"{indent}Returning sorted: {arr}")
        return arr  # Base case: already sorted

    pivot = arr[len(arr) // 2]  # Choosing pivot (middle element)
    left = [x for x in arr if x < pivot]  # Elements smaller than pivot
    middle = [x for x in arr if x == pivot]  # Elements equal to pivot
    right = [x for x in arr if x > pivot]  # Elements greater than pivot
    
    print(f"{indent}Pivot: {pivot}")
    print(f"{indent}Left: {left}")
    print(f"{indent}Middle: {middle}")
    print(f"{indent}Right: {right}")
    
    sorted_left = quick_sort(left, depth + 1)
    sorted_right = quick_sort(right, depth + 1)
    
    sorted_array = sorted_left + middle + sorted_right
    print(f"{indent}Merged: {sorted_array}")
    
    return sorted_array  # Recursively sort and merge

# Example usage:
arr = [10, 7, 8, 9, 1, 5]
print("Starting QuickSort...\n")
sorted_arr = quick_sort(arr)
print("\nFinal sorted array:", sorted_arr)
```