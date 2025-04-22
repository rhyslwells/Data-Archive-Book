

# Java Vs Javascript {#java-vs-javascript}


### Difference Between [Java](#java) and [JavaScript](#javascript)

Although their names are similar, **Java** and **JavaScript** are fundamentally different languages designed for different purposes. Below is a comparison between the two:

| Aspect               | Java                                           | JavaScript                                       |
|----------------------|------------------------------------------------|--------------------------------------------------|
| **Type**             | Object-Oriented Programming Language           | Scripting Language                               |
| **Use**              | General-purpose, used for desktop, mobile, and enterprise applications | Primarily used for web development (front-end and back-end) |
| **Execution**        | Runs on the Java Virtual Machine (JVM)         | Runs in the browser or on server-side (Node.js)  |
| **Compiled or Interpreted** | Compiled to bytecode, then executed by the JVM | Interpreted directly by the browser or Node.js    |
| **Syntax**           | Strongly typed; requires defining data types   | Loosely typed; variables can change types        |
| **Concurrency**      | Supports multithreading                        | Single-threaded, but supports asynchronous programming (e.g., with callbacks, promises) |
| **Platform Dependency** | Platform-independent (write once, run anywhere) | Platform-independent, mainly within the context of the web |
| **Main Use Case**    | Enterprise applications, Android development, large systems | Dynamic web pages, front-end and server-side scripting for web applications |
| **Libraries/Frameworks** | Spring, Hibernate, JavaFX, Android SDK      | React, Angular, Vue.js (front-end), Node.js (back-end) |
| **Syntax Example**   | `System.out.println("Hello, World!");`         | `console.log("Hello, World!");`                  |

### Key Points:
- **[Java](#java)** is used for building **large-scale applications**, including desktop apps and Android apps. It is strongly typed, compiled, and can handle multithreading.
- **[JavaScript](#javascript)** is mainly used for **web development**, both for the front-end (managing user interfaces) and back-end (using Node.js), and is more flexible with dynamic typing and asynchronous behavior.

# Javascript {#javascript}



## Johnson–Lindenstrauss lemma

#math 

https://youtu.be/9-Jl0dxWQs8?list=PLZx_FHIHR8AwKD9csfl6Sl_pgCXX19eer&t=1125

THe number of vectors that can be fit into a spaces grows exponentially.

Useful for [LLM](#llm) in storing ideas. 

Plotting M>N almost orthogonal vectors in N-dim space

Optimisation process that nudges then towards being perpendicular between 89-91 degrees

```python
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# List of vectors in some dimension, with many
# more vectors than there are dimensions
num_vectors = 10000
vector_len = 100
big_matrix = torch.randn(num_vectors, vector_len)
big_matrix /= big_matrix.norm(p=2, dim=1, keepdim=True)
big_matrix.requires_grad_(True)

# Set up an optimization loop to create nearly-perpendicular vectors
optimizer = torch.optim.Adam([big_matrix], lr=0.01)
num_steps = 250

losses = []

dot_diff_cutoff = 0.01
big_id = torch.eye(num_vectors, num_vectors)

for step_num in tqdm(range(num_steps)):
    optimizer.zero_grad()

    dot_products = big_matrix @ big_matrix.T
    # Punish deviation from orthogonality
    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()

    # Extra incentive to keep rows normalized
    loss += num_vectors * diff.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Plot loss curve
plt.plot(losses)
plt.grid(True)
plt.show()

# Compute angle distribution
dot_products = big_matrix @ big_matrix.T
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
angles_degrees = torch.rad2deg(torch.acos(normed_dot_products.detach()))

# Use this to ignore self-orthogonality
self_orthogonality_mask = ~(torch.eye(num_vectors, num_vectors).bool())
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000, range=(0, 180))
plt.grid(True)
plt.show()

```

### Joining Datasets

In [DE_Tools](#de_tools) see:
- https://github.com/rhyslwells/DE_Tools/blob/main/Explorations/Transformation/Joining.ipynb

```python
# Merge
df1 = pd.DataFrame({'key': ['A', 'B'], 'value': [1, 2]})
df2 = pd.DataFrame({'key': ['A', 'B'], 'value': [3, 4]})
merged_df = pd.merge(df1, df2, on='key')

# Concat
concat_df = pd.concat([df1, df2])

# Join
df1.set_index('key', inplace=True)
df2.set_index('key', inplace=True)
joined_df = df1.join(df2, lsuffix='_left', rsuffix='_right')
```

Merging datasets for completeness (also see [SQL Joins](#sql-joins)). 


# Json To Yaml {#json-to-yaml}


[Json](#json)

[yaml](#yaml)

```JSON
{
  "json": [
    "rigid",
    "better for data interchange"
  ],
  "yaml": [
    "slim and flexible",
    "better for configuration"
  ],
  "object": {
    "key": "value",
    "array": [
      {
        "null_value": null
      },
      {
        "boolean": true
      },
      {
        "integer": 1
      },
      {
        "alias": "aliases are like variables"
      },
      {
        "alias": "aliases are like variables"
      }
    ]
  },
  "paragraph": "Blank lines denote\nparagraph breaks\n",
  "content": "Or we\ncan auto\nconvert line breaks\nto save space",
  "alias": {
    "bar": "baz"
  },
  "alias_reuse": {
    "bar": "baz"
  }
}
```

```YAML
---
# <- yaml supports comments, json does not
# did you know you can embed json in yaml?
# try uncommenting the next line
# { foo: 'bar' }

json:
  - rigid
  - better for data interchange
yaml:
  - slim and flexible
  - better for configuration
object:
  key: value
  array:
    - null_value: null
    - boolean: true
    - integer: 1
    - alias: aliases are like variables
    - alias: aliases are like variables
paragraph: |
  Blank lines denote
  paragraph breaks
content: |-
  Or we
  can auto
  convert line breaks
  to save space
alias:
  bar: baz
alias_reuse:
  bar: baz 
```

# Json {#json}

Stands for [javascript object notation](https://www.json.org/json-en.html)

- records separated by commas
- keys & strings wrapped by double quotes
- good choice for data transport


JSON data embedded inside of a string, is an example of semi-structured data. The string contains all the information required to understand the structure of the data, but is still for the moment just a string -- it hasn't been structured yet. The Raw JSON stored by Airbyte during ELT is an example of semi-structured data. This looks as follows:  

|               |  **\_airbyte_data**|
|---------| -----------|
|Record 1| \"{'id': 1, 'name': 'Mary X'}\" |
|Record 2| \"{'id': 2, 'name': 'John D'}\"|

# Junction Tables {#junction-tables}



# Jinja Template {#jinja-template}


### Resources

[LINK](https://www.youtube.com/watch?v=OraYXEr0Irg)

### Practical

jinja2 works with python 3.

![Pasted image 20240922201606.png](./images/Pasted%20image%2020240922201606.png)

Renders templates with variable substitutions 

You can use tags too.

![Pasted image 20240922202345.png](./images/Pasted%20image%2020240922202345.png)

Get gpt to generate example if necessary.

can get a csv to export the data.

context dictionaries are used can do html and flask.

jinja used to manage web pages

[Flask](#flask)

makes me think about how [Quartz](#quartz) is constructed.

### About

Jinja is a fast, expressive, extensible templating engine. Special placeholders in the template allow writing code similar to [Python](term/python.md) syntax. Then the template is passed data to render the final document.

Most popularized by [dbt](#dbt).  Read more on the [Jinja Documentation](https://jinja.palletsprojects.com/).

integrates with [Flask](#flask).
