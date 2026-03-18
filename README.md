# The ID3 Algorithm demo for Mathematics for Computer Science's presentation

The **Iterative Dichotomiser 3 (ID3)** algorithm builds a decision tree from a dataset by employing a top-down, greedy search. Its core mechanism relies entirely on Shannon's Information Theory—specifically the concepts of **Entropy** and **Information Gain**—to select the optimal attribute for partitioning the data at each node.

Below is the detailed mathematical formulation of the algorithm formatted in GitHub Flavored Markdown (GFM).

---

## 1. Mathematical Preliminaries

Let $S$ represent the training dataset consisting of $|S|$ examples.
Let the target classification attribute have $k$ distinct class labels, denoted by the set $C = \{c_1, c_2, \dots, c_k\}$.

### 1.1 Entropy (Measure of Impurity)
Entropy, denoted as $H(S)$, quantifies the uncertainty or impurity of the dataset $S$. 

If $p(c_i)$ is the proportion of examples in $S$ that belong to class $c_i$, it is mathematically defined as:

$$p(c_i) = \frac{|\{x \in S \mid class(x) = c_i\}|}{|S|}$$

The Entropy of the set $S$ is given by:

$$H(S) = -\sum_{i=1}^{k} p(c_i) \log_2(p(c_i))$$

*Note: If $p(c_i) = 0$ for any class, the term $0 \log_2(0)$ is conventionally evaluated as $0$ using limits in information theory.*

### 1.2 Information Gain (Reduction in Entropy)
Information Gain, denoted as $IG(S, A)$, measures the expected reduction in entropy caused by partitioning the examples according to a given attribute $A$.

Let attribute $A$ have a set of mutually exclusive values $V = \{v_1, v_2, \dots, v_m\}$.
Let $S_v$ be the subset of $S$ for which attribute $A$ has the value $v$:

$$S_v = \{x \in S \mid A(x) = v\}$$

The Information Gain of attribute $A$ relative to the collection of examples $S$ is defined as:

$$IG(S, A) = H(S) - \sum_{v \in V} \frac{|S_v|}{|S|} H(S_v)$$

The second term in this equation represents the **conditional entropy** of $S$ given $A$, denoted as $H(S|A)$. Thus, the equation can also be expressed elegantly as:

$$IG(S, A) = H(S) - H(S|A)$$

---

## 2. The ID3 Algorithm Step-by-Step Formalization

Let $Attributes$ be the set of all descriptive features available for splitting. The algorithm $\text{ID3}(S, Attributes)$ proceeds recursively as follows:

### Step 1: Evaluate Base Cases (Stopping Criteria)
The recursion terminates and returns a leaf node if any of the following conditions are met:
1. **Pure Node:** $\exists c_i \in C$ such that $p(c_i) = 1$. Return a leaf node labeled with $c_i$. (Mathematically, $H(S) = 0$).
2. **Empty Attribute Set:** $Attributes = \emptyset$. Return a leaf node labeled with the most frequent class in $S$: $\arg\max_{c_i \in C} p(c_i)$.
3. **Empty Dataset:** $S = \emptyset$. Return a leaf node labeled with the most frequent class in the parent node's dataset.

### Step 2: Select the Optimal Attribute
If the base cases are not met, compute the Information Gain for every attribute $A_j \in Attributes$. Select the attribute $A^*$ that maximizes the Information Gain:

$$A^* = \arg\max_{A_j \in Attributes} IG(S, A_j)$$

Create a root node for the current tree (or subtree) labeled with $A^*$.

### Step 3: Partition the Dataset
For each distinct value $v \in V_{A^*}$ (where $V_{A^*}$ is the set of possible values for $A^*$), partition $S$ into subsets:

$$S_v = \{x \in S \mid A^*(x) = v\}$$

### Step 4: Recursive Call
For each value $v$, recursively call the algorithm to generate a subtree, removing the splitting attribute $A^*$ from the set of available attributes to prevent redundant splitting on the same branch:

$$\text{Subtree}_v = \text{ID3}(S_v, Attributes \setminus \{A^*\})$$

Attach each $\text{Subtree}_v$ as a branch to the root node $A^*$ corresponding to the condition $A^* = v$.
