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

>Note: If $p(c_{i}) = 0$ for any class, the term $0 \log_{2}(0)$ is conventionally evaluated as 0 using limits in information theory.

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
For each distinct value $v \in V_{A^\ast}$ (where $V_{A^\ast}$ is the set of possible values for $A^\ast$), partition $S$ into subsets:

$$S_v = \{x \in S \mid A^*(x) = v\}$$

### Step 4: Recursive Call
For each value $v$, recursively call the algorithm to generate a subtree, removing the splitting attribute $A^*$ from the set of available attributes to prevent redundant splitting on the same branch:

$$\text{Subtree}_v = \text{ID3}(S_v, Attributes \setminus \{A^*\})$$

Attach each $\text{Subtree}_v$ as a branch to the root node _A\*_ corresponding to the condition _A\* = v_.
## 3. The C4.5 Algorithm (Overcoming ID3's Limitations)

While ID3 is highly effective, it possesses a significant inherent bias: it strongly favors attributes with a large number of distinct values (e.g., a "Candidate ID" or "Date" column). Such attributes partition the dataset into many small, perfectly pure subsets, yielding an artificially high Information Gain but resulting in severe overfitting.

The **C4.5 algorithm** (developed by Ross Quinlan as an extension to ID3) addresses this bias by introducing a normalization factor called **Split Information** to compute the **Gain Ratio**.

### 3.1 Split Information (Intrinsic Information)
Split Information, denoted as $SplitInfo(S, A)$, measures the entropy of the dataset $S$ with respect to the values of the attribute $A$ itself, rather than the target classification. It acts as a penalty term for attributes that scatter the data into too many fragmented branches.

Using the established notation, where $V$ is the set of distinct values for attribute $A$, the Split Information is mathematically defined as:

$$SplitInfo(S, A) = -\sum_{v \in V} \frac{|S_v|}{|S|} \log_2 \left( \frac{|S_v|}{|S|} \right)$$

### 3.2 Gain Ratio
The Gain Ratio normalizes the Information Gain by dividing it by the Split Information. This ensures that attributes generating a massive number of splits are proportionately penalized.

$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

> **Mathematical Note:** As the number of partitions increases, the $SplitInfo$ value grows larger, thereby reducing the overall $GainRatio$. In practical implementations, if $SplitInfo(S, A) = 0$ (which occurs if an attribute has the exact same value for all instances in $S$), the attribute is typically discarded to prevent division by zero.

### 3.3 Algorithmic Adjustments for C4.5
To transition from the formalized ID3 algorithm to C4.5, the core recursive structure remains the same, but with crucial enhancements:

1.  **Modified Step 2 (Optimal Selection Criteria):** Instead of maximizing Information Gain, C4.5 evaluates all attributes and selects the attribute $A^*$ that maximizes the Gain Ratio:
    
    $$A^* = \arg\max_{A_j \in Attributes} GainRatio(S, A_j)$$

2.  **Handling Continuous Data:** ID3 is strictly limited to discrete/categorical data. C4.5 introduces the ability to process continuous numerical attributes. For a continuous attribute $A_c$, the algorithm dynamically sorts the numerical values and evaluates potential cut-off thresholds $t$ to partition the dataset into two binary subsets ($A_c \le t$ and $A_c > t$). The threshold $t$ that yields the highest Gain Ratio is dynamically selected during the splitting step.
