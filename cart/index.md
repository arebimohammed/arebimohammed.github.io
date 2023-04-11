# Decision Trees: The unfounded strength of recursive decision rules

<!--more-->
<span style="font-size:1.125rem">

## Introduction

Decision Trees are popular supervised machine learning algorithms. They're popular due to their simplicity of interpretation and wide range of applications. They are applicable to both regression and classification problems (with differences on how they are trained, more on this later). 

A Decision Tree is made up of a series of sequential decisions, or decision nodes, on the features of a data set. Conditional control statements, or if-then rules, are used to navigate the resulting flow-like structure, which divides each decision node into two or more subnodes. The model's prediction outputs are represented by leaf nodes, also known as terminal nodes.

Training a Decision Tree from data entails determining the order in which decisions should be assembled from the root to the leaves. New data can then be passed down from the top until it reaches a leaf node, which represents a prediction for that data point. The predictor space is stratified or segmented into a number of simple regions by decision trees.

We typically use the mean or mode target (response) value for the training observations in the region to which it belongs to make a prediction for a given observation. These approaches are known as decision tree methods because the set of splitting rules used to segment the predictor space can be summarized in a tree.

## Building a Decision Tree 

Let's make a decision tree.
Assume we're farmers with a new plot of land. We must determine whether a tree is an Apple, Cherry, or Oak tree based solely on its diameter and height. We'll use a Decision Tree to accomplish this.

```python
import json
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.style.use("seaborn")

with open("./data/data.json", "r") as f:
    data = f.read()

df = pd.DataFrame(
    eval(data.split("[")[1].split("]")[0].replace("\n", "").replace(" ", ""))
)
df.Family = df.Family.astype("category")

scatter = plt.scatter(
    x=df.Diameter,
    y=df.Height,
    c=df.Family.cat.codes,
    edgecolor="white",
    linewidth=1.5,
    cmap=matplotlib.cm.gnuplot,
)
plt.xlabel("Diameter")
plt.ylabel("Height")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=df.Family.cat.categories.values.tolist(),
    bbox_to_anchor=(1, 1.2),
    ncol=3,
)
```
<img src = "/posts/Decision Trees (CART)/farm.png" height= "100%" width="100%">


Now we can begin splitting!

Almost all trees with a diameter ≥ 0.45 are oak trees! As a result, we can probably assume that any other trees we find in that area are also oak trees.

Our root node will be this first decision node. We'll draw a vertical line at this Diameter and label everything above it as Oak (our first leaf node), then continue partitioning our data on the left.

```python
scatter = plt.scatter(
    x=df.Diameter,
    y=df.Height,
    c=df.Family.cat.codes,
    edgecolor="white",
    linewidth=1.5,
    cmap=matplotlib.cm.gnuplot,
)
plt.xlabel("Diameter")
plt.ylabel("Height")
plt.axvspan(0.45, 1.1, alpha=0.3, facecolor="y", lw=1, edgecolor="black")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=df.Family.cat.categories.values.tolist(),
    bbox_to_anchor=(1, 1.2),
    ncol=3,
)
```
<img src = "/posts/Decision Trees (CART)/farm2.png" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree.PNG" height= "100%" width="100%">

We then can split some more

We continue hoping to divide our plot of land in the most advantageous way possible. We notice that adding a new decision node at Height ≤ 4.88 results in a nice section of Cherry trees, so we partition our data there.

Our Decision Tree is updated as a result, with a new leaf node for Cherry.

```python
scatter = plt.scatter(
    x=df.Diameter,
    y=df.Height,
    c=df.Family.cat.codes,
    edgecolor="white",
    linewidth=1.5,
    cmap=matplotlib.cm.gnuplot,
)
plt.xlabel("Diameter")
plt.ylabel("Height")
plt.axvspan(0.45, 1.1, alpha=0.3, facecolor="y", lw=1, edgecolor="black")
plt.axhspan(0, 4.88, 0, 0.39, alpha=0.3, facecolor="#ae36ff", edgecolor="black")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=df.Family.cat.categories.values.tolist(),
    bbox_to_anchor=(1, 1.2),
    ncol=3,
)
```

<img src = "/posts/Decision Trees (CART)/farm3.png" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree2.PNG" height= "100%" width="100%">

More Splitting!

Following this second split, we are left with an area densely forested with Apple and Cherry trees. No problem: a vertical division can be drawn to better separate the Apple trees.

Once again, our Decision Tree is updated.

```python
scatter = plt.scatter(
    x=df.Diameter,
    y=df.Height,
    c=df.Family.cat.codes,
    edgecolor="white",
    linewidth=1.5,
    cmap=matplotlib.cm.gnuplot,
)
plt.xlabel("Diameter")
plt.ylabel("Height")
plt.axvspan(0.45, 1.1, alpha=0.3, facecolor="y", lw=1, edgecolor="black")
plt.axhspan(0, 4.88, 0, 0.39, alpha=0.3, facecolor="#ae36ff", edgecolor="black")
plt.axvspan(
    0.25, 0.45, ymin=0.36, alpha=0.3, facecolor="#5eab86", lw=1, edgecolor="black"
)
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=df.Family.cat.categories.values.tolist(),
    bbox_to_anchor=(1, 1.2),
    ncol=3,
)
```

<img src = "/posts/Decision Trees (CART)/farm4.png" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree3.PNG" height= "100%" width="100%">

And still some more
The remaining region simply requires another horizontal division, and our job is done! We've found the best set of nested decisions.

Nonetheless, some regions still contain a few misclassified points. Should we keep splitting and dividing into smaller sections?

Hmm... :thinking:

```python
scatter = plt.scatter(
    x=df.Diameter,
    y=df.Height,
    c=df.Family.cat.codes,
    edgecolor="white",
    linewidth=1.5,
    cmap=matplotlib.cm.gnuplot,
)
plt.xlabel("Diameter")
plt.ylabel("Height")
plt.axvspan(0.45, 1.1, alpha=0.3, facecolor="y", lw=1, edgecolor="black")
plt.axhspan(0, 4.88, 0, 0.42, alpha=0.3, facecolor="#ae36ff", edgecolor="black")
plt.axvspan(
    0.25, 0.45, ymin=0.36, alpha=0.3, facecolor="#5eab86", lw=1, edgecolor="black"
)
plt.axvspan(0, 0.25, ymin=0.5, alpha=0.3, facecolor="#5eab86", lw=1, edgecolor="black")
plt.axhspan(4.88, 7.14, 0, 0.252, alpha=0.3, facecolor="#ae36ff", edgecolor="black")
plt.legend(
    handles=scatter.legend_elements()[0],
    labels=df.Family.cat.categories.values.tolist(),
    bbox_to_anchor=(1, 1.2),
    ncol=3,
)
```
<img src = "/posts/Decision Trees (CART)/farm5.png" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree4.PNG" height= "100%" width="100%">

We shouldn't go too deep!

If we do, the resulting regions will become increasingly complex, and our tree will grow unreasonably large. A Decision Tree of this type would learn far too much from the noise of the training examples and far too few generalizable rules.

Is this something you've heard before? Well it is the well-known tradeoff called The Bias Variance Tradeoff! Going too deep in this case results in a tree that overfits our data, so we'll stop here.

We've finished! Simply pass the Height and Diameter values of any new data point through the newly created Decision Tree to classify them as an Apple, Cherry, or Oak tree!


## Where Should We Divide?

We just saw how a Decision Tree works at a high level: it generates a series of sequential rules from the top down that divide the data into well-separated regions for classification. But, given the large number of possible partitions, how does the algorithm decide where to partition the data? Before we can understand how that works, we must first understand Entropy.

Entropy quantifies the amount of information contained in a variable or event. It will be used to identify regions that contain a large number of similar (pure) or dissimilar (impure) elements.

Given a set of events with probabilities $(p_1, p_2, {\dots}, p_n)$, the total entropy H can be expressed as the negative sum of weighted probabilities:

$$\\displaystyle  H = - \\sum\\limits_{i=1}^{n} p_i \\log_2(p_i)$$

The quantity has several intriguing properties:

{{< admonition type=info  title="Entropy Properties" open=true >}}
1. $H=0$ Only if all but one $p_i$ is zero, with this one having a value of 1. Thus, entropy disappears only when there is no uncertainty in the outcome, implying that the sample is completely predictable.

2. $H$ is greatest when all $p_i$ are equal. This is the most ambiguous, or 'impure' situation.

3. Any change towards the equalisation of probabilities $(p_1, p_2, {\dots}, p_n)$ raises $H$.
{{< /admonition >}}

The entropy of a collection of labelled data points can be used to quantify its impurity: a node with multiple classes is impure, whereas a node with only one class is pure.

```python
import numpy as np
from math import log, e


def entropy_np(labels, base=None):
    """Computes entropy of label distribution."""
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.0

    base = e if base is None else base

    for i in probs:
        ent -= i * log(i, base)

    return ent

pst_labels = list("bbbbbbbbbbbbbbbbbbbb")
ents = [0]
proportions = [0]
for i in range(len(pst_labels)):
    temp = pst_labels[: i + 1]
    pst_labels[: i + 1] = list("a" * len(temp))
    ent = entropy_np(pst_labels, base=2)
    value, counts = np.unique(pst_labels, return_counts=True)
    probs = counts / len(pst_labels)
    ents.append(ent)
    proportions.append(probs[0])

plt.scatter(proportions, ents, c="g", edgecolor="white", lw=2)
plt.plot(proportions, ents)
plt.xlabel("Proportion of Positive Class")
plt.ylabel("Entropy")
_ = plt.title("Distribution of Entropy against various class proportions")
```
<br>
<img src = "/posts/Decision Trees (CART)/entropy.png" height= "100%" width="100%">

Above we can see how the entropy of a set of labelled data points divided into two classes (these are simply implemented as A or B's in Python lists with various distribution), as is common in binary classification problems, changes with various class distributions.

Have you noticed that pure samples have zero entropy while impure samples have higher entropy values? This is what entropy does for us: it measures how pure (or impure) a collection of samples is. By defining the Information Gain, we'll use it in the algorithm to train Decision Trees.

## Information Gain

We can now describe the logic for training Decision Trees using the intuition gained from the above graph. As the name implies, information gain quantifies the amount of information we obtain. It accomplishes this through the use of entropy. The idea is to subtract from our data's entropy before splitting, the entropy of each possible partition. The split that results in the greatest reduction in entropy, or equivalently, the greatest increase in information, is then chosen.

[ID3](https://link.springer.com/article/10.1007/BF00116251) is the name of the core algorithm used to calculate information gain (although not the only one). It is a recursive procedure that begins at the root node of the tree and iterates greedily top-down on all non-leaf branches, calculating the difference in entropy at each depth:

$$\\displaystyle \\Delta IG = H_{\\text{parent}}  - \\frac{1}{N}\\sum\\limits_{\\text{children}} N_{\\text{child}} \\cdot H_{\\text{child}}$$

To be more specific, the steps of the algorithm are as follows:

{{< admonition type=info  title="The ID3 Algorithm Steps" open=true >}}

1. Determine the entropy associated with each feature of the data set.

2. Divide the data set into subsets based on various features and cutoff values. Using the formula above, compute the information gain $\\Delta IG$ as the difference in entropy before and after the split for each. Use the weighted average taking $N_{\\text{child}}$ into account for the total entropy of all child nodes after the split, i.e. how many of the $N$ samples end up on each child branch.

3. Determine which partition provides the most information gain. Make a decision node for that feature and divide the value.

4. When there are no more splits possible on a subset, create a leaf node and label it with the most common class of the data points within it if doing classification or the average value if doing regression.

5. Iterate through all subsets. Recursion is terminated if, following a split, all elements in a child node are of the same type. Additional stopping conditions, such as requiring a minimum number of samples per leaf to continue splitting, or finishing when the trained tree reaches a specified maximum depth, may be imposed.

{{< /admonition >}}

Reading the steps of an algorithm isn't always the most intuitive thing to do. To make things clearer, consider how information gain was used to determine the first decision node in our tree.

Remember how our first decision node split on Diameter ≤ 0.45? How did we come up with this condition? It was the result of trying to gain as much information as possible.

Each of the possible data splits on its two features (Diameter and Height) and cutoff values results in a different information gain value.

The line graph depicts the various split values for the Diameter feature. Changing the decision boundary will show how the entropy to the right and left of the split changes. The corresponding entropy values of both child nodes, as well as the total information gain, are shown.

```python
ent_left = []
ent_right = []
info_gain = []
N = df.shape[0]
initial_ent = entropy_np(df.Family, base=2)
splits = np.linspace(0, 1, 150)
for split in splits:
    right = df.loc[df.Diameter >= split].Family
    left = df.loc[df.Diameter < split].Family
    ent_r = entropy_np(right, base=2)
    ent_right.append(ent_r)
    ent_l = entropy_np(left, base=2)
    ent_left.append(ent_l)
    info_weighted = (right.shape[0] / N) * ent_r + (left.shape[0] / N) * ent_l
    info_gain.append(initial_ent - info_weighted)
  
plt.plot(splits, info_gain, label="Information Gain")
plt.plot(splits, ent_right, label="Entropy Right")
plt.plot(splits, ent_left, label="Entropy Left")
plt.xlabel("Diameter Split Value")
plt.annotate(
    f"Highest Information Gain Here = {max(info_gain):.2f} \n with split = {splits[np.argmax(info_gain)]:.3f}",
    (splits[np.argmax(info_gain)], max(info_gain)),
    xytext=(splits[np.argmax(info_gain)] + 0.07, max(info_gain) + 0.01),
    arrowprops=dict(facecolor="black", shrink=0.05, width=2, headwidth=8),
)
plt.legend(bbox_to_anchor=(1, 1.2), ncol=3)
```
<br>
<img src = "/posts/Decision Trees (CART)/entropy2.png" height= "100%" width="100%">

The ID3 algorithm will choose the split point with the highest information gain, which is represented by the peak of the blue line in the above chart of 0.85 at Diameter = 0.45.

{{< admonition type=note  title="A word about information measures" open=true >}}
As stated above there are other algorithms to calculate information gain.
The Gini impurity is an alternative to entropy in the construction of Decision Trees. This quantity is also a measure of information and is similar to Shannon's entropy. Decision trees trained with entropy or Gini impurity are comparable, with only a few exceptions showing significant differences. Entropy may be more prudent in the case of imbalanced data sets. Gini, on the other hand, may train faster because it does not use logarithms.
{{< /admonition >}}

## A Second Look at Our Decision Tree
Let's go over what we've learned thus far. We first observed how a Decision Tree categorises data by repeatedly dividing the feature space into regions in accordance with a set of conditional rules. Second, we studied entropy, a well-liked metric for assessing the purity (or lack thereof) of a particular sample of data. Third, we discovered that the precise conditional series of rules to choose is determined by Decision Trees using the ID3 algorithm and entropy in information gain. The three sections together detail the typical Decision Tree algorithm.

Let's take a look at our Decision Tree from a different angle to reinforce concepts.

We'll use scikit-learn's ```DecisionTreeClassifier``` with entropy (not the default Gini) to train the decision tree and the lovely [dtreeviz](https://github.com/parrt/dtreeviz) library to beautifully visualize our tree 

```python
from sklearn.tree import DecisionTreeClassifier
import dtreeviz

tree_classifier = DecisionTreeClassifier(max_depth=4)
tree_classifier.fit(df[["Height", "Diameter"]], df.Family.cat.codes)

viz_model = dtreeviz.model(
    tree_classifier,
    X_train=df[["Height", "Diameter"]],
    y_train=df.Family.cat.codes,
    feature_names=["Height", "Diameter"],
    target_name="Family",
    class_names=["apple", "cherry", "oak"],
)

v = viz_model.view()
```


<img src = "/posts/Decision Trees (CART)/tree5.svg" height= "100%" width="100%">

To begin, we can see that the original data set has the highest possible entropy value of 1.58 for a sample of three equally-sized classes.
Then, at the cost of two Apple data points, our first leaf node successfully separates out all Oak samples. Our data is well-partitioned at this point, but we can try going deeper if we want to separate the classes even further (overfit).
Each decision node, including those above and below the Height ≤ 4.88 node, is chosen based on information gain, which is a function of the entropy of the tree at the current and prior depths.
The second leaf node divides a large number of Cherry trees while misclassifying one Apple tree. We attempt to divide the remaining Apple and Cherry data points in our third or second to last decision node (Diameter ≤ 0.318). Finally the last decision node (Height ≤ 7.14) partitions the remaining data points, although it is not always successful, a Decision Tree tries to divide data at the leaf nodes into groups that are as "pure" as possible, as seen here and above.

Our sample of data points to classify shrinks from the top down as it is partitioned to different decision and leaf nodes. If we wanted, we could trace the entire path taken by a training data point in this manner. Also, not every leaf node is pure: as previously discussed (and in the following section), we don't want the structure of our Decision Trees to be too deep, as such a model is unlikely to generalise well to unseen data.

## The Pertubations Issue


Without a doubt, Decision Trees have a lot going for them. They are simple models that are simple to understand. They train quickly and require little data preprocessing. They also handle outliers with ease. However, they have a significant limitation in comparison to other predictors, and that is their instability. They can be extremely sensitive to small changes in the data: a minor change in the training examples can result in a drastic change in the Decision Tree's structure.

See for yourself how small random Gaussian perturbations on just 5% of the training examples result in a set of Decision Trees that are completely different:

```python
n_perturp = 3
for i in range(n_perturp):
    sub_smp = df.sample(frac=0.05).iloc[:, :2]
    perturped = np.random.normal(0, 1, (sub_smp.shape[0], 2)) + sub_smp
    new_df = df.copy()
    new_df.iloc[perturped.index, :2] = perturped

    tree_classifier = DecisionTreeClassifier(max_depth=4, criterion="entropy")
    tree_classifier.fit(new_df[["Height", "Diameter"]], new_df.Family.cat.codes)

    viz_model = dtreeviz.model(
        tree_classifier,
        X_train=new_df[["Height", "Diameter"]],
        y_train=new_df.Family.cat.codes,
        feature_names=["Height", "Diameter"],
        target_name="Family",
        class_names=["apple", "cherry", "oak"],
    )

    viz_model.view()
```
<img src = "/posts/Decision Trees (CART)/tree6.svg" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree7.svg" height= "100%" width="100%">
<img src = "/posts/Decision Trees (CART)/tree8.svg" height= "100%" width="100%">

### Why Is This An Issue?
Decision Trees are inherently unstable in their most basic form.

If left alone, the ID3 algorithm for training Decision Trees will work indefinitely to minimise entropy. It will keep splitting the data until all leaf nodes are completely pure - that is, they only contain one class. Such a process could result in extremely deep and complex Decision Trees. Furthermore, we just saw that when exposed to small perturbations in the training data, Decision Trees exhibit high variance.

Both of these issues are undesirable because they result in predictors that fail to distinguish between persistent and random patterns in the data, a problem known as overfitting. This is a problem because it implies that our model will underperform when exposed to new data.

Pruning Decision Trees can prevent excessive growth by limiting their maximum depth, limiting the number of leaves that can be created, or setting a minimum size for the amount of items in each leaf and not allowing leaves with too few items.

What about the issue of high variance? Unfortunately, it is an inherent feature when training a single Decision Tree.

The Importance of Moving Beyond Decision Trees
Ironically, one method for mitigating the instability caused by perturbations is to introduce an extra layer of randomness into the training process. In practise, this can be accomplished by assembling groups of Decision Trees trained on slightly different versions of the data set, the combined predictions of which are less prone to high variance. This method paves the way for one of the most successful Machine Learning algorithms to date: Random Forests. The random forest algorithm uses an ensemble technique known as bagging. There are other ensembling techniques that utilize the decision tree algorithm such as boosting (gradient or adaptive) and stacking, which just further shows the strength of this versatile algorithm and how it can be improved. 

Here is a python implementation of the decision tree algorithm (using recursion):
```python
class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _is_finished(self, depth):
        if (
            depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split
        ):
            return True
        return False

    def _entropy(self, y):
        n_labels = len(labels)

        if n_labels <= 1:
            return 0

        value, counts = np.unique(labels, return_counts=True)
        probs = counts / n_labels
        n_classes = np.count_nonzero(probs)

        if n_classes <= 1:
            return 0

        ent = 0.0

        base = 2

        for i in probs:
            ent -= i * log(i, base)

        return ent

    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        parent_loss = self._entropy(y)
        left_idx, right_idx = self._create_split(X, thresh)
        n, n_left, n_right = len(y), len(left_idx), len(right_idx)

        if n_left == 0 or n_right == 0:
            return 0

        child_loss = (n_left / n) * self._entropy(y[left_idx]) + (
            n_right / n
        ) * self._entropy(y[right_idx])
        return parent_loss - child_loss

    def _best_split(self, X, y, features):
        split = {"score": -1, "feat": None, "thresh": None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split["score"]:
                    split["score"] = score
                    split["feat"] = feat
                    split["thresh"] = thresh

        return split["feat"], split["thresh"]

    def _build_tree(self, X, y, depth=0):
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)

        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)
        return Node(best_feat, best_thresh, left_child, right_child)

    def _traverse_tree(self, x, node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)
```

We can test it on the same farm data we had earlier.

```python
tree = DecisionTree(max_depth=4)
tree.fit(df.iloc[:, :2].values, df.loc[:, "Family"].cat.codes.values)
print(tree.root.threshold)
```
    0.4488555701323009
We see that we get the same diameter threshold of 0.45 as our root node.

## Further Topics

The ID3 algorithm isn't the only algorithm used to train a decision tree, there are other decision tree training algorithms that differ in the use of information measures such as gini or entropy, the way thresholding is sampled (discrete vs continous) and the way features are sampled if sampled at all. Other decision tree training algorithms are: 
- CART (Classification and Regression Trees)
- C4.5 which is the successor of ID3 that can handle missing values, both continuous and discrete attributes and prunes trees after creation
- CHAID (Chi-square automatic interaction detection) which executes multi-level splits when computing classification trees and uses chi-square for splitting rather than gini or entropy.

Decision trees can also be used for regression problems not only classification (predicting a continous target rather than a discrete one). Here the mean response of observations falling in that region is the value obtained by leaf nodes in the training data. Reduction in variance or mean squared error is used to determine the splitting as information measures can't be used for continous targets without the option of discretization the target. The mean absolute error can also be used. 

There are other topics not discussed in this article to keep it more compact that I might write about in the future, these include other tree-specific hyperparameters, pruning, other splitting criterion and ensembling techniques.

Thank you for taking the time to read this! I hope you found the article useful. 
</span>
