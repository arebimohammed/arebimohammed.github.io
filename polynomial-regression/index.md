# Polynomial Regression



### Introduction
<span style="font-size:1.125rem"> The data we have frequently cannot be fitted linearly, and we must use a higher degree polynomial (such as a quadratic or cubic one) to be able to fit the data. When the relationship between the dependent ($Y$) and the independent ($X$) variables is curvilinear, as seen in the following figure (generated using numpy based on a quadratic equation), we can utilise a polynomial model.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import operator

X = 6 * np.random.rand(200, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(200, 1)

fig = plt.figure(figsize=(10,8))
_ = plt.scatter(X,y,s=10)
_ = plt.xlabel("$X$", fontsize=16)
_ = plt.ylabel("$y$", rotation=0, fontsize=16)
```

<img style = "width: 100%" src = "/posts/Polynomial Regression/example-data.png">



</span>

<span style="font-size:1.125rem"> Understanding linear regression and the mathematics underlying it is necessary for this topic. You can read my previous article on [Linear Regression](https://arebimohammed.github.io/linear-regression-all-you-need-to-know/) if you aren't familiar with it.</span>

<span style="font-size:1.125rem">
Applying a linear regression model to this dataset first will allow us to gauge how well it will perform.

```python
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

fig = plt.figure(figsize=(10,8))
_ = plt.scatter(X, y, s=10)
_ = plt.plot(X, y_pred, color='r')
plt.show()
```
The plot of the best fit line is:

<img style = "width: 100%" src = "/posts/Polynomial Regression/linear-fit.png">


It is clear that the straight line is incapable to depict the data's patterns. This is an example of [underfitting](https://www.ibm.com/cloud/learn/underfitting) (<cite>"Underfitting is a scenario in data science where a data model is unable to capture the relationship between the input and output variables accurately, generating a high error rate.."</cite>). Computing the R<sup>2</sup> score of the linear line gives: 

```python
from sklearn.metrics import r2_score

print(f"R-Squared of the model is {r2_score(y, y_pred)}")
```
    R-Squared of the model is 0.42763591651827204

{{< admonition note "Combatting Underfitting" true >}}
The complexity of the model must be increased in order to combat underfitting.
{{< /admonition >}}

### Why Polynomial Regression?

We can add powers of the original features as additional features to create a higher order equation. A linear model,

$\hat{y} = {\alpha} + {\beta}_1x$

can be transformed to

$\hat{y} = {\alpha} + {\beta}_1x + {\beta}_2x^{2}$

{{< admonition note "Linear Model or Not?" true >}}
Given that the coefficients and weights assigned to the features are still linear, this model is still regarded as linear. $x^{2}$ is still only a feature. But the curve we are trying to fit is quadratic in nature.
{{< /admonition >}}

What we are only doing here is adding powers of each feature as new features (interactions between multiple features can also be added as well depending on the implementation), then simply train a linear model on this extended set of features. This is the essence of Polynomial Regression (details later :wink:)

The scikit-Learn ``` PolynomialFeatures ``` class can be used to transform the original features into their higher order terms. Then we can train a linear model with those new generated features. Numpy also offers a polynomial regression implementation using ``` numpy.polyfit ``` (check my [github](https://github.com/arebimohammed/code-for-articles) for usage).

```python
polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
r2 = r2_score(y,y_poly_pred)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
fig = plt.figure(figsize=(10,8))
_ = plt.scatter(X, y, s=10)
_ = plt.plot(x, y_poly_pred, color='r')
plt.show()
```
<br>
Fitting a linear regression model on the transformed features gives the following plot.

<img style = "width: 100%" src = "/posts/Polynomial Regression/poly-fit.png">

The figure makes it quite evident that the quadratic curve can fit the data more accurately than the linear line. Calculating the quadratic plot's R<sup>2</sup> score results in:

```python
print(f"R-Squared of the model is {r2}")
```
    R-Squared of the model is 0.8242378566950601

Which is a great improvement from the previous R<sup>2</sup> score

When we try to fit a curve with degree 10 to the data, we can observe that it "passes through" more data points than the quadratic and linear plots.

```python
polynomial_features= PolynomialFeatures(degree=10)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)
r2 = r2_score(y,y_poly_pred)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
fig = plt.figure(figsize=(10,8))
_ = plt.scatter(X, y, s=10)
_ = plt.plot(x, y_poly_pred, color='r')
plt.show()

print(f"R-Squared of the model is {r2}")
```

<img style = "width: 100%" src = "/posts/Polynomial Regression/poly-fit10.png">

    R-Squared of the model is 0.831777739222978

We can observe that increasing the degree to 30 causes the curve to pass through more data points.

```python
polynomial_features= PolynomialFeatures(degree=30)
x_poly = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred30 = model.predict(x_poly)
r2 = r2_score(y,y_poly_pred30)

sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(X,y_poly_pred30), key=sort_axis)
x, y_poly_pred30 = zip(*sorted_zip)
fig = plt.figure(figsize=(10,8))
_ = plt.scatter(X, y, s=10)
_ = plt.plot(x, y_poly_pred30, color='r')
plt.show()

print(f"R-Squared of the model is {r2}")
```
<img style = "width: 100%" src = "/posts/Polynomial Regression/poly-fit30.png">

    R-Squared of the model is 0.7419383093794893

For degree 30, the model also accounts for noise in the data. This is an example of [overfitting](https://www.ibm.com/cloud/learn/overfitting) (<cite> "Overfitting is a concept in data science, which occurs when a statistical model fits exactly against its training data. When this happens, the algorithm unfortunately cannot perform accurately against unseen data, defeating its purpose." </cite>). Despite the fact that this model passes through many of the data, it will fail to generalise on unseen data, as observed on the decrease in R<sup>2</sup> score.

{{< admonition note "Combatting Overfitting" true >}}
To avoid overfitting, we may either increase the number of training samples so that the algorithm does not learn the noise in the system and can become more generalised (adding more data can potentially be an issue if the data is itself noise), or we can reduce the model's complexity (smaller degree in this example).
{{< /admonition >}}

Before we jump into the details of polynomial regression we must answer an important question, How do we choose an optimal model? And to answer this question we need to understand the bias vs variance trade-off.

### The Bias vs Variance Trade-Off

<b>Bias</b> refers to the error due to the model’s simplistic assumptions in fitting the data. A high bias means that the model is unable to capture the patterns in the data and this results in underfitting.

<b>Variance</b> refers to the error due to the complex model trying to fit the data. High variance means the model passes through most of the data points and it results in overfitting the data.

The following figure summarizes this concept:

<img style = "width: 100%" src = "/posts/Polynomial Regression/summary.png">

The graph below shows that as model complexity increases, bias decreases and variance increases, and vice versa. A machine learning model should ideally have low variance and bias. However, it is nearly difficult to have both. As a result, a trade-off is made in order to produce a good model that performs well on both train and unseen data.

<figure>
<img style = "width: 100%" src = "/posts/Polynomial Regression/tradeoff.png">
<figcaption style = "font-size:15px">Source: <a href = "http://scott.fortmann-roe.com/docs/BiasVariance.html">Bias-Variance Tradeoff</a></figcaption>
</figure>

<br>

### Polynomial Fitting Details
#### Rank of a matrix

Suppose we are given a $mxn$ matrix $A$ with its columns to be $[ a_1, a_2, a_3 ……a_n ]$. The column $a_i$ is called linearly dependent if we can write it as a linear combination of other columns i.e. $a_i = w_1a_1 + w_2a_2 + ……. + w_{i-1}a_{i-1} + w_{i+1}a_{i+1} +….. + w_{n}a_{n}$ where at least one $w_{i}$ is non-zero. Then we define the rank of the matrix as the number of independent columns in that matrix. $Rank(A)$ = number of independent columns in $A$.

However there is another interesting property that the number of linearly independent columns is equal to the number of independent rows in a matrix ([proof](https://en.wikibooks.org/wiki/Linear_Algebra/Row_and_column_spaces#Proof)). Hence $Rank(A) ≤ min(m, n)$. A matrix is called full rank if $Rank(A) = min(m, n)$ and is called rank deficient if $Rank(A) < min(m, n)$.

#### Pseudo-Inverse of a matrix

An $nxn$ square matrix $A$ has an inverse $A^{-1}$ if and only if $A$ is a full rank matrix. However a rectangular $mxn$ matrix $A$ does not have an inverse. If $A^{T}$ denotes the transpose of matrix $A$ then $A^{T}A$ is a square matrix and Rank of $(A^{T}A)$ = $Rank (A)$ ([proof](https://math.stackexchange.com/questions/349738/prove-operatornamerankata-operatornameranka-for-any-a-in-m-m-times-n)).

Therefore if $A$ is a full-rank matrix then the inverse of $A^{T}A$ exists. And $(A^{T}A)^{-1}A^{T}$ is called the pseudo-inverse of $A$. We’ll see soon why it is called so.

#### Details

As discussed earlier, in polynomial regression, the original features are converted into polynomial features of required degree $(2,3,..,n)$ and then modeled using a linear model. Suppose we are given $n$ data points $pi = [ x_{i1} ,x_{i2} ,……, x_{im} ]^{T} , 1 ≤ i ≤ n$ , and their corresponding values $vi$. Here $m$ denotes the number of features that we are using in our polynomial model. Our goal is to find a nonlinear function $f$ that minimizes the error

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq1.png">

Hence $f$ is nonlinear over $pi$. 

So let’s take an example of a quadratic function i.e. with $n$ data points and 2 features. Then the function would be

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq2.png">

The objective is to learn the coefficients. Hence we have $n$ points $pi$ and their corresponding values $vi$; we have to minimize

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq3.png">

For each data point we can write equations as

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq4.png">

Hence we can form the following matrix equation $$Da = v$$

where

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq5.png">

However the equation is nonlinear with respect to the data points $pi$, it is linear with respect to the coefficients $a$. So, we can solve for a using the linear least square method.

We have $$Da = v$$

multiply $D^{T}$ on both sides $$D^{T}Da = D^{T}v$$

Suppose $D$ has a full rank, that is when the columns in $D$ are linearly independent, then $D^{T}D$ has an inverse.Therefore $$(D^{T}D)^{-1}(D^{T}D)a = (D^{T}D)^{-1}D^{T}v$$

We now have $$a = (D^{T}D)^{-1}D^{T}v$$

Comparing it with $Da = v$, we can see that $(D^{T}D)^{-1}D^{T}$ acts like the inverse of $D$. So it is called the pseudo-inverse of $D$.

The above used quadratic polynomial function can be generalised to a polynomial function of order or degree $m$.

<img style = " display: block;margin-left: auto;margin-right: auto" src = "/posts/Polynomial Regression/eq6.png">


Thank you for reading and I hope now that you are clear with the working of polynomial regression and the mathematics behind. I have used polynomial regression on the Covid-19 data and wrote an article about it, you can read it [here](https://arebimohammed.github.io/covid19-interactive-analysis/).

Also check the [github repo](https://github.com/arebimohammed/code-for-articles) for the complete code.
</span>
