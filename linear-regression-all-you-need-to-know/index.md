# Linear Regression

<!--more-->

What a better way to start my blogging journey, than with one of the most fundamental statistical learning techniques, Linear Regression. A straightforward method for supervised learning - supervised learning is the process of training a model on data where the outcome is known before applying it to data where the outcome is unknown -, learning linear regression also helps to understand the overall process of what supervised learning looks like. Linear regression, in particular, is a powerful tool for predicting a quantitative response. It's been around for a while and is the subject of a slew of textbooks. Linear Regression is part of the Generalized Linear Models family (GLM for short). Despite the fact that it may appear tedious in comparison to some of the more current statistical learning approaches, linear regression remains an effective and extensively used statistical learning method. It also provides as a useful starting point for emerging approaches. Many fancy statistical learning methods can be thought of as extensions or generalizations of linear regression. As a result, the necessity of mastering linear regression before moving on to more advanced learning approaches cannot be emphasized.


 The response to the question "Is the variable $X$ associated with a variable $Y$, and if so, what is the relationship, and can we use it to predict Y?" is perhaps the most prevalent goal in statistics and linear regression tries to answer that, it does this based on linear relationships between the independent ($X$) and dependent ($Y$) variables. Simple linear regression creates a model of the relationship between the magnitudes of two variables—for example, as X increases, Y increases as well. Alternatively, when X increases, Y decreases. I've used the term simple here because we are only talking about one variable X, but instead of one variable X, we can of course use multiple predictor variables $ X_{1}...X_{n} $, which is often termed Multiple Linear Regression or just simply Linear Regression. 

 I assure you that I have a multitude of examples and explanations that go through all of the finer points of linear regression. But first, let's go through the fundamental concepts. 


### The fundamental concepts behind linear regression
<br>

- The first step in linear regression is to fit a line to the data using least squares
- The second step is to compute R<sup>2</sup>
- Finally, compute the p-value for the computed R squared in the previous step

The first concept we are going to tackle is fitting a line to the data, what exactly does that mean and how can we do it? To effectively explain that we are going to need some data. I will be using the fish market data from Kaggle which is available [here](https://www.kaggle.com/aungpyaeap/fish-market). The dataset contains sales records for seven common fish species seen in fish markets. Some of the features (columns) in dataset are: Species, Weight, Different Lengths, Height and Width. We'll be using it to estimate the weight of the fish, as we are trying to explain linear regression we'll first use only one feature to predict the weight, the height (simple linear regression).

We'll first load the dataset using pandas ``` read_csv ```:

```python
df = pd.read_csv("data/Fish.csv")
```
All code in this article can be found in my Github in this [repository](https://github.com/arebimohammed/code-for-articles/tree/master/Linear%20Regression)

Now back to our first concept, fitting a line to the data, what does that mean? 

#### Fitting a line to the data

To clearly explain this concept let's first get only one species of fish, in this example we'll use Perch:

```python
df_perch = df[df.Species == 'Perch']
```
We'll then simply create a scatter plot of Weight vs. Height.
```python
fig = px.scatter(df_perch, x='Height', y='Weight')
fig.show()
```

<img style = "width: 100%" src = "/posts/Linear Regression/Scatter_x_y.png">

And then we'll plot a horizontal line across the data, at the average weight (our target variable). 

```python
mean_val= df_perch.Weight.mean()
fig = fig.add_hline(y = mean_val,row= None, col = None)
fig.show()
```

<img style = "width: 100%" src = "/posts/Linear Regression/Scatter_mean.png">


Then we calculate the difference between the actual values, the dots representing the weights in the figure, and the line (the mean value). This can be thought of as the distance between them (if you would like to think geometrically). These distances are also called the residuals. Here are the residuals plotted.

```python
for weight,height in zip(df_perch.Weight,df_perch.Height):
    fig.add_shape(type='line',x0=height,x1=height,y0=weight,y1=mean_val,line=dict(width=1,dash='dash',color='red'))
fig.show()
```
<img style = "width: 100%" src = "/posts/Linear Regression/Scatter_residuals.png">

 Then we'll square each distance (residual) from the line to the data and sum them all up. We square the residuals so that the negative and positive residuals don't cancel each other out. 

 ```python
 residuals_squared = (df_perch.Weight - mean_val)**2
 residuals_squared.sum()
 ```
We get a value of `` 6646094.253571428 ``. That's a very large number, but we don't want that we want it to be as low as possible, and how can we do that? That's the next step.

Third we rotate and/or shift the line a little bit to be able to decrease this sum of squared distances or residuals to its lowest possible value, that is where the name least squares (or ordinary least squares) comes from. To do that we need to understand what exactly a line is mathematically.

You probably remember from linear algebra that a line is just this equation $ y = mx +b $, where $m$ is the slope of the line $b$ is the y-intercept, $x$ is the x-axis value and $y$ is the y-axis value. Well to rotate the line we need to iteratively - or mathematically - adjust the y-intercept ($b$) and the slope ($m$) so that the sum of the squared residuals - the error - is the lowest. In linear regression the equation is usually expressed in a different way like so: $\hat{y} = {\alpha} + {\beta}x$, where $ {\alpha} $ is the y-intercept (sometimes also written as $ {\beta}_0 $) and $ {\beta} $ is the slope. In simple linear regression there is only one independent variable - feature - and therefore only one slope - or coefficient -, but in many cases we have many different features that we would like to use in our equation so for every feature $X_n$ added we add its coefficient $ {\beta}_n$, to be estimated with the y-intercept. 
In general, such a relationship may not hold perfectly for the mostly unobserved population of values of the feature and target variables; these unobserved variations from the above equation are referred to as random errors or noise, which we assume on average is zero, so during modelling of the linear regression line it is omitted from the parameters to estimate. And they are usually added to the equation as well to get: $\hat{y} = {\alpha} + {\beta}x  + {\epsilon}_i$. 

But now comes the real question how does linear regression adjust the y-intercept and coefficients to get the "best-fitting" line which yields the lowest sum of squared residuals. Recall that I mentioned that these parameters ($\hat{\alpha}$ and $\hat{\beta}$) can be either estimated iteratively or mathematically. In the third step I mentioned that we rotate the line a little bit to be able to decrease the sum of squared residuals, well that is actually the iterative approach, which is conducted using an optimization technique named gradient descent. The mathematical approach uses derived equations using calculus to estimate these parameters, analytically straight away to get the line. In this article I'll talk about these equations i.e. the mathematical approach, and will leave the iterative approach using gradient descent for another article where I talk about gradient descent in general as it is widely used in many other machine learning algorithms (not only linear regression) to minimize a loss or cost function (or gradient ascent to maximize an objective function) and it does that by iteratively adjusting these parameter, in linear regression the sum of squared residuals is the loss function. 

First let's define the sum of squared residuals equations, we know that the equation for the line is: $ \hat{y} = {\alpha} + {\beta}x$, and we know that the residuals is the actual value $y$ minus the estimated value which we will give the symbol $\hat{y}$ then squared. So the sum of squared residuals is defined by: 

$$ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2  = \sum_{i=1}^{n} (y_i -  (\hat{\alpha} + {\hat{\beta}}x_i))^2 $$

where $n$ is the sample size.

we can shorten this to a function with $ \hat{\alpha} $ and $ \hat{\beta} $ as parameters, $ F(\hat{\alpha},\hat{\beta}) $. To minimize this equation, we use calculus, which is great in such problems (if the function is differentiable). 

How does it do that?

- It takes the partial derivatives of the function ($ F(\hat{\alpha},\hat{\beta}) $) with respect to  $ \hat{\alpha} $ and $ \hat{\beta} $
- Sets the partial derivatives equal to 0 
- Finally, solve for  $ \hat{\alpha} $ and $ \hat{\beta} $


We could examine the second-order conditions to make sure we identified a minimum and not a maximum or a saddle point. I'm not going to do that here, but rest assured that our calculations will result in the smallest sum of squared residuals achievable. I will also not explain all the derivation steps of the partial derivatives here but simply give them. 
The first partial derivative with respect to $\hat{\alpha}$ is:
$\frac {\partial} {\partial\hat{\alpha}} = -2 \sum_{i=1}^{n} (y_i -  (\hat{\alpha} + {\hat{\beta}}x_i)) $
 
and the second partial derivative with respect to $\hat{\beta}$ is:
$\frac {\partial} {\partial\hat{\beta}} = -2 \sum_{i=1}^{n} x_i(y_i -  (\hat{\alpha} + {\hat{\beta}}x_i)) $

Now we set these partial derivatives equal to zero:

For $\hat{\alpha}$ : $ -2 \sum_{i=1}^{n} (y_i -  (\hat{\alpha} + {\hat{\beta}}x_i)) = 0 $ (1)

For $\hat{\beta} $ : $ -2 \sum_{i=1}^{n} x_i(y_i -  (\hat{\alpha} + {\hat{\beta}}x_i)) = 0 $ (2)

Now we have 2 equations and 2 unknowns, and we are going to solve these to find our optimal parameters, how do we do that? First we get an expression for $ \hat{\alpha} $ that solves for it using the first equation and, which we will then substitute in the second equation to solve for $ \hat{\beta} $. 

The first expression that solves for $\hat{\alpha}$ ends up to be: $ \hat{\alpha} = \bar{y} - \hat{\beta}\bar{x} $, where $\bar{y}$ and $\bar{x}$ are the averages of both variables. When we substitute this expression for the partial derivative equation for $ \hat{\beta} $ we get: $\frac {\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i-\bar{x})^2} $

With these equations in hand we can easily estimate the parameters and fit a line to the data that will give us the minimum sum of squared residuals. Let's implement that in python:

```python
def LS(x, y):

  x_bar = np.average(x)
  y_bar = np.average(y)

  numerator = np.sum((x - x_bar)*(y - y_bar))
  denominator = np.sum((x - x_bar)**2)
  beta_hat = numerator / denominator

  alpha_hat = y_bar - beta_hat*x_bar

  best_model = {'alpha_hat':alpha_hat, 'beta_hat':beta_hat}

  return best_model
```

We can then use this function for our data to get the best fitting line:

```python 
x= df_perch.Height
y = df_perch.Weight
best_model  = LS(x,y)
```
The least squares function determines that the best values for the parameters, $\hat{\alpha}$ and $\hat{\beta}$ that minimize the sum of squared residuals are: -537.3275192931233 and 116.96540985551397 respectively.

We can also visualize that using plotly (my favourite plotting library 😊):

```python
y_hat = best_model['alpha_hat'] + best_model['beta_hat'] * x
fig.add_scatter(x=x,y=y_hat, line=dict(color='green'),name='LS Fit')
```

<img style = "width: 100%" src = "/posts/Linear Regression/LS_fit.png">

We can compare these estimated/learned parameters with scikit-learns ``` LinearRegression ``` solution as well:

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(x.values.reshape(-1, 1),y)
print("Intercept: " + str(model.intercept_) , "Coefficient: " + str(model.coef_[0]))
```
    Intercept: -537.3275192931234 Coefficient: 116.96540985551398

They are similar as scikit-learns ``` LinearRegression ``` also uses scipys ``` linalg.lstsq ```, which computes the least squares solution.

We can also visualize the residuals with respect to this line:

``` python
for weight,height,res in zip(df_perch.Weight,df_perch.Height,y_hat):
    fig.add_shape(type='line',x0=height,x1=height,y0=weight,y1=res,line=dict(width=1,dash='dash',color='red'))

fig.show()
```
<img style = "width: 100%" src = "/posts/Linear Regression/LS_fit_residuals.png">

We can see they are very small residuals which is exactly what we want. We can also quantify the residuals, using our sum of squared residuals to get the lowest possible value which in this case is: 412872.84783624834 which is a great improvement from 6646094.253571428, but linear regression is usually evaluated using the mean squared error. The mean squared error is just the sum of squared residuals divided by $ n $, the sample size. Which in this simple case is 56, that gives us a mean squared error of: 7372.7294256472915.

```python
ls_fit_residuals = (df_perch.Weight - y_hat)**2
print(ls_fit_residuals.sum())
print(ls_fit_residuals.sum()/len(x))
```
    412872.84783624834
    7372.7294256472915.

or by using scikit-learns ``` mean_squared_error ``` in the metrics module

```python
from sklearn.metrics import mean_squared_error
mean_squared_error(y, y_hat)
```
    7372.7294256472915.



Other evaluation metrics such as mean absolute error, root mean squared error and R<sup>2</sup> are used for linear regression as well. Which brings us to our second concept calculating R<sup>2</sup>.


#### Computing R<sup>2</sup> or Coefficient of Determination

This metric represents the proportion of the dependent variable's variation explained by the model's independent variables. It assesses the adequacy of your model's relationship with the dependent variable.
Consider the following scenario in which we estimate the error of the model with and without knowledge of the independent variables to better grasp what R<sup>2</sup> really means.

When we know the values of the independent variable ($X$), we can calculate the sum of squared residuals (SSR for short) as discussed above. Which we know is the difference between the actual value and the value on the line (predicted value) squared, summed over all data points. And with consider the situation where the independent variable's values are unknown and only the $y$ values are available. The mean of the $y$ values is then calculated, then the mean can be represented by a horizontal line, as seen above, we now calculate the sum of squared residuals between the mean $y$ value and all other $y$ values. This sum of squared residuals between the mean $y$ value and all other $y$ values is termed as the total sum of squares (TSS for short), which is the total variation in $y$. Why the mean? Because the mean is the simplest model we can fit and hence serves as the model to which the least squares regression line is compared to.

$$ TSS = \sum_{i=1}^{n} (y_i - \bar{y}) ^ 2 $$

As a result, we'd like to know what proportion of the total variation of $y$ is explained by the independent variable $x$. We may derive the coefficient of determination or R<sup>2</sup> by subtracting the proportion of the total variation of $y$ that is not represented by the regression line from 1, to get:

$$ R^2 = 1 - \frac {SSR}{TSS} $$ 

which represents that part of the variance of $y$ that is described by the independent variable $x$.
In our case we can calculate it like so:

```python
1 - (ls_fit_residuals.sum() / residuals_squared.sum())
```
    0.9378773709665068

or by using scikit-learns ``` r2_score ```

```python
from sklearn.metrics import r2_score
r2_score(y,y_hat)
```
    0.9378773709665068

The sum of squared residuals (SSR) to total sum of squares (TSS) ratio indicates how much overall error is left in your regression model. By subtracting that ratio from one, you can calculate how much error you were able to eliminate using regression analysis. This is R<sup>2</sup>. If R<sup>2</sup> is high (it's between 0 and 1), the model accurately depicts the dependent variable's variance. If R<sup>2</sup> is very low, the model does not represent the dependent variable's variance, and regression is no better than taking the mean value, i.e. no information from the other variables is used. A negative R<sup>2</sup> indicates that you are performing worse than the average. It can be negative if the predictors fail to explain the dependent variables at all. As a result, R<sup>2</sup> assesses the data points that are scattered about the regression line.
A model with an R<sup>2</sup> of 1 is almost impossible to find. In that situation, all projected values are identical to actual values, implying that all values lie along the regression line or that the model is over-fitting the data, which in this case might require some regularisation (Lasso or Ridge Regression), I'll talk about these in a seperate blog post.
However, don't be fooled by the R<sup>2</sup> value. A good model may have a low R<sup>2</sup> value, while a biased model may have a high R<sup>2</sup>. It is for this reason that residual plots should be used:

```python
import plotly.graph_objects as go
res_plot = go.Figure(go.Scatter(x=y_hat, y = y-y_hat,mode='markers'))

res_plot.show()
```

<img style = "width: 100%" src = "/posts/Linear Regression/LS_fit_resplot.png">

And as we can see they are randomly scattered around zero.

#### How significant is the R<sup>2</sup>?

We need a way to determine if the R<sup>2</sup> value is statistically significant. We need a p-value to be more precise. The main idea behind the p-value for R<sup>2</sup> is determined by the F-statistic. Which is a statistic that expresses how much of the model has improved (compared to the mean) given the inaccuracy of the model, which we can calculate a p-value for to get the significance of R<sup>2</sup>. The F-statistic is expressed in the following way:

$ F = \frac {(TSS- RSS)/(p-1)}{RSS/(n-p)} $

where $p$ is the number of predictors (including the constant) and $n$ is the number of observations or sample size, these are the degrees of freedom that turn the sums of squares into variances. We can interpret the nominator as the variation in fish weight explained by height and the denominator as the variation in fish weight NOT explained by fish height i.e. the variation that remains after fitting the line. If the "fit" was good then the nominator will be a large number and the denominator a small number which should yield a large F-statistic, the opposite otherwise. The question remains, where is the p-value? or more precisely how do we turn this number - F-statistic - to a p-value. 

To determine the p-value, we generate a set of random data, calculate the mean and sum of squares around the mean, and then calculate the fit and sum of squares around the fit. After that, plug all of those values into the F-statistic equation to get a number. Then plot that number in a histogram, and repeat the process as many times as necessary. We return to our original data set once we've finished with our random data sets. The numbers are then entered into the F-statistic equation in this case. The p-value is calculated by dividing the number of more extreme values by the total number of values. Our F-statistic is:

```python
RSS = ls_fit_residuals.sum()
TSS = residuals_squared.sum()
p = 2
n= len(x)

F_statistic = ((TSS - RSS)/(p-1))/(RSS/(n-p))
print(F_statistic)
```
    815.2484661408347

All of this is usually done with software and we can do that in python using statsmodels ``` OLS ```  like so:

```python
import statsmodels.api as sm
x = sm.add_constant(x.values.reshape(-1,1))
model_sm = sm.OLS(y,x).fit()
model_sm.summary()
```
which results in the following table with all information including the coefficients, the intercept, R<sup>2</sup>, F-statistic and the p-value for it, which in this case is $2.9167 × 10^{-34}$ and if the p-value is less than a significance level (usually 0.05) then the model fits the data well (true in this case).

<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>         <td>Weight</td>      <th>  R-squared:         </th> <td>   0.938</td>
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.937</td>
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   815.2</td>
</tr>
<tr>
  <th>Date:</th>             <td>Fri, 18 Mar 2022</td> <th>  Prob (F-statistic):</th> <td>2.92e-34</td>
</tr>
<tr>
  <th>Time:</th>                 <td>14:21:40</td>     <th>  Log-Likelihood:    </th> <td> -328.82</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>    56</td>      <th>  AIC:               </th> <td>   661.6</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>    54</td>      <th>  BIC:               </th> <td>   665.7</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>   
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>   
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td> -537.3275</td> <td>   34.260</td> <td>  -15.684</td> <td> 0.000</td> <td> -606.015</td> <td> -468.640</td>
</tr>
<tr>
  <th>x1</th>    <td>  116.9654</td> <td>    4.096</td> <td>   28.553</td> <td> 0.000</td> <td>  108.752</td> <td>  125.178</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>11.275</td> <th>  Durbin-Watson:     </th> <td>   0.678</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.004</td> <th>  Jarque-Bera (JB):  </th> <td>  11.319</td>
</tr>
<tr>
  <th>Skew:</th>          <td> 0.954</td> <th>  Prob(JB):          </th> <td> 0.00349</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.099</td> <th>  Cond. No.          </th> <td>    24.8</td>
</tr>
</table>

These are the main and most fundamental ideas behind linear regression, but of course there are more in-depth topics regarding linear regression such as multicolinearity, categorical variables, cofounding variables, outliers, influential values, heteroskedasticity, non-normality, correlated errors, regularisation and polynomial regression. Polynomial regression is normal linear regression but just with polynomial features which are feature crosses (combinations & exponents). Polynomial features are usually computed by multiplying features together and/or raising features to a user-selected degree. Scikit-learn preprocessing module implements this using its [`PolynomialFeatures`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) class that also gives you the option to keep interaction features, which are <cite> "...features that are products of at most degree distinct input features, i.e. terms with power of 2 or higher of the same input feature are excluded"</cite>. I will try my best to write about these different topics in seperate blog posts.

Thank you very much for reading all the way through, and I hope you enjoyed the article. See you in the next article and, Stay Safe! 


