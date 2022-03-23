# COVID-19: An Interactive Analysis

<!--more-->

A new virus has plauged us and caused turmoil in the world. This virus has caused shut-downs, lockdowns and in the worst-case will unfortunately cause deaths. It is our job as responsible citizens to do our best to stop it from further spreading, and what better way than for us data scientists to dive deep into the data (of course while wearing masks and keeping the safe distance).
So lets dive in.

## Introduction 
<hr>

The SARS-CoV-2 virus causes Coronavirus Disease (COVID-19), an infectious respiratory disease.
The majority of those infected with the virus will have mild to moderate respiratory symptoms and will recover without the need for medical attention. Some, on the other hand, will become critically unwell and require medical assistance. Serious sickness is more likely to strike the elderly and those with underlying medical disorders such as cardiovascular disease, diabetes, chronic respiratory disease, or cancer. COVID-19 can make anyone sick and cause them to get very ill or die at any age. 

#### But how does the virus spread?

Coughing, sneezing, and talking are the most common ways for the virus to spread through small droplets. Although the droplets are not normally airborne, persons who are in close proximity to them may inhale them and become infected. By contacting a contaminated surface and subsequently touching their face, people can become sick. Aerosols that can stay suspended in the air for prolonged periods of time in confined places may also be a source of transmission. It is most contagious in the first three days after symptoms develop, but it can also spread before symptoms appear and from asymptomatic people. Which strongly explains the value of wearing well fitted masks and as further illustrated in the GIF below.


<figure>
  <img src="/posts/COVID19- An Interactive Analysis/COVID19-Spread.gif" alt="Covid-19 Spread"/>
  <figcaption style = "font-size:15px">GIF Source: <a href = "https://www.youtube.com/watch?v=xEp-Sdgl9AU">Infrared video shows the risks of airborne coronavirus spread | Visual Forensics</a></figcaption>
</figure>

So if we ask ourselves how can we prevent COVID-19 from spreading, we'll find these main precautions to take:

- Keep a safe distance away from somebody coughing or sneezing.
- When physical separation isn't possible, wear a mask.
- Seek medical help if you have a fever, cough, or difficulty breathing.
- If you're sick, stay at home.
- Hands should be washed frequently. Use soap and water or an alcohol-based hand rub to clean your hands.
- Keep your hands away from your eyes, nose, and mouth.
- When you cough or sneeze, cover your nose and mouth with your bent elbow or a tissue.

<br>

#### How is the virus detected?

Real-time reverse transcription polymerase chain reaction (rRT-PCR) from a nasopharyngeal swab is the usual method of diagnosis. Although chest CT imaging may be useful for diagnosis in patients with a strong suspicion of infection based on symptoms and risk factors, it is not recommended for routine use screening ([Wikipedia](https://en.wikipedia.org/wiki/COVID-19_testing#:~:text=Chest%20CT%20scans%20and%20chest%20x%2Drays%20are%20not%20recommended%20for%20diagnosing%20COVID%2D19.)), which may present an option to utilize computer vision for example by using convolutional neural networks to detect the virus in CT image scans, we'll explore this another time in a different article.


## Diving Deep into the data
<hr>

All these previous precautions mentioned are of tremendous importance to stop the virus from further spreading but these won't allow us to study more concisely where it spreads, why it spreads in areas more than others and how we can flatten the curve, as they are proactive measures, we are trying to analyze the historical (even though the virus is still fairly young) data through reactive measures of data analysis and machine learning, of course proactive (unsupervised learning) measures can also be deployed. That's where data analysis comes into play. I dug deep into the data and all the accompaying code in this article can be found in this Github repository:
[Covid-19-Analysis](https://github.com/arebimohammed/Covid-19-Analysis). 

The data is provided by John Hopkins University, it is available in their [Github repo](https://github.com/CSSEGISandData/COVID-19). The data is updated daily. 
I will also be using data from [Our World In Data](https://ourworldindata.org/coronavirus) and [acaps](https://www.acaps.org/covid-19-government-measures-dataset) for further data exploration.

Lets start by first importing all required libraries. And setting some default settings

```python
import pandas as pd
import numpy as np
import itertools
import os
import warnings 
from itertools import tee 
warnings.filterwarnings('ignore') 

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.offline as py
import ipywidgets as widgets

py.init_notebook_mode()
pio.renderers.default = "notebook"
pd.options.plotting.backend = "plotly"
pd.options.display.max_rows = 20

from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import  r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

#from countryinfo import CountryInfo #Populations not up to date

import pypopulation
import pycountry

```

I'll be using [plotly](https://plotly.com/python/getting-started/) for interactive plotting. [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) for interactive user data filtering and selection. [Pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/) for data manipulation. [scikit-learn](https://scikit-learn.org/stable/) for machine learning and further corresponding libraries for other utilities.

We'll then load the datasets using Pandas straight from the source URL, because in this way everytime the code is ran it will have the most up to date data, rather than a downloaded Excel or CSV file.

```python
# Only use the columns we need 
cols = ['iso_code','continent','location','date'
         ,'total_cases','new_cases','total_deaths',
         'new_deaths','total_cases_per_million',
         'new_cases_per_million','total_deaths_per_million',
         'new_deaths_per_million','new_tests',
         'total_tests','total_tests_per_thousand',
         'new_tests_per_thousand','new_tests_smoothed',
         'new_tests_smoothed_per_thousand',
         'tests_units','stringency_index',
         'population','population_density',
         'median_age','aged_65_older','aged_70_older',
         'gdp_per_capita','extreme_poverty',
         'cardiovasc_death_rate','diabetes_prevalence',
         'female_smokers','male_smokers',
         'handwashing_facilities',
         'hospital_beds_per_thousand','life_expectancy']

df_stats =  pd.read_csv('https://covid.ourworldindata.org/data/owid-covid-data.csv',delimiter=',')
df_stats = df_stats[cols] 
df_measures = pd.read_excel('https://www.acaps.org/sites/acaps/files/resources/files/acaps_covid19_government_measures_dataset_0.xlsx', header=0,sheet_name='Dataset')

df_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
df_recoveries = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

```

    Shape for Statistics DataFrame: (167554, 34)
    Shape for Measures DataFrame: (23923, 18)
    Shape for Confirmed DataFrame: (284, 783)
    Shape for Deaths DataFrame: (284, 783)
    Shape for Recovories DataFrame: (269, 783)

After loading the data and printing their shapes I had a quick look at it using several Pandas DataFrame methods such as,
``` head ```, ``` sample ``` , ``` describe ``` and ``` info ```. I then grouped each case status by Country/Region which I will use later.

```python
#Group by each country for later use
df_confirmed_group = df_confirmed.groupby(by='Country/Region',as_index=False).sum()
df_deaths_group = df_deaths.groupby(by='Country/Region',as_index=False).sum()
df_recoveries_group = df_recoveries.groupby(by='Country/Region',as_index=False).sum()

# Getting active cases by substracting the recovories and deaths from all cases
df_active_group = pd.concat([df_deaths_group.iloc[: , :3],pd.DataFrame(df_confirmed_group.iloc[: , 3:] - (df_recoveries_group.iloc[: ,3:] +  df_deaths_group.iloc[:, 3:]))],axis=1)
```
Due to the shape of the data (see above; dates are columns), I created a combined dataframe for easier manipulation, analysis and visualization.

```python
#Creating Dataframe for further visualization and analysis
stats = pd.DataFrame(columns=['Date','Confirmed','Deaths','Recovered','Active'])
stats['Date'] = df_confirmed.columns[4:]


stats['Confirmed'] = stats['Date'].apply(lambda x: df_confirmed[x].sum())
stats['Deaths'] = stats['Date'].apply(lambda x: df_deaths[x].sum())
stats['Recovered'] = stats['Date'].apply(lambda x: df_recoveries[x].sum())
stats.reset_index(drop=False, inplace=True)
stats['Active'] = stats['index'].apply(lambda x: (stats['Confirmed'][x]-(stats['Recovered'][x]+stats['Deaths'][x])))
stats['Date'] = pd.to_datetime(stats['Date'])

```
I then created a figure, a Tree Map to be precise to get the latest world-wide Covid-19 status data. The figure was produced using plotlys [ `graph_objects`](https://plotly.com/python/graph-objects/) API, which has support for many different types of figures (TreeMap in this example,using the general Figure)

```python
latest_stats_fig = go.Figure()
latest_stats_fig.add_trace(go.Treemap(labels = ['Confirmed','Active','Recovered','Deaths'],
                                     parents = ['','Confirmed','Confirmed','Confirmed'],
                                     values = [stats['Confirmed'].sum(), stats['Active'].sum(), stats['Recovered'].sum(), stats['Deaths'].sum()],
                                      branchvalues="total", marker_colors = ['#b3e3ff','#ff0000','#26ff00','#000000'],
                                      textinfo = "label+value",
                                      outsidetextfont = {"size": 35},
                                      marker = {"line": {"width": 3}}
                                     ))
latest_stats_fig.update_layout(height=500, title_text = 'Latest Covid-19 Stats')
latest_stats_fig.show()
```

<iframe src= "/posts/COVID19- An Interactive Analysis/latest.html" height="525" width="100%"></iframe>

Just a quick side note, you might realize that the recovery data is odd, as there is this sudden plummet on the 5th of August 2021. This is because this data has been discontinued from that day onwards (here is the issue: [Recovery data to be discountinued](https://github.com/CSSEGISandData/COVID-19/issues/4465))

I have added this figure for illustration on how plotly works in general but all other interactive plots have been deployed to [Binder](https://mybinder.org/) and [Heroku](https://www.heroku.com/) using [voila](https://voila.readthedocs.io/en/stable/using.html). These can be attractive deployment options for machine learning apps and/or interactive dashboard apps for analysis (I will be using them quite often in the future 😊). The links for the interactive dashboard are:

- [Binder](https://mybinder.org/v2/gh/arebimohammed/Covid-19-Analysis/master?urlpath=voila%2Frender%2FML-Covid19-Interact-app.ipynb)
- [Heroku](https://covid19-interactive-app.herokuapp.com/)

Both apps are using Voila as an interface to make the notebook interactive so there isn't any difference in how the app looks (see demo video below). The app will take a bit of time to launch as it has to execute all code cells in the notebook, some which take a bit longer due to creating animations in the figure (as we'll see later 😉). So be patient and you'll be able to interact with the figures, you can continue reading in the meantime.

<iframe src="https://drive.google.com/file/d/1hsrj-yOMJWZi8rlz1ublFc4ZUd4JVLzL/preview" width="640" height="480" allow="autoplay"></iframe>


Ok I hope you got the dashboard app up and running, now moving on. The next analysis I decided to conduct is to get the daily change in Covid-19 cases per status, which are either "Confirmed", "Recovored", "Active" or "Deaths". This figure is produced with this piece of code below:

```python 

daily_case_fig = go.FigureWidget(make_subplots(rows=2, cols=2, vertical_spacing=0.1, horizontal_spacing=0.1,
                           subplot_titles=('Confirmed','Active','Recovered','Deaths'),
                            x_title='Date', y_title='Count of Cases',shared_xaxes=True))
                                                                                  
daily_case_fig.add_trace(go.Bar(x=stats['Date'], y=stats['index'].apply(lambda x: stats['Confirmed'][x]-stats['Confirmed'][x-1:x].sum()),
                              name='Confirmed',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><b>Daily Change of Confirmed Cases</b>: %{y}',
                                marker=dict(color='#b3e3ff', line = dict(width=0.5, color = '#b3e3ff'))),row=1, col=1)

daily_case_fig.add_trace(go.Bar(x=stats['Date'], y=stats['index'].apply(lambda x: stats['Active'][x]-stats['Active'][x-1:x].sum()), 
                             name='Active',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><b>Daily Change of Active Cases</b>: %{y}',
                               marker=dict(color='#ff0000', line = dict(width=0.5, color = '#ff0000'))),row=1, col=2)

daily_case_fig.add_trace(go.Bar(x=stats['Date'], y=stats['index'].apply(lambda x: stats['Recovered'][x]-stats['Recovered'][x-1:x].sum()), 
                              name='Recovered',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><b>Daily Change of Recovered Cases</b>: %{y}',
                               marker=dict(color='#26ff00', line = dict(width=0.5, color = '#26ff00'))),row=2, col=1)


daily_case_fig.add_trace(go.Bar(x=stats['Date'], y=stats['index'].apply(lambda x: stats['Deaths'][x]-stats['Deaths'][x-1:x].sum()), 
                              name='Deaths',hovertemplate = '<br><b>Date</b>: %{x}'+'<br><b>Daily Change of Deaths</b>: %{y}',
                               marker=dict(color='#000000', line = dict(width=0.5, color = '#000000'))),row=2, col=2)




daily_case_fig.update_xaxes(showticklabels=False)
daily_case_fig = daily_case_fig.update_layout(title_text="Daily Change in Cases of Covid-19", title_x=0.5,title_y=0.97, title_font_size=20,
                            legend=dict(orientation='h',yanchor='top',y=1.15,xanchor='right',x=1))                

options = pd.date_range(start =stats.Date.min(),end = stats.Date.max()).strftime("%#d/%#m/%Y")
index = (0, len(options)-1)
w_date2 = widgets.SelectionRangeSlider(options=options,index=index,description='Dates',orientation='horizontal',
                            layout=widgets.Layout(width='400px'), continuous_update =False)
interactive_box  = widgets.HBox([w_date2])          

def update_axes(dates):
    with daily_case_fig.batch_update():
        for i in range(0, len(daily_case_fig.data)):
            daily_case_fig.data[i].x = pd.date_range(start=dates[0], end=dates[1]).strftime("%#m/%#d/%y")
            daily_case_fig.data[i].y = stats[(stats.Date >= dates[0]) & (stats.Date <= dates[1])]['index'].apply(lambda x: stats[daily_case_fig.data[i].name][x]-stats[daily_case_fig.data[i].name][x-1:x].sum())

out = widgets.interactive_output(update_axes, {'dates': w_date2})
widgets.VBox([interactive_box, daily_case_fig])

```
The figure this time is produced using the ``` FigureWidget ``` object instead, to be able to have interactions with ipywidgets, as we will see below. The y-axis is produced by utilizing the index column, and to use it in the ``` apply ``` function, as an index (duh) to compute the daily difference in cases for each status. I am also using ipywidgets as mentioned above for interactive user data filtering and selection, in this case only to filter the date of the figures (but many more in the dashboard app). This can be done by adding a ``` SelectionRangeSlider ``` widget that outputs a tuple of dates (the start and end date). These dates can then be an input to the function ``` update_axes ``` which updates (guess what?) the axes. Plotlys figures have a very convenient function named ``` batch_update ```, which according to the [documentation](https://plotly.github.io/plotly.py-docs/generated/plotly.html#:~:text=batch_update(),the%20context%20exits.), is <cite>"A context manager that batches up trace and layout assignment operations into a singe plotly_update message that is executed when the context exits."</cite> So I have used it as well to update the figure based on the user selected dates. Finally this is wrapped in an interactive output using ipywidgets ``` interactive_output ``` function to get the interactive figure. All the interactive graphs that have these selection/filtering figures are produced using the same workflow.
- First create a figure
- Then create the widgets (adding options from the data/figure)
- Writing the update logic in a callable function
- Finally wrapping the widgets, update logic and figure in a final output widget.

I have added several other interactive figures in the dashboard app that I will not discuss here. But you can definitely check out the code for them in the Github repository linked above, and interact with the figures in the app also linked above.  

The next analysis I wanted to work on is trying to predict the number of cases per status (Confirmed, Recovered, Active, Deaths). I opted for Linear regression to conduct such analysis. If you haven't read my blog post on linear regression read it [here](https://arebimohammed.github.io/2020/03/10/Linear-Regression-All-you-need-to-know.html). The blog posts tries to cover most of the details of linear regression, but also includes polynomial regression, which is linear regression with polynomial features, It's also possible to think of it as a linear regression with a feature space mapping (aka a polynomial kernel). I will try to dedicate a seperate post for polynomial features and polynomial regression, but all you need to know ahead here is that the time series data of the Covid-19 cases is not linear, it can have various shapes, particularly for the daily changes per status data, the others are fairly linear especially the 'Confirmed' and 'Deaths', but of course they are if we are looking at the cumulative count, as seen below (click the 'Confirmed' or 'Deaths' button). But if you look at the daily changes data (also by clicking their respective buttons), it is not that linear. 

<iframe src= "/posts/COVID19- An Interactive Analysis/stats_fig.html" height="525" width="100%"></iframe>


The code below shows how Polynomial Regression can be done using scikit-learns [``` PolynomialFeatures ```](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html) 
```python
stats['Confirmed Daily Change'] = stats['index'].apply(lambda x: stats['Confirmed'][x]-stats['Confirmed'][x-1:x].sum())
stats['Deaths Daily Change'] = stats['index'].apply(lambda x: stats['Deaths'][x]-stats['Deaths'][x-1:x].sum())
stats['Recovered Daily Change'] = stats['index'].apply(lambda x: stats['Recovered'][x]-stats['Recovered'][x-1:x].sum())
stats['Active Daily Change'] = stats['index'].apply(lambda x: stats['Active'][x]-stats['Active'][x-1:x].sum())

days = np.array(stats[['index']]).reshape(-1, 1)
days_idx = []
for i in range(len(days)+30):
    days_idx = days_idx+[[i]]

df_prediction = pd.DataFrame(columns=['Index', 'Confirmed Prediction', 'Deaths Prediction', 'Recovered Prediction', 'Active Prediction', 
                                      'Confirmed Daily Change Prediction', 'Deaths Daily Change Prediction', 
                                      'Recovered Daily Change Prediction', 'Active Daily Change Prediction'])
df_prediction['Index'] = np.array(days_idx).flatten() # could use pandas.core.common.flatten as well


for col in stats.columns[2:]:

    y = np.array(stats[col]).reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(days, y, test_size=0.1, shuffle=False)

    mae, mse, R2 = [], [], []
    for j in range(1,10):
        #creating the model
        poly = PolynomialFeatures(degree=j)
        X_train_poly = poly.fit_transform(X_train)
        
        
        regr_poly = LinearRegression()
        regr_poly.fit(X_train_poly, y_train)
        
        y_pred_poly = regr_poly.predict(poly.fit_transform(X_test))
        mae.append(mean_absolute_error(y_test, y_pred_poly))
        mse.append(mean_squared_error(y_test,y_pred_poly))
        R2.append(r2_score(y_test , y_pred_poly))
        
    deg = mse.index(min(mse))+1 # Get best degree with lowest Mean Squared Error
    print(f"Best Polynomial Degree for feature '{col}' is {deg}")

    poly = PolynomialFeatures(degree=deg)
    X_train_poly = poly.fit_transform(X_train)

    regr_poly = LinearRegression()
    regr_poly.fit(X_train_poly, y_train)
    col_name = col+' Prediction'
    df_prediction[col_name] = np.array(regr_poly.predict(poly.fit_transform(days_idx))).flatten()
```

    Best Polynomial Degree for feature 'Confirmed' is 9
    Best Polynomial Degree for feature 'Deaths' is 1
    Best Polynomial Degree for feature 'Recovered' is 2
    Best Polynomial Degree for feature 'Active' is 4
    Best Polynomial Degree for feature 'Confirmed Daily Change' is 8
    Best Polynomial Degree for feature 'Deaths Daily Change' is 7
    Best Polynomial Degree for feature 'Recovered Daily Change' is 3
    Best Polynomial Degree for feature 'Active Daily Change' is 2

First I added the daily change to the dataframe and then I computed our input feature which is the days. After that we simply run a for loop to try different degrees of Polynomials for the different Covid-19 status. The results for the best Polynomial per status are printed, the results are obtained by finding the lowest [Mean Squared Error](https://en.wikipedia.org/wiki/Mean_squared_error) and finally the prediction results are added to a dataframe (```df_prediction```) for visualization. 

```python
prediction_fig = go.Figure()

color_pred_dict = {
  "Confirmed": ['#b3e3ff', 'Confirmed', 'Predicted Confirmed','#357387'],
  "Deaths": ['#000000', 'Deaths', 'Predicted Deaths','#62b5d1'],
  "Recovered": ['#26ff00', 'Recovered', 'Predicted Recovered','#00c491'],
  "Active": ['#ff0000', 'Active', 'Predicted Active', '#fa4d76'],
  "Confirmed Daily Change": ['#141011', 'Confirmed Daily Change', 'Predicted Confirmed Daily Change','#5e5e5e'],
  "Deaths Daily Change": ['#303030', 'Deaths Daily Change','Predicted Deaths Daily Change', '#8c8b8b'],
  "Recovered Daily Change": ['#363636', 'Recovered Daily Change', 'Predicted Recovered Daily Change', '#969595'],
  "Active Daily Change": ['#1a1a1a', 'Active Daily Change', 'Predicted Active Daily Change','#c2c2c2']
    }

for col in stats.columns[2:]:
    
    col_pred = col+' Prediction'
    prediction_fig.add_trace(go.Scatter(x=np.array(days).flatten(), y=stats[col],
                                       line=dict(color=color_pred_dict[col][0]), name = color_pred_dict[col][1],
                                       hovertemplate ='<br><b>Day Number</b>: %{x}'+'<br><b>No.of Cases </b>:'+'%{y}'))
    
    prediction_fig.add_trace(go.Scatter(x=np.array(days_idx).flatten(), y=df_prediction[col_pred],
                                       line=dict(dash="longdash", color=color_pred_dict[col][3]),name = color_pred_dict[col][2],
                                       hovertemplate ='<br><b>Day Number</b>: %{x}'+'<br><b>Predicted No.of Cases </b>:'+'%{y}'))

    
    
    
prediction_fig.update_layout(
    updatemenus=[
        dict(
        buttons=list(
            [dict(label = 'All',
                  method = 'update',
                  args = [{'visible': [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]},
                          {'title': 'All Cases',
                           'showlegend':True}]),
             dict(label = 'Confirmed',
                  method = 'update',
                  args = [{'visible': [True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False]},
                          {'title': 'Confirmed Cases',
                           'showlegend':True}]),
             dict(label = 'Deaths',
                  method = 'update',
                  args = [{'visible': [False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False]},
                          {'title': 'Deaths Cases',
                           'showlegend':True}]),
             dict(label = 'Recovered',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False]},
                          {'title': 'Recovered Cases',
                           'showlegend':True}]),
             dict(label = 'Active',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False]},
                          {'title': 'Active Cases',
                           'showlegend':True}]),
             dict(label = 'Confirmed Daily Change',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False]},
                          {'title': 'Daily Change Cases',
                           'showlegend':True}]),
            dict(label = 'Deaths Daily Change',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, False, False, False, False, True, True, False, False, False, False]},
                          {'title': 'Daily Change Cases',
                           'showlegend':True}]),
            dict(label = 'Recovered Daily Change',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, False]},
                          {'title': 'Daily Change Cases',
                           'showlegend':True}]),
            dict(label = 'Active Daily Change',
                  method = 'update',
                  args = [{'visible': [False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True]},
                          {'title': 'Daily Change Cases',
                           'showlegend':True}]),
            ]),
             type = "buttons",
             direction="down",
             showactive=True,
             x=-0.5,
             xanchor="left",
             y=1.1,
             yanchor="top"
        )
    ])


prediction_fig.update_xaxes(showticklabels=False)
prediction_fig.update_layout(
    title_text="Prediction for Covid19 Cases", title_x=0.5, title_font_size=20,
                            legend=dict(orientation='h',yanchor='top',y=1.12,xanchor='right',x=1),
                            xaxis_title="Number of Days", yaxis_title="Count")
prediction_fig.show()
```

<iframe src= "/posts/COVID19- An Interactive Analysis/prediction_fig.html" height="525" width="100%"></iframe>

The results may not look so great. But it did a fairly good job for a quick implementation, other methods might perform much better for example a recurrent neural network with LSTM or GRU layers, or even a 1-dimensional convolutional neural network. These options might be better for time-series data such as Covid-19 cases. XGBoost, or other ensemble boosted methods can also perform very well on time series data. Given that good feature engineering is conducted to get lagging/leading values, rolling averages etc. (I will write about these different topics in a seperate blog post)

I also created a World Map using plotlys ```Choropleth``` figure object, to get the status stats per country in a map. I also added an interactive figure with widgets to get different Covid-19 rates per country and filter by dates. I then used plotly to create animations to visualize the development of Covid-19 cases for the top 20 countries with the highest number of cases. All these figures can be found in the app linked above.

The final analysis I conducted was to see if a measure introduced by the government to reduce/limit the number of Covid-19 infection is effective or not. As I am living in Germany I only did this analysis for Germany, but it can of course be reproduced for other countries as well. 

```python
# Filtering just for Germany 
df_measures.rename(columns={'DATE_IMPLEMENTED': 'date'}, inplace = True)
df_stats_de = df_stats[(df_stats.location == 'Germany')].reset_index(drop=True)
df_measures_de = df_measures[(df_measures.COUNTRY == 'Germany')].reset_index(drop=True)
df_stats_de.date = pd.to_datetime(df_stats_de.date)
```

Firstly, I filtered the datasets just for Germany (as mentioned), had a look at the data types and data in general, removed unnecessary columns, dropped NA rows and finally merged the two datasets for analysis.

```python
df_all_de = pd.merge(df_measures_de, df_stats_de, how="outer", on="date")
```
I wanted to get the week of Covid-19 since its start to add to the dataframe for further analysis as we will see later. I didn't want the week of the year, but that can also be used if conducting the analysis for just one year, but I want this analysis to be reproducible anytime it is ran, maybe if the virus is still here after several years, hopefully not (UPDATE: It is still here after 2 years 😕), then the analysis would still be feasible and it is done with this piece of code:

```python
#Function to get all pairs of weeks (see below)
def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return list(zip(a, b))


start,end = df_all_de.date.min(), df_all_de.date.max()
weeks = pd.date_range(start=start, end =end, freq='W-MON') # Every Monday since startof Covid 

#Week dictionary to lookup week number since Covid starts
week_dict = {}
for i, week in enumerate(weeks):
    week_dict[week] = i+1

week_pairs = pairwise(weeks)

for pairs in week_pairs:
   start,end = pairs[0] ,pairs[1]
   indexes = df_all_de.date[df_all_de.date.between(start,end)].index.values #Getting all dataframe indexes betweenin this week 
   week_no = week_dict[start]
   df_all_de.loc[indexes,'Week'] = week_no # Creating new column Week to the week of covid based on date
```

I first used the pairwise function to get consecutive pairs of weeks for example, Week 1 and 2, Week 2 and 3, Week 3 and 4 etc.
I then got the first and last dates in the dataframe to create a weekly date range using Pandas ```date_range``` and with frequency ```W-MON```, which is weekly every Monday. This week range is then used to create a dictionary that maps these weeks to their number since the start of Covid. Finally we loop through these pairs of weeks and check if the date is between these pairs of weeks then we assign it the week number of the start week.

We can then get the average of new or total cases per week for Germany by doing this

```python
df_week_avg = df_all_de.groupby('Week',as_index=False).agg({'total_cases':'mean','new_cases':'mean'})
new_week_fig = px.bar(df_week_avg, x= 'Week', y ='new_cases')
new_week_fig = new_week_fig.update_layout(title_text = 'Average of New Cases Per Week of Covid-19',title_x = 0.5)
new_week_fig.show()
```

<iframe src= "/posts/COVID19- An Interactive Analysis/new_week.html" height="525" width="100%"></iframe>

We can then see these sudden rises and falls of new cases in certain weeks, we will try to see why. I chose only a subset of measures to analyse which I think are the mostly employed measures, these are:
- Schools closure,
- Limit public gatherings
- Isolation and quarantine policies
- Visa restrictions
- Closure of businesses and public services

I then added (annotated) when these measures where first introduced to the previous plot:
```python
sub_measures = ['Schools closure','Limit public gatherings','Isolation and quarantine policies',
'Visa restrictions','Closure of businesses and public services']

first_measures = df_all_de[df_all_de.MEASURE.isin(sub_measures)].drop_duplicates(subset=['MEASURE'],keep='first')[['MEASURE','Week']]

position_dict = {'Schools closure':[-100,-30],'Limit public gatherings':[-90,-30],'Requirement to wear protective gear in public':[-70,80],
'Isolation and quarantine policies':[-70,-10],'Visa restrictions':[-60,-5],'Closure of businesses and public services':[-40,10]}
for idx,row in first_measures.iterrows():
    text = row['MEASURE']
    week_no = row['Week']
    y_val = df_week_avg[df_week_avg.Week == week_no]['new_cases'].values[0] + 100

    new_week_fig.add_annotation(x=week_no, y=y_val, text=text, font_size =10,
                                hoverlabel=dict(bgcolor='#e5ff3b'),hovertext=text,ay=position_dict[text][0],ax=position_dict[text][1])

new_week_fig.show()
```

<iframe src= "/posts/COVID19- An Interactive Analysis/new_week_annot.html" height="525" width="100%"></iframe>

And luckily with plotly you can slightly zoom in to see when these measures where first introduced. When they were first introduced only the Visa Restrictions measure had weeks after it with a decreased average of new cases.

Next I did a naive analysis to see if the measures mentioned are effective or not. This is done by looking at the average of new cases 4 weeks before the measure was introduced and the average of new cases 4 week after the measure was introduced (for everytime it was introduced/employed). This method has a flaw which we will discuss in just a second but first lets see how we can do that:

```python
# Looking at only 5 measures to check for their effectiveness 
# For every week the measure was introduced/employed we calculate the mean of new cases 4 weeks before and 4 weeks after and during
# The means are then averaged to get an overall measure of effectiveness then they are compared to see if the measure was effective or not

df_imp_measures = df_all_de[df_all_de.MEASURE.isin(sub_measures)].sort_values(by='date')[['MEASURE','date','Week','Week_of_year']].reset_index(drop=True)
df_week_sum = df_all_de.groupby('Week',as_index =False).agg({'new_cases':'sum'}).sort_values(by='Week')
measure_dict = {}
for measure in sub_measures:
    dftmp = df_imp_measures[df_imp_measures.MEASURE == measure].drop_duplicates(subset=['MEASURE','Week'])
    mean_before,mean_after = [],[]
    for week in dftmp.Week: 
        mean_before.append(df_week_sum[df_week_sum.Week.isin([week-4,(week-4)+1,(week-4)+2,(week-4)+3])].new_cases.sum()/4)
        mean_after.append(df_week_sum[df_week_sum.Week.isin([week,week+1,week+2,week+3])].new_cases.sum()/4)
    measure_dict[measure] = {'mean_before':np.mean(mean_before), 'mean_after':np.mean(mean_after)}
  
result_df = pd.DataFrame(measure_dict).T.reset_index()
result_df.columns = ['MEASURE','Average Infection 4 weeks Before', 'Average Infection 4 weeks After']
result_df['Effective'] = np.where((result_df['Average Infection 4 weeks Before'] > result_df['Average Infection 4 weeks After']),1,0)
```

We can then see the results below and that the out of these 5 measures 3 were effective, Schools closure, Visa restrictions and Closure of businesses and public services. But like I said above this analysis has a flaw because even if the rise has already occurred, a countermeasure is usually implemented in response to it. If the measure was implemented during a period of rapid growth in new infections and only took effect after 4 weeks, it is evident that the number of new infections 4 weeks before the measure was implemented will be lower than the number of new infections 4 weeks after it was implemented. 

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MEASURE</th>
      <th>Average Infection 4 weeks Before</th>
      <th>Average Infection 4 weeks After</th>
      <th>Effective</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Schools closure</td>
      <td>34754.687500</td>
      <td>26264.750000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Limit public gatherings</td>
      <td>24231.833333</td>
      <td>31325.944444</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Isolation and quarantine policies</td>
      <td>12751.886364</td>
      <td>23022.522727</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Visa restrictions</td>
      <td>18329.333333</td>
      <td>17110.750000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Closure of businesses and public services</td>
      <td>23464.150000</td>
      <td>23450.275000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>

With that being said we know from [Linear Regression](https://arebimohammed.github.io/2020/03/10/Linear-Regression-All-you-need-to-know.html) that it is trying to estimate the coefficients of the input variables, these coefficients help us determine the relationship between the input variable (independent variable/predictor/feature) and the output variable (dependent variable/response/target). So I have used Linear Regression to estimate the coefficient for the 'effectiveness' of the measure as we will see below.

```python
# Reindexing the average cases to calculate weekly trend (difference between subsequent weeks)

cases = df_week_avg['new_cases']
cases.loc[-1] = 0
cases.index = cases.index +1
cases.sort_index(inplace =  True)

weekly_trend = cases - cases.shift(1)
weekly_trend.fillna(0, inplace=True)

# Using these 4 features + 1 feature representing the measure for a particular week,
# for linear regression to predict average new cases 

df_reg = pd.DataFrame()
df_reg['New Cases'] = cases
df_reg['Weekly Trend'] = weekly_trend
df_reg['Week'] = df_week_avg['Week'].copy()
df_reg['Week_sqr'] = df_reg['Week'] **2
df_reg.drop([len(df_reg)-1],inplace=True)

df_school = df_reg.copy()
df_school['Schools Closure'] = np.where((df_school.Week.isin(df_imp_measures[df_imp_measures.MEASURE == 'Schools closure'].Week.values)),True,False)

X_school = df_school.values
y = df_week_avg['new_cases']
lin_reg_school = LinearRegression().fit(X_school,y)

df_limit = df_reg.copy()
df_limit['Limit public gatherings'] = np.where((df_school.Week.isin(df_imp_measures[df_imp_measures.MEASURE == 'Limit public gatherings'].Week.values)),True,False)
X_limit = df_limit.values
lin_reg_limit = LinearRegression().fit(X_limit,y)


df_isolation = df_reg.copy()
df_isolation['Isolation and quarantine policies'] = np.where((df_isolation.Week.isin(df_imp_measures[df_imp_measures.MEASURE == 'Isolation and quarantine policies'].Week.values)),True,False)
X_isolation = df_isolation.values
lin_reg_isolation = LinearRegression().fit(X_isolation,y)

df_visa = df_reg.copy()
df_visa['Limit public gatherings'] = np.where((df_visa.Week.isin(df_imp_measures[df_imp_measures.MEASURE == 'Visa restrictions'].Week.values)),True,False)
X_visa = df_visa.values
lin_reg_visa = LinearRegression().fit(X_visa,y)

df_closure = df_reg.copy()
df_closure['Limit public gatherings'] = np.where((df_closure.Week.isin(df_imp_measures[df_imp_measures.MEASURE == 'Closure of businesses and public services'].Week.values)),True,False)
X_closure = df_closure.values
lin_reg_closure = LinearRegression().fit(X_closure,y)

```

I first get the average new cases per week and reindex them to calculate the weekly trend in average new cases by substracting subsequent weeks from previous weeks. I have also used the Week and Week squared as input features as well. Finally the input feature we are mostly interested in, is the measure feature which is different for every measure hence the different input variables (``` X_<measure>```) for the different measures. This will give us this Linear Regression equation:

$$ \hat{y}_{t} = {\alpha} + {\beta}_{0}N_{t-1} + {\beta}_{1}W + {\beta}_{2}t + {\beta}_{3}t^2 + {\beta}_{4}M $$

Where: 
- $ {\alpha} $ is the y-intercept
- $ {\beta}_{i} $ is for the coefficients to be estimated
- $ N_{t-1} $ is for the new infections 1 week ago 
- $ W $ is for the weekly trend $ (N_{t-1}  –  N_{t-2}) $
- $ t $ is for the week
- $ t^2 $ is for the week squared
- $ M $ is for the measure dummy variable

Then how can we interpret this coefficient? It is fairly simple, we can interpret this coefficient by its sign if it is negative then it is reducing the weekly average of new cases, if it positive it is either not reducing it or the average is still on the rise. Here are the results:

```python
print("Schools Closure Coefficient: ", lin_reg_school.coef_[-1])
print("Limit public gatherings Coefficient: ", lin_reg_limit.coef_[-1])
print("Isolation and quarantine policies: ", lin_reg_isolation.coef_[-1])
print("Visa restrictions Coefficient: ", lin_reg_visa.coef_[-1])
print("Closure of businesses and public services Coefficient: ", lin_reg_closure.coef_[-1])
```
      Schools Closure Coefficient:  -136.60953972008906
      Limit public gatherings Coefficient:  178.78724726294902
      Isolation and quarantine policies:  272.8940135554292
      Visa restrictions Coefficient:  -574.2499316102472
      Closure of businesses and public services Coefficient:  -97.1873527294096

And coincidently the naive approach had the same results as the Linear Regression approach the same 3 measures are effective here as well. 

We can also use ```statsmodels``` to estimate the coefficients like so:
```python
import statsmodels.api as sm

modeltst = sm.OLS(y, sm.add_constant(np.array(df_school, dtype =float))).fit()
modeltst.summary()
```

We add a constant to get the intercept of the Linear Regression ($ {\alpha} $), the results are very similar to scikit-learns ``` LinearRegression ``` class which uses scipys [`linalg.lstsq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lstsq.html) in the backend.


Thank you very much for reading all the way through, and I hope you enjoyed the article and hopefully we can all defeat this virus together, don't forget to check out the interactive dashboard app linked above. You can of course write me if you have any questions or just want to chat :), all my contact details are at the end of the page. See you in the next article and, Stay Safe! 
