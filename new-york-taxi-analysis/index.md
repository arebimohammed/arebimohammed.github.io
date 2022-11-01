# Digging Deep Into the New York Taxi Dataset

<!--more-->
<span style="font-size:1.125rem">

This article goes in detail through one of the data science projects I worked on, the New York Taxi dataset which is made available by the [New York City Taxi and Limousine Commission (TLC)](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page). New York is partly known for its yellow taxis (even though there are green taxis we'll focus on only the yellow taxis) and there are millions of taxi rides taken every month as New York is one of the most populous cities in the United States which makes for a very interesting dataset to dive into as there are a lot of analysis possibilities (Geospatial Data Analysis, Time Analysis, Taxi Ride Price Analysis, Traffic Analysis, etc.). 

## Introduction 
<hr>

When seen as a whole, the precise trip-level data is more than simply a long list of taxi pickup and drop-off locations: it tells a tale of New York. How much do taxis make each trip if they can only travel 30 or 50 kilometres per trip, as opposed to taxis with no limit? Is it more common for taxi journeys to begin in the north or south? What is the average speed of New York traffic (with regards to taxis)? How does the speed of traffic in New York change during the day? All of these questions, and many more, are addressed in the dataset. And we will answer them as well as others. 

## The Process

The usual data science process starts with business and feasability study, data collection, data cleaning, exploratory data analysis (EDA) and sometimes modelling. Of course there is more to this simple process abstraction, such as model deployment and monitoring but in our case these are skipped as we will focus on the data collection, cleaning and exploratory analysis. We'll now go through the different phases one by one.

## Data Collection

The data is taken from this link: [NYC Yellow Taxi](http://www.andresmh.com/nyctaxitrips/) provided by Andrés Monroy. The link contains data contains compressed files of yellow taxi rides for the year 2013. There are two compressed files one named ```trip_data``` is for trip specific data such as the distance, time, etc while the other one named ```trip_fare``` contains data regarding the trip fare (hence the name), surcharge, tip, etc. Both files divide the data in 12 CSV files (one for each month). The data is also available in Kaggle for the trip data and trip fare [here](https://www.kaggle.com/c/nyc-taxi-trip-duration) and [here](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction) respectively. Kaggle wasn't used as it doesn't contain all the data (even though we are also not using all of it, more below) and for the fare data the data is shuffled between several years. The data dictionary/description can be found here: [Data Dictionary](https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf), note that not all columns are in the data dictionary from the data source somehow but looking further in the analysis they are all made clear and are mostly Intuitive.

After downloading the compressed files and unzipping them they are about 5.4 GB in total, which is quite hefty for my local machine as I am doing the analysis locally on a small machine and not on the cloud. This is where data sampling comes into play. Reservoir random sampling was used to sample 150K records/trips per month of 2013. The code for this is shown here and is also available in the projects [github repo](https://github.com/arebimohammed/New-York-Yellow-Taxi-Data-Analysis-and-ML-Project).

```python
import pandas as pd
import numpy as np
import os
from itertools import islice
from io import StringIO
from tqdm.notebook import tqdm

pd.set_option("display.max_columns",None)
np.random.seed(42)


def reservoir_sample(iterable, random1, random2,random3,random4, k=1):
    
    '''
    From : https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm
    
    '''
    
    iterator = iter(iterable)
    values = list(islice(iterator, k))

    W = np.exp(np.log(random1/k))
    while True:
        skip = int(np.floor(np.log(random2)/np.log(1-W)))
        selection = list(islice(iterator, skip, skip+1))
        if selection:
            values[random3] = selection[0]
            W *= np.exp(np.log(random4)/k)
        else:
            return values

def df_sample(filepath1,filepath2, k):
    
    r1,r2,r3,r4 = np.random.random(), np.random.random() ,np.random.randint(k), np.random.random()
    
    with open(filepath1, 'r') as f1, open(filepath2, 'r') as f2: 
        
        header1 = next(f1)
        header2 = next(f2)
        
        values1 = reservoir_sample(f1,r1,r2,r3,r4,k)  
        values2 = reservoir_sample(f2,r1,r2,r3,r4,k)  
        
        result1 = [header1] + values1
        result2 = [header2] + values2
        
    df1 = pd.read_csv(StringIO(''.join(result1)))
    df2 = pd.read_csv(StringIO(''.join(result2)))
    
    df1 = sample_preprocessing(df1)
    df2 = sample_preprocessing(df2, fare= True) #f2 has to be fare data
    
    return df1,df2


def sample_preprocessing(df_sample, fare= False):
    
    df_sample.columns = df_sample.columns.str.replace(" ","")
    
    if fare:
        df_sample.drop(columns=["hack_license","vendor_id"],inplace= True)  
    
    return df_sample


def reduce_memory(df, verbose=True):
    numerics = ["int8", "int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df

data_file_list = sorted(os.listdir("../Data/NYC Trip Data/"), key = lambda x: int(x.split(".")[0].split("_")[-1]))
data_file_list = [os.path.join("../Data/NYC Trip Data/",file) for file in data_file_list]

fare_file_list = sorted(os.listdir("../Data/NYC Trip Fare/"), key = lambda x: int(x.split(".")[0].split("_")[-1]))
fare_file_list = [os.path.join("../Data/NYC Trip Fare/",file) for file in fare_file_list]

N = 150000 # Number of records per month chosen randomly 
df_data = []
df_fare = []
zipped_data = zip(data_file_list,fare_file_list)
for data_file, fare_file in tqdm(zipped_data, total = len(data_file_list)):
    
    df_data_sample, df_fare_sample = df_sample(data_file,fare_file,N)
    
    df_data.append(df_data_sample)
    df_fare.append(df_fare_sample)

fare = pd.concat(df_fare,axis=0,ignore_index=True)
data = pd.concat(df_data,axis=0,ignore_index=True)
data.drop_duplicates(subset=["medallion","pickup_datetime"],inplace=True)
fare.drop_duplicates(subset=["medallion","pickup_datetime"],inplace=True)

final_df = data.merge(fare, how="inner", on = ["medallion","pickup_datetime"])
final_df["pickup_datetime"] = pd.to_datetime(final_df["pickup_datetime"])
final_df["dropoff_datetime"] = pd.to_datetime(final_df["dropoff_datetime"])

final_df.to_csv("../Data/Pre-processed Data/data.csv")
```
The main sampling is conducted in the ```df_sample``` function which utilizes the reservoir sampling function itself. The main idea of reservoir sampling is explained [here](https://en.wikipedia.org/wiki/Reservoir_sampling) along with the pseudocode for its implementation. What we need to know is that it chooses a random sample (row) without replacement from the CSV file in a single pass over the whole file. Finally the fare and trip data are then joined together on the trips medallion and pickup datatime as they the represent the same trip just different data which gives us the final data to work with. It has about 2 million records (1.8M to be exact).

## Data Cleaning 

In this phase of the project we'll check the data for any inconsistencies, outliers, erroneous values, etc. The cleaning itself is divided into several parts such as location data (pickup, dropoff) cleaning, trip duration cleaning, trip distance cleaning, trip speed cleaning, trip total amount, fare Amount, surcharge, tax, tolls and tip cleaning and categorical columns cleaning. We'll also use this phase to calculate and add some other features/columns as well. Let's get straight to it :heart_eyes:!

### Import Libraries and Data Loading

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime 
import requests
import json
import os
from tqdm.notebook import tqdm
import geopy.distance

from shapely.geometry import Point
import folium
import geopandas as gpd
from geopandas import GeoDataFrame
import geoplot
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.set_option("display.max_columns",None)
np.random.seed(42)


df = pd.read_csv("../Data/Pre-processed Data/data.csv")
df.head()
```

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
      <th>Unnamed: 0</th>
      <th>medallion</th>
      <th>hack_license</th>
      <th>vendor_id</th>
      <th>rate_code</th>
      <th>store_and_fwd_flag</th>
      <th>pickup_datetime</th>
      <th>dropoff_datetime</th>
      <th>passenger_count</th>
      <th>trip_time_in_secs</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>payment_type</th>
      <th>fare_amount</th>
      <th>surcharge</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>89D227B655E5C82AECF13C3F540D4CF4</td>
      <td>BA96DE419E711691B9445D6A6307C170</td>
      <td>CMT</td>
      <td>1</td>
      <td>N</td>
      <td>2013-01-01 15:11:48</td>
      <td>2013-01-01 15:18:10</td>
      <td>4</td>
      <td>382</td>
      <td>1.0</td>
      <td>-73.978165</td>
      <td>40.757977</td>
      <td>-73.989838</td>
      <td>40.751171</td>
      <td>CSH</td>
      <td>6.5</td>
      <td>0.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0BD7C8F5BA12B88E0B67BED28BEA73D8</td>
      <td>9FD8F69F0804BDB5549F40E9DA1BE472</td>
      <td>CMT</td>
      <td>1</td>
      <td>N</td>
      <td>2013-01-06 00:18:35</td>
      <td>2013-01-06 00:22:54</td>
      <td>1</td>
      <td>259</td>
      <td>1.5</td>
      <td>-74.006683</td>
      <td>40.731781</td>
      <td>-73.994499</td>
      <td>40.750660</td>
      <td>CSH</td>
      <td>6.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0BD7C8F5BA12B88E0B67BED28BEA73D8</td>
      <td>9FD8F69F0804BDB5549F40E9DA1BE472</td>
      <td>CMT</td>
      <td>1</td>
      <td>N</td>
      <td>2013-01-05 18:49:41</td>
      <td>2013-01-05 18:54:23</td>
      <td>1</td>
      <td>282</td>
      <td>1.1</td>
      <td>-74.004707</td>
      <td>40.737770</td>
      <td>-74.009834</td>
      <td>40.726002</td>
      <td>CSH</td>
      <td>5.5</td>
      <td>1.0</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>DFD2202EE08F7A8DC9A57B02ACB81FE2</td>
      <td>51EE87E3205C985EF8431D850C786310</td>
      <td>CMT</td>
      <td>1</td>
      <td>N</td>
      <td>2013-01-07 23:54:15</td>
      <td>2013-01-07 23:58:20</td>
      <td>2</td>
      <td>244</td>
      <td>0.7</td>
      <td>-73.974602</td>
      <td>40.759945</td>
      <td>-73.984734</td>
      <td>40.759388</td>
      <td>CSH</td>
      <td>5.0</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>DFD2202EE08F7A8DC9A57B02ACB81FE2</td>
      <td>51EE87E3205C985EF8431D850C786310</td>
      <td>CMT</td>
      <td>1</td>
      <td>N</td>
      <td>2013-01-07 23:25:03</td>
      <td>2013-01-07 23:34:24</td>
      <td>1</td>
      <td>560</td>
      <td>2.1</td>
      <td>-73.976250</td>
      <td>40.748528</td>
      <td>-74.002586</td>
      <td>40.747868</td>
      <td>CSH</td>
      <td>9.5</td>
      <td>0.5</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.5</td>
    </tr>
  </tbody>
</table>
</div>

```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1799998 entries, 0 to 1799997
    Data columns (total 22 columns):
     #   Column              Dtype  
    ---  ------              -----  
     0   Unnamed: 0          int64  
     1   medallion           object 
     2   hack_license        object 
     3   vendor_id           object 
     4   rate_code           int64  
     5   store_and_fwd_flag  object 
     6   pickup_datetime     object 
     7   dropoff_datetime    object 
     8   passenger_count     int64  
     9   trip_time_in_secs   int64  
     10  trip_distance       float64
     11  pickup_longitude    float64
     12  pickup_latitude     float64
     13  dropoff_longitude   float64
     14  dropoff_latitude    float64
     15  payment_type        object 
     16  fare_amount         float64
     17  surcharge           float64
     18  mta_tax             float64
     19  tip_amount          float64
     20  tolls_amount        float64
     21  total_amount        float64
    dtypes: float64(11), int64(4), object(7)
    memory usage: 302.1+ MB
    


```python
df.describe()
```

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
      <th>Unnamed: 0</th>
      <th>rate_code</th>
      <th>passenger_count</th>
      <th>trip_time_in_secs</th>
      <th>trip_distance</th>
      <th>pickup_longitude</th>
      <th>pickup_latitude</th>
      <th>dropoff_longitude</th>
      <th>dropoff_latitude</th>
      <th>fare_amount</th>
      <th>surcharge</th>
      <th>mta_tax</th>
      <th>tip_amount</th>
      <th>tolls_amount</th>
      <th>total_amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799930e+06</td>
      <td>1.799930e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
      <td>1.799998e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>8.999985e+05</td>
      <td>1.035788e+00</td>
      <td>2.020107e+00</td>
      <td>7.653680e+02</td>
      <td>6.113157e+00</td>
      <td>-7.286966e+01</td>
      <td>4.013907e+01</td>
      <td>-7.280162e+01</td>
      <td>4.009319e+01</td>
      <td>1.243297e+01</td>
      <td>2.739124e-01</td>
      <td>4.982975e-01</td>
      <td>1.338854e+00</td>
      <td>2.654285e-01</td>
      <td>1.480971e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.196148e+05</td>
      <td>2.800236e-01</td>
      <td>1.660724e+00</td>
      <td>1.153058e+04</td>
      <td>3.744392e+03</td>
      <td>9.001916e+00</td>
      <td>6.591034e+00</td>
      <td>9.397887e+00</td>
      <td>7.702412e+00</td>
      <td>4.701487e+01</td>
      <td>3.383309e-01</td>
      <td>2.912650e-02</td>
      <td>2.238275e+00</td>
      <td>1.239370e+00</td>
      <td>4.750405e+01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-9.885000e+01</td>
      <td>-3.114279e+03</td>
      <td>-1.145424e+03</td>
      <td>-3.113789e+03</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>4.499992e+05</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>3.600000e+02</td>
      <td>1.080000e+00</td>
      <td>-7.399219e+01</td>
      <td>4.073427e+01</td>
      <td>-7.399150e+01</td>
      <td>4.073344e+01</td>
      <td>6.500000e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>8.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>8.999985e+05</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.000000e+02</td>
      <td>1.820000e+00</td>
      <td>-7.398193e+01</td>
      <td>4.075231e+01</td>
      <td>-7.398040e+01</td>
      <td>4.075279e+01</td>
      <td>9.500000e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e-01</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.100000e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.349998e+06</td>
      <td>1.000000e+00</td>
      <td>2.000000e+00</td>
      <td>9.600000e+02</td>
      <td>3.300000e+00</td>
      <td>-7.396665e+01</td>
      <td>4.076756e+01</td>
      <td>-7.396390e+01</td>
      <td>4.076763e+01</td>
      <td>1.400000e+01</td>
      <td>5.000000e-01</td>
      <td>5.000000e-01</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.650000e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.799997e+06</td>
      <td>9.000000e+00</td>
      <td>2.080000e+02</td>
      <td>4.293410e+06</td>
      <td>5.005073e+06</td>
      <td>4.082401e+01</td>
      <td>2.234990e+03</td>
      <td>1.428738e+03</td>
      <td>6.527231e+02</td>
      <td>6.155086e+04</td>
      <td>5.830000e+00</td>
      <td>5.000000e-01</td>
      <td>5.586200e+02</td>
      <td>9.713000e+01</td>
      <td>6.155303e+04</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    Unnamed: 0                  0
    medallion                   0
    hack_license                0
    vendor_id                   0
    rate_code                   0
    store_and_fwd_flag    1493331
    pickup_datetime             0
    dropoff_datetime            0
    passenger_count             0
    trip_time_in_secs           0
    trip_distance               0
    pickup_longitude            0
    pickup_latitude             0
    dropoff_longitude          68
    dropoff_latitude           68
    payment_type                0
    fare_amount                 0
    surcharge                   0
    mta_tax                     0
    tip_amount                  0
    tolls_amount                0
    total_amount                0
    dtype: int64

After getting a quick overview of the data let's get into the actual cleaning

### Convert dates to datetime type

```python
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
df.drop(df.columns[0], axis =1, inplace = True)
```

### Location Data Cleaning 

```python
print(min(df.pickup_longitude.min(), df.dropoff_longitude.min()), \
max(df.pickup_longitude.max(), df.dropoff_longitude.max()))

print(min(df.pickup_latitude.min(), df.dropoff_latitude.min()), \
max(df.pickup_latitude.max(), df.dropoff_latitude.max()))
```

    -1145.4238 1428.7383
    -3114.2788 2234.9895

We see some very unexpected and erroneous latitude and longitude (location) data, let's clean that up.

#### Trips only within New York City


```python
NYBB = (-74.4505, -72.8478, 40.2734, 41.256) # https://boundingbox.klokantech.com/

before = df.shape[0]
print(f'Rows/Size Before: {before}')

df = df[(df.pickup_longitude >= NYBB[0]) & (df.pickup_longitude <= NYBB[1]) & \
           (df.pickup_latitude >= NYBB[2]) & (df.pickup_latitude <= NYBB[3]) & \
           (df.dropoff_longitude >= NYBB[0]) & (df.dropoff_longitude <= NYBB[1]) & \
           (df.dropoff_latitude >= NYBB[2]) & (df.dropoff_latitude <= NYBB[3])].reset_index(drop=True)

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1799998
    Rows/Size After: 1770605
    Removed: 29393

We remove 29393 incorrect location records.

### Trip Dates and Duration Cleaning

```python
(df["dropoff_datetime"] - df["pickup_datetime"]).describe()
```




    count                      1770605
    mean     0 days 00:12:17.623533763
    std      0 days 00:11:57.343927277
    min                0 days 00:00:00
    25%                0 days 00:06:00
    50%                0 days 00:10:00
    75%                0 days 00:16:00
    max                7 days 01:13:13
    dtype: object

We're seeing weird trip durations of 7 days and 0 seconds here.

```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

invalid_trip_time_idx = np.where((df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds().astype(np.int64) != df["trip_time_in_secs"])[0]
trip_time_in_secs_ser = (df["dropoff_datetime"] - df["pickup_datetime"]).dt.total_seconds().astype(np.int64)
df.iloc[invalid_trip_time_idx, df.columns.tolist().index("trip_time_in_secs")] = trip_time_in_secs_ser.iloc[invalid_trip_time_idx]

invalid_duration_idx = np.where(((df["dropoff_datetime"] - df["pickup_datetime"]) == datetime.timedelta()) | ((df["dropoff_datetime"] - df["pickup_datetime"]) < datetime.timedelta()))[0]
df.drop(invalid_duration_idx, inplace=True)
df.reset_index(drop=True, inplace=True)

df.insert(loc=df.columns.tolist().index("trip_time_in_secs")+1,column="trip_time_in_min",value = (df["dropoff_datetime"] - df["pickup_datetime"]).astype("timedelta64[m]"))

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1770605
    Rows/Size After: 1768663
    Removed: 1942

Here we correct the invalid trip duration by calculating it from the pickup and dropoff datetime as well as remove the negative and trips with zero seconds. We do that using numpy's ```np.where``` to get all the rows where the duration is invalid and correct that in line 6. We finally add the trip duration in minutes as well.

```python
(df["dropoff_datetime"] - df["pickup_datetime"]).describe()
```




    count                      1768663
    mean     0 days 00:12:18.433447751
    std      0 days 00:11:57.320887829
    min                0 days 00:00:01
    25%                0 days 00:06:00
    50%                0 days 00:10:00
    75%                0 days 00:16:00
    max                7 days 01:13:13
    dtype: object

We corrected the incorrect trip duration, removed the zero and negative duration trips but we still see very low trip durations of 1 second and very high trip duration of 7 days. We can inspect the percentiles of the trip duration distribution to get an idea where to cut the maximum and minimum values.

```python
trip_duration_percentiles = df["trip_time_in_secs"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
```


```python
trip_duration_percentiles.loc[0.00:0.1]
```




    0.00      1.0
    0.01    107.0
    0.02    120.0
    0.03    120.0
    0.04    180.0
    0.05    180.0
    0.06    180.0
    0.07    180.0
    0.08    212.0
    0.09    240.0
    0.10    240.0
    Name: trip_time_in_secs, dtype: float64

```python
trip_duration_percentiles.loc[0.90:]
```




    0.90      1380.0
    0.91      1440.0
    0.92      1500.0
    0.93      1560.0
    0.94      1680.0
    0.95      1740.0
    0.96      1860.0
    0.97      2040.0
    0.98      2280.0
    0.99      2700.0
    1.00    609193.0
    Name: trip_time_in_secs, dtype: float64

```python
df_short_duration.trip_distance.describe()
```




    count    337.000000
    mean       0.763501
    std        3.118799
    min        0.000000
    25%        0.000000
    50%        0.000000
    75%        0.000000
    max       31.700000
    Name: trip_distance, dtype: float64




```python
fig_short = px.scatter(df_short_duration, x="pickup_longitude",y="pickup_latitude")
fig_short = fig_short.update_traces(marker=dict(size= 1),name="Short Duration Pickups")
fig_short.data[0].showlegend = True
fig_short = fig_short.add_trace(go.Scatter(x=df_short_duration["dropoff_longitude"], 
                         y=df_short_duration["dropoff_latitude"], mode='markers', marker=dict(color="red", size=1),
                                          name="Short Duration Dropoffs", opacity=0.5))
fig_short = fig_short.update_layout(xaxis_title ="Longitude", yaxis_title = "Latitude")

fig_short.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/fig1.html" height="525" width="100%"></iframe>

We're seeing the outline of New York so these weird durations might just be problems with the meter, we'll drop them.

```python
df_large_duration = df[(df.trip_time_in_secs >= 43200)] # 12 hours max, See : https://www1.nyc.gov/assets/tlc/downloads/pdf/rule_book_current_chapter_58.pdf#page=34
```


```python
df_large_duration.trip_distance.describe()
```




    count         3.000000
    mean      96236.666667
    std      166681.513487
    min           0.200000
    25%           3.050000
    50%           5.900000
    75%      144354.900000
    max      288703.900000
    Name: trip_distance, dtype: float64




```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

invalid_duration_idx = df[(df.trip_time_in_secs.between(1,5)) | (df.trip_time_in_secs >= 43200)].index.tolist()

df.drop(invalid_duration_idx, inplace=True)
df.reset_index(drop=True, inplace= True)

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1768663
    Rows/Size After: 1768323
    Removed: 340

340 records have been removed with erroneous trip durations.

### Trip Distance Cleaning

We follow the same percentile inspection procedure along with the describe function to check for outliers and invalid records.

```python
df.trip_distance.describe().apply(lambda x: '%.5f' % x)
```




    count    1768323.00000
    mean           6.02009
    std         3771.53508
    min            0.00000
    25%            1.09000
    50%            1.83000
    75%            3.31000
    max      5005073.00000
    Name: trip_distance, dtype: object




```python
trip_distance_percentiles = df["trip_distance"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
```


```python
trip_distance_percentiles.loc[0.00:0.1]
```




    0.00    0.00
    0.01    0.24
    0.02    0.38
    0.03    0.44
    0.04    0.50
    0.05    0.53
    0.06    0.58
    0.07    0.60
    0.08    0.64
    0.09    0.68
    0.10    0.70
    Name: trip_distance, dtype: float64




```python
trip_distance_percentiles.loc[0.90:]
```




    0.90          6.48
    0.91          7.02
    0.92          7.72
    0.93          8.52
    0.94          9.30
    0.95         10.11
    0.96         11.09
    0.97         12.62
    0.98         16.48
    0.99         18.31
    1.00    5005073.00
    Name: trip_distance, dtype: float64




```python
px.histogram(df.sample(500000), x= "trip_distance") #obvious outliers (ran several times due to sampling)
```

<iframe src= "/posts/New-York-Taxi-Analysis/fig2.html" height="525" width="100%"></iframe>

As the comment above states there are obvious outliers.

```python
df.trip_distance.where(lambda x: x > df.trip_distance.min()).min() # Second smallest distance is 0.01 miles or 16 meters
```




    0.01




```python
df.trip_distance.nsmallest(500000).unique()
```




    array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1 ,
           0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2 , 0.21,
           0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3 , 0.31, 0.32,
           0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4 , 0.41, 0.42, 0.43,
           0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5 , 0.51, 0.52, 0.53, 0.54,
           0.55, 0.56, 0.57, 0.58, 0.59, 0.6 , 0.61, 0.62, 0.63, 0.64, 0.65,
           0.66, 0.67, 0.68, 0.69, 0.7 , 0.71, 0.72, 0.73, 0.74, 0.75, 0.76,
           0.77, 0.78, 0.79, 0.8 , 0.81, 0.82, 0.83, 0.84, 0.85, 0.86, 0.87,
           0.88, 0.89, 0.9 , 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98,
           0.99, 1.  , 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
           1.1 , 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17])




```python
df.trip_distance.nlargest(10)
```




    1059974    5005073.0
    1090317     318000.0
    1085978      40005.0
    1055761       1320.6
    1167755        335.9
    1125147        302.0
    1055917        275.8
    1134537        272.9
    1166016        238.8
    1095309        225.2
    Name: trip_distance, dtype: float64

```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

invalid_distance_idx = df[(df.trip_distance > 100) | (df.trip_distance == 0)].index.tolist() # Remove distance of 0 and greater than 100 miles https://www.google.com/maps/dir/40.5055432,-74.2381798/41.0191685,-73.7871866/@40.9358817,-74.3487811,8.92z/data=!3m1!5s0x89c2f2be234865c9:0x12533050f3f45b3c!4m2!4m1!3e0

df.drop(invalid_distance_idx, inplace=True)
df.reset_index(drop=True, inplace= True)

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1768323
    Rows/Size After: 1765510
    Removed: 2813

### Trip Speed Cleaning

Helps detect abnormal relationships between distance and time (combining the last 2 cleaning procedures)


```python
df.insert(loc=df.columns.tolist().index("trip_distance")+1,column="trip_distance_km",value = np.round(df["trip_distance"] * 1.609344,3))
df.insert(loc=df.columns.tolist().index("trip_distance_km")+1,column="trip_speed_mi/hr",value = np.round((df["trip_distance"]/df["trip_time_in_secs"])*60*60,3))
```


```python
df["trip_speed_mi/hr"].describe().apply(lambda x: '%.5f' % x)
```




    count    1765510.00000
    mean          13.79154
    std           19.18644
    min            0.03000
    25%            9.12000
    50%           12.28000
    75%           16.50000
    max         7280.00000
    Name: trip_speed_mi/hr, dtype: object




```python
speed_percentiles = df["trip_speed_mi/hr"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
```


```python
speed_percentiles.loc[0:0.1]
```




    0.00    0.030
    0.01    2.864
    0.02    4.045
    0.03    4.665
    0.04    5.115
    0.05    5.480
    0.06    5.800
    0.07    6.076
    0.08    6.327
    0.09    6.560
    0.10    6.771
    Name: trip_speed_mi/hr, dtype: float64




```python
speed_percentiles.loc[0.90:] # very odd maxmimum speed
```




    0.90      22.439
    0.91      23.191
    0.92      24.000
    0.93      25.000
    0.94      26.107
    0.95      27.424
    0.96      29.023
    0.97      31.024
    0.98      33.715
    0.99      37.835
    1.00    7280.000
    Name: trip_speed_mi/hr, dtype: float64




```python
df["trip_speed_mi/hr"].nlargest(30)
```




    1455995    7280.000
    1440280    7155.000
    1150060    7069.091
    1150552    6222.857
    1143343    5811.429
    1053937    5340.000
    1374463    5130.000
    1157752    4551.429
    1173308    4385.455
    1041760    4075.714
    1459064    3894.545
    1433255    3577.500
    1458672    3570.000
    1438038    3350.769
    1058099    3305.455
    1149133    3276.000
    1374628    3120.000
    1339895    3105.000
    1439642    3067.200
    1345988    2674.286
    1374860    2595.000
    1151062    2520.000
    1094388    2487.273
    1076385    2392.941
    1094307    2310.000
    1148658    2243.077
    1150085    2243.077
    1405731    2229.231
    1344046    2160.000
    1078396    2052.000
    Name: trip_speed_mi/hr, dtype: float64




```python
speeds = df["trip_speed_mi/hr"].values
speeds = np.sort(speeds,axis = None)

for i in np.arange(0.0, 1.0, 0.1):
    print("{} percentile is {}".format(99+i, speeds[int(len(speeds) * (float(99+i) / 100) )]))
print("100 percentile value is ",speeds[-1])
del speeds
```

    99.0 percentile is 37.835
    99.1 percentile is 38.4
    99.2 percentile is 39.025
    99.3 percentile is 39.729
    99.4 percentile is 40.518
    99.5 percentile is 41.4
    99.6 percentile is 42.45
    99.7 percentile is 43.722
    99.8 percentile is 45.408
    99.9 percentile is 48.027
    100 percentile value is  7280.0
    


```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

df = df[(df["trip_speed_mi/hr"] > 0) & (df["trip_speed_mi/hr"] <= 50)].reset_index(drop=True) # 99.9th percentile is 48.027

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1765510
    Rows/Size After: 1764465
    Removed: 1045
    
### Trip Total Amount, Fare Amount, Surcharge, Tax, Tolls and Tip Cleaning


```python
df["total_amount"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean          14.66869
    std           11.83378
    min            0.00000
    25%            8.00000
    50%           11.00000
    75%           16.50000
    max          563.12000
    Name: total_amount, dtype: object




```python
df["fare_amount"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean          12.30551
    std            9.79532
    min            0.00000
    25%            6.50000
    50%            9.50000
    75%           14.00000
    max          330.00000
    Name: fare_amount, dtype: object




```python
df["surcharge"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean           0.27433
    std            0.33809
    min            0.00000
    25%            0.00000
    50%            0.00000
    75%            0.50000
    max            5.83000
    Name: surcharge, dtype: object




```python
df["mta_tax"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean           0.49890
    std            0.02340
    min            0.00000
    25%            0.50000
    50%            0.50000
    75%            0.50000
    max            0.50000
    Name: mta_tax, dtype: object




```python
df["tip_amount"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean           1.32689
    std            2.16548
    min            0.00000
    25%            0.00000
    50%            1.00000
    75%            2.00000
    max          558.62000
    Name: tip_amount, dtype: object




```python
df["tolls_amount"].describe().apply(lambda x: '%.5f' % x)
```




    count    1764465.00000
    mean           0.26283
    std            1.22958
    min            0.00000
    25%            0.00000
    50%            0.00000
    75%            0.00000
    max           97.13000
    Name: tolls_amount, dtype: object




```python
total_amount_percentiles = df["total_amount"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
total_amount_percentiles.loc[0:0.1], total_amount_percentiles.loc[0.90:]
```




    (0.00    0.00
     0.01    4.50
     0.02    4.50
     0.03    5.00
     0.04    5.00
     0.05    5.50
     0.06    5.50
     0.07    5.62
     0.08    6.00
     0.09    6.00
     0.10    6.00
     Name: total_amount, dtype: float64,
     0.90     26.50
     0.91     28.50
     0.92     30.50
     0.93     33.33
     0.94     36.12
     0.95     39.33
     0.96     43.03
     0.97     48.90
     0.98     57.83
     0.99     65.60
     1.00    563.12
     Name: total_amount, dtype: float64)




```python
df.total_amount.nsmallest(20)
```




    1031853    0.0
    1035254    0.0
    1041181    0.0
    1043898    0.0
    1044443    0.0
    1047473    0.0
    1050744    0.0
    1053522    0.0
    1055058    0.0
    1055089    0.0
    1056811    0.0
    1060794    0.0
    1061535    0.0
    1064759    0.0
    1071199    0.0
    1072267    0.0
    1074670    0.0
    1076662    0.0
    1092351    0.0
    1098501    0.0
    Name: total_amount, dtype: float64




```python
df.total_amount.nlargest(20)
```




    1122353    563.12
    1470048    330.00
    1407367    325.00
    1429526    300.60
    1445864    300.00
    685414     298.83
    1291224    287.84
    1255443    269.00
    910977     267.60
    74407      266.55
    456863     261.75
    1114236    260.00
    632331     256.63
    1555698    255.33
    897057     254.50
    1721477    254.23
    1617697    252.00
    1223427    250.63
    1239784    250.00
    1183020    249.00
    Name: total_amount, dtype: float64




```python
totals = df["total_amount"].values
totals = np.sort(totals,axis = None)

for i in np.arange(0.0, 1.0, 0.1):
    print("{} percentile is {}".format(99+i, totals[int(len(totals) * (float(99+i) / 100) )]))
print("100 percentile value is ",totals[-1])
del totals
```

    99.0 percentile is 65.6
    99.1 percentile is 67.7
    99.2 percentile is 67.93
    99.3 percentile is 68.23
    99.4 percentile is 68.23
    99.5 percentile is 69.38
    99.6 percentile is 70.33
    99.7 percentile is 72.28
    99.8 percentile is 78.0
    99.9 percentile is 91.66
    100 percentile value is  563.12
    


```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

df = df[(df["total_amount"] >= 0) & (df["total_amount"] <= 150)].reset_index(drop=True)   # See: https://www.introducingnewyork.com/taxis

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1764465
    Rows/Size After: 1764321
    Removed: 144
    


```python
fare_amount_percentiles = df["fare_amount"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
fare_amount_percentiles.loc[0:0.1], fare_amount_percentiles.loc[0.90:]
```




    (0.00    0.0
     0.01    3.5
     0.02    4.0
     0.03    4.0
     0.04    4.0
     0.05    4.5
     0.06    4.5
     0.07    4.5
     0.08    5.0
     0.09    5.0
     0.10    5.0
     Name: fare_amount, dtype: float64,
     0.90     23.0
     0.91     24.0
     0.92     25.5
     0.93     27.5
     0.94     29.5
     0.95     32.0
     0.96     34.5
     0.97     40.0
     0.98     52.0
     0.99     52.0
     1.00    150.0
     Name: fare_amount, dtype: float64)




```python
before = df.shape[0]
print(f'Rows/Size Before: {before}')

df = df[(df["fare_amount"] >= 0)].reset_index(drop=True)

after = df.shape[0]
print(f'Rows/Size After: {after}')
print(f'Removed: {before - after}')
```

    Rows/Size Before: 1764321
    Rows/Size After: 1764321
    Removed: 0
    


```python
surcharge_percentiles = df["surcharge"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
surcharge_percentiles.loc[0:0.1], surcharge_percentiles.loc[0.9:]
```




    (0.00    0.0
     0.01    0.0
     0.02    0.0
     0.03    0.0
     0.04    0.0
     0.05    0.0
     0.06    0.0
     0.07    0.0
     0.08    0.0
     0.09    0.0
     0.10    0.0
     Name: surcharge, dtype: float64,
     0.90    1.00
     0.91    1.00
     0.92    1.00
     0.93    1.00
     0.94    1.00
     0.95    1.00
     0.96    1.00
     0.97    1.00
     0.98    1.00
     0.99    1.00
     1.00    5.83
     Name: surcharge, dtype: float64)




```python
df.surcharge.nlargest(10)
```




    1434721    5.83
    1458665    2.50
    1092410    2.00
    1426960    2.00
    1035864    1.50
    1116033    1.50
    1138106    1.50
    1150637    1.50
    1358139    1.50
    1370980    1.50
    Name: surcharge, dtype: float64




```python
mta_percentiles = df["mta_tax"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
mta_percentiles.loc[0:0.1], mta_percentiles.loc[0.9:]
```




    (0.00    0.0
     0.01    0.5
     0.02    0.5
     0.03    0.5
     0.04    0.5
     0.05    0.5
     0.06    0.5
     0.07    0.5
     0.08    0.5
     0.09    0.5
     0.10    0.5
     Name: mta_tax, dtype: float64,
     0.90    0.5
     0.91    0.5
     0.92    0.5
     0.93    0.5
     0.94    0.5
     0.95    0.5
     0.96    0.5
     0.97    0.5
     0.98    0.5
     0.99    0.5
     1.00    0.5
     Name: mta_tax, dtype: float64)




```python
tip_amount_percentiles = df["tip_amount"].quantile(np.round(np.arange(0.00, 1.01, 0.01), 2))
tip_amount_percentiles.loc[0.0:0.1], tip_amount_percentiles.loc[0.9:]
```




    (0.00    0.0
     0.01    0.0
     0.02    0.0
     0.03    0.0
     0.04    0.0
     0.05    0.0
     0.06    0.0
     0.07    0.0
     0.08    0.0
     0.09    0.0
     0.10    0.0
     Name: tip_amount, dtype: float64,
     0.90      3.37
     0.91      3.60
     0.92      3.88
     0.93      4.10
     0.94      4.50
     0.95      5.00
     0.96      5.60
     0.97      6.40
     0.98      7.75
     0.99     10.40
     1.00    121.20
     Name: tip_amount, dtype: float64)




```python
df.tip_amount.nlargest(20)
```




    1513550    121.20
    600519     115.50
    843787     111.00
    1013173    111.00
    466914     110.00
    676289     110.00
    909348     110.00
    244908     100.33
    408036     100.00
    1255130     96.00
    414779      93.71
    1675505     91.00
    655186      88.00
    1279699     88.00
    609128      87.94
    60783       85.80
    1549039     81.00
    1321652     80.80
    1214863     80.00
    1697546     80.00
    Name: tip_amount, dtype: float64




```python
tolls_percentiles = df["tolls_amount"].quantile(np.round(np.arange(0.00, 1.01, 0.01),2 ))
tolls_percentiles.loc[0.0:0.1], tolls_percentiles.loc[0.9:]
```




    (0.00    0.0
     0.01    0.0
     0.02    0.0
     0.03    0.0
     0.04    0.0
     0.05    0.0
     0.06    0.0
     0.07    0.0
     0.08    0.0
     0.09    0.0
     0.10    0.0
     Name: tolls_amount, dtype: float64,
     0.90     0.00
     0.91     0.00
     0.92     0.00
     0.93     0.00
     0.94     0.00
     0.95     0.00
     0.96     4.80
     0.97     5.33
     0.98     5.33
     0.99     5.33
     1.00    94.83
     Name: tolls_amount, dtype: float64)




```python
df.tolls_amount.nlargest(10)
```




    1036970    94.83
    1144668    90.31
    1147141    62.26
    1034771    53.49
    1080876    53.01
    1150660    48.69
    1137765    34.39
    1163542    28.00
    1121143    21.32
    1167016    20.91
    Name: tolls_amount, dtype: float64



### Passenger Count Cleaning


```python
df["passenger_count"].describe().apply(lambda x: '%.5f' % x) # Looks clean but we have passenger counts of 0
```




    count    1764321.00000
    mean           2.02239
    std            1.65448
    min            0.00000
    25%            1.00000
    50%            1.00000
    75%            2.00000
    max            6.00000
    Name: passenger_count, dtype: object




```python
print(df[df['passenger_count'] == 0].shape[0]) #7 trips with passenger count 0, replace with mean value

df.loc[df[df['passenger_count'] == 0].index,'passenger_count'] = df.passenger_count.mean()
```

    7
    


```python
df.passenger_count.quantile(np.round(np.arange(0.00,1.01, 0.01),2)).loc[0.9:]
```




    0.90    5.0
    0.91    5.0
    0.92    5.0
    0.93    5.0
    0.94    6.0
    0.95    6.0
    0.96    6.0
    0.97    6.0
    0.98    6.0
    0.99    6.0
    1.00    6.0
    Name: passenger_count, dtype: float64



### Categorical Columns Cleaning


```python
df.select_dtypes(exclude=["number","datetime"]).columns #rate code is missing as its numerical, but will be treated as categorical
```




    Index(['medallion', 'hack_license', 'vendor_id', 'store_and_fwd_flag',
           'payment_type'],
          dtype='object')




```python
df.rate_code.unique() # Should only be 1,2,3,4,5,6 from data description, See: https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf
```




    array([1, 2, 4, 3, 5, 6, 0], dtype=int64)




```python
allowed = [1,2,3,4,5,6]

abnormal_rate_code = df[~df.rate_code.isin(allowed)]
print(abnormal_rate_code.shape[0])
```

    33
    
We can use the [Folium](https://python-visualization.github.io/folium/) library to plot maps with markers and lines on it (and much more). We'll use it to inspect some of these (14) abnormal rate code (rate code = 0) trips 

```python
newyork_center= folium.Map(location=[40.7345, -73.8697]) # https://www.google.com/search?q=middle+of+new+york+coordinates&rlz=1C1CHBF_enDE1000DE1000&oq=middle+of+new+york+coordina&aqs=chrome.1.69i57j33i22i29i30l6.7294j0j7&sourceid=chrome&ie=UTF-8

for pickup_lat,pick_long, dropoff_lat, dropoff_long, code in zip(abnormal_rate_code["pickup_latitude"],
                                                                 abnormal_rate_code["pickup_longitude"],
                                                                 abnormal_rate_code["dropoff_latitude"],
                                                                 abnormal_rate_code["dropoff_longitude"],
                                                                 abnormal_rate_code["rate_code"][0:14]):
    
    folium.CircleMarker([pickup_lat, pick_long],popup=code,tooltip = "Pickup",fill_color= "#d1005b",color="#d1005b",
                       fill = True,radius = 5).add_to(newyork_center)
                  
    folium.CircleMarker([dropoff_lat, dropoff_long],popup=code,tooltip = "Dropoff",fill_color= "#616161",color="#616161",
                       fill = True,radius = 5).add_to(newyork_center)
    
    folium.PolyLine([[pickup_lat, pick_long],[dropoff_lat,dropoff_long]]).add_to(newyork_center)
    
newyork_center 
```

<iframe src= "/posts/New-York-Taxi-Analysis/abnormal_rate.html" height="525" width="100%"></iframe>

They look normal trips, there are 33 so I'll replace them with rate code 1 i.e. standard rate

```python
abnormal_rate_code_idx = abnormal_rate_code.index.tolist()
df.loc[abnormal_rate_code_idx, 'rate_code'] = 1
```

```python
df.payment_type.unique() # Looks fine CSH = Cash, DIS = Dispute, CRD = Credit Card, UNK = Unknown, NOC = No charge
```




    array(['CSH', 'DIS', 'CRD', 'UNK', 'NOC'], dtype=object)




```python
df.store_and_fwd_flag.unique() # Looks fine Y= store and forward trip, N= not a store and forward trip
```




    array(['N', 'Y', nan], dtype=object)




```python
df.vendor_id.unique() # Looks fine CMT = Creative Mobile Technologies, VTS = VeriFone Inc.
```




    array(['CMT', 'VTS'], dtype=object)

```python
df.to_csv("../Data/Pre-processed Data/cleaned_data.csv",index=False)
```
They all look as expected so we finally save our clean dataset and onto EDA!

## Data Analysis

Exploratory Data Analysis is the process of studying data and extracting insights from it in order to examine its major properties. EDA may be accomplished through the use of statistics and graphical approaches (using my favourite plotting library Plotly! With a bit of matplotlib as well :grin:). Why is it important? We simply can’t make sense of such huge datasets if we don’t explore the data. Exploratory Data Analysis helps us look deeper and see if our intuition matches with the data. It helps us see if we are asking the right questions. Exploring and analyzing the data is important to see how features are distributed or if and how they are related with each other. How they are contributing to the target variable (varable to be predicted/modelled), if there is any or simply analyzing the features without it. This will eventually help us answer the questions we have around the dataset, allow us to formulate new questions and check if we are even asking the right questions. So let's start as usual by importing and loading our the cleaned dataset.

### Import Libraries and Data Loading

```python
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import scipy.stats as ss
from statsmodels.distributions.empirical_distribution import ECDF

import matplotlib.pyplot as plt
import seaborn as sns
import datetime 
import requests
import json
import os
from tqdm.notebook import tqdm
from tabulate import tabulate
import geopy.distance

from sklearn.model_selection import train_test_split

from shapely.geometry import Point,Polygon, MultiPoint,MultiPolygon
import folium
from folium.plugins import HeatMapWithTime
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="taxi-analyis")
import geopandas as gpd
from geopandas import GeoDataFrame
import geoplot
import pygeohash as pgh
import pyproj
from area import area

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
pyo.init_notebook_mode(connected=True)

from PIL import Image

pd.set_option("display.max_columns",None)
np.random.seed(42)

df = pd.read_csv("../Data/Pre-processed Data/cleaned_data.csv")
```

### Convert dates to datetime



```python
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])


# Too many addresses to geocode, Nominatim usage policy doesn't allow bulk geocoding and will timeout (was going to be used in hover)

#df["pickup_address"] = df.apply(lambda x: geolocator.reverse(str(x.pickup_latitude) + "," + str(x.pickup_longitude),timeout=5)[0],axis=1)
#df["dropoff_address"] = df.apply(lambda x: geolocator.reverse(str(x.dropoff_latitude) + "," + str(x.dropoff_longitude),timeout=5)[0],axis=1)
```

### Splitting data for test and train sets 

Prevent exploration on test set (data-snooping)


```python
df_train, df_test = train_test_split(df, test_size=0.33,random_state=32)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

df_train.shape, df_test.shape
```
    ((1182095, 24), (582226, 24))


We start by simply plotting a map of a sample (10,000) pickups and dropoffs to get a rough idea if there is a significant difference.

```python
def getBB(df, BB):

    df_BB = df[(df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) & 
               (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &
               (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &
               (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])].reset_index(drop=True)
    
    return df_BB

def drop_pick_map(df, BB, sample_size = 1000, title = '' ,size=5, opacity=0.8):
    
    
    df_plt = df.sample(sample_size)
    df_plt = getBB(df_plt, BB)
    
    data = [go.Scattermapbox(
            lat= df_plt['pickup_latitude'] ,
            lon= df_plt['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(df_plt['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(df_plt['trip_speed_mi/hr'][i]) for i in range(df_plt.shape[0])],
            hovertext=df_plt['trip_time_in_secs'],
            customdata = df_plt['total_amount'],
            mode='markers',
            marker=dict(
                size= size,
                color = 'red',
                opacity = opacity),
            name ='Pickups',
            subplot='mapbox',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          ),
        go.Scattermapbox(
            lat= df_plt['dropoff_latitude'] ,
            lon= df_plt['dropoff_longitude'],
            text = ['Trip Distance (mi): '+ str(df_plt['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(df_plt['trip_speed_mi/hr'][i]) for i in range(df_plt.shape[0])],
            hovertext=df_plt['trip_time_in_secs'],
            customdata = df_plt['total_amount'],
            mode='markers',
            marker=dict(
                size= size,
                color = 'blue',
                opacity = opacity),
            name ='Dropoffs',
            subplot ='mapbox2',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          )]

    layout = go.Layout(autosize=False,
                       mapbox= dict(center= dict(
                                             lat=40.721319,
                                             lon=-73.987130),
                                   style = 'open-street-map',
                                   zoom = 8,
                                   domain={'x': [0, 0.5], 'y': [0, 1]}),
                       mapbox2 = dict(center= dict(
                                             lat=40.721319,
                                             lon=-73.987130),
                                   style = 'open-street-map',
                                   zoom=8,
                                   domain={'x': [0.51, 1.0], 'y': [0, 1]}),
                        width=1000,
                        height=600, title = title)

    fig = go.Figure(data=data, layout=layout)
    
    
    fig.show()
    
    return 

#All hires computationaly expensive, can't use interactive plots opting from matplotlib
def all_hires(df, BB, figsize=(12, 12), ax=None, c=('r', 'b')):
    if ax == None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    df_sub = getBB(df,BB)
    ax.scatter(df_sub.pickup_longitude, df_sub.pickup_latitude, c=c[0], s=0.1, alpha=0.5)
    ax.scatter(df_sub.dropoff_longitude, df_sub.dropoff_latitude, c=c[1], s=0.1, alpha=0.5)
    plt.legend(["Pickups","Dropoffs"])

NYBB = (-74.15, -73.7004, 40.5774, 40.9176)

```


```python
drop_pick_map(df_train, NYBB, title = 'Map of Pickups & Dropoffs',sample_size=10000)
```

<iframe src= "/posts/New-York-Taxi-Analysis/pick_drop.html" height="525" width="100%"></iframe>

We see a slight difference where the pickups are more on the West of New York in the Manhattan area but the dropoffs are more to the East of New York in the Queens and Brooklyn area.

We can also see all the pickups and dropoffs in one graphic below

```python
all_hires(df_train,NYBB)
```

<img src="/posts/New-York-Taxi-Analysis/all hires.png" width="100%">

We'll segment the EDA sections by question and we'll start with our first question

### Q1: How much do taxis earn per trip if they can only drive 30km or 50km per trip, compared to taxis that have no limit? (i.e. what is the average income for trips <30km or <50km compared to the total revenue?)

```python
columns=['trip_distance','trip_speed_mi/hr','trip_time_in_secs']
correlation_amount = pd.DataFrame(index=["total_amount"],columns=columns)
for i,col in enumerate(columns):
    cor = df_train.total_amount.corr(df_train[col])
    correlation_amount.loc[:,columns[i]] = cor
plt.figure(figsize=(25,5))
g = sns.heatmap(correlation_amount,annot=True,fmt='.4f', annot_kws={"fontsize":16})
g.set_yticklabels(['Tip Amount'], fontsize = 16)
g.set_xticklabels(['Trip Distance','Trip Speed', 'Trip Duration'], fontsize = 16)
g.set_yticklabels(['Total Amount'], fontsize = 16)
plt.title("Correlation Matrix between Total Amount and Numerical Features", fontdict={'fontsize':20})
plt.plot()
```

<img src="/posts/New-York-Taxi-Analysis/Amount Correlation.jpeg" height= "525px" width="100%">

Total Amount is highly positively correlated with the trip distance (0.94) and trip duration (0.83)

Significant Difference between Trips less than 5km and all trips, no significant difference from the other restrictions, this is mainly due to the fact that <30km and <50km capture most trips
We can also see the difference below in the box plots and histograms


```python
df_30_less = df_train[(df_train['trip_distance_km'] <= 30)].reset_index(drop=True)
df_50_less = df_train[(df_train['trip_distance_km'] <= 50)].reset_index(drop=True)

#more severe
df_5_less = df_train[(df_train['trip_distance_km'] <= 5)].reset_index(drop=True)
```


```python
print("Mean Total Amount for 30 km or less: ", df_30_less.total_amount.mean())
print("Mean Total Amount for 50 km or less: ", df_50_less.total_amount.mean())
print("Mean Total Amount for 5 km or less: ", df_5_less.total_amount.mean())
print("Mean Total Amount for no distance restriction: ", df_train.total_amount.mean())
```
    Mean Total Amount for 30 km or less:  14.223142284432392
    Mean Total Amount for 50 km or less:  14.645665338921816
    Mean Total Amount for 5 km or less:  9.734024318446
    Mean Total Amount for no distance restriction:  14.6549403897318

```python
fig = go.Figure()

fig.add_trace(go.Box(y=df_30_less["total_amount"], name= "30 km or less earnings distribution"))
fig.add_trace(go.Box(y=df_50_less["total_amount"], name= "50 km or less earnings distribution"))
fig.add_trace(go.Box(y=df_5_less["total_amount"], name= "5 km or less earnings distribution"))

fig.add_trace(go.Box(y=df_train["total_amount"],name= "No distance limit earnings distribution"))

fig = fig.update_layout(title = "T1: How much do taxis earn per trip according to distance restrictions")
fig.write_html("../Figures/T1/T1_fig.html")
```

<img src="/posts/New-York-Taxi-Analysis/Boxplot Amount.png" width="100%">



```python
all_log = np.log(df_train.total_amount.values)
km30_log = np.log(df_30_less.total_amount.values)
km50_log = np.log(df_50_less.total_amount.values)

fig = plt.figure(figsize=(15,8))
plt.title("Log Distribution of Total Amount")
plt.hist(all_log[np.isfinite(all_log)],bins=60, label='No restriction', alpha = 0.5)
plt.hist(km30_log[np.isfinite(km30_log)],bins=65,label='30 km or less', alpha=0.5)
plt.hist(km50_log[np.isfinite(km50_log)],bins=70, label = '50 km or less',alpha=0.5)
plt.legend(loc='upper right')
plt.show()
```

<img src="/posts/New-York-Taxi-Analysis/Log Distribution of Total Amount.png" width="100%">

```python
fig = plt.figure(figsize=(15,8))
plt.title("Distribution of Total Amount")
plt.hist(df_train.total_amount.values,bins=60, label='No restriction', alpha = 0.5)
plt.hist(df_30_less.total_amount.values,bins=65,label='30 km or less', alpha=0.5)
plt.hist(df_50_less.total_amount.values,bins=70, label = '50 km or less',alpha=0.5)
plt.legend(loc='upper right')
plt.show()
```
<img src="/posts/New-York-Taxi-Analysis/Distribution of Total Amount.png" width="100%">

Kolmogorov–Smirnov Test is used to test for the significance in the difference between the distributions and again same results are observed analytically

```python
ks_30vs_all = ss.ks_2samp(df_train['total_amount'].values, df_30_less['total_amount'].values)
ks_50vs_all = ss.ks_2samp(df_train['total_amount'].values, df_50_less['total_amount'].values)
ks_5vs_all = ss.ks_2samp(df_train['total_amount'].values, df_5_less['total_amount'].values)


print(f"KS-Statistics between 30 km restriction and no restriction: {ks_30vs_all.statistic}, p-value: {ks_30vs_all.pvalue}")
print(f"KS-Statistics between 50 km restriction and no restriction: {ks_50vs_all.statistic}, p-value: {ks_50vs_all.pvalue}")
print(f"KS-Statistics between 5 km restriction and no restriction: {ks_5vs_all.statistic}, p-value: {ks_5vs_all.pvalue}")
```
    KS-Statistics between 30 km restriction and no restriction: 
    0.008042675652436548, p-value: 1.6896187294453e-33

    KS-Statistics between 50 km restriction and no restriction: 
    0.00011191457137738059, p-value: 1.0

    KS-Statistics between 5 km restriction and no restriction: 
    0.23128162369336924, p-value: 0.0

The Kolmogorov–Smirnov Test is based on the cumulative distribution so we will look at that

```python
ecdf_norestrict = ECDF(df_train.total_amount.values)
ecdf_30 = ECDF(df_30_less.total_amount.values)
ecdf_50 = ECDF(df_50_less.total_amount.values)
ecdf_5 = ECDF(df_5_less.total_amount.values)

fig = plt.figure(figsize=(15,8))
plt.plot(ecdf_norestrict.x, ecdf_norestrict.y, label='No Restriction', alpha=0.5)
plt.plot(ecdf_30.x, ecdf_30.y, label = '30 km or less', alpha=0.5)
plt.plot(ecdf_50.x, ecdf_50.y, label = '50 km or less',alpha=0.5)
plt.plot(ecdf_5.x, ecdf_5.y, label = '5 km or less',alpha=0.5)
_= plt.xlabel('Total Amount',fontdict=dict(size=14))
_= plt.ylabel('Probability',fontdict=dict(size=14))
plt.legend()
plt.show()
```

<img src="/posts/New-York-Taxi-Analysis/Cumulative Distribution of Total Amount.png" width="100%">

Another metric we can use to quanitfy the difference in the distributions is the Wasserstein Distance, which we inspect below visually and analytically

```python
def wassersteindist_plot(a, b):
    all_values = sorted(set(a) | set(b))
    
    a_cdf = np.vstack(
        [[np.mean(a < v) for v in all_values],
        [np.mean(a <= v) for v in all_values]]
    ).flatten("F")
    
    b_cdf = np.vstack(
        [[np.mean(b < v) for v in all_values],
        [np.mean(b <= v) for v in all_values]]
    ).flatten("F")
    
    all_values = np.repeat(all_values, 2)
    
    return all_values, a_cdf, b_cdf

av, cdf_all, cdf_30 = wassersteindist_plot(df_train.total_amount.values, df_30_less.total_amount.values)
av, cdf_all, cdf_50 = wassersteindist_plot(df_train.total_amount.values, df_50_less.total_amount.values)
av, cdf_all, cdf_5 = wassersteindist_plot(df_train.total_amount.values, df_5_less.total_amount.values)

fig, ax = plt.subplots(figsize=(15,8))

ax.plot(av, cdf_all, label="No Restrict")
ax.plot(av, cdf_30, label="30 km or less")
ax.fill_between(av, cdf_all, cdf_30, color="grey", alpha=0.6, label="Wasserstein\ndistance")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Total Amount', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
ax.legend(fontsize=14, loc="lower right")
```
<img src="/posts/New-York-Taxi-Analysis/SWD30.png" width="100%">

```python
fig, ax = plt.subplots(figsize=(15,8))

ax.plot(av, cdf_all, label="No Restrict")
ax.plot(av, cdf_50, label="50 km or less")
ax.fill_between(av, cdf_all, cdf_50, color="grey", alpha=0.6, label="Wasserstein\ndistance")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Total Amount', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
ax.legend(fontsize=14, loc="lower right")
```

<img src="/posts/New-York-Taxi-Analysis/SWD50.png" width="100%">


```python
fig, ax = plt.subplots(figsize=(15,8))

ax.plot(av, cdf_all, label="No Restrict")
ax.plot(av, cdf_5, label="5 km or less")
ax.fill_between(av, cdf_all, cdf_5, color="grey", alpha=0.6, label="Wasserstein\ndistance")
ax.tick_params(axis='both', which='major', labelsize=14)
ax.set_xlabel('Total Amount', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
ax.legend(fontsize=14, loc="lower right")
```

<img src="/posts/New-York-Taxi-Analysis/SWD5.png" width="100%">

Again from the wasserstein distance plots we see the same result, no significant difference (low wasserstein distance) between trips of less than 30 or 50km and all trips, but the opposite (high wasserstein distance) between trips of less than 5km and all trips.

We'll also look at the standerdized wasserstein distance

```python
def standardized_wasserstein_distance(u, v, method="std"):
    u, v = np.array(u), np.array(v)
    numerator = ss.wasserstein_distance(u, v)
    concat = np.concatenate([u, v])
    if method == 'std':
        denominator = np.std(concat)
    elif method == 'minmax':
        denominator = np.max(concat) - np.min(concat)
    elif method == 'mean':
        denominator = max(np.max(concat) - np.mean(concat), np.mean(concat) - np.min(concat))
    elif method == 'median':
        denominator = max(np.min(concat) - np.median(concat), np.median(concat) - np.min(concat))
    elif method == 'iqr':
        denominator = np.diff(np.quantile(concat, [.25, .75]))[0]
    return numerator / denominator if denominator != .0 else .0



print(f"SWD (mean method) between No Restriction and 30 km or less: {standardized_wasserstein_distance(df_train.total_amount.values, df_30_less.total_amount.values,method='mean')}")
print(f"SWD (mean method) between No Restriction and 50 km or less: {standardized_wasserstein_distance(df_train.total_amount.values, df_50_less.total_amount.values,method='mean')}")
print(f"SWD (mean method) between No Restriction and 5 km or less: {standardized_wasserstein_distance(df_train.total_amount.values, df_5_less.total_amount.values,method='mean')}")
```
    SWD (mean method) between No Restriction and 30 km or less: 
    0.0031855750868686784
    
    SWD (mean method) between No Restriction and 50 km or less: 
    6.868595381679189e-05
    
    SWD (mean method) between No Restriction and 5 km or less: 
    0.035809828654724

Trip Distance matters!

Obviously as we have seen severe restrictions on distance affect the total trip amount and as seen in the correlation chart above, the trip distance is highly positevely correlated with the total amount


### Q2: Do more taxi trips start at the North or South?

By taking a arbitrary north and south cutoff we get these results

```python
south = df_train[df_train.pickup_latitude < 40.66240321639479].reset_index(drop=True)
north = df_train[df_train.pickup_latitude > 40.803265534217225].reset_index(drop=True)

print("Number of trips starting in the south: ",south.shape[0])
print("Number of trips starting in the north: ",north.shape[0])
```
    Number of trips starting in the south:  21708
    Number of trips starting in the north:  19359

```python
data = [go.Scattermapbox(
            lat= south['pickup_latitude'] ,
            lon= south['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(south['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(south['trip_speed_mi/hr'][i]) for i in range(south.shape[0])],
            hovertext=south['trip_time_in_secs'],
            customdata = south['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'red',
                opacity = 0.8),
            name ='South Pickups',
            subplot='mapbox',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          ),
        go.Scattermapbox(
            lat= north['pickup_latitude'] ,
            lon= north['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(north['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(north['trip_speed_mi/hr'][i]) for i in range(north.shape[0])],
            hovertext=north['trip_time_in_secs'],
            customdata = north['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'blue',
                opacity = 0.8),
            name ='North Pickups',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          )]

layout = go.Layout(autosize=False,
                       mapbox= dict(center= dict(
                                             lat=40.721319,
                                             lon=-73.987130),
                                   style = 'open-street-map',
                                   zoom = 8,
                                   domain={'x': [0, 1], 'y': [0, 1]}),
                        width=1000,
                        height=750, title = 'South vs North Pickups')

fig = go.Figure(data=data, layout=layout)
    
    
fig.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/cutoff.html" height="525" width="100%"></iframe>

By taking the center of Manhatten as a cutoff we get a more intutive answer

```python
south2 = df_train[df_train.pickup_latitude < 40.78035634806236].reset_index(drop=True)
north2 = df_train[df_train.pickup_latitude > 40.78035634806236].reset_index(drop=True)

print("Number of trips starting in the south: ",south2.shape[0])
print("Number of trips starting in the north: ",north2.shape[0])
```
    Number of trips starting in the south:  1062763
    Number of trips starting in the north:  119332

```python
data = [go.Scattermapbox(
            lat= south2.iloc[0:150000]['pickup_latitude'] ,
            lon= south2.iloc[0:150000]['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(south2.iloc[0:150000]['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(south2.iloc[0:150000]['trip_speed_mi/hr'][i]) for i in range(south2.iloc[0:150000].shape[0])],
            hovertext=south['trip_time_in_secs'],
            customdata = south['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'red',
                opacity = 0.8),
            name ='South Pickups',
            subplot='mapbox',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          ),
        go.Scattermapbox(
            lat= north2.iloc[0:50000]['pickup_latitude'] ,
            lon= north2.iloc[0:50000]['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(north2.iloc[0:50000]['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(north2.iloc[0:50000]['trip_speed_mi/hr'][i]) for i in range(north2.iloc[0:50000].shape[0])],
            hovertext=north['trip_time_in_secs'],
            customdata = north['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'blue',
                opacity = 0.8),
            name ='North Pickups',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          )]

layout = go.Layout(autosize=False,
                       mapbox= dict(center= dict(
                                             lat=40.721319,
                                             lon=-73.987130),
                                   style = 'open-street-map',
                                   zoom = 8,
                                   domain={'x': [0, 1], 'y': [0, 1]}),
                        width=1000,
                        height=750, title = 'South vs North Pickups')

fig = go.Figure(data=data, layout=layout)
    
    
fig.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/center.html" height="525" width="100%"></iframe>

Some trips are in water! These will be removed and numbers checked again (should have been done in the cleaning phase but these weren't detected until now)

```python
BB = (-74.5, -72.8, 40.5, 41.8)
nyc_water = plt.imread('water_mask.png')[:,:,0] > 0.9 # Mask 

def getXY(lon, lat, xdim, ydim, BB):
    
    X = (xdim * (lon - BB[0]) / (BB[1]-BB[0])).astype('int')
    y = (ydim - ydim * (lat - BB[2]) / (BB[3]-BB[2])).astype('int')

    
    return X,y

before = df_train.shape[0]
print("Rows before bounding box", df_train.shape[0])


df_train = df_train[(df_train.pickup_longitude >= BB[0]) & (df_train.pickup_longitude <= BB[1]) & 
         (df_train.pickup_latitude >= BB[2]) & (df_train.pickup_latitude <= BB[3]) &
         (df_train.dropoff_longitude >= BB[0]) & (df_train.dropoff_longitude <= BB[1]) &
         (df_train.dropoff_latitude >= BB[2]) & (df_train.dropoff_latitude <= BB[3])].reset_index(drop=True)


print("Rows after bounding box", df_train.shape[0])

pickup_x, pickup_y = getXY(df_train.pickup_longitude, df_train.pickup_latitude, nyc_water.shape[1], nyc_water.shape[0], BB)
dropoff_x, dropoff_y = getXY(df_train.dropoff_longitude, df_train.dropoff_latitude, nyc_water.shape[1], nyc_water.shape[0], BB)
land_idx = (nyc_water[pickup_y, pickup_x] & nyc_water[dropoff_y, dropoff_x])

print(f"{np.sum(~land_idx)} Trips in water!")

print("Rows before water trips removal", df_train.shape[0])

df_train = df_train[land_idx].reset_index(drop=True)

after=df_train.shape[0]
print("Rows after water trips removal", df_train.shape[0])
print("Total removed: ",before - after)
```
    Rows before bounding box 1182095
    Rows after bounding box 1182079
    49 Trips in water!
    Rows before water trips removal 1182079
    Rows after water trips removal 1182030
    Total removed:  65

I removed the trips in water by loading in a mask (shown below) that was created using GIMP and following this [tutorial](https://www.fsdeveloper.com/forum/attachments/using-gimp-to-create-water-mask-pdf.39106/) from an image of New York. 


<figure>
  <img src="/posts/New-York-Taxi-Analysis/NYC.png" alt="New York"/>
  <figcaption style = "font-size:15px">Map of NYC</figcaption>
</figure>

<figure>
  <img src="/posts/New-York-Taxi-Analysis/water_mask.png" alt="New York"/>
  <figcaption style = "font-size:15px">Water Mask of NYC</figcaption>
</figure>

```python
south2 = df_train[df_train.pickup_latitude < 40.78035634806236].reset_index(drop=True)
north2 = df_train[df_train.pickup_latitude > 40.78035634806236].reset_index(drop=True)

print("Number of trips starting in the south after water removal: ",south2.shape[0])
print("Number of trips starting in the north after water removal: ",north2.shape[0])
```
    Number of trips starting in the south after water removal:  1062702
    Number of trips starting in the north after water removal:  119328
  
```python
data = [go.Scattermapbox(
            lat= south2.iloc[0:150000]['pickup_latitude'] ,
            lon= south2.iloc[0:150000]['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(south2.iloc[0:150000]['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(south2.iloc[0:150000]['trip_speed_mi/hr'][i]) for i in range(south2.iloc[0:150000].shape[0])],
            hovertext=south['trip_time_in_secs'],
            customdata = south2['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'red',
                opacity = 0.8),
            name ='South Pickups',
            subplot='mapbox',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          ),
        go.Scattermapbox(
            lat= north2.iloc[0:50000]['pickup_latitude'] ,
            lon= north2.iloc[0:50000]['pickup_longitude'],
            text = ['Trip Distance (mi): '+ str(north2.iloc[0:50000]['trip_distance'][i]) + "<br>" + 'Trip Speed (mi/hr): ' + str(north2.iloc[0:50000]['trip_speed_mi/hr'][i]) for i in range(north2.iloc[0:50000].shape[0])],
            hovertext=north['trip_time_in_secs'],
            customdata = north['total_amount'],
            mode='markers',
            marker=dict(
                size= 5,
                color = 'blue',
                opacity = 0.8),
            name ='North Pickups',
            hovertemplate = "Latitude (°): %{lat} <br>Longitude (°): %{lon} <br>Trip Duration (s): %{hovertext} <br>%{text} <br>Trip Amount ($): %{customdata}"
          )]

layout = go.Layout(autosize=False,
                       mapbox= dict(center= dict(
                                             lat=40.721319,
                                             lon=-73.987130),
                                   style = 'open-street-map',
                                   zoom = 8,
                                   domain={'x': [0, 1], 'y': [0, 1]}),
                        width=1000,
                        height=750, title = 'South vs North Pickups')

fig = go.Figure(data=data, layout=layout)
    
    
fig.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/correct_center.html" height="525" width="100%"></iframe>

We see there are no longer data points on the water. And that there are more trips starting at the South of New York.

We'll now add more features that are required for later analysis

### Adding date features

```python
def get_time_of_day(time):
    
    if time >= datetime.time(6, 0, 1) and time <= datetime.time(12, 0, 0):
        return 'Morning'
    elif time >= datetime.time(12, 0, 1) and time <= datetime.time(16, 0, 0):
        return 'Afternon'
    elif time >= datetime.time(16, 0, 1) and time <= datetime.time(22, 0, 0):
        return 'Evening'
    elif time >= datetime.time(22, 0, 1) or time <=datetime.time(1, 0, 0):
        return 'Night'
    elif time >= datetime.time(1, 0, 1) or time <=datetime.time(6, 0, 0):
        return 'Late night'

df_train['pickup_day'] = df_train['pickup_datetime'].dt.day_name()
df_train['dropoff_day']= df_train['dropoff_datetime'].dt.day_name()
df_train['pickup_month'] = df_train['pickup_datetime'].dt.month_name()
df_train['dropoff_month'] = df_train['dropoff_datetime'].dt.month_name()

df_train['pickup_hour']=df_train['pickup_datetime'].dt.hour
df_train['dropoff_hour']=df_train['dropoff_datetime'].dt.hour

df_train['pickup_week_year'] = df_train['pickup_datetime'].dt.isocalendar().week
df_train['dropoff_week_year'] = df_train['dropoff_datetime'].dt.isocalendar().week


df_train['pickup_month_day'] = df_train['pickup_datetime'].dt.day
df_train['dropoff_month_day'] = df_train['dropoff_datetime'].dt.day

df_train["pickup_time_of_day"] = df_train["pickup_datetime"].dt.time.apply(lambda x: get_time_of_day(x))
df_train["dropoff_time_of_day"] = df_train["dropoff_datetime"].dt.time.apply(lambda x: get_time_of_day(x))
```

### Adding direction feature

```python
def get_direction(pickup_coords, dropoff_coords):
    
    def get_bearing(pickup_coords, dropoff_coords):
        
        lat1 = np.radians(pickup_coords[0])
        lat2 = np.radians(dropoff_coords[0])

        diffLong = np.radians(dropoff_coords[1] - pickup_coords[1])

        x = np.sin(diffLong) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1)
                * np.cos(lat2) * np.cos(diffLong))

        initial_bearing = np.arctan2(x, y)
        initial_bearing = np.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing
    
    directions = ['↑ N', '↗ NE', '→ E', '↘ SE', '↓ S', '↙ SW', '← W', '↖ NW']
    
    bearing = get_bearing(pickup_coords, dropoff_coords)
    
    idx = round(bearing / 45) % 8
    
    return directions[idx]

df_train["trip_direction"] = df_train.apply(lambda x: get_direction([x.pickup_latitude,x.pickup_longitude],
                                                                    [x.dropoff_latitude,x.dropoff_longitude]),axis=1)
```

We can get the trip direction in terms of compass bearing using the above formula.

### Q3: Traffic Speed Analysis
- Provide information about the average speed of traffic in New York
Analyze of the traffic speed changes in New York throughout the day
- Analyze of the traffic speed changes in New York throughout the day

```python
print(f"Average overall taxi speed in New York in 2013: {round(df_train['trip_speed_mi/hr'].mean(),2)} miles per hour or {round(df_train['trip_speed_mi/hr'].mean() * 1.609,2)} km per hour")
print(f"Maximum taxi speed in New York in 2013: {df_train['trip_speed_mi/hr'].max()} miles per hour or {df_train['trip_speed_mi/hr'].max() * 1.609} km per hour")
print(f"Minimum taxi speed in New York in 2013: {df_train['trip_speed_mi/hr'].min()} miles per hour or {df_train['trip_speed_mi/hr'].min() * 1.609} km per hour")
```
    Average overall taxi speed in New York in 2013: 
    13.66 miles per hour or 21.99 km per hour
    
    Maximum taxi speed in New York in 2013: 
    50.0 miles per hour or 80.45 km per hour
    
    Minimum taxi speed in New York in 2013: 
    0.03 miles per hour or 0.04827 km per hour


```python
fig_speed = px.histogram(df_train, x="trip_speed_mi/hr",nbins = 100,title='Histogram of Trip Speeds')
fig_speed = fig_speed.update_layout(xaxis_title = "Trip Speed", yaxis_title= 'Count')

fig_speed_pdf = px.histogram(df_train, x="trip_speed_mi/hr",nbins = 100,title='PDF of Trip Speeds', histnorm='probability density')
fig_speed_pdf = fig_speed_pdf.update_layout(xaxis_title = "Trip Speed", yaxis_title= 'Probability Density')
# Data skewed to the right, this is usually a result of a lower boundary in a data set 
# So if the dataset's lower bounds are extremely low relative to the rest of the data, this will cause the data to skew right.

fig_speed.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/Speed.html" height="525" width="100%"></iframe>

```python
table =[df_train['trip_speed_mi/hr'].mean(),df_train['trip_speed_mi/hr'].median(),df_train['trip_speed_mi/hr'].mode().values[0]]
print("Right Skewed Distribution which means mean > median > mode",end="\n\n")
print(tabulate([table],headers=["Speed Mean","Speed Median","Speed Mode"]))
```
    Right Skewed Distribution which means mean > median > mode

    Speed Mean    Speed Median    Speed Mode
    ------------  --------------  ------------
     13.6649          12.275            12

#### Trip Speed Per Hour of Day

We'll inspect the trip speed per hour of the day next

```python
pispeed_per_hourofday = df_train.groupby("pickup_hour",as_index=False)['trip_speed_mi/hr'].mean()
drspeed_per_hourofday = df_train.groupby("dropoff_hour",as_index=False)['trip_speed_mi/hr'].mean()

speed_hour = go.Figure()
speed_hour.add_trace(go.Scatter(x=pispeed_per_hourofday.pickup_hour, y=pispeed_per_hourofday['trip_speed_mi/hr'],
                             name='Average Pickup Speed per hour',mode='markers+lines'))
speed_hour.add_trace(go.Scatter(x=drspeed_per_hourofday.dropoff_hour, y=drspeed_per_hourofday['trip_speed_mi/hr'],
                             name='Average Dropoff Speed per hour',mode='markers+lines'))

speed_hour = speed_hour.update_layout(title = "Average Trip Speed Per Hour of Day", yaxis_title = "Average Trip Speed", xaxis_title ="Hour of Day")
speed_hour = speed_hour.update_xaxes(type='category')
speed_hour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_hour.html" height="525" width="100%"></iframe>

We see 5 AM being the fastest hour on average, as expected it is early in the morning and traffic is at its lowest. We also see that the slowest time on average is 9AM.

#### Trip Speed Per Day of Week

```python
# Avergae speed distribution per day

order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
pispeed_per_day = df_train.groupby("pickup_day",as_index=False)['trip_speed_mi/hr'].mean()
pispeed_per_day.sort_values(by=["pickup_day"],key=lambda x: x.map(order_dict),inplace=True)

drspeed_per_day = df_train.groupby("dropoff_day",as_index=False)['trip_speed_mi/hr'].mean()
drspeed_per_day.sort_values(by=["dropoff_day"],key=lambda x: x.map(order_dict),inplace=True)


speed_day = go.Figure()
speed_day.add_trace(go.Scatter(x=pispeed_per_day.pickup_day, y=pispeed_per_day['trip_speed_mi/hr'],
                             name='Average Pickup Speed per day',mode='markers+lines'))
speed_day.add_trace(go.Scatter(x=drspeed_per_day.dropoff_day, y=drspeed_per_day['trip_speed_mi/hr'],
                             name='Average Dropoff Speed per day',mode='markers+lines'))

speed_day = speed_day.update_layout(title = "Average Trip Speed Per Day", yaxis_title = "Average Trip Speed", xaxis_title ="Day")
speed_day = speed_day.update_xaxes(type='category')
speed_day.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_day.html" height="525" width="100%"></iframe>

Sunday is the fastest day, which is a bit unusual as a lot of people usually go out on Sundays due to it being a holiday but maybe in New York it's the opposite and people stay inside. Friday is the slowest day of the week.

#### Trip Speed per both Hour of Day and Day of Week

Going more granular with both day and time will allow us to see if this changes our findings and that there are combinations that weren't discovered.

```python
order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
pispeed_per_dayhour = df_train.groupby(["pickup_day","pickup_hour"],as_index=False)['trip_speed_mi/hr'].mean()
pispeed_per_dayhour.sort_values(by=["pickup_day","pickup_hour"],key=lambda x: x.map(order_dict),inplace=True)


speed_dayhour = px.bar(pispeed_per_dayhour, x="pickup_hour",y="trip_speed_mi/hr",facet_row="pickup_day",color='pickup_day',
                       barmode='stack',height=1000,facet_row_spacing=0.03)

speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Monday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Monday Mean Speed",annotation_position="bottom right",row=7,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Tuesday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Tuesday Mean Speed",annotation_position="bottom right",row=6,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Wednesday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Wednesday Mean Speed",annotation_position="bottom right",row=5,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Thursday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Thursday Mean Speed",annotation_position="bottom right",row=4,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Friday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Friday Mean Speed",annotation_position="bottom right",row=3,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Saturday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Saturday Mean Speed",annotation_position="bottom right",row=2,line_color='red')
speed_dayhour.add_hline(y=df_train[df_train['pickup_day']== 'Sunday']["trip_speed_mi/hr"].mean(), line_dash="dot",
                        annotation_text="Sunday Mean Speed",annotation_position="bottom right",row=1,line_color='red')

speed_dayhour = speed_dayhour.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
speed_dayhour = speed_dayhour.update_xaxes(title = "Hour",type='category')

for axis in speed_dayhour.layout:
    if type(speed_dayhour.layout[axis]) == go.layout.YAxis:
        speed_dayhour.layout[axis].title.text = ''
    if type(speed_dayhour.layout[axis]) == go.layout.XAxis:
        speed_dayhour.layout[axis].title.text = ''
        
speed_dayhour = speed_dayhour.update_layout(annotations = list(speed_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=-0.07,y=0.5,font=dict(size=14),
                                                  showarrow=False,text="Average Trip Speed",textangle=-90,
                                                  xref="paper",yref="paper")])

speed_dayhour = speed_dayhour.update_layout(annotations = list(speed_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=0.5,y=-0.05,font=dict(size=14),
                                                  showarrow=False,text="Hour",xref="paper",yref="paper")])

speed_dayhour = speed_dayhour.update_layout(legend_title = 'Day')

speed_dayhour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_dayhour.html" height="525" width="100%"></iframe>

Again we see on average 4 and 5AM are the fastest for all the days but we also see that compared to the other days Sunday has a higher speed at 11PM on average.

#### Traffic Speed per Time of Day

We increase the granularity further by binning the hours into different time of day
- Morning is from 6AM to 11:59PM
- Afternoon is from 12PM to 3:59PM
- Evening is from 4PM to 9:59PM
- Night is from 10PM to 12:59AM
- Late Night is from 1AM to 5:59AM

```python
# Average speed distribution per time of day

pispeed_per_tday = df_train.groupby("pickup_time_of_day",as_index=False)['trip_speed_mi/hr'].mean()
drspeed_per_tday = df_train.groupby("dropoff_time_of_day",as_index=False)['trip_speed_mi/hr'].mean()


speed_tday = go.Figure()
speed_tday.add_trace(go.Scatter(x=pispeed_per_tday.pickup_time_of_day, y=pispeed_per_tday['trip_speed_mi/hr'],
                             name='Average Pickup Speed per Time of Day',mode='markers+lines'))
speed_tday.add_trace(go.Scatter(x=drspeed_per_tday.dropoff_time_of_day, y=drspeed_per_tday['trip_speed_mi/hr'],
                             name='Average Dropoff Speed per Time of Day',mode='markers+lines'))

speed_tday = speed_tday.update_layout(title = "Average Trip Speed Per Time of Day", yaxis_title = "Average Trip Speed", xaxis_title ="Time of Day")
speed_tday = speed_tday.update_xaxes(type='category')
speed_tday.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_tday.html" height="525" width="100%"></iframe>

Late night is intuitively the fastest time and afernoon the slowest.

#### Traffic Speed per Month

```python
# Average speed distribution per month

order_dict = {"January":1, "February":2, "March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,
              "November":11,"December":12}
pispeed_per_month = df_train.groupby("pickup_month",as_index=False)['trip_speed_mi/hr'].mean()
pispeed_per_month.sort_values(by=["pickup_month"],key=lambda x: x.map(order_dict),inplace=True)

drspeed_per_month = df_train.groupby("dropoff_month",as_index=False)['trip_speed_mi/hr'].mean()
drspeed_per_month.sort_values(by=["dropoff_month"],key=lambda x: x.map(order_dict),inplace=True)

speed_month= go.Figure()
speed_month.add_trace(go.Scatter(x=pispeed_per_month.pickup_month, y=pispeed_per_month['trip_speed_mi/hr'],
                             name='Average Pickup Speed per Month',mode='markers+lines'))
speed_month.add_trace(go.Scatter(x=drspeed_per_month.dropoff_month, y=drspeed_per_month['trip_speed_mi/hr'],
                             name='Average Dropoff Speed per Month',mode='markers+lines'))

speed_month = speed_month.update_layout(title = "Average Trip Speed Per Month", yaxis_title = "Trip Speed", xaxis_title ="Month")
speed_month = speed_month.update_xaxes(type='category')
speed_month.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_month.html" height="525" width="100%"></iframe>

#### Traffic Speed per Direction of Trip

```python
# Average speed distribution per direction

speed_per_dir = df_train.groupby("trip_direction",as_index=False)['trip_speed_mi/hr'].mean()

speed_dir= go.Figure()
speed_dir.add_trace(go.Scatter(x=speed_per_dir.trip_direction, y=speed_per_dir['trip_speed_mi/hr'],
                             name='Average Speed per Direction',mode='markers+lines'))

speed_dir = speed_dir.update_layout(title = "Average Trip Speed Per Direction", yaxis_title = "Trip Speed", xaxis_title ="Direction")
speed_dir = speed_dir.update_xaxes(type='category')
speed_dir.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/speed_dir.html" height="525" width="100%"></iframe>

#### Adding borough information

I have added the pickup and dropoff boroughs to be used in the analysis as well. I have also tried to add the pickup and dropoff neighbourhood (using New Yorks around 200 neighborhoods), for more granularity but this deemed to be highly computationaly expensive. 

Adding the borough information can be seen as a challenging task at first but thanks to the [Shapley](https://shapely.readthedocs.io/en/stable/manual.html) library and to the New York boroughs geojson file found [here](https://github.com/dwillis/nyc-maps) it is possible to calculate where the trip started and ended like so:

```python
borough_nyc = json.loads(requests.get("https://raw.githubusercontent.com/dwillis/nyc-maps/master/boroughs.geojson").content)

boroughs = {}
for feature in borough_nyc['features']:
    name = feature['properties']['BoroName']
    code = feature['properties']['BoroCode']
    polygons = []
    for poly in feature['geometry']['coordinates']:
        polygons.append(Polygon(poly[0]))
    boroughs[code] = {'name':name,'polygon':MultiPolygon(polygons=polygons)}

boroughs = dict(sorted(boroughs.items()))


def get_borough(lat,lon):
    
    point = Point(lon,lat)
    boro = 'Outside Boroughs'
    
    for key,value in boroughs.items():
        if value['polygon'].contains(point):
            boro = value['name']
            
    return boro

trip_borough_list = []

for idx, row in df_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']].iterrows():
    trip_borough_list.append((get_borough(row['pickup_latitude'],row['pickup_longitude']),
                              get_borough(row['dropoff_latitude'],row['dropoff_longitude'])))
    
    
df_boroughs = pd.DataFrame(trip_borough_list, columns=["pickup_borough","dropoff_borough"])
df_boroughs.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_train = pd.concat([df_train,df_boroughs],axis=1)
df_train.reset_index(drop=True, inplace=True)
df_train.to_csv("../Data/Pre-processed Data/cleaned_data2.csv") #Saving for future use
```

We can then use this information for analyzing the traffic speed, traffic density and much more.

#### Traffic Speed between New York City Boroughs

```python
avg_speed_boroughs = df_train.groupby(["pickup_borough","dropoff_borough"],as_index=False)['trip_speed_mi/hr'].mean()
avg_speed_boroughs_img = pd.pivot_table(avg_speed_boroughs, index='pickup_borough', columns='dropoff_borough', values = 'trip_speed_mi/hr')

avg_speed_boroughs_fig = px.imshow(avg_speed_boroughs_img)
for i,r in enumerate(avg_speed_boroughs_img.values):
    for k,c in enumerate(r):
        if pd.isna(c):
            c = 'No Trips'
            avg_speed_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c}</b>',
                               showarrow=False,
                               font = dict(color='black'))
        else:
            avg_speed_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c:.2f}</b>',
                               showarrow=False,
                               font = dict(color='black'))
avg_speed_boroughs_fig = avg_speed_boroughs_fig.update_layout(title = 'Average Trip Speed between Borough',
                                                              xaxis_title = 'Dropoff Borough', yaxis_title = 'Pickup Borough')

avg_speed_boroughs_fig.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/avg_speed_boroughs_fig.html" height="525" width="100%"></iframe>

We can see that Trips from Queens to Staten Island are the fastest on average (wth a whopping 37 miles per hour) while trips from Staten Island to Manhattan are the slowest on average along with trips from the Outskirts to Staten Island.

```python
def make_pretty(styler):
    styler.set_caption("Borough Average Speed")
    styler.background_gradient(axis=None, vmin=1, vmax=30, cmap="plasma")
    return styler

borough_speed =  df_train.groupby('pickup_borough',as_index=False)['trip_speed_mi/hr'].mean()
borough_speed.columns  = ['Borough', 'Average Speed']
borough_speed.sort_values(by="Average Speed", ascending= False,inplace = True)
display(borough_speed.style.pipe(make_pretty))
```

<style type="text/css">
#T_46bff_row0_col1 {
  background-color: #fca338;
  color: #000000;
}
#T_46bff_row1_col1 {
  background-color: #ec7754;
  color: #f1f1f1;
}
#T_46bff_row2_col1 {
  background-color: #d14e72;
  color: #f1f1f1;
}
#T_46bff_row3_col1 {
  background-color: #cc4977;
  color: #f1f1f1;
}
#T_46bff_row4_col1 {
  background-color: #c33d80;
  color: #f1f1f1;
}
#T_46bff_row5_col1 {
  background-color: #b42e8d;
  color: #f1f1f1;
}
</style>
<table id="T_46bff">
  <caption>Borough Average Speed</caption>
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_46bff_level0_col0" class="col_heading level0 col0" >Borough</th>
      <th id="T_46bff_level0_col1" class="col_heading level0 col1" >Average Speed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_46bff_level0_row0" class="row_heading level0 row0" >4</th>
      <td id="T_46bff_row0_col0" class="data row0 col0" >Queens</td>
      <td id="T_46bff_row0_col1" class="data row0 col1" >23.922178</td>
    </tr>
    <tr>
      <th id="T_46bff_level0_row1" class="row_heading level0 row1" >5</th>
      <td id="T_46bff_row1_col0" class="data row1 col0" >Staten Island</td>
      <td id="T_46bff_row1_col1" class="data row1 col1" >20.205111</td>
    </tr>
    <tr>
      <th id="T_46bff_level0_row2" class="row_heading level0 row2" >0</th>
      <td id="T_46bff_row2_col0" class="data row2 col0" >Bronx</td>
      <td id="T_46bff_row2_col1" class="data row2 col1" >16.200803</td>
    </tr>
    <tr>
      <th id="T_46bff_level0_row3" class="row_heading level0 row3" >1</th>
      <td id="T_46bff_row3_col0" class="data row3 col0" >Brooklyn</td>
      <td id="T_46bff_row3_col1" class="data row3 col1" >15.686250</td>
    </tr>
    <tr>
      <th id="T_46bff_level0_row4" class="row_heading level0 row4" >3</th>
      <td id="T_46bff_row4_col0" class="data row4 col0" >Outside Boroughs</td>
      <td id="T_46bff_row4_col1" class="data row4 col1" >14.567957</td>
    </tr>
    <tr>
      <th id="T_46bff_level0_row5" class="row_heading level0 row5" >2</th>
      <td id="T_46bff_row5_col0" class="data row5 col0" >Manhattan</td>
      <td id="T_46bff_row5_col1" class="data row5 col1" >12.998500</td>
    </tr>
  </tbody>
</table>

### Q4: Traffic Volume Analysis

#### Traffic Volume Per Hour of Day

```python
traffic_density_pihour = df_train.groupby("pickup_hour",as_index=False).size()
traffic_density_drhour = df_train.groupby("dropoff_hour",as_index=False).size()


density_hour = go.Figure()
density_hour.add_trace(go.Bar(y=traffic_density_pihour.pickup_hour, x=traffic_density_pihour['size'],
                             name='Pickup Density per hour',orientation='h',width=0.3))
density_hour.add_trace(go.Bar(y=traffic_density_drhour.dropoff_hour, x=traffic_density_drhour['size'],
                             name='Dropoff Density per hour',orientation='h',width=0.3))

density_hour = density_hour.update_layout(title = "Taxi traffic Density Per Hour", yaxis_title = "Hour", 
                                          xaxis_title ="Number of trips",bargap =0.3,bargroupgap = 1,height=800)
density_hour = density_hour.update_yaxes(type='category',categoryorder ='array',categoryarray = traffic_density_pihour.pickup_hour.values.tolist()[::-1])
density_hour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_hour.html" height="525" width="100%"></iframe>

Mornings and Evening are the busiest times.

#### Traffic Volume Per Day of Week

```python
# Average speed distribution per day

order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}

traffic_density_piday = df_train.groupby("pickup_day",as_index=False).size()
traffic_density_piday.sort_values(by=["pickup_day"],key=lambda x: x.map(order_dict),inplace=True)

traffic_density_drday = df_train.groupby("dropoff_day",as_index=False).size()
traffic_density_drday.sort_values(by=["dropoff_day"],key=lambda x: x.map(order_dict),inplace=True)


density_day = go.Figure()
density_day.add_trace(go.Bar(y=traffic_density_piday.pickup_day, x=traffic_density_piday['size'],
                             name='Pickup Density per Day',orientation='h',width=0.3))
density_day.add_trace(go.Bar(y=traffic_density_drday.dropoff_day, x=traffic_density_drday['size'],
                             name='Dropoff Density per Day',orientation='h',width=0.3))
                      
density_day = density_day.update_layout(title = "Taxi traffic Density Per Day", yaxis_title = "Day", 
                                          xaxis_title ="Number of trips",bargap =0.3,bargroupgap = 1,height=800)
density_day = density_day.update_yaxes(type='category')
density_day.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_day.html" height="525" width="100%"></iframe>

#### Traffic Volume per both Hour of Day and Day of Week

```python
order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
density_per_dayhour = df_train.groupby(["pickup_day","pickup_hour"],as_index=False).size()
density_per_dayhour.sort_values(by=["pickup_day","pickup_hour"],key=lambda x: x.map(order_dict),inplace=True)


density_dayhour = px.bar(density_per_dayhour, x="pickup_hour",y="size",facet_row="pickup_day",color='pickup_day',
                       barmode='stack',height=1000,facet_row_spacing=0.03)

density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Monday Mean Density",annotation_position="bottom right",row=7,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Tuesday Mean Density",annotation_position="bottom right",row=6,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Wednesday Mean Density",annotation_position="bottom right",row=5,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Thursday Mean Density",annotation_position="bottom right",row=4,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Friday Mean Density",annotation_position="bottom right",row=3,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Saturday Mean Density",annotation_position="bottom right",row=2,line_color='red')
density_dayhour.add_hline(y=density_per_dayhour[density_per_dayhour.pickup_day == 'Monday']['size'].mean(), line_dash="dot",
                        annotation_text="Sunday Mean Density",annotation_position="bottom right",row=1,line_color='red')

density_dayhour = density_dayhour.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
density_dayhour = density_dayhour.update_xaxes(title = "Hour",type='category')

for axis in density_dayhour.layout:
    if type(density_dayhour.layout[axis]) == go.layout.YAxis:
        density_dayhour.layout[axis].title.text = ''
    if type(density_dayhour.layout[axis]) == go.layout.XAxis:
        density_dayhour.layout[axis].title.text = ''
        
density_dayhour = density_dayhour.update_layout(annotations = list(density_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=-0.07,y=0.5,font=dict(size=14),
                                                  showarrow=False,text="Number of Trips",textangle=-90,
                                                  xref="paper",yref="paper")])

density_dayhour = density_dayhour.update_layout(annotations = list(density_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=0.5,y=-0.05,font=dict(size=14),
                                                  showarrow=False,text="Hour",xref="paper",yref="paper")])

density_dayhour = density_dayhour.update_layout(legend_title = 'Day')

density_dayhour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_dayhour.html" height="525" width="100%"></iframe>

The exceptions regarding traffic volume mentioned previously are conceptualised in this graph. Friday and Saturday night are the busiest for days, while the afternoon is the busiest for Sunday.

#### Traffic Volume per Time of Day

```python
densitypi_per_tday = df_train.groupby("pickup_time_of_day",as_index=False).size()
densitydr_per_tday = df_train.groupby("dropoff_time_of_day",as_index=False).size()


density_tday = go.Figure()
density_tday.add_trace(go.Bar(y=densitypi_per_tday.pickup_time_of_day, x=densitypi_per_tday['size'],
                             name='Pickup Density per Time of Day',orientation='h'))
density_tday.add_trace(go.Bar(y=densitydr_per_tday.dropoff_time_of_day, x=densitydr_per_tday['size'],
                             name='Dropoff Density per Time of Day',orientation='h'))

density_tday = density_tday.update_layout(title = "Density Per Time of Day", yaxis_title = "Time of Day", xaxis_title ="Number of Trips")
density_tday = density_tday.update_yaxes(type='category')
density_tday.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_tday.html" height="525" width="100%"></iframe>

#### Traffic Volume per Month

```python
order_dict = {"January":1, "February":2, "March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,
              "November":11,"December":12}
densitypi_per_month = df_train.groupby("pickup_month",as_index=False).size()
densitypi_per_month.sort_values(by=["pickup_month"],key=lambda x: x.map(order_dict),inplace=True)

densitydr_per_month = df_train.groupby("dropoff_month",as_index=False).size()
densitydr_per_month.sort_values(by=["dropoff_month"],key=lambda x: x.map(order_dict),inplace=True)

density_month= go.Figure()
density_month.add_trace(go.Bar(y=densitypi_per_month.pickup_month, x=densitypi_per_month['size'],
                             name='Pickup Density per Month',orientation='h'))
density_month.add_trace(go.Bar(y=densitydr_per_month.dropoff_month, x=densitydr_per_month['size'],
                             name='Dropoff Density per Month',orientation='h'))

density_month = density_month.update_layout(title = "Density Per Month", yaxis_title = "Month", xaxis_title ="Number of Trips")
density_month = density_month.update_yaxes(type='category',categoryorder ='array',categoryarray = densitypi_per_month.pickup_month.values.tolist()[::-1])
density_month.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_month.html" height="525" width="100%"></iframe>

As expected due to our sampling (150K per month) we get a uniform distribution for the months.

#### Traffic Volume per Direction of Trip

```python
density_per_dir = df_train.groupby("trip_direction",as_index=False).size()

density_dir= go.Figure()
density_dir.add_trace(go.Bar(y=density_per_dir.trip_direction, x=density_per_dir['size'],
                             name='Density per Direction',orientation='h'))

density_dir = density_dir.update_layout(title = "Density Per Direction", yaxis_title = "Trip Direction", xaxis_title ="Number of Trips")
density_dir = density_dir.update_yaxes(type='category')
density_dir.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_dir.html" height="525" width="100%"></iframe>


#### Traffic Volume between NYC Boroughs

```python
density_boroughs = df_train.groupby(["pickup_borough","dropoff_borough"],as_index=False).size()
density_boroughs_img = pd.pivot_table(density_boroughs, index='pickup_borough', columns='dropoff_borough', values = 'size')

density_boroughs_fig = px.imshow(density_boroughs_img,color_continuous_scale=px.colors.sequential.Agsunset_r)
for i,r in enumerate(density_boroughs_img.values):
    for k,c in enumerate(r):
        if pd.isna(c):
            c = 'No Trips'
            density_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c}</b>',
                               showarrow=False,
                               font = dict(color='black',family='Open Sans'))
        else:
            density_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c:.0f}</b>',
                               showarrow=False,
                               font = dict(color='black',family='Open Sans'))
density_boroughs_fig = density_boroughs_fig.update_layout(title = 'Density between Borough',
                                                              xaxis_title = 'Dropoff Borough', yaxis_title = 'Pickup Borough')

density_boroughs_fig.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/density_boroughs_fig.html" height="525" width="100%"></iframe>

Looking at the trips between boroughs we see that trips within Manhattan dominate, while trips from Manhattan to Brooklyn and Queens and Queens to Manhattan are also high. Only 1 trip between Staten Island and Queens, Brooklyn and Manhattan.

#### Traffic Density per Borough and Area

```python
borough_area = {}
boroughs = borough_nyc["features"]
for b in boroughs:
    name = b["properties"]["BoroName"]
    a = area(b["geometry"])/(1609*1609) # converts from m^2 to mi^2
    borough_area[name] = a

pidensity_borough = df_train.groupby(["pickup_borough"],as_index=False).size()
pidensity_borough['area'] = pidensity_borough.pickup_borough.map(borough_area)
pidensity_borough['density_area'] = pidensity_borough['size']/pidensity_borough['area']

drdensity_borough = df_train.groupby(["dropoff_borough"],as_index=False).size()
drdensity_borough['area'] = drdensity_borough.dropoff_borough.map(borough_area)
drdensity_borough['density_area'] = drdensity_borough['size']/drdensity_borough['area']

borough_density = go.Figure()

borough_density.add_trace(go.Bar(x=pidensity_borough['pickup_borough'], y=pidensity_borough['size'],
                                     name='Pickup Borough Traffic Density'))

borough_density.add_trace(go.Bar(x=drdensity_borough['dropoff_borough'], y=drdensity_borough['size'],
                                     name='Dropoff Borough Traffic Density'))


borough_density = borough_density.update_layout(title = "Borough Traffic Density",
                                                         xaxis_title = "Borough", yaxis_title = "Density")

borough_density.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/borough_density.html" height="525" width="100%"></iframe>

Brooklyn has more dropoffs than pickups

Strategic positioning to meet the demand to Brooklyn can be beneficial

```python
borough_density_area = go.Figure()

borough_density_area.add_trace(go.Bar(x=pidensity_borough['pickup_borough'], y=pidensity_borough['density_area'],
                                     name='Pickup Borough Traffic Density per Area'))

borough_density_area.add_trace(go.Bar(x=drdensity_borough['dropoff_borough'], y=drdensity_borough['density_area'],
                                     name='Dropoff Borough Traffic Density per Area'))


borough_density_area = borough_density_area.update_layout(title = "Borough Traffic Density per Area",
                                                         xaxis_title = "Borough", yaxis_title = "Density/Area")

borough_density_area.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/borough_density_area.html" height="525" width="100%"></iframe>

Queens is by far the largest borough so we see here that Brooklyn is slightly higher on Traffic Density Per Area

We can also animate the traffic volume changes throughout the day using the Folium library as below, but the file is too big to publish below is a screen recording of how it looks like:

```python
df_density_animation = df_train[['pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','pickup_hour',
                                'dropoff_hour']]
pickup_coords_per_hour = []
for i in tqdm(range(0,24)):
    temp=[]
    for idx, row in df_density_animation[df_density_animation['pickup_hour'] == i].iterrows():
        temp.append([row['pickup_latitude'],row['pickup_longitude']])
    pickup_coords_per_hour.append(temp)

dropoff_coords_per_hour = []
for i in tqdm(range(0,24)):
    temp=[]
    for idx, row in df_density_animation[df_density_animation['dropoff_hour'] == i].iterrows():
        temp.append([row['dropoff_latitude'],row['dropoff_longitude']])
    dropoff_coords_per_hour.append(temp)


    
density_hourmap = folium.Map(location=[40.70977,-74.016609],zoom_start=10)
labels = ["Hour = " + str(i) for i in range(0,24)]
HeatMapWithTime(pickup_coords_per_hour,radius=3,auto_play=False,position='bottomright',index = labels,
               name="Pickup Traffic Volume").add_to(density_hourmap)
HeatMapWithTime(dropoff_coords_per_hour,radius=3,auto_play=False,position='bottomright',index = labels,
               name="Dropoff Traffic Volume").add_to(density_hourmap)

folium.LayerControl().add_to(density_hourmap)
display(density_hourmap)
```

<video width="100%" controls>
  <source src="/posts/New-York-Taxi-Analysis/heatmap.mp4" type="video/mp4">
</video>


We notice the same patterns as well in that most pickups are focused in the Manhattan area and that Dropoffs are more spread out to Queens, Brooklyn and JFK Airport

I have also animated the traffic volume per day and hour as well to get a more granular look but again its file is way too big to publish here, but the code to reproduce it can be found in my GitHub repo linked above.

#### Traffic Density Throughout the Year

```python
df_train['pickup_datetime'] = pd.to_datetime(df_train['pickup_datetime'])
df_train['dropoff_datetime'] = pd.to_datetime(df_train['dropoff_datetime'])
date_time_density = df_train.groupby(['pickup_hour',df_train.pickup_datetime.dt.date],as_index=False).size()
date_time_density.sort_values(by=["pickup_datetime","pickup_hour"],inplace=True)
date_time_density_img = date_time_density.pivot_table(columns='pickup_datetime',index="pickup_hour",values= "size")
date_time_density_img = date_time_density_img[::-1]


traffic_datehour = px.imshow(date_time_density_img,color_continuous_scale='OrRd',origin='lower')
traffic_datehour = traffic_datehour.update_xaxes(showticklabels = False)
traffic_datehour = traffic_datehour.update_layout(title= "Pickup Density 2013",xaxis_title= "Date",yaxis_title= "Hour")

highlighted_dates = ['2013-02-01','2013-02-08','2013-07-04','2013-09-13','2013-12-25','2013-12-31']
highlighted_hours = [20, 12, 17, 8, 14, 22]
text= ['<b>New York Knicks Final</b>', '<b>Weather Storm</b>','<b>4th of July</b>','<b>New York Yankees Final</b>','<b>Christmas</b>',
       '<b>New Years Eve</b>']

for x,y,text in zip(highlighted_dates, highlighted_hours,text):
    traffic_datehour.add_annotation(x=x, y=y, text=text,showarrow=True,arrowhead = 2,arrowwidth=1.5,xshift=2.5)
    
traffic_datehour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/traffic_datehour.html" height="525" width="100%"></iframe>

Looking at the whole year we can see the traffic volumne pattern on specific dates

### Q4: Tip Analysis

```python
fig_tip = px.histogram(df_train, x="tip_amount",nbins = 100,title='Histogram of Tip Amount')
fig_tip = fig_tip.update_layout(xaxis_title = "Tip Amount", yaxis_title= 'Count')

fig_tip.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/fig_tip.html" height="525" width="100%"></iframe>

```python
table =[df_train['tip_amount'].mean(),df_train['tip_amount'].median(),df_train['tip_amount'].mode().values[0]]
print("Right Skewed Distribution which means mean > median > mode",end="\n\n")
print(tabulate([table],headers=["Tip Amount Mean","Tip Amount Median","Tip Amount Mode"]))
```
    Right Skewed Distribution which means mean > median > mode

    Tip Amount Mean    Tip Amount Median    Tip Amount Mode
    -----------------  -------------------  -----------------
          1.32447                    1                  0

We see that most people don't tip.

#### Tip per Hour of Day

```python
tip_per_hourofday = df_train.groupby("dropoff_hour",as_index=False)['tip_amount'].mean()

tip_hour = go.Figure()
tip_hour.add_trace(go.Scatter(x=tip_per_hourofday.dropoff_hour, y=tip_per_hourofday['tip_amount'],
                             name='Average Tip Amount per hour',mode='markers+lines'))

tip_hour = tip_hour.update_layout(title = "Average Tip Amount Per Hour of Day", yaxis_title = "Average Tip Amount", xaxis_title ="Hour of Day")
tip_hour = tip_hour.update_xaxes(type='category')
tip_hour.show() 
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_hour.html" height="525" width="100%"></iframe>


#### Tip per Day of Week

```python
order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
tip_per_day = df_train.groupby("dropoff_day",as_index=False)['tip_amount'].mean()
tip_per_day.sort_values(by=["dropoff_day"],key=lambda x: x.map(order_dict),inplace=True)


tip_day = go.Figure()
tip_day.add_trace(go.Scatter(x=tip_per_day.dropoff_day, y=tip_per_day['tip_amount'],
                             name='Average Tip Amount per day',mode='markers+lines'))


tip_day = tip_day.update_layout(title = "Average Tip Amount Per Day", yaxis_title = "Average Tip Amount", xaxis_title ="Day")
tip_day = tip_day.update_xaxes(type='category')
tip_day.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/tip_day.html" height="525" width="100%"></iframe>


#### Tip per Hour of Day and Day of Week

```python
order_dict = {"Monday":1, "Tuesday":2, "Wednesday":3,"Thursday":4,"Friday":5,"Saturday":6,"Sunday":7}
tip_per_dayhour = df_train.groupby(["dropoff_day","dropoff_hour"],as_index=False)['tip_amount'].mean()
tip_per_dayhour.sort_values(by=["dropoff_day","dropoff_hour"],key=lambda x: x.map(order_dict),inplace=True)


tip_dayhour = px.bar(tip_per_dayhour, x="dropoff_hour",y="tip_amount",facet_row="dropoff_day",color='dropoff_day',
                       barmode='stack',height=1000,facet_row_spacing=0.03)

tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Monday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Monday Mean Tip Amount",annotation_position="bottom right",row=7,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Tuesday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Tuesday Mean Tip Amount",annotation_position="bottom right",row=6,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Wednesday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Wednesday Mean Tip Amount",annotation_position="bottom right",row=5,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Thursday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Thursday Mean Tip Amount",annotation_position="bottom right",row=4,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Friday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Friday Mean Tip Amount",annotation_position="bottom right",row=3,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Saturday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Saturday Mean Tip Amount",annotation_position="bottom right",row=2,line_color='red')
tip_dayhour.add_hline(y=df_train[df_train['dropoff_day']== 'Sunday']["tip_amount"].mean(), line_dash="dot",
                        annotation_text="Sunday Mean Tip Amount",annotation_position="bottom right",row=1,line_color='red')

tip_dayhour = tip_dayhour.for_each_annotation(lambda x: x.update(text=x.text.split("=")[-1]))
tip_dayhour = tip_dayhour.update_xaxes(title = "Hour",type='category')

for axis in tip_dayhour.layout:
    if type(tip_dayhour.layout[axis]) == go.layout.YAxis:
        tip_dayhour.layout[axis].title.text = ''
    if type(tip_dayhour.layout[axis]) == go.layout.XAxis:
        tip_dayhour.layout[axis].title.text = ''
        
tip_dayhour = tip_dayhour.update_layout(annotations = list(tip_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=-0.07,y=0.5,font=dict(size=14),
                                                  showarrow=False,text="Average Tip Amount",textangle=-90,
                                                  xref="paper",yref="paper")])

tip_dayhour = tip_dayhour.update_layout(annotations = list(tip_dayhour.layout.annotations) + 
                            [go.layout.Annotation(x=0.5,y=-0.05,font=dict(size=14),
                                                  showarrow=False,text="Hour",xref="paper",yref="paper")])

tip_dayhour = tip_dayhour.update_layout(legend_title = 'Day')

tip_dayhour.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_dayhour.html" height="525" width="100%"></iframe>

This if a slightly harder figure to interpret so let's simplify it:

```python
tip_per_dayhour_img = tip_per_dayhour.pivot_table(columns='dropoff_day', index='dropoff_hour',values='tip_amount')
tip_per_dayhour_img = tip_per_dayhour_img[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
tip_per_dayhour_img = tip_per_dayhour_img[::-1]
tip_per_dayhour_img_fig = px.imshow(tip_per_dayhour_img,origin='lower')
tip_per_dayhour_img_fig = tip_per_dayhour_img_fig.update_layout(title = 'Heatmap of Average Tip Amount', xaxis_title='Day',
                                                               yaxis_title= 'Hour')
tip_per_dayhour_img_fig.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_per_dayhour_img_fig.html" height="525" width="100%"></iframe>

#### Tip per Time of Day

```python
tip_per_tday = df_train.groupby("dropoff_time_of_day",as_index=False)['tip_amount'].mean()


tip_tday = go.Figure()
tip_tday = go.Figure()
tip_tday.add_trace(go.Scatter(x=tip_per_tday.dropoff_time_of_day, y=tip_per_tday['tip_amount'],
                             name='Average Tip Amount per Time of Day',mode='markers+lines'))

tip_tday = tip_tday.update_layout(title = "Average Tip Amount per Time of Day", yaxis_title = "Average Tip Amount", 
                                  xaxis_title ="Time of Day")
tip_tday = tip_tday.update_xaxes(type='category')
tip_tday.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_tday.html" height="525" width="100%"></iframe>

#### Tip per Month

```python
# Average tipping per month

order_dict = {"January":1, "February":2, "March":3,"April":4,"May":5,"June":6,"July":7,"August":8,"September":9,"October":10,
              "November":11,"December":12}

tip_per_month = df_train.groupby("dropoff_month",as_index=False)['tip_amount'].mean()
tip_per_month.sort_values(by=["dropoff_month"],key=lambda x: x.map(order_dict),inplace=True)

tip_month= go.Figure()

tip_month.add_trace(go.Scatter(x=tip_per_month.dropoff_month, y=tip_per_month['tip_amount'],
                             name='Average Tip Amount per Month',mode='markers+lines'))

tip_month = tip_month.update_layout(title = "Average Tip Amount Per Month", yaxis_title = "Average Tip Amount", xaxis_title ="Month")
tip_month = tip_month.update_xaxes(type='category')
tip_month.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_month.html" height="525" width="100%"></iframe>

#### Tip per Direction

```python
# Average tipping per direction

tip_per_dir = df_train.groupby("trip_direction",as_index=False)['tip_amount'].mean()

tip_dir= go.Figure()
tip_dir.add_trace(go.Scatter(x=tip_per_dir.trip_direction, y=tip_per_dir['tip_amount'],
                             name='Average Tip Amount per Direction',mode='markers+lines'))

tip_dir = tip_dir.update_layout(title = "Average Tip Amount Per Direction", yaxis_title = "Average Tip Amount", 
                                xaxis_title ="Direction")
tip_dir = tip_dir.update_xaxes(type='category')
tip_dir.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_dir.html" height="525" width="100%"></iframe>

On average people tip the highest early in the morning (maybe going home drunk from a nightout :laughing:), lowest at the afternoon. People also tip highest on Wednesdays and lowest on Saturdays but these differences aren't that significant (of a few cents). We can also analyze the tipping behaviour with regards to geographic data.

#### Tip per Borough

```python
avg_tip_boroughs = df_train.groupby(["pickup_borough","dropoff_borough"],as_index=False)['tip_amount'].mean()
avg_tip_boroughs_img = pd.pivot_table(avg_tip_boroughs, index='pickup_borough', columns='dropoff_borough', values = 'tip_amount')

avg_tip_boroughs_fig = px.imshow(avg_tip_boroughs_img)
for i,r in enumerate(avg_tip_boroughs_img.values):
    for k,c in enumerate(r):
        if pd.isna(c):
            c = 'No Trips'
            avg_tip_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c}</b>',
                               showarrow=False,
                               font = dict(color='black'))
        else:
            avg_tip_boroughs_fig.add_annotation(x=k,y=i,
                               text=f'<b>{c:.2f}</b>',
                               showarrow=False,
                               font = dict(color='black'))
avg_tip_boroughs_fig = avg_tip_boroughs_fig.update_layout(title = 'Average Tip Amount between Borough',
                                                              xaxis_title = 'Dropoff Borough', yaxis_title = 'Pickup Borough')

avg_tip_boroughs_fig.show()
```
<iframe src= "/posts/New-York-Taxi-Analysis/avg_tip_boroughs_fig.html" height="525" width="100%"></iframe>

```python
tip_borough = df_train.groupby('dropoff_borough',as_index=False)['tip_amount'].mean()
tip_borough.sort_values(by='tip_amount',inplace=True)

tip_borough_fig = px.bar(tip_borough, y='dropoff_borough', x='tip_amount')
tip_borough_fig = tip_borough_fig.update_layout(title= 'Average Tip Amount Per Borough', xaxis_title = 'Average Tip Amount',
                                               yaxis_title= 'Borough')
tip_borough_fig.show()
```

<iframe src= "/posts/New-York-Taxi-Analysis/tip_borough_fig.html" height="525" width="100%"></iframe>

Queens is the highest tip paying borough when disregarding the low density of trips in Staten Island and the Outside Boroughs

We saw earlier that there isn't a significant difference between the tipping of various time of the day on average but we can also investigate the correlation as well

```python
columns=list(range(0,24))
correlation_hour = pd.DataFrame(index=["tip_amount"],columns=columns)
hour_one_hot = pd.get_dummies(df_train.dropoff_hour)
for i,col in enumerate(columns):
    cor = df_train.tip_amount.corr(hour_one_hot[col])
    correlation_hour.loc[:,columns[i]] = cor
plt.figure(figsize=(25,5))
g = sns.heatmap(correlation_hour,annot=True,fmt='.4f', annot_kws={"fontsize":12})
g.set_yticklabels(['Tip Amount'], fontsize = 16)
plt.title("Correlation Matrix between Tip Amount and Hour of Day", fontdict={'fontsize':20})
plt.plot()

columns=df_train.dropoff_time_of_day.unique()
correlation_tday = pd.DataFrame(index=["tip_amount"],columns=columns)
tday_onehot = pd.get_dummies(df_train.dropoff_time_of_day)
for i,col in enumerate(columns):
    cor = df_train.tip_amount.corr(tday_onehot[col])
    correlation_tday.loc[:,columns[i]] = cor
plt.figure(figsize=(25,5))
g = sns.heatmap(correlation_tday,annot=True,fmt='.4f', annot_kws={"fontsize":16})
g.set_yticklabels(['Tip Amount'], fontsize = 16)
plt.title("Correlation Matrix between Tip Amount and Time of Day", fontdict={'fontsize':20})
plt.plot()
```

<img src="/posts/New-York-Taxi-Analysis/corr_thour.png" width="100%">


<img src="/posts/New-York-Taxi-Analysis/corr_tday.png" width="100%">

Again no significant correlation, but we can still go even further with simple linear regression

```python
import statsmodels.api as sm

lr = sm.OLS(df_train.tip_amount, sm.add_constant(hour_one_hot.drop(23, axis=1))).fit()
lr2 = sm.OLS(df_train.tip_amount, sm.add_constant(tday_onehot.drop('Night',axis=1))).fit()
```


```python
lr.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>tip_amount</td>    <th>  R-squared:         </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.002</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   108.0</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 30 Oct 2022</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>06:35:25</td>     <th>  Log-Likelihood:    </th> <td>-2.5450e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>1182030</td>     <th>  AIC:               </th>  <td>5.090e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>1182006</td>     <th>  BIC:               </th>  <td>5.090e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>    23</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    1.4465</td> <td>    0.008</td> <td>  181.745</td> <td> 0.000</td> <td>    1.431</td> <td>    1.462</td>
</tr>
<tr>
  <th>0</th>     <td>    0.0436</td> <td>    0.012</td> <td>    3.664</td> <td> 0.000</td> <td>    0.020</td> <td>    0.067</td>
</tr>
<tr>
  <th>1</th>     <td>   -0.0119</td> <td>    0.014</td> <td>   -0.861</td> <td> 0.389</td> <td>   -0.039</td> <td>    0.015</td>
</tr>
<tr>
  <th>2</th>     <td>   -0.0992</td> <td>    0.015</td> <td>   -6.514</td> <td> 0.000</td> <td>   -0.129</td> <td>   -0.069</td>
</tr>
<tr>
  <th>3</th>     <td>   -0.1342</td> <td>    0.017</td> <td>   -7.934</td> <td> 0.000</td> <td>   -0.167</td> <td>   -0.101</td>
</tr>
<tr>
  <th>4</th>     <td>   -0.1116</td> <td>    0.019</td> <td>   -5.870</td> <td> 0.000</td> <td>   -0.149</td> <td>   -0.074</td>
</tr>
<tr>
  <th>5</th>     <td>    0.2021</td> <td>    0.022</td> <td>    9.390</td> <td> 0.000</td> <td>    0.160</td> <td>    0.244</td>
</tr>
<tr>
  <th>6</th>     <td>    0.0184</td> <td>    0.015</td> <td>    1.189</td> <td> 0.235</td> <td>   -0.012</td> <td>    0.049</td>
</tr>
<tr>
  <th>7</th>     <td>   -0.1275</td> <td>    0.012</td> <td>  -10.354</td> <td> 0.000</td> <td>   -0.152</td> <td>   -0.103</td>
</tr>
<tr>
  <th>8</th>     <td>   -0.0405</td> <td>    0.011</td> <td>   -3.557</td> <td> 0.000</td> <td>   -0.063</td> <td>   -0.018</td>
</tr>
<tr>
  <th>9</th>     <td>   -0.0629</td> <td>    0.011</td> <td>   -5.489</td> <td> 0.000</td> <td>   -0.085</td> <td>   -0.040</td>
</tr>
<tr>
  <th>10</th>    <td>   -0.1860</td> <td>    0.012</td> <td>  -16.050</td> <td> 0.000</td> <td>   -0.209</td> <td>   -0.163</td>
</tr>
<tr>
  <th>11</th>    <td>   -0.2671</td> <td>    0.012</td> <td>  -22.231</td> <td> 0.000</td> <td>   -0.291</td> <td>   -0.244</td>
</tr>
<tr>
  <th>12</th>    <td>   -0.2721</td> <td>    0.012</td> <td>  -22.997</td> <td> 0.000</td> <td>   -0.295</td> <td>   -0.249</td>
</tr>
<tr>
  <th>13</th>    <td>   -0.2713</td> <td>    0.012</td> <td>  -22.860</td> <td> 0.000</td> <td>   -0.295</td> <td>   -0.248</td>
</tr>
<tr>
  <th>14</th>    <td>   -0.2296</td> <td>    0.012</td> <td>  -18.927</td> <td> 0.000</td> <td>   -0.253</td> <td>   -0.206</td>
</tr>
<tr>
  <th>15</th>    <td>   -0.2090</td> <td>    0.012</td> <td>  -16.809</td> <td> 0.000</td> <td>   -0.233</td> <td>   -0.185</td>
</tr>
<tr>
  <th>16</th>    <td>   -0.1547</td> <td>    0.013</td> <td>  -12.346</td> <td> 0.000</td> <td>   -0.179</td> <td>   -0.130</td>
</tr>
<tr>
  <th>17</th>    <td>   -0.1704</td> <td>    0.012</td> <td>  -13.808</td> <td> 0.000</td> <td>   -0.195</td> <td>   -0.146</td>
</tr>
<tr>
  <th>18</th>    <td>   -0.1385</td> <td>    0.011</td> <td>  -12.242</td> <td> 0.000</td> <td>   -0.161</td> <td>   -0.116</td>
</tr>
<tr>
  <th>19</th>    <td>   -0.1376</td> <td>    0.011</td> <td>  -12.407</td> <td> 0.000</td> <td>   -0.159</td> <td>   -0.116</td>
</tr>
<tr>
  <th>20</th>    <td>   -0.1295</td> <td>    0.011</td> <td>  -11.259</td> <td> 0.000</td> <td>   -0.152</td> <td>   -0.107</td>
</tr>
<tr>
  <th>21</th>    <td>   -0.0831</td> <td>    0.012</td> <td>   -7.147</td> <td> 0.000</td> <td>   -0.106</td> <td>   -0.060</td>
</tr>
<tr>
  <th>22</th>    <td>   -0.0468</td> <td>    0.011</td> <td>   -4.097</td> <td> 0.000</td> <td>   -0.069</td> <td>   -0.024</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1253930.783</td> <th>  Durbin-Watson:     </th>   <td>   2.001</td>   
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>    <th>  Jarque-Bera (JB):  </th> <td>385938280.509</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 4.829</td>    <th>  Prob(JB):          </th>   <td>    0.00</td>   
</tr>
<tr>
  <th>Kurtosis:</th>        <td>90.993</td>    <th>  Cond. No.          </th>   <td>    21.6</td>   
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.




```python
lr2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>tip_amount</td>    <th>  R-squared:         </th>  <td>   0.001</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.001</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>   380.3</td>  
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 30 Oct 2022</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>06:35:26</td>     <th>  Log-Likelihood:    </th> <td>-2.5454e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>1182030</td>     <th>  AIC:               </th>  <td>5.091e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>1182025</td>     <th>  BIC:               </th>  <td>5.091e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     4</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
       <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>      <td>    1.4439</td> <td>    0.005</td> <td>  300.591</td> <td> 0.000</td> <td>    1.434</td> <td>    1.453</td>
</tr>
<tr>
  <th>Afternon</th>   <td>   -0.2440</td> <td>    0.007</td> <td>  -36.964</td> <td> 0.000</td> <td>   -0.257</td> <td>   -0.231</td>
</tr>
<tr>
  <th>Evening</th>    <td>   -0.1314</td> <td>    0.006</td> <td>  -22.143</td> <td> 0.000</td> <td>   -0.143</td> <td>   -0.120</td>
</tr>
<tr>
  <th>Late night</th> <td>   -0.0448</td> <td>    0.008</td> <td>   -5.577</td> <td> 0.000</td> <td>   -0.061</td> <td>   -0.029</td>
</tr>
<tr>
  <th>Morning</th>    <td>   -0.1181</td> <td>    0.006</td> <td>  -19.483</td> <td> 0.000</td> <td>   -0.130</td> <td>   -0.106</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1253963.407</td> <th>  Durbin-Watson:     </th>   <td>   2.001</td>   
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>    <th>  Jarque-Bera (JB):  </th> <td>385045385.766</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 4.830</td>    <th>  Prob(JB):          </th>   <td>    0.00</td>   
</tr>
<tr>
  <th>Kurtosis:</th>        <td>90.890</td>    <th>  Cond. No.          </th>   <td>    6.67</td>   
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


From this analysis we see that 1AM and 6AM are insignificant while most other hours of the day contribute negatively to the average tipping amount except for 5AM as expected.

We can go even further by trying to model the tipping amount with the features we have. I'll opt for s Random Forest Model, which I will probably write about in the future along with tree models. The code for this is as follows:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
categorical_columns = ['payment_type','dropoff_day','dropoff_month','dropoff_time_of_day','dropoff_borough']

hour_transformer = ColumnTransformer(
    transformers=[
        ("categorical", one_hot_encoder, categorical_columns),
        ("hour_sin", sin_transformer(24), ["dropoff_hour"]),
        ("hour_cos", cos_transformer(24), ["dropoff_hour"])
    ],
    remainder=StandardScaler()
)

rf_pipe = make_pipeline(
    hour_transformer,
    RandomForestRegressor(),
)

X = df_train[['payment_type','dropoff_day','dropoff_month','dropoff_time_of_day','dropoff_borough','dropoff_hour','trip_distance',
             'trip_time_in_secs','pickup_longitude','pickup_latitude','fare_amount']]

y = df_train.tip_amount

_  = rf_pipe.fit(X,y)
```

We also cyclically encode the time feature (dropoff hour), this is usually done when we have cyclical features such as hour of the day, days of the week, and others. This can be achieved by taking the sine and cosine of the cyclical feature divided by the period (total number/maximum number of the feature, i,e. 7 if days of the week and 24 if hours). We also one-hot encode our categorical variables. I'll also write a future article about cyclical encoding of features in the future.  We evaluate how well the model performs on our training data.

```python
print("Coefficient of determination (R^2) of the model is: ", rf_pipe.score(X,y))
```
    Coefficient of determination (R^2) of the model is: 
    0.962619575415421
  
Eventually we would like to see the most important deatures that contribute to predicting the tipping amount (More on this in a future article)

```python
features = pd.concat([pd.get_dummies(X[categorical_columns]),X[X.columns.difference(categorical_columns)]],axis=1).columns.tolist()
features[features.index('dropoff_hour')] = 'dropoff_hour_sin'
features.insert(features.index('dropoff_hour_sin') +1,'dropoff_hour_cos')
feature_importance = rf_pipe.steps[-1][1].feature_importances_
feature_importance_df = pd.DataFrame({'Feature Name':features, 'Importance':feature_importance})
feature_importance_df.sort_values(by='Importance',ascending=False,inplace=True)
feature_importance_df.reset_index(drop=True, inplace=True)

feature_importance_fig = px.bar(feature_importance_df, x='Importance', y='Feature Name',
       category_orders={'Feature Name':feature_importance_df['Feature Name']},height=900)
feature_importance_fig.show()
```


<iframe src= "/posts/New-York-Taxi-Analysis/feature_importance_fig.html" height="525" width="100%"></iframe>

We see that a major influencing factor in determining the tipping amount is how long the trip took and what payment method was used (Cash in our case as intuitively that's what most people tip taxi rides with). We can see the correlation between the tip, time taken and distance to confirm that there is a correlation but a weak one.  

<img src="/posts/New-York-Taxi-Analysis/tip_corr.png" width="100%">

Seeing this we analyze this again with linear regression modelling the average tipping amount

```python
lr_distance = sm.OLS(df_train.tip_amount, sm.add_constant(df_train.trip_distance)).fit()
lr_distance.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>tip_amount</td>    <th>  R-squared:         </th>  <td>   0.259</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.259</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>4.131e+05</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 30 Oct 2022</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>08:03:25</td>     <th>  Log-Likelihood:    </th> <td>-2.3691e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>1182030</td>     <th>  AIC:               </th>  <td>4.738e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>1182028</td>     <th>  BIC:               </th>  <td>4.738e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
        <td></td>           <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>         <td>    0.3976</td> <td>    0.002</td> <td>  181.351</td> <td> 0.000</td> <td>    0.393</td> <td>    0.402</td>
</tr>
<tr>
  <th>trip_distance</th> <td>    0.3109</td> <td>    0.000</td> <td>  642.736</td> <td> 0.000</td> <td>    0.310</td> <td>    0.312</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1150484.269</td> <th>  Durbin-Watson:     </th>   <td>   2.002</td>   
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>    <th>  Jarque-Bera (JB):  </th> <td>903227788.715</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 3.820</td>    <th>  Prob(JB):          </th>   <td>    0.00</td>   
</tr>
<tr>
  <th>Kurtosis:</th>        <td>138.207</td>   <th>  Cond. No.          </th>   <td>    6.15</td>   
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



A positive contribution

```python
scaler = StandardScaler()

scaled_duration = scaler.fit_transform(df_train.trip_time_in_secs.values.reshape(-1,1)) 
lr_duration = sm.OLS(df_train.tip_amount, sm.add_constant(scaled_duration)).fit()
lr_duration.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>tip_amount</td>    <th>  R-squared:         </th>  <td>   0.201</td>  
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   0.201</td>  
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>2.976e+05</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Sun, 30 Oct 2022</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>   
</tr>
<tr>
  <th>Time:</th>                 <td>08:03:26</td>     <th>  Log-Likelihood:    </th> <td>-2.4135e+06</td>
</tr>
<tr>
  <th>No. Observations:</th>      <td>1182030</td>     <th>  AIC:               </th>  <td>4.827e+06</td> 
</tr>
<tr>
  <th>Df Residuals:</th>          <td>1182028</td>     <th>  BIC:               </th>  <td>4.827e+06</td> 
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>      <td> </td>     
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>     
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>       <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th> <td>    1.3245</td> <td>    0.002</td> <td>  772.412</td> <td> 0.000</td> <td>    1.321</td> <td>    1.328</td>
</tr>
<tr>
  <th>x1</th>    <td>    0.9355</td> <td>    0.002</td> <td>  545.563</td> <td> 0.000</td> <td>    0.932</td> <td>    0.939</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>1204548.432</td> <th>  Durbin-Watson:     </th>   <td>   2.001</td>   
</tr>
<tr>
  <th>Prob(Omnibus):</th>   <td> 0.000</td>    <th>  Jarque-Bera (JB):  </th> <td>758035989.092</td>
</tr>
<tr>
  <th>Skew:</th>            <td> 4.241</td>    <th>  Prob(JB):          </th>   <td>    0.00</td>   
</tr>
<tr>
  <th>Kurtosis:</th>        <td>126.771</td>   <th>  Cond. No.          </th>   <td>    1.00</td>   
</tr>
</table><br/><br/>Notes:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

```python
print(f"On Average {np.round(scaler.inverse_transform(lr_duration.params['x1'].reshape(1,-1))[0][0],0)} seconds or 20 minutes contribute to a 1 dollar tip") 
```
    On Average 1249.0 seconds or 20 minutes contribute to a 
    1 dollar tip

This sums up our findings and this article. I have conducted other analysis as well as a prediction model for predicting the pickup density after clustering New York into several buckets/regions and binning of time. All of this can be found in the projects GitHub repository.

Thank you for reading and if you have any questions regarding the project I'm an email away. Until next time, Bye!
</span>
