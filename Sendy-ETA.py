#!/usr/bin/env python
# coding: utf-8

# # Notebook Submission for KamiLimu DS-Track
# 
# ### Project : Forecasting  the expected time of arrival of deliveries using local data from the Sendy platform
# 
# #### Table of Contents
# 
# <ul>
# <li><a href="#introduction">Introduction</a></li>&nbsp;
# <li><a href="#questions">Question(s) for Analysis</a></li>&nbsp;
# <li><a href="#imports">Imports</a></li>&nbsp;
# <li><a href="#variable_desc">Variables Description</a></li>&nbsp;
# <li><a href="#load_data">Load Data</a></li>&nbsp;
# <li><a href="#data_cleaning">Data Cleaning</a></li>&nbsp;
# <li><a href="#feature_engineering">Feature Engineering</a></li>&nbsp;
# <li><a href="#eda">Exploratory Data Analysis</a></li>&nbsp;
# <li><a href="#modelling">Modelling</a></li>&nbsp;
# <li><a href="#conclusions">Conclusion</a></li>&nbsp;
# </ul>
#     

# <a id='introduction'><a>
#  
# ## Introduction
#     
# <p>
# This projects aims to get an accurate prediction of the time of arrival of deliveries to the customers  
#     
# It takes into account the following time deltas :
#        
# 
# >Time the customer made the order  
# Time the order was confirmed by the rider  
# Time the rider arrived at the pickup station  
# Time the rider took at the pickup station
# Time the rider took after leaving the pickup station to arrive at the customer's destination
#     
#     
#     
#  
# Breaking down of the timelines to multiple data points enables the company to clearly spotlight  which section of its logistics channels  has the most holdup.  
#     
# This enables the company to work on improving the holdups and consequently optimizing their logistics while increasing customer satisfaction 
#     
# </p>
#     
#  
# 

# <a id='questions'><a>
#     
# ## Question(s) for Analysis 

# > What factors are important in predicting ETA ?  
# How does distance affect ETA ?  
# What kind of spatial analysis can we perform on the data ?  
# Which spatial data features can we add to improve the model ?  
# 

# <a id='imports'><a>
# 
# ## Imports

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime
import os
import descartes
import geopandas as gpd
from shapely.geometry import Point ,Polygon

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import altair as alt
alt.data_transformers.disable_max_rows()

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree 
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
import matplotlib.mlab as mlab
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook as tqdm
import pickle
import warnings
import json
import pymorphy2
warnings.filterwarnings("ignore")
import requests

pd.set_option('display.max_columns', None)


# <a id='variable_desc'><a>
# 
# ## Variables Description

# In[2]:


variable_definitions =  pd.read_csv("VariableDefinitions.csv")
variable_definitions


# ![image-4.png](attachment:image-4.png)

# <a id='load_data'><a>
#     
# ## Load Data

# In[3]:


train_df = pd.read_csv("Train.csv")
test_df = pd.read_csv("Test.csv")
riders_df =  pd.read_csv("Riders.csv")
counties_map = gpd.read_file("kenyan-counties/County.shp")


file = os.listdir("planet_36.633,-1.415_37.037,-1.149-shp/shape")
path = [os.path.join("planet_36.633,-1.415_37.037,-1.149-shp/shape", i) for i in file if ".shp" in i]

gdf = gpd.GeoDataFrame(pd.concat([gpd.read_file(i) for i in path], 
                        ignore_index=True), crs=gpd.read_file(path[0]).crs)

crs = {'init':'epsg:4326'}


# In[4]:


train_df.info()
train_df.head()


# In[5]:


test_df.info()
test_df.head()


# In[6]:


len_train = len(train_df)
data = pd.concat([train_df, test_df], axis = 0).reset_index(drop = True)


# In[7]:


print("Length of train_data" ,len(train_df))
print("Length of test_data",len(test_df))
print("Length of train_test_data" , len(data))


# In[8]:


data.shape


# ### OSRM - Open Source Routing Machine
# 
# OSRM API  is a modern C++ routing engine for shortest paths in road networks
# 
# Link - http://project-osrm.org/docs/v5.24.0/api/?language=Python#
# 
# You make a GET request to the API and you get a JSON reponse 
# 
# >GET-REQUEST:  
# http://router.project-osrm.org/{service}/{version}/{profile}/{coordinates}[.{format}]?option=value&option=value  
# >>Service - route,nearest,match,trip  route service - Finds the fastest route between coordinates in the supplied order  
# >>Version -  default v1  
# >>Profile -  Mode of transportation : car,bike,foot  
# >>Format  - default JSON
# 
# >JSON RESPONSE:
# >>Code  : if the request was successful Ok  
# >>Waypoints: Array of Waypoint objects representing all waypoints in order .Object used to describe waypoint on a route.  
# >>Routes: An array of Route objects, ordered by descending recommendation rank.Represents a route through (potentially multiple) waypoints.  
# 
#     
# >Example of output response    
# >>ul = http://router.project-osrm.org/route/v1/bike/36.829741,-1.3004062;36.8303703,-1.3177547  
# >>rsp = {
#    >>><B>"code"</B>:"Ok",  
#     >>><B>"routes"</B>:[{"geometry":"Fahx_F~DeBiBzFkCxL`[lAt@h]aMTaBaAcDEoDnGzPhAz@jDb@vBiNfASvLrARcB}AYDUdEr@",  
#                 "legs":[{"steps":[],"summary":"","weight":419.4,"duration":419.4,"distance":3213.2}],  
#                 "weight_name":"routability",  
#                 "weight":419.4,  
#                 "duration":419.4,
#                 "distance":3213.2}],  
#       >>><B> "waypoints"</B>:  [{"hint":"TmTIgTOrfIdCAAAAMwAAAGwAAAAAAAAADjI3Qs3QDEKXVZRCAAAAAEIAAAAzAAAAbAAAAAAAAAA8FgAArvkxAgoo7P8t-jECSijs_wMAnwlTJhh7",
#                    "distance":15.806449,
#                    "name":"Dunga Close",
#                    "location":[36.829614,-1.30047]},
#                   {"hint":"fIB9h3-AfYcRAQAAAAAAAAAAAAAAAAAAtHHiQgAAAAAAAAAAAAAAABEBAAAAAAAAAAAAAAAAAAA8FgAAxPwxApDm6_-i_DECheTr_wAALwxTJhh7",
#                    "distance":57.950648,
#                    "name":"",
#                    "location":[36.830404,-1.317232]}]  
#     }  
# >>><B>Indexing into an output response </B>   
# list indices must be integers or slices, not str    
# rsp['routes'][0]['legs'][0]['duration'] = 419.4
# 
# 
# 
# <!-- >>>Properties Waypoints
# >>>Name : Name of the street the coordinate snapped to  
# >>>Location : Array that contains the [longitude, latitude] pair of the snapped coordinate  
# >>>Distance :  The distance, in metres, from the input coordinate to the snapped coordinate  
# >>>Hint :  Unique internal identifier of the segment (ephemeral, not constant over data updates) This can be used on subsequent request to significantly speed up the query and to connect multiple services. E.g. you can use the hint value obtained by the nearest query as hint values for route inputs.   -->
# 
# <!-- >>>Properties Routes
# >>>Distance: The distance traveled by the route, in float meters.
# >>>Duration: The estimated travel time, in float number of seconds.
# >>>Geometry: The whole geometry of the route value depending on overview parameter, format depending on the geometries parameter. 
# >>>Weight: The calculated weight of the route.
# >>>Weight_name: The name of the weight profile used during extraction phase.
# >>>Legs: The legs between the given waypoints, an array of RouteLeg objects. -->

# In[9]:


duration_list = []
distance_list = []

for x_lat, x_lon, y_lat, y_lon in tqdm(zip(data['Destination Lat'], data['Destination Long'], data['Pickup Lat'], data['Pickup Long'])):
    dest_coordinates = str(x_lon) + ',' + str(x_lat) + ';'
    source_coordinates = str(y_lon) + ',' + str(y_lat)
    url =  'http://router.project-osrm.org/route/v1/driving/'+dest_coordinates+source_coordinates
#     print(url)

    # payload = {"steps":"true","geometries":"geojson"}

    response = requests.get(url)

    rsp = response.json()
    while 'routes' not in rsp:
        response = requests.get(url)
        rsp = response.json()
    
    duration_list += [rsp['routes'][0]['legs'][0]['duration']]
    distance_list += [rsp['routes'][0]['legs'][0]['distance']]

data['calc_distance'] = distance_list
data['calc_duration'] = duration_list


# In[ ]:


data.head()


# In[10]:


data.to_csv("data_osrm.csv",index=None)


# In[11]:


df = pd.read_csv("data_osrm.csv")


# In[12]:


df.info()


# In[13]:


riders_df.info()
riders_df.head()


# In[14]:


# drop the no_of_ratings as it is optional for the customer 
riders_df.drop(columns=['No_of_Ratings'],inplace=True)


# In[15]:


riders_df.shape


# In[16]:


df.shape


# In[17]:


# do a left merge with the df column onn rider_id
df =  df.merge(riders_df,on='Rider Id',how='left')


# In[18]:


df.shape


# <a id='data_cleaning'></a>
# 
# ## Data Cleaning

# In[19]:


# rename the columns 
df.rename(columns=lambda x : x.strip().lower().replace(" ","_").replace("-",""),inplace=True)


# In[20]:


df.isnull().sum()


# In[21]:


# what percentage of precipitation_in_millimeters is null

df['precipitation_in_millimeters'].isnull().sum()/df.shape[0]


# >97% of that column is null 

# In[22]:


df['precipitation_in_millimeters'].corr(df['time_from_pickup_to_arrival'])


# >It has a really small positive correlation with the target column 
# For the  model , drop the column 
# 
# 

# In[23]:


df.drop(columns=['precipitation_in_millimeters'], inplace=True)


# >We have null values in temperature . Null values indicate that the values were not recorded  for that day . 
# For temperature , the value cannot be  zero , we can fill it with the median/mean value  

# In[24]:


df['temperature'].describe()


# In[25]:


alt.Chart(df).mark_bar().encode(
    alt.X("temperature:Q"),
    y='count()',
).properties(
    title="Histogram of Temperatures"
)


# In[26]:


sns.boxplot(x=df['temperature']);


# In[27]:


# values have a normal distirbution , no outliers use mean

# train data
df[:len_train]['temperature'].fillna(df['temperature'].mean(),inplace=True)

# test data
df[len_train:]['temperature'].fillna(df['temperature'].mean(),inplace=True)


# In[28]:


df.describe()


# Most of the values seem reasobale 
# 
# Let's look further into some columns 

# In[29]:


df['time_from_pickup_to_arrival'].describe()


# In[30]:


# x = df['time_from_pickup_to_arrival'].min()
# df.query('time_from_pickup_to_arrival == @x')
# df.drop(df[df['time_from_pickup_to_arrival'] == 1].index ,inplace=True)


# In[31]:


((df['placement__day_of_month'] == df['confirmation__day_of_month']) & (df['arrival_at_pickup__day_of_month'] == df['arrival_at_destination__day_of_month'])).any()


# In[32]:


((df['placement__weekday_(mo_=_1)'] == df['confirmation__weekday_(mo_=_1)']) & (df['arrival_at_pickup__weekday_(mo_=_1)'] == df['arrival_at_destination__weekday_(mo_=_1)'])).any()


# In[33]:


# drop this columns as they say the same thing 
df.drop(columns=['confirmation__day_of_month','arrival_at_pickup__day_of_month','arrival_at_destination__day_of_month','confirmation__weekday_(mo_=_1)','arrival_at_pickup__weekday_(mo_=_1)','arrival_at_destination__weekday_(mo_=_1)','pickup__day_of_month','pickup__weekday_(mo_=_1)'],inplace=True)


# In[34]:


df.rename(columns={'placement__day_of_month':'day_of_month','placement__weekday_(mo_=_1)':'day_of_week'},inplace=True)


# In[35]:


# there is only one vehicle type
df.drop(columns=['vehicle_type'],inplace=True)


# In[36]:


df.duplicated().sum()


# In[37]:


df.dtypes


# In[38]:


# convert the time to correct format
list_time = ['placement__time','confirmation__time','arrival_at_pickup__time','pickup__time','arrival_at_destination__time']
for col in list_time:
    df[col] = pd.to_datetime(df[col], format='%I:%M:%S %p').dt.time


# <a id='feature_engineering'></a>
# 
# ## Feature Engineering

# In[39]:


# get time deltas between 

df['time_from_placement_to_confirmation']= (pd.to_timedelta(df["confirmation__time"].astype(str)) - 
                             pd.to_timedelta(df["placement__time"].astype(str))).dt.total_seconds()
df['time_from_confirmation_to_arrival_at_pickup']= (pd.to_timedelta(df["arrival_at_pickup__time"].astype(str)) - 
                             pd.to_timedelta(df["confirmation__time"].astype(str))).dt.total_seconds()
df['time_from_arrival_at_pickup_to_pickup']= (pd.to_timedelta(df["pickup__time"].astype(str)) - 
                             pd.to_timedelta(df["arrival_at_pickup__time"].astype(str))).dt.total_seconds()
df['time_from_pickup_to_arrival_at_destination']= (pd.to_timedelta(df["arrival_at_destination__time"].astype(str)) - 
                             pd.to_timedelta(df["pickup__time"].astype(str))).dt.total_seconds()


# In[40]:


# the column time_from_pickup_to_arrival is correct
(df['time_from_pickup_to_arrival'] == df['time_from_pickup_to_arrival_at_destination']).any()


# In[41]:


df.drop(columns=['time_from_pickup_to_arrival_at_destination'],inplace=True)


# In[42]:


# sum the time difference between placement and pickup
df['time_from_placement_to_pickup']= (pd.to_timedelta(df['pickup__time'].astype(str)) - 
                             pd.to_timedelta(df['placement__time'].astype(str))).dt.total_seconds()


# In[43]:


# The hour of  pickup affects the delevery time

df['hour'] = pd.to_datetime(df["pickup__time"].astype(str)).dt.hour


# In[44]:


df.drop(columns = ['placement__time','confirmation__time','arrival_at_pickup__time','pickup__time','arrival_at_destination__time'], inplace=True)


# In[45]:


df['personal_or_business'].unique()


# In[46]:


dummies = pd.get_dummies(df.personal_or_business)
# Concatenate the dummies to original dataframe
df = pd.concat([df, dummies], axis='columns')


# In[47]:


# df['platform_type'].unique()
# df['platform_type'].replace([4,3,2,1],['a','b','c','d'], inplace=True)
# df['platform_type'].unique()
# dummies_ = pd.get_dummies(df.platform_type)

# # Concatenate the dummies to original dataframe
# df = pd.concat([df, dummies_], axis='columns')
 
# # drop the values
# df.drop(columns = ['platform_type'], inplace=True)


# In[48]:


# column showing the number of trips each rider has done 
dict_vc = df['rider_id'].value_counts().to_dict()
df["rider_trips"] = df['rider_id'].map(dict_vc)


# >There is a specific time delta that is not affected by distance   
# The time from <B>arrival at pickup time</B>  to <B>pickup time</B>   
# What affects this duration of time ?May depend on :
# ><ul>
#     <li>Rider</li>
#     <li>what the customer orders</li>
#     <li>Effeciency of logistics at the pickup location</li>
# </ul>  
# This data can help mprove optimizations at this stage 

# In[49]:


df.head()


# In[50]:


# Per customer , what is the mean time their order takes between arrival_at_pickup to pickup, independent of rider
df['cust_mean_time_at_pickup'] =  df['user_id'].map(df.groupby('user_id')['time_from_arrival_at_pickup_to_pickup'].mean().to_dict())


# In[51]:


# Per rider,what is the mean time they take to pickup an order at pickup loaction  ,independent of cust
df['rider_mean_time_at_pickup'] =  df['rider_id'].map(df.groupby('rider_id')['time_from_arrival_at_pickup_to_pickup'].mean().to_dict())


# In[52]:


# for different riders serving the same customer , what is the ratio of the mean time 
# they take at the pickup station to the mean time of cust at pickup 
df['ratio_cust_rider_at_pickup'] = df['cust_mean_time_at_pickup'] / df['rider_mean_time_at_pickup']


# In[53]:


# for an order , for a customer ,what was the ratio of the mean time it takes for their order to picked 
#  to the actual time it took to be picked 
df['ratio_user_diff_pickup'] = df['cust_mean_time_at_pickup'] / df['time_from_arrival_at_pickup_to_pickup']


# In[54]:


# for an order , for a rider  ,what was the ratio of the mean time it takes for them to pick an  order 
# to the actual time it took to be pick the order
df['ratio_rider_diff_pickup'] = df['rider_mean_time_at_pickup'] / df['time_from_arrival_at_pickup_to_pickup']


# In[55]:


df['len_lat'] = (df['pickup_lat'] - df['destination_lat']).abs()
df['len_long'] = (df['pickup_long'] - df['destination_long']).abs()


# In[56]:


df.head()


# <a id='eda'><a>
# 
# ## Exploratory Data Analysis

# ### Location data 

# In[57]:


# fig,ax = plt.subplots(figsize=(15,15))
# gdf.plot(ax=ax);


# In[58]:


geometry_pickup = [Point(xy) for xy in zip( df['pickup_long'],df['pickup_lat'])]
geometry_dest = [Point(xy) for xy in zip( df['destination_long'],df['destination_lat'])]


# In[59]:


geo_df_pickup = gpd.GeoDataFrame(df[['distance_(km)']],
                         crs=crs,
                         geometry=geometry_pickup)


geo_df_dest = gpd.GeoDataFrame(df[['distance_(km)']],
                         crs=crs,
                         geometry=geometry_dest)


# In[60]:


geo_df_dest.dtypes


# In[61]:


geo_df_dest= geo_df_dest.rename(columns={'geometry': 'geometry_dest'}).set_geometry('geometry_dest')
geo_df_pickup = geo_df_pickup.rename(columns={'geometry': 'geometry_pickup'}).set_geometry('geometry_pickup')


# In[62]:


joined_df = pd.concat([geo_df_dest, geo_df_pickup], axis=1)


# In[63]:


joined_df.dtypes


# In[64]:


fig,ax=plt.subplots(figsize = (15,15))
counties_map.plot(ax=ax,alpha=0.4,color='green')
joined_df.set_geometry('geometry_dest').plot(ax=ax,markersize=20,color="red",marker="o",label="Destination")
joined_df.set_geometry('geometry_pickup').plot(ax=ax,markersize=20,color="blue",marker="^",label="Pickup")
plt.title("Plot of Pickup and Destination points on the Kenyan map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(prop={'size':15});


# In[65]:


fig,ax=plt.subplots(figsize = (15,15))
gdf.plot(ax=ax,alpha=0.4,color='grey')
joined_df.set_geometry('geometry_dest').plot(ax=ax,markersize=20,color="red",marker="o",label="Destination")
joined_df.set_geometry('geometry_pickup').plot(ax=ax,markersize=20,color="blue",marker="^",label="Pickup")
plt.title("Plot of Pickup and Destination points in Nairobi county")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(prop={'size':15})


# >Our orders are in Nairobi county  
# Most pickup points and destination points are closely located  
# Destination location have a larger standard deviation 

# #### Aanlysing "platform_type" and "personal_or_business"

# In[66]:


alt.Chart(df).mark_bar().encode(
    alt.X("platform_type"),
    y='count()',
).properties(
    title="Histogram of platform_type"
).interactive()


# In[67]:


df.platform_type.value_counts()


# In[68]:


alt.Chart(df).mark_bar(size=30).encode(
    alt.X("personal_or_business"),
    y='count()',
).properties(
    title="Histogram of platform_type",
    width=400,
)


# In[69]:


g = df.groupby(["platform_type","personal_or_business"])['order_no'].count().reset_index()
alt.Chart(g).mark_bar().encode(
    alt.X("platform_type"),
    y='order_no',
    color="personal_or_business"
).properties(
    title="'Distribution of repeat trips by Rider'"
)


# >Most of our orders are  from businesses  
# The common platform type  used is 3  
# Most of the businesss use platform 3 , For personal orders , platform 1 and 2 are common    
# Platform Type 4 hardly has any traffic   
# Recommendation : Optimize platform 3 for businesses and 1 ,2 for personal orders
# 

# #### Anlysing repeat orders "per rider_id" and "user_id"

# In[70]:


f = df.groupby('rider_id').count()
alt.Chart(f).mark_bar().encode(
    alt.X("order_no"),
    y='count()',
).properties(
    title="'Distribution of repeat trips by Rider'"
)


# In[71]:


h = df.groupby('user_id').count()
alt.Chart(h).mark_bar().encode(
    alt.X("order_no"
#     bin = alt.Bin(maxbins=70),
    ),
    y='count()',
).properties(
    title='Distribution of repeat orders by Customers'
    
)


# >Most Riders have taken trips betwen 0 and 20  
# Most customers have only 1 order , the number of repeat cutomers is wanting  
# Recommndation : Figure out ways to keep customers 

# In[72]:


alt.Chart(df[:len_train]).mark_line().encode(
    x='distance_(km)',
    y='time_from_pickup_to_arrival'
).properties(
     title="Line chart of ETA and distance in KM"
)


# In[73]:


alt.Chart(df[:len_train]).mark_line().encode(
    x='day_of_week',
    y='time_from_pickup_to_arrival'
).properties(
     title="Line chart of ETA and Day of the week"
)


# In[74]:


alt.Chart(df[:len_train]).mark_circle(size=60).encode(
    x="day_of_week",
    y='time_from_pickup_to_arrival',
    color='personal_or_business'
   
).properties(
     title="Scatter plot of day of week and time_from_pickup_to_arrival"
).interactive()


# In[75]:


alt.Chart(df[:len_train]).mark_circle(size=60).encode(
    x="day_of_month",
    y='time_from_pickup_to_arrival',
    color='personal_or_business'
   
).properties(
     title="Scatter plot of day of week and time_from_pickup_to_arrival"
).interactive()


# In[ ]:





# In[ ]:





# <a id='modelling'><a>
# ## Modelling

# ##### Feature Selection
# 
# In the dataset, are there any columns that are not useful or have very low correlation?
# <ul>
# <li> A correlation value can range between -1 to 1.</li> 
# <li> Value closer to -1 means high negative correlation between two variables </li> 
# <li> Value closer to +1 means high positive correlation between two variables </li> 
# <li> Value closer to 0 means no or very low correlation between two variables </li> 
# </ul>

# In[76]:


df.corr()


# In[77]:


df.head()


# In[78]:


train = df[:len_train]
test  =  df[len_train:]


# ### Testing different models

# In[79]:


X = train.drop(['time_from_pickup_to_arrival','order_no', 'user_id','rider_id','personal_or_business'], axis = 1) 


y = train['time_from_pickup_to_arrival']


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# In[81]:


print("Shape of the original data: ", train.shape)
print("Shape of the train data = ", X_train.shape)
print("Shape of the test data = ", X_test.shape)


# In[82]:


X_train = X_train.reset_index()
X_train= X_train.drop(columns=['index'])
X_test = X_test.reset_index()
X_test= X_test.drop(columns=['index'])


# #### Linear regression

# In[83]:


from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)


# In[84]:


y_test_pred = linear_model.predict(X_test)


# In[85]:


MAE = mean_absolute_error(y_test, y_test_pred)
MSE = mean_squared_error(y_test, y_test_pred)
R2 =   r2_score(y_test, y_test_pred)

print('LINEAR REGRESSION')
print('The mean_absolute_error =  ' +str(MAE))
print('The mean_squared_error =  ' +str(MSE))
print('The r2_score =  ' +str(R2))


# #### Random forest

# In[86]:


forest = RandomForestRegressor(random_state=1)
 
forest.fit(X_train, y_train)

y_test_forest_pred = forest.predict(X_test)

MAE_Forest = mean_absolute_error(y_test, y_test_forest_pred)
MSE_Forest = mean_squared_error(y_test, y_test_forest_pred)
R2_Forest =   r2_score(y_test, y_test_forest_pred)

print('RANDOM FOREST REGRESSOR')
print('The mean_absolute_error =  ' +str(MAE_Forest))
print('The mean_squared_error =  ' +str(MSE_Forest))
print('The r2_score =  ' +str(R2_Forest))


# #### Decison tree

# In[87]:


decision_tree = DecisionTreeRegressor(random_state = 1, max_depth = 5)

decision_tree.fit(X_train,y_train)

y_test_tree_pred = decision_tree.predict(X_test)

MAE_Tree = mean_absolute_error(y_test, y_test_tree_pred)
MSE_Tree = mean_squared_error(y_test, y_test_tree_pred)
R2_Tree =   r2_score(y_test, y_test_tree_pred)

print('DECISION TREE REGRESSOR')
print('The mean_absolute_error =  ' +str(MAE_Tree))
print('The mean_squared_error =  ' +str(MSE_Tree))
print('The r2_score =  ' +str(R2_Tree))


# #### Grid search with Random Forest

# In[88]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
     'bootstrap': [True],
     'max_depth': [10, 20],
     'min_samples_leaf': [3, 4],
     'min_samples_split': [4, 6],
    'n_estimators': [100, 200]
     }


rf = RandomForestRegressor(random_state = 1)

# Grid search cv
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)

grid_search.fit(X_train, y_train)

y_test_gs_pred = grid_search.predict(X_test)

MAE_gs = mean_absolute_error(y_test, y_test_gs_pred)
MSE_gs = mean_squared_error(y_test, y_test_gs_pred)
R2_gs =   r2_score(y_test, y_test_gs_pred)

print('GRID SEARCHCV ')
print('The mean_absolute_error =  ' +str(MAE_gs))
print('The mean_squared_error =  ' +str(MSE_gs))
print('The r2_score =  ' +str(R2_gs))


# In[92]:


grid_search.best_estimator_.feature_importances_


# In[97]:


feat_importances = pd.Series(grid_search.best_estimator_.feature_importances_, index=X_train.columns)
feat_importances.nlargest(4).plot(kind='bar',title="Model feature importance");


# #### Light GBM

# In[84]:


import graphviz


# In[85]:


import lightgbm as lgb
model_lgb = lgb.LGBMClassifier(learning_rate=0.09,max_depth=5,random_state=42)
model_lgb.fit(X_train,y_train)


# In[86]:


y_test_gs_pred = model_lgb.predict(X_test)

MAE_gs = mean_absolute_error(y_test, y_test_gs_pred)
MSE_gs = mean_squared_error(y_test, y_test_gs_pred)
R2_gs =   r2_score(y_test, y_test_gs_pred)

print('LIGHT GBM ')
print('The mean_absolute_error =  ' +str(MAE_gs))
print('The mean_squared_error =  ' +str(MSE_gs))
print('The r2_score =  ' +str(R2_gs))


# In[87]:


print('Training accuracy {:.4f}'.format(model_lgb.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(model_lgb.score(X_test,y_test)))


# In[88]:


lgb.plot_importance(model_lgb)


# In[ ]:


lgb.plot_metric(model_lgb)


# In[89]:


lgb.plot_tree(model_lgb,figsize=(30,40));


# In[1]:


metrics.plot_confusion_matrix(model_lgb,X_test,y_test,cmap='Blues_r')


# In[ ]:


print(metrics.classification_report(y_test,model_lgb.predict(X_test)))


# #### XGBOOST

# In[ ]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error

data_dmatrix = xgb.DMatrix(data=X,label=y)

xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                max_depth = 5, alpha = 10, n_estimators = 10)


xg_reg.fit(X_train,y_train)

y_preds = xg_reg.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, y_preds))
print("RMSE: %f" % (rmse))


# In[ ]:


y_test_gs_pred = xg_reg.predict(X_test)

MAE_gs = mean_absolute_error(y_test, y_test_gs_pred)
MSE_gs = mean_squared_error(y_test, y_test_gs_pred)
R2_gs =   r2_score(y_test, y_test_gs_pred)

print('XG BOOST ')
print('The mean_absolute_error =  ' +str(MAE_gs))
print('The mean_squared_error =  ' +str(MSE_gs))
print('The r2_score =  ' +str(R2_gs))


# In[ ]:


# xg boost with cross validation

params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)


# In[ ]:


cv_results.head()


# In[ ]:


print((cv_results["test-rmse-mean"]).tail(1))


# In[ ]:


xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)


# In[ ]:


xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()


# In[ ]:


xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()


# ### Final part

# In[ ]:


# grid search seems to be a better model 
# lets use this 


# In[ ]:


test.head()


# In[ ]:


final_df_fetures = train.drop(['time_from_pickup_to_arrival','order_no', 'user_id','rider_id','personal_or_business'], axis = 1) 

final_df_labels = train['time_from_pickup_to_arrival']


# In[ ]:


# final_df_fetures = final_df.drop('time_from_pickup_to_arrival', axis = 1)
# final_df_labels = final_df['time_from_pickup_to_arrival']


# In[ ]:


final_df_fetures = final_df_fetures.reset_index()
final_df_fetures = final_df_fetures.drop(columns=['index'])


# In[ ]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
     'bootstrap': [True],
     'max_depth': [10, 20],
     'min_samples_leaf': [3, 4],
     'min_samples_split': [4, 6],
    'n_estimators': [100, 200]
     }


rf = RandomForestRegressor(random_state = 1)

# Grid search cv
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 2, n_jobs = -1, verbose = 2)

grid_search.fit(final_df_fetures, final_df_labels)


# In[94]:


import joblib
joblib.dump(grid_search, 'model.pkl')
print("Model dumped!")


# In[95]:


# Saving the data columns from training
model_columns = list(final_df_fetures.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")


# In[96]:


# Load the model that you just saved
grid_search = joblib.load('model.pkl')


# In[ ]:


test.shape


# In[ ]:


Order_No = test['order_no']
Order_No = pd.Series(list(Order_No))


# In[ ]:


len(Order_No)


# In[ ]:


test_df = test.drop(['time_from_pickup_to_arrival','order_no', 'user_id','rider_id','personal_or_business'], axis = 1) 
test_df = test_df.reset_index()
test_df = test_df.drop(columns=['index'])


# In[ ]:


time_from_pickup_to_arrival = grid_search.predict(test_df)


# In[ ]:


time_from_pickup_to_arrival = pd.Series(list(time_from_pickup_to_arrival))


# In[ ]:


ouput_df = pd.DataFrame(columns = ['Order_No', 'Time from Pickup to Arrival'])


# In[ ]:


ouput_df['Order_No'] = Order_No
ouput_df['Time from Pickup to Arrival'] = time_from_pickup_to_arrival

ouput_df.head()


# In[ ]:


# ouput_df = pd.DataFrame({'Order_No':list(Order_No),'Time from Pickup to Arrival':list(time_from_pickup_to_arrival)})


# In[ ]:


ouput_df.to_csv("submission_fizz_4.csv",index=False)


# <a id='conclusions'></a>
# 
# ## Conclusions
# 
# > Distance is highly correlated with ETA  
# Additional data from OSRM really complemented the data
# Creating features and
# Identifying how a unique rider / customer affects the ETA from their previous orders improves the perfoemnace of the model 
# Grid search with random forest was a quick win for the model 
# 
# 
# 
# ##### Limitations
# > We were not able to get traffic data for the exact dates since the date did not include the year column  
# 
# 
# 
# ##### Suggestions
# > More data is always better for the model

# ### Notes
