# High-Level of Steps I need:
# 1. Determine how many Dublin Bikes were used per day
# 2. Merge the Weather data for each day
# 3. Compare Bike usage level to Weather conditions
# 4. Use this comparison to create a model that will forecast demand based on Weather forecast for the week ahead

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns
from scipy import stats

# Dublin Bikes
# Dublin Bikes historic data is held in CSV format and does not require a password.
# There is an API also available which would probably be better but for the purposes of
# this assignment I am using CSV file so the Tutor can run the code

# CSV files from Dublin Bikes are sourced directly from https://data.gov.ie/dataset/dublinbikes-api
# There is a one file per quarter. File name tells you which quarter is covered.
# File name format: dublinbikes_yyyymmdd_yyyymmdd.csv

# Each row represents a snapshot of the number of slots and the number of available bikes at each Station
# The snapshots are taken at 5-minute intervals.

# **********************************************************************************************************
# Read in the Dublin Bikes data. I am using a dictionary dtypes to force Station_ID to be an integer datatype and
# force Time to be a String (str). Once Time is read-in I can parse it to be more usable.
dtypes = {'STATION ID':'int', 'TIME':'str'}

# Going to pass Time as the parse_dates argument to the Pandas read_csv function. This will convert it to date
parse_dates = ['TIME']

# Read the CSVin using Pandas Read_CSV. Need dayfirst argument to avoid treating it as an american date
initial_df = pd.read_csv("dublinbikes_20190401_20190701.csv", dtype=dtypes, parse_dates=parse_dates, dayfirst=True)

# Check the file
print(initial_df.head())
print(initial_df.info())
print(initial_df.describe())

# (1.)
# To determine number of bikes used each day I need to extract the date from the Time field
# Using Pandas DatetimeIndex to extract its and store in DATE
initial_df['DATE'] = pd.DatetimeIndex(initial_df['TIME']).date
initial_df['DATE'] = pd.to_datetime(initial_df['DATE'], format='%Y-%m-%d')
# Check new field is as expected
print(initial_df['DATE'])
print(initial_df.head())
print(initial_df.info())

#Sort the dataframe so that I can get the final sum of bike usage for each day (end of day will have the cumulative sum)
initial_df = initial_df.sort_values(['STATION ID', 'TIME'], ascending=(True, True))
print(initial_df.head())

# Initial_df now has the number of bikes that are available at each 5 minute interval
# sorted by Station_ID and Time.
# To calculate the total number times bikes are taken each day I am going to
# subtract the value in Available Bikes from the value in previous row (5 mins before).
# Each time the difference is negative will be a proxy for the number of bikes taken in that
# 5 minute window. I am going to assume that the sum of  these negative values for an entire day
# will be the number of bikes taken in that day
initial_df['Interactions'] = initial_df.groupby(['STATION ID', 'DATE'])['AVAILABLE BIKES'].diff().fillna(0)
# Sense Check values
print(initial_df.head())
print(initial_df['DATE'].max())
print(initial_df['Interactions'].max())
print(initial_df['Interactions'].min())

# I only want to count the negative values in Interactions as positives represent bikes being returned
# This will create a new field called Check_Neg that will hold the count of negatives converted to positive
initial_df['Check_Neg'] = np.where(initial_df['Interactions'] < 0,initial_df['Interactions']*-1, 0 )
# Sense check
print(initial_df['Check_Neg'].max())
print(initial_df['Check_Neg'].min())

# Num_Taken will be the number of bikes taken by Station per day. I am using the CumSum to work out total
# Using this means that the last line per station, per day will have the total
initial_df['Num_Taken']=initial_df.groupby(['STATION ID', 'DATE'])['Check_Neg'].cumsum().fillna(0)
print(initial_df['Num_Taken'])

# Filtering the dataframe to create Summary_DF that will have just the last line by Station ID and Date
def filter_last_timevalue(g):
        return g.groupby(['STATION ID', 'DATE']).last().agg({'Num_Taken':'sum'})

summary_df = initial_df.groupby(['STATION ID', 'DATE']).apply(filter_last_timevalue)


#Sense check
print(summary_df.head())
print(summary_df.info())
print(summary_df.describe())
# Reset Index values
summary_df = summary_df.reset_index(level=0)

# Creating a Average of Num_taken value across all stations for each day
# This will help me categorise in to a Low, Medium or High later
summary_df['Daily Average'] = summary_df.groupby('DATE')['Num_Taken'].mean()
print(summary_df.head())
print(summary_df.info())
print(summary_df.describe())


# **********************************************************************************************************
# Weather Data
# Read in the Historical Weather Data to capture Rainfall and Temperatures.
# I have selected Dublin City Centre (Merrion Row) as the location for the data.
# This data is manually downloaded from https://www.met.ie/climate/available-data/historical-data
# For the Period that I am analysing the bike data (2019) I have Rainfall (rain mm),
# Max Temp (maxt C) & Min Temp (mint C) to use.
# The future forecast data is available via API. That will be used later in the model

parse_dates_w = ['date']
initial_weather = pd.read_csv("dly3923.csv", parse_dates=parse_dates_w,  dayfirst=True, skip_blank_lines=True)
# Making Date the same as that on the Bikes data so they can bve merged on that as a key
initial_weather['DATE'] = pd.DatetimeIndex(initial_weather['date']).date
initial_weather['DATE'] = pd.to_datetime(initial_weather['DATE'], format='%Y-%m-%d')
print(initial_weather.head())
print(initial_weather.info())
print(initial_weather.describe())






# (2. Merge Historic Bike and Weather data)
# Merge the summary data with the weather data on DATE and left join to only include values that exist on Bikes data
# This will discard the obsolete weather data
merged_data = pd.merge(summary_df, initial_weather, how='left', on='DATE')
print(merged_data.head())
print(merged_data.info())
print(merged_data.describe())



#Check the MetEireann API for weather forecast info
#Co-ordinates for Dublin City Centre taken from Google Maps
#response = requests.get("http://metwdb-openaccess.ichec.ie/metno-wdb2ts/locationforecast?lat=53.348366;long=-6.254815")
#print(response.status_code)
#print(response.text)

#Do a trend graph showing the total usage in the period (Matplotlib)
plt.figure(0)
plt.title("Number of Bikes Used per Day")
plt.xlabel("Date")
plt.ylabel("Number Used")
plt.plot('DATE', 'Num_Taken', data=merged_data, )
plt.show()
