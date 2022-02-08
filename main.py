# High-Level of Steps I need:
# 1. Determine how many Dublin Bikes were used per day
# 2. Merge the Weather data for each day
# 3. Compare Bike usage level to Weather conditions
# 4. Use this comparison to create a model that will forecast demand based on Weather forecast for the week ahead

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
