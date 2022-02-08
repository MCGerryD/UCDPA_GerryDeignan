
#Check the MetEireann API for weather forecast info
import requests
import numpy as np
import pandas as pd
#Co-ordinates for Dublin City Centre taken from Google Maps
response = requests.get("http://metwdb-openaccess.ichec.ie/metno-wdb2ts/locationforecast?lat=53.348366;long=-6.254815")
print(response.status_code)
print(response.text)



#Dublin Bikes API is not open so I am downloading CSV files
# directly from https://data.gov.ie/dataset/dublinbikes-api

#Read in the Dublin Bikes data and use dictionary to force the following two datatypes
dtypes = {'STATION ID':'int', 'TIME':'str'}
#Convert TIME field from OBJECT to DATETIME on the fly
parse_dates = ['TIME']

#Parse the TIME field to a Date field. Need dayfirst argument to avoid treating it as an american date
#Data in the CSV's only goes up to Q2 2020
initial_df = pd.read_csv("dublinbikes_20190401_20190701.csv", dtype=dtypes, parse_dates=parse_dates, dayfirst=True)
print(initial_df.head())
#Check datatypes
print(initial_df.info())
print(initial_df.describe())

#Extract the Date from TIME so that we can calulate how many bikes were used in the entire day
initial_df['DATE'] = pd.DatetimeIndex(initial_df['TIME']).date
print(initial_df['DATE'])

#Sort the dataframe so that I can get the final sum of bike usage for each day (end of day will have the cumulative sum)
initial_df = initial_df.sort_values(['STATION ID', 'TIME'], ascending=(True, True))

#The raw data has the number of bikes that are available at each 5 minute interval
#To calculate the number times a bike is taken each day it is necessary
#to subtract the current number of available bikes from the previous one (5 mins before)
#Each time the difference is negative, this signifies the number of bikes taken in that
#5 minute window. Summing these for an entire day will give the number of bikes taken in that day
initial_df['Interactions']=initial_df.groupby(['STATION ID', 'DATE'])['AVAILABLE BIKES'].diff().fillna(0)
#only count negative differences. The positive differences mean a bike was left there (not taken)
initial_df['Check_Neg'] = np.where(initial_df['Interactions'] < 0,initial_df['Interactions']*-1, 0 )
#Only using this if doing analysis by Station ID but too many for this project so will just do total usage
initial_df['Num_Taken']=initial_df.groupby(['STATION ID', 'DATE'])['Check_Neg'].cumsum().fillna(0)
#Total Usage for all stations by Date
initial_df = initial_df.sort_values(['DATE'], ascending=(True))
initial_df['Total_Num_Taken'] = initial_df.groupby('DATE')['Check_Neg'].cumsum().fillna(0)
print(initial_df.head())
#Check datatypes
print(initial_df.info())


#I want to group the data by Day i.e. cumulative sum all 5 minute intervals
# in each day and only keep the last value
def filter_last_timevalue(g):
    return g.groupby('DATE').last().agg({'Total_Num_Taken':'sum'})
#apply the filter
summary_df = initial_df.groupby(['DATE']).apply(filter_last_timevalue)
#reset index values
summary_df = summary_df.reset_index(level=0)
print(summary_df.head())
#Check datatypes

