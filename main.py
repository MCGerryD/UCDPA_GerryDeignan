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
from sklearn.linear_model import LinearRegression

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

# Read the CSV in using Pandas Read_CSV. Need dayfirst argument to avoid treating it as an american date
# Two files will give me 6 months of data (from 1/1/2019 - 30/6/2019)
File1 = pd.read_csv("dublinbikes_20190101_20190401.csv", dtype=dtypes, parse_dates=parse_dates, dayfirst=True )
File2 = pd.read_csv("dublinbikes_20190401_20190701.csv", dtype=dtypes, parse_dates=parse_dates, dayfirst=True)

# Stack both dataframes on top of each other. Fields are the same so no need to worry about data not lining up
initial_df = pd.concat([File1, File2], ignore_index=True)

# Sense check the file
print(initial_df.head())
print(initial_df.info())
print(initial_df.describe())

# (1.)
# To determine number of bikes used each day I need to extract the date from the Time field
# Using Pandas DatetimeIndex to extract it and store in DATE
initial_df['DATE'] = pd.DatetimeIndex(initial_df['TIME']).date
# Formatting the date now into YYYmmdd
initial_df['DATE'] = pd.to_datetime(initial_df['DATE'], format='%Y-%m-%d')
# Check new field is as expected
print(initial_df['DATE'])
print(initial_df.head())
print(initial_df.info())

#Sort the dataframe so that I can get the final sum of bike usage for each day (row at the
# end of day will have the cumulative sum)
initial_df = initial_df.sort_values(['STATION ID', 'TIME'], ascending=(True, True))
print(initial_df.head())

# Initial_df now has the number of bikes that are available at each 5 minute interval
# sorted by Station_ID and Time.
# To calculate the total number times bikes are taken each day I am going to
# subtract the value in Available Bikes from the value in previous row (5 mins before).
# Each time the difference is negative will be a proxy for the number of bikes taken in that
# 5 minute window. I am going to assume that the sum of  these negative values for an entire day
# will be the number of bikes taken in that day.
# The Interactions will identify if a bike was either taken or replaced. I'll then just keep the ones taken
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
# Using this means that the last line per station, per day will have the total taen
initial_df['Num_Taken']=initial_df.groupby(['DATE'])['Check_Neg'].cumsum().fillna(0)
print(initial_df['Num_Taken'])


# Filtering the dataframe to create Summary_DF that will have just the last line by Station ID and Date
# Its a summary of initial_df grouped by Date
def filter_last_timevalue(g):
        return g.groupby(['DATE']).last().agg({'Num_Taken':'sum'})

#Summary_df summarises for each Date
summary_df = initial_df.groupby(['DATE']).apply(filter_last_timevalue)

#Sense check
print(summary_df.head())
print(summary_df.info())
print(summary_df.describe())

# Reset Index values
summary_df = summary_df.reset_index(level=0)

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
# Making Date the same as that on the Bikes data so they can be merged on that as a key
initial_weather['DATE'] = pd.DatetimeIndex(initial_weather['date']).date
initial_weather['DATE'] = pd.to_datetime(initial_weather['DATE'], format='%Y-%m-%d')
print(initial_weather.head())
print(initial_weather.info())
print(initial_weather.describe())

#Checking dataframe by outputtingto CSV and viewing in Excel
initial_weather.to_csv('initial_weather.csv')


# (2. Merge Historic Bike and Weather data)
# Merge the summary data with the weather data on DATE and left join to only include values that exist on Bikes data
# This will discard the obsolete weather data
merged_data = pd.merge(summary_df, initial_weather, how='left', on='DATE')

print(merged_data.head())
print(merged_data.info())
print(merged_data.describe())

#Checking if there are missing values in any columns
print(merged_data.isna().sum())

# Convert Maxt and Mint to Floats
# The CSV had some blanks where there was no maxt or mint so
# I have used a lambda function to replace them with zero
# otherwise pd.to_numeric was failing on the blank
merged_data['maxt'] = merged_data['maxt'].apply(lambda x: 0 if x == ' ' else x)
merged_data['maxt'] = pd.to_numeric(merged_data['maxt'], downcast="float")
merged_data['mint'] = merged_data['mint'].apply(lambda x: 0 if x == ' ' else x)
merged_data['mint'] = pd.to_numeric(merged_data['mint'], downcast="float")
# Check
print(merged_data.head())
print(merged_data.info())
print(merged_data.describe())
merged_data.to_csv('merged_data.csv')

# Checking data visually with matplotlib
# This chart shows peaks and troughs at regular intervals that could potentially
# distort results. I suspected that the peaks would be weekdays and troughs would be weekends
plt.figure(1)
plt.title("Number of Bikes Used per Day")
plt.xlabel("Date")
plt.ylabel("Number Used")
plt.plot('DATE', 'Num_Taken', data=merged_data, )


# Splitting my dataset into Weekdays and weekends to see if it smooths the data
# It does make a difference but Bank Holidays are showing in the weekdays
# Ideally I would add them to the weekend dataframe and keep working days separate
# dayofweek has numeric values, Mon = 0, Tue = 1, ... Sat = 5 and Sun = 6
myMask = merged_data['DATE'].dt.dayofweek.isin([5, 6])
weekdays_only = merged_data[~myMask]
weekends_only = merged_data[myMask]





# Check Weekdays now
plt.figure(2)
plt.title("Number of Bikes Used per WeekDay")
plt.xlabel("Date")
plt.ylabel("Number Used")
plt.plot('DATE', 'Num_Taken', data=weekdays_only, )
#plt.show()
plt.figure(3)
plt.title("Number of Bikes Used per Weekend Day")
plt.xlabel("Date")
plt.ylabel("Number Used")
plt.plot('DATE', 'Num_Taken', data=weekends_only, )
#plt.show()




weekdays_only.to_csv('weekdays_only.csv')

#Investigate correlation between Bike usage on weekdays and rainfall
#Show the correlation of Rainfall to Bike Usage along with printing the R Value (Seaborn)
slope, intercept, r_value, p_value, std_err = stats.linregress(weekdays_only['rain'],weekdays_only['Num_Taken'])
plt.figure(4)
chart = sns.regplot(weekdays_only['rain'],weekdays_only['Num_Taken'])
chart.set(title='Number of Bikes Taken on Weekdays vs Rainfall(mm)',ylabel='Number Taken',xlabel='Rainfall(mm)')
#Print the R Value
print('The linear coefficient for weekdays is',r_value)

#Investigate correlation between Bike usage on weekdays and rainfall
#Show the correlation of Rainfall to Bike Usage along with printing the R Value (Seaborn)
slope, intercept, r_value, p_value, std_err = stats.linregress(weekends_only['rain'],weekends_only['Num_Taken'])
plt.figure(5)
chart = sns.regplot(weekends_only['rain'],weekends_only['Num_Taken'])
chart.set(title='Number of Bikes Taken on Weekends vs Rainfall(mm)',ylabel='Number Taken',xlabel='Rainfall(mm)')

#Print the R Value
print('The linear coefficient for weekends is',r_value)
#plt.show()


#Check the MetEireann API for weather forecast info
#Co-ordinates for Dublin City Centre taken from Google Maps
#response = requests.get("http://metwdb-openaccess.ichec.ie/metno-wdb2ts/locationforecast?lat=53.348366;long=-6.254815")
#print(response.status_code)
#print(response.text)

print(weekdays_only.head())
print(weekdays_only.info())
print(weekdays_only.describe())
# Building the model
# Setting X to all features in the dataset except the response variable Num_taken and others that I'm not interested in
x = weekdays_only.drop(['Num_Taken', 'ind', 'ind.1', 'ind.2', 'gmin', 'soil'], axis=1)
# Y variable is the response variable Num_Taken. This is what I want to predict eventually
y = weekdays_only['Num_Taken']

#cycle through the features and create a plot for each to see what the relationship is like
for col in x.columns:
        if(col != ['Num_Taken']):
                plt.scatter(x[col],y)
                plt.xlabel(col)
                plt.ylabel('Bikes Taken')
                plt.show()

#From Lecture notes;
# If COEF is Positive: A unit increase in FEATURE isassociated with a COEFICIENT "unit" increase in RESPONSE
# If COEF is NEGATIVE: A unit increase in FEATURE is associated with a COEFICIENT "unit" DECREASE in RESPONSE
# y = b + mx
# y = Intercept + Coef X Feature

# Predict using a value for Rainfall
feature_cols = ['rain']
X = weekdays_only[feature_cols]
y = weekdays_only.Num_Taken
lm = LinearRegression()
lm.fit(X,y)
print("Intercept for Rain is ", lm.intercept_) #Intercept for Rain is  7491.34866595658
print("Coefficient for Rain is ", lm.coef_) #Coefficient for Rain is  [-108.54511026]

Prediction_using_rain = pd.DataFrame({'rain':[10]})
Prediction_using_rain.head()
print(lm.predict((Prediction_using_rain)))

X_new = pd.DataFrame({'rain': [weekdays_only.rain.min(), weekdays_only.rain.max()]})
preds = lm.predict(X_new)
print(preds)
weekdays_only.plot(kind='scatter', x='rain',y='Num_Taken')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()



# Predict using a value for Max Temp
feature_cols = ['maxt']
X = weekdays_only[feature_cols]
y = weekdays_only.Num_Taken
lm = LinearRegression()
lm.fit(X,y)
print("Intercept for maxt is ", lm.intercept_)#Intercept for maxt is  6777.0273
print("Coefficient for maxt is ", lm.coef_) #Coefficient for maxt is  [45.20709]

Prediction_using_maxt = pd.DataFrame({'maxt':[5]})
Prediction_using_maxt.head()
print(lm.predict((Prediction_using_maxt)))


X_new = pd.DataFrame({'maxt': [weekdays_only.maxt.min(), weekdays_only.maxt.max()]})
preds = lm.predict(X_new)
print(preds)
weekdays_only.plot(kind='scatter', x='maxt',y='Num_Taken')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()

# Predict using a value for Min Temp
feature_cols = ['mint']
X = weekdays_only[feature_cols]
y = weekdays_only.Num_Taken

lm = LinearRegression()
lm.fit(X,y)
print("Intercept for mint is ", lm.intercept_) #Intercept for mint is  7340.511
print("Coefficient for mint is ", lm.coef_) #Coefficient for mint is  [-1.9628823]
Prediction_using_mint = pd.DataFrame({'mint':[25]})
Prediction_using_mint.head()
print(lm.predict((Prediction_using_mint)))
X_new = pd.DataFrame({'mint': [weekdays_only.mint.min(), weekdays_only.mint.max()]})
preds = lm.predict(X_new)
print(preds)
weekdays_only.plot(kind='scatter', x='mint',y='Num_Taken')
plt.plot(X_new, preds, c='red', linewidth=2)
plt.show()
# Plotting the least Squares Line to see the regression visually







