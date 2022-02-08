

import requests
import numpy as np
import pandas as pd

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

