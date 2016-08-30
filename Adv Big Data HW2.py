import sys
import datetime as date
import datetime as dt
from sklearn import gaussian_process
import pandas as pd
import pandas.io.data as pddata
from pandas import Series
import numpy as np
from yahoo_finance import Share
import yahoo_finance as yahoo
import itertools

# Initialize Arrays
array_1 = []
array_2 = []
curdata = []
average=[]
percentage=[]
diff=0
sum=0

# Set the list of Firm Stocks
firmlist = ['BAC','C','IBM','AAPL','GE','T','MCD','NKE','TWTR','TSLA']

# Gather the Prediction Data 
def gatherpreddata(companyname, today):
    # Gather the Yahoo Prediction Data, where the start date is a specified date to start aquiring data and the end date is today's date
    yahoo_data = pddata.get_data_yahoo(companyname,start= dt.datetime(2016, 2, 1),end = dt.datetime(today.year, today.month, today.day))
    # Gather the data for today - stock when the market opens, the highest stock during the day, etc.
    yahoo_data.loc[today] = (Share(companyname).get_open(), Share(companyname).get_days_high(), Share(companyname).get_days_low(),Share(companyname).get_price(),Share(companyname).get_volume(), float(Share(companyname).get_price()))
    # Closing Data
    price = yahoo_data['Adj Close']
    # Moving average of the last 14 days
    moving_average = pd.rolling_mean(price, 14)
    # The exponential moving average gives importance to the stocks closer to today
    exp_moving_avg = pd.ewma(price, span=5)
    # Golden Ratio
    fibonacci_ratio = ((1-5**0.5)/2)
    # Take the last two of the moving average and the last two of exponential moving average and multiply them by the Fibonacci Golden Ratio
    predict_price = (((moving_average[-1]-moving_average[-2])*(1-fibonacci_ratio)+(exp_moving_avg[-1]-exp_moving_avg[-2])*(fibonacci_ratio)+price[-1]))
    # Store these values into an array
    array_1.append(str(predict_price))
    return array_1

def main():
	output = []
	# Determine Today's Date	
	today = dt.date.today()-dt.timedelta(days=0)
    	# Gather the Prediction Data
	for companyname in firmlist:
        	gatherpreddata(companyname, today)
	# Put the output data for the Stocks
	output = gatherpreddata(companyname, today)
	for x,n in zip(output,firmlist):
		print (x,n)
main()
