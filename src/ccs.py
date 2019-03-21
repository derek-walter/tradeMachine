# Import Stuff Here

'''
This is a data handler object. It's job is to Collect, Clean, and Serve data.
When passing in a stock name, it will:
Collect data from the relevant API's
    Stock Price API
    News Data API
Clean the data
    Price - Difference and Activate
    News - NLP on sentiment
Serve
    In consistent step, sequential array form.

May want to initialize a database object each time.
This would look like:
- Initialize DB
- Populate DB
- Clean and Remove DB on stock kill

May want to use OOP for data. Init data object for each stock?
i.e. stockname.values
This would make bot references to stocks that other bots are on nice and easy
It would complicate dependencies and data saving.

'''
import os
import numpy as np
import requests

# if oop I could set api key as param

av_pl = {
    'function':'TIME_SERIES_DAILY_ADJUSTED',
    'outputsize':'compact',
    'symbol':'TRX',
    'apikey':avkey
    }
av_loc = 'https://www.alphavantage.co/query'
iex_loc = 'https://cloud.iexapis.com'


data = requests.get(url, params = av_pl)
trx = data.json()

#  new

def call_api(url, stock):
    # Hoping this is a streamlined retrieval from numerous APIs
    with requests.get(url, params = payload, stream=True) as r:

# intrino
from __future__ import print_function
import time
from datetime import datetime, date, timedelta
import intrinio_sdk
from intrinio_sdk.rest import ApiException
from pprint import pprint

intrinio_sdk.ApiClient().configuration.api_key['api_key'] = 'YOUR_API_KEY'

security_api = intrinio_sdk.SecurityApi()

identifier = 'AAPL' # str | A Security identifier (Ticker, FIGI, ISIN, CUSIP, Intrinio ID)
start_date = str(date.today() - timedelta(days = 90)) # date | Get historical data on or after this date (optional)
end_date = str(date.today()) # date | Get historical date on or before this date (optional)
frequency = 'daily' # str | Sort by date `asc` or `desc` (optional)
next_page = '' # str | Gets the next page of data from a previous API call (optional)

try:
    api_response = security_api.get_security_stock_prices(identifier, start_date=start_date, end_date=end_date, frequency=frequency)
    pprint(api_response)
except ApiException as e:
    print("Exception when calling SecurityApi->get_security_historical_data: %s\n" % e)
