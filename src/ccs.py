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

def call_api(url, payload):
    with requests.get(url, params = payload, stream=True) as r:
