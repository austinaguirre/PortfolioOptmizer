"""
getTickerPriceData.py

Script to pull Ticker Symbol Data from specified website (yahoo).
1) In "Scripting" Section enter the desired Stock Index to Pull Price Data
2) Once data has been pulled it will be written to a .csv file.
"""

"""
==============================================================================
Import Modules
------------------------------------------------------------------------------
"""
import os
import datetime as dt
import numpy as np
import pandas_datareader.data as pdr
import yfinance as yf

from getTickerSymbolsHTML import dup_delete, get_tickers


"""
==============================================================================
Functions
------------------------------------------------------------------------------
"""
def get_price_data(ticker_symbols, start_date, end_date):
    """
    Function uses a yahoo finance API call to pull price data:
        open, high, low, close, adj close, volume
        for the ticker_symbols specified
    :param ticker_symbols: list format
    :param start_date: datetime format
    :param end_date: datetime format
    :return: price in dataframe format
    """

    # --- Make API call for price data ---
    try:
        print('INFO: API Call for Price Data.')
        yf.pdr_override()  # <== that's all it takes :-) This is needed for wb.get_data_yahoo())
        data = pdr.get_data_yahoo(ticker_symbols, start_date, end_date)
        data = data.replace(np.nan, 0)
        print('INFO: API Call Success.  Price Data Acquired.')
        return data

    except:
        print('ERROR: >>> Could not get price data from API call.')



"""
==============================================================================
Scripting
------------------------------------------------------------------------------
"""
# --- Specify desired index to pull stock data: 'SP500', 'DOW30', or 'NASDAQ'
index = 'NASDAQ'

# --- Get ticker symbols
tickers = dup_delete(get_tickers(index))
# Add the selected index price data to position ZERO of the tickers data frame
# index_ticker = []
# if (index == 'DOW30'): index_ticker = '^DJI'
# if (index == 'SP500'): index_ticker = '^GSPC'
# if (index == 'NASDAQ'): index_ticker = '^IXIC'
index_tickers = ['^GSPC', '^DJI', '^IXIC']
print(50* '=')
print('INFO: Getting Ticker Symbols...')
print(tickers)
print('INFO: Index Ticker Symbols = ', index_tickers)
print('INFO: Complete.')
print(50* '^')


# --- Specify desired time period.
start_date = dt.datetime(2018,1,1)        #Year:Month:Day
# end_date   = dt.datetime(2020,1,1)        #Year:Month:Day
end_date   = dt.datetime.today()


# --- Get Price Data
print()
print(50* '=')
print('INFO: Pulling Price Data For Following Dates.')
print(f" start date = {start_date}, end date = {end_date}")
df1 = get_price_data(tickers,start_date,end_date)
df2 = get_price_data(index_tickers,start_date,end_date)
print('Print Price Data Head(5)')
print(df1.head(5))
print(df2.head(5))
print('INFO: Complete.')
print(50* '^')


# ---  Write to .csv file
print()
print(50* '=')
print('INFO: Writing Price Data to .csv...')
#  Reformat dates for .csv filename
start_date = start_date.strftime("%d %b, %Y")
end_date = end_date.strftime("%d %b, %Y")
# Define filename format
ticker_base_filename = 'PriceData ' + index
ticker_filename = ticker_base_filename + ' From ' + start_date + ' To ' + end_date + '.csv'
index_base_filename = 'IndexData ' 
index_filename = index_base_filename + ' From ' + start_date + ' To ' + end_date + '.csv'
# Specify root path for csv data files
root_path = os.getcwd()
relative_path = 'AIPortfolioOptimizer/data'
file_path = os.path.join(root_path, relative_path)

df1.to_csv(f'{file_path}/{ticker_filename}', index=True, index_label='Date')
df2.to_csv(f'{file_path}/{index_filename}', index=True, index_label='Date')
print('Ticker Filename: ' + ticker_filename)
print('Index Filename: ' + index_filename)
print('INFO: Complete.')
print(50* '^')
# %%
