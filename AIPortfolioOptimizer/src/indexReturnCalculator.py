"""
indexReturnCalculator.py

1) Read in .csv price data
2) Calculate Returns and Risk For Specified Time Period
3) Plot and display rate of return

"""


"""
=============================================================================
Import Modules
-----------------------------------------------------------------------------
"""
import os
import datetime as dt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import yfinance as yf
import math
import matplotlib.pyplot as plt

from scipy import stats

# Custom Modules
import sys
root_path = os.getcwd()
# Add project folders to root path
sys.path.append(root_path + '/AIPortfolioOptimizer')
sys.path.append(root_path + '/AIPortfolioOptimizer/data')
sys.path.append(root_path + '/AIPortfolioOptimizer/src')
sys.path.append(root_path + '/AIPortfolioOptimizer/utilities')
from loadTickerPriceData import load_price_data


"""
==============================================================================
Functions
------------------------------------------------------------------------------
"""
def clear_terminal():
    """
    Clear terminal window.
    """
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


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


def log_returns_data(df):
    """
    Define daily log returns function.
    
    Argument: Data Frame (df) of time series data
    Returns: Logarithmic returns data frame (in this case time series price data)
    """
    print()
    print(50 * '=')
    print('INFO: Calculating Log Returns.')
    log_returns =  np.log(df / df.shift(1))    
    #Clean data
    log_returns.replace(np.nan,0.000, inplace = True)                                
    log_returns.replace(np.inf,0.000, inplace = True)
    log_returns.replace(-np.inf,0.000, inplace = True) 
    print(log_returns.head(5))
    print('INFO:  Log Returns Complete!' )
    print(50 * '^')
    return log_returns

def annual_risk_return(df):
    """
    Define annualized risk and returns functions.
    
    Argument: Data Frame (df) of time series data
    Returns: Object of annualized risk and returns
    
    Annualized Returns
        log_returns_a = log_returns.mean() * 252                                                         
    Risk - Standard Deviation of Returns   
    Annualized STD                         
        log_returns_std = log_returns.std() * np.sqrt(252)   
    
    """
    
    #Calculate Annualized Returns & Risks
    print()
    print(50 * '=')
    print('INFO: Calculating Annualized Returns & Risks For Comparison.')
    annual = df.agg(["mean", "std"]).T
    annual.columns = ["Return", "Risk"]
    annual.Return = annual.Return * 252
    annual.Risk = annual.Risk * np.sqrt(252)
    print(annual.head(5))
    print()
    print('INFO:  Annualized Calculations Complete!')
    print(50 * '^')
        
    return annual


"""
==============================================================================
MAIN PROGRAM
------------------------------------------------------------------------------
"""
def main():
    #Use for code development and debugging
    #Comment out when not needed.
    clear_terminal()

    print('*** Calculate Index Returns and Risk ***')
    # --- Calculate Log Returns
    # Specify root path for csv data files
    root_path = os.getcwd()
    relative_path = 'AIPortfolioOptimizer/data'
    filename = 'IndexData  From 01 Jan, 2018 To 21 Apr, 2023.csv'
    file_path = os.path.join(root_path, relative_path, filename)

    # Load csv data file and format data frame
    price_data = load_price_data(file_path)

    # Pull Ticker Symbols from MultiIndex and put into a list
    symbols = price_data.columns.levels[1].unique().tolist()
    symbols.remove('Unnamed: 0_level_1')

    #Calculate Daily Returns with log_returns_data() function
    daily_log_returns = log_returns_data(price_data['Adj Close'])

    #Calculate Annualized Risk and Returns with annual_risk_return() funciton
    annual =  annual_risk_return(daily_log_returns)









if __name__ == '__main__':
    main()