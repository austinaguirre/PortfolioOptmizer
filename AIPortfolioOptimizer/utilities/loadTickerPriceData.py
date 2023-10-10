"""
loadTickerPriceData.py

Script loads price data from a .csv file and formats the data frame.
"""

"""
==============================================================================
Import Modules
------------------------------------------------------------------------------
"""
import os
import datetime as dt
import numpy as np
import pandas as pd

"""
==============================================================================
Functions
------------------------------------------------------------------------------
"""
def load_price_data(filename):
    """
    Load csv file of price data and format data frame for analysis.
    
    Argument: "filename.csv"
    Returns: Data Frame (df) of time series data with Date column as index.
    """
    print()
    print(50 * '=')
    print('INFO: Loading Price Data.')
    print('  filename = ' + filename)
    # Read in csv data file
    df = pd.read_csv(filename, header=[0,1])
    # Format data frame
    # Make Date column the index    
    df.set_index(df.columns[0], inplace=True)
    # Replace NaN with ''
    df.replace(np.nan, '', inplace=True)
    # Convert columns to float
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Print head of date frame to confirm format is good
    print(df.head(5))
    print('INFO:  Loading Price Data Complete!')
    print(50 * '^')
    return df





"""
=============================================================================
FOR TESTING
Load CSV Price Data and Format Data Frame
-----------------------------------------------------------------------------   
"""
# # Specify root path for csv data files
# root_path = os.getcwd()
# relative_path = 'AIPortfolioOptimizer/data'
# filename = 'PriceData DOW30 From 01 Dec, 2022 To 12 Mar, 2023.csv'
# file_path = os.path.join(root_path, relative_path, filename)

# # Load csv data file and format data frame
# price_data = load_price_data(file_path)




