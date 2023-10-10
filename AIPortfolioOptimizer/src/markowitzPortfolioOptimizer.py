"""
markowitzPortfolioOptimizer.py

1) Read in .csv price data
2) Calculate Returns and Risk
3) Optimize Portfolio to Max Sharpe Ratio via Monte Carlo Sim

"""


"""
=============================================================================
Import Modules
-----------------------------------------------------------------------------
"""
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# import statsmodels.api as sm
from pandas_datareader import data as wb
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
        
def normalize_and_plot_data(df):
    """
    Normalize data frame to 100 allows for easy comparison of time series data
    of various price scales.
            
    Argument: Data Frame (df) of time series data
    Returns: Normalized data frame
    """
    print()
    print(50 * '=')
    print('INFO: Normalizing Price Data For Comparison Plot.')
    norm = (df / df.iloc[2]) * 1
    #Clean data
    norm.replace(np.nan,0.000)                                               
    norm.replace(np.inf,0.000)     
    norm.replace(-np.inf,0.000)  
    print(norm.head(5))
    print('INFO:  Normalizing Price Data Complete! Close Figure to Continue.')
    print(50 * '^')
    
    #Plot norm data
    norm['Adj Close'].plot(figsize = (15,8))
    # #Formatting
    plt.ylabel('Normalized Price (USD)')
    plt.xlabel('Date')
    plt.title('Normalized Adj Close Price Plot')
    plt.grid('on','both')
    plt.show()
    
    return norm


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
    print('INFO: Calculating Annualized Returns & Risks For Comparison Plot.')
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
=============================================================================
Clear Terminal Window
-----------------------------------------------------------------------------   
"""
#Use for code development and debugging
#Comment out when not needed.
clear_terminal()



"""
=============================================================================
Load CSV Price Data 
-----------------------------------------------------------------------------   
"""
# Specify root path for csv data files
root_path = os.getcwd()
relative_path = 'AIPortfolioOptimizer/data'
filename = 'PriceData SP500 From 01 Jan, 2018 To 21 Apr, 2023.csv'
file_path = os.path.join(root_path, relative_path, filename)

# Load csv data file and format data frame
price_data = load_price_data(file_path)

# Pull Ticker Symbols from MultiIndex and put into a list
symbols = price_data.columns.levels[1].unique().tolist()
symbols.remove('Unnamed: 0_level_1')

# print("Debugging -> Ticker Symbols Read in:\n " + str(symbols))


"""
#=============================================================================
# Normalize & Plot
#-----------------------------------------------------------------------------
"""
# Normalize data frame to 100 allows for easy comparison of time series data
# of 
norm = normalize_and_plot_data(price_data)



"""
#=============================================================================
# Calculate Log Returns & Risk of Stocks
#-----------------------------------------------------------------------------
"""
#Calculate Daily Returns with log_returns_data() function
daily_log_returns = log_returns_data(price_data['Adj Close'])

#Calculate Annualized Risk and Returns with annual_risk_return() funciton
annual =  annual_risk_return(daily_log_returns)

print('\nINFO: Close Figure to Continue.\n')
#Plot Annual Risk and Returns on a Scatter Plot 
annual.plot(figsize = (20,10), kind = "scatter", x = "Risk", y = "Return", s = 50, fontsize = 15)
for i in annual.index:
    plt.annotate(i, xy=(annual.loc[i,"Risk"]+0.005, annual.loc[i,"Return"]+0.005))
#Plot Formatting
plt.xlabel("Ann. Risk(std)", fontsize = 15)
plt.ylabel("Ann. Return", fontsize = 15)  
plt.title("Annualized Stock Risk vs Return", fontsize = 20)
plt.grid()
plt.show()


"""
#=============================================================================
# Generate a Portfolio of Random Weights and Find Max Sharpe Ratio
#-----------------------------------------------------------------------------
"""

#Define the number of assets from loaded data
assets = symbols
# Remove S&P 500, DJI, and NASDAQ from assets list if desired.
# if '^GSPC' in assets: assets.remove('^GSPC')
# if '^DJI' in assets: assets.remove('^DJI')
# if '^IXIC' in assets: assets.remove('^IXIC')
num_assets = len(assets)
#Define the number of portfolios to generate for Monte Carlo Sim
num_portfolios = 10000
print()
print(25 * '*-*')
print(25 * '*-*')
print('INFO: Begin Portfolio Optimization.')
print("\tNumber of Assets = "+ str(num_assets))
print("\tNumber of Portfolios = "+ str(num_portfolios))
print("-----------------------------------\n")
# Initialize empty arrays to hold portfolio returns, risks, and weights
port_returns = np.zeros(num_portfolios)
port_risks = np.zeros(num_portfolios)
port_weights = np.zeros(shape=(num_portfolios, num_assets))
port_sharpe = np.zeros(num_portfolios)

# Loop and generate random portfolios
for ind in range(num_portfolios):
    # Generate random weights for a portfolio
    weights = np.array(np.random.random(num_assets))
    #Force sum of weights for each portfolio to equal ONE
    weights = weights / np.sum(weights)     

    # Store each weight array in the port_weights array
    port_weights[ind,:] = weights

    # Calculate the expected return for each portfolio
    port_returns[ind] = np.sum(daily_log_returns.mean() * weights) * 252

    # Calculate the expected risk for each portfolio
    port_risks[ind] = np.sqrt(np.dot(weights.T, np.dot(daily_log_returns.cov() * 252, weights)))

    # # Calculate the Sharpe Ratio for each portfolio
    port_sharpe[ind] = port_returns[ind] / port_risks[ind]

# Get Portfolio with the Highest (MAX) Sharpe Ratio    
ind_max_sharpe = np.argmax(port_sharpe)
# Get the weights for the portfolio with the highest Sharpe Ratio
max_sharpe_weight = port_weights[ind_max_sharpe,:]
# Get the expected return for the portfolio with the highest Sharpe Ratio
max_sharpe_return = port_returns[ind_max_sharpe]
# Get the expected risk for the portfolio with the highest Sharpe Ratio
max_sharpe_risk = port_risks[ind_max_sharpe]

print()
print('Max Sharpe Risk ='+ str(max_sharpe_risk) )
print('Max Sharpe Return = ' + str(max_sharpe_return) )
print('Max Sharpe Weight = ' + str(max_sharpe_weight) )
print()


print('INFO:  Portfolio Optimization Complete!')
print(25 * '^^^')


# Plot the results of the Monte Carlo Simulation
print()
print(50 * '=')
print('INFO: Plotting Portfolio Optimization Results.')
print('INFO: Close Figure to Continue.')

plt.figure(figsize=(20,10))
plt.scatter(port_risks, port_returns, c=port_sharpe, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')

# Plot the portfolio with the highest Sharpe Ratio
plt.scatter(max_sharpe_risk, max_sharpe_return, c='red', s=50, edgecolors='black')
# Plot Formatting
plt.xlabel("Risk", fontsize = 15)
plt.ylabel("Returns", fontsize = 15)  
plt.title("Portfolio Optimization", fontsize = 20)
plt.grid()
plt.show()

print()
print('INFO: Plotting Complete!')
print(50 * '^')





