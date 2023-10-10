
The usage of this code can be described as follows:


The director structure is described here:
AIPortfolioOptimizer
    -data
        -.csv files located here
    -src
        genetricAlgorithmOptimizer.py
        indexReturnCalculator.py
        markowitzPortfolioOptimizer.py
    -utilities
        getTickerPriceData.py
        getTickerSymbolHTML.py
        loadTickerPriceData.py


Utilities Explanation:
    getTickerSymbolHTML.py  
        -runs as a stand alone python file
        -scraps HTML wiki pages to pull stock ticker symbols
    getTickerPriceData.py
        -runs as a stand alone python file
        -uses function from getTickerSymbolHTML.py
        -uses yahoo API call to pull historical stock price data
        -writes yahoo API call to .csv files
    loadTickerPriceData.py
        -contains function to read and load .csv files of price data

App Explanation:
    indexReturnCalculator.py    
        -runs as a stand alone python file
        -pull index historical price data over specified time period
        -calculates the rates of return and risk for each index
    markowitzPortfolioOptimizer.py
        -runs as a stand alone python file
        -loads price data .csv
        -calculates return, risk and sharpe ratio
        -performs a Monte Carlo analysis to maximize the Sharpe ratio
        -returns Optimize Portfolio which has the Max Sharpe Ratio
    genetricAlgorithmOptimizer.py
        -runs as a stand alone python file
        -loads price data .csv
        -runs Genetic Algorithm 
        -returns GA optimized Portfolio
        -take GA portfolio and Optimizes that using a Simulated Annealing algorithm
        -returns the results of the GA and SA algorithms