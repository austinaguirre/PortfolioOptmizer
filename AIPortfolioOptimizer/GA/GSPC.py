import yfinance as yf
import pandas as pd

ticker = '^GSPC'
start_date = '2015-01-01'
end_date = '2023-04-06'

# Get the price data for the S&P 500 index
prices = yf.download(ticker, start_date, end_date)['Close']
returns = prices.pct_change().fillna(0)

# Create a dataframe with price and return columns
data = pd.DataFrame({'Price': prices, 'Return': returns})

# Save the dataframe to a csv file
data.to_csv('GSPC.csv')
