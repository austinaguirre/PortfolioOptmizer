import yfinance as yf
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
data = pd.read_html(url)
table = data[0]
tickers = table['Symbol'].tolist()
tickers.remove('BF.B')
tickers.remove('BRK.B')

start_date = '2015-01-01'
end_date = '2023-04-06'

prices_df = pd.DataFrame()
for ticker in tickers:
    try:
        ticker_df = yf.download(ticker, start=start_date, end=end_date)
        if ticker_df.empty:
            raise ValueError("empty dataframe")
        ticker_df = ticker_df.reset_index()
        ticker_df['Ticker'] = ticker
        prices_df = pd.concat([prices_df, ticker_df], ignore_index=True)
    except:
        pass

prices_df = prices_df[['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
prices_df.to_csv('prices.csv', index=False)



