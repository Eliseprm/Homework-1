import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# A)

data = pd.read_csv('Stock_data.csv', parse_dates=['date'])

# Filter data
start_date = '2013-12-02'
end_date = '2022-12-02'
tickers = ['XOM', 'ZTS', 'NKE']
data_filtered = data[(data['date'] >= start_date) & (data['date'] <= end_date) & (data['TICKER'].isin(tickers))]

# Calculate descriptive statistics for each ticker
desc_stats = data_filtered.groupby('TICKER')['RET'].describe()
print("Descriptive Statistics:\n", desc_stats)

# Plot time-series data for each ticker
plt.figure(figsize=(12, 8))
for ticker in tickers:
    plt.plot(data_filtered[data_filtered['TICKER'] == ticker]['date'], 
             data_filtered[data_filtered['TICKER'] == ticker]['RET'], 
             label=ticker)
plt.title("Daily Returns for XOM, ZTS, and NKE (2013-2022)")
plt.xlabel("Date")
plt.ylabel("Daily Return")
plt.legend()
plt.show()

#   Mean : Nike has a 0.0665% daily average return
#          Exxon has a 0.0400% daily average return
#          Zoetis has a 0.0873% daily average return
#          This suggests that on average, ZTS provided highest returns per day, indicating stronger growth potential during this period compared to Nike & Exxon.
#
#   STD : The standard deviation represents the volatility of the stock returns, Nike has the highest volatility (0.017972), that suggests more fluctuations regarding the daily returns compared to the other stocks Exxon (0.017525) and Zoetis (0.016187).
# 
#   Min (worst daily return) : Nike : -12.81%
#                              Exxon : -12.22%
#                              Zoetis : -14.70%
#                              Zoetis experienced the biggest daily drop while supposed to be the more stable (lowest std).
#
#   Max (highest daily return) : Nike : 15.53%
#                                Exxon : 12.69%
#                                Zoetis : 11.98%
#                                That indicates Nike's potential for large positive retuns.
#
#   Quartiles : We clearly see that the quartiles for the three stocks are similar, which suggests a similar trend/distribution over the years. The median for each is around 0 (slightly positive), that suggests similar amount of positive and negative returns.
#
#  The plot : The plot shows similar distribution of the returns for the three stocks, with some upward and downward peaks sometimes. Also, we clearly notice some rare events or crisis as for the Covid-19 crisis where we observe high negative returns for all three stocks suggesting higher standard deviation at that time, followed by high returns. """

# B)

import numpy as np

def risk_measures(returns, alpha=0.05):
    """
    Calculating the Value at Risk (VaR) and Expected Shortfall (ES) for given returns.
    
    Parameters:
    - returns: array-like, daily returns
    - alpha: float, risk level (e.g., 0.05 for 5%)
    
    Returns:
    - var_alpha: VaR at the given alpha level
    - es_alpha: Expected Shortfall at the given alpha level
    """
    # Sorting returns
    sorted_returns = np.sort(returns)
    
    # VaR
    var_alpha = np.percentile(sorted_returns, (1 - alpha) * 100)
    
    # ES
    es_alpha = sorted_returns[sorted_returns <= var_alpha].mean()
    
    return var_alpha, es_alpha

# Example usage:
alpha = 0.05
for ticker in tickers:
    ticker_returns = data_filtered[data_filtered['TICKER'] == ticker]['RET'].dropna()
    var, es = risk_measures(ticker_returns, alpha)
    print(f"{ticker} VaR at {alpha*100}% level: {var:.6f}")
    print(f"{ticker} ES at {alpha*100}% level: {es:.6f}")



          
