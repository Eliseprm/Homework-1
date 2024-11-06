
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde


## a)

stock_df = pd.read_csv('Merged_Market_and_Stock_Returns.csv', parse_dates=['date'])

def bootstrapped_VaR(returns, alpha=0.05, n=10000):

    # We convert the returns data into an array
    returns = np.array(returns)

    # We generate 10 000 bootstrapped samples from the return array
    bootstrapped_samples = np.random.choice(returns, size=(n, len(returns)), replace=True)

    # We compute the VaR for each bootstrapped sample by looking for the corresponding percentile
    bootstrapped_VaRs = np.percentile(bootstrapped_samples, alpha * 100, axis=1)
    
    # We get the average of the bootstrapped VaRs obtained
    VaR = np.mean(bootstrapped_VaRs)
    return VaR

# Example - Compute the average of bootstrapped VaRs for EIX 

returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['stock_return'].dropna().values
output_4a = bootstrapped_VaR(returns_eix)
print(f"The 95% VaR is approximately {output_4a:.4f}.")

## b) 

def plot_bootstrapped_var(returns, alpha=0.05, n_draws=10000):
    
    # We generate 10 000 bootstrapped samples from the return array
    bootstrapped_samples = np.random.choice(returns, size=(n_draws, len(returns)), replace=True)
    # We compute the VaR for each bootstrapped sample by looking for the corresponding percentile
    bootstrapped_VaRs = np.percentile(bootstrapped_samples, alpha * 100, axis=1)

    # We add a KDE curve to better understand the shape of the distribution
    kde = gaussian_kde(bootstrapped_VaRs)
    x_range = np.linspace(min(bootstrapped_VaRs), max(bootstrapped_VaRs), 1000)

    # We plot the graph (KDE, histogram, title, x-axis, y-axis)
    plt.plot(x_range, kde(x_range), color='darkred', label="KDE")
    plt.hist(bootstrapped_VaRs, bins=50, edgecolor='darkblue')
    plt.title(f'Bootstrapped VaR at {(1-alpha)*10}% Confidence')
    plt.xlabel('VaR')
    plt.ylabel('Frequency')

    # We display the graph
    plt.show()

    return    

# Example - Plot the distribution of the bootstrapped VaRs of the EIX ticker
returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['stock_return'].dropna().values
output_4b = plot_bootstrapped_var(returns_eix)


## c)

def bootstrap_var_confidence_interval(returns, alpha=0.05, n_draws=10000, conf_level=0.95):
    
    # We generate bootstrap samples
    bootstrapped_samples = np.random.choice(returns, size=(n_draws, len(returns)), replace=True)
    # We compute the VaR for each bootstrapped sample by looking for the corresponding percentile
    bootstrapped_VaRs = np.percentile(bootstrapped_samples, alpha * 100, axis=1)
    # We compute the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(bootstrapped_VaRs, (1 - conf_level) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_VaRs, (1 + conf_level) / 2 * 100)
    return [round(float(lower_bound), 6), round(float(upper_bound), 6)]

# Example - Compute the 95% confidence interval of the VaR of the EIX ticker
returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['stock_return'].dropna().values

# Calculate the 95% confidence interval for the bootstrapped VaR
output_4c = bootstrap_var_confidence_interval(returns_eix)
print(f"95% Confidence Interval for VaR: {output_4c}")


