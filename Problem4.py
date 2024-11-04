
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde


## a)

stock_df = pd.read_csv('Stock_data.csv', parse_dates=['date'])

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

returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['RET'].dropna().values
output_4a = bootstrapped_VaR(returns_eix)
print(f"The 95% VaR is approximately {output_4a:.4f}.")

## b) 

def plot_bootstrapped_var(returns, alpha=0.05, n_draws=10000):
    bootstrap_samples = np.random.choice(returns, size=(n_draws, len(returns)), replace=True)
    bootstrap_vars = np.percentile(bootstrap_samples, alpha * 100, axis=1)

    kde = gaussian_kde(bootstrap_vars)
    x_range = np.linspace(min(bootstrap_vars), max(bootstrap_vars), 1000)
    plt.plot(x_range, kde(x_range), color='darkred', label="KDE")

    plt.hist(bootstrap_vars, bins=50, edgecolor='darkblue')
    plt.title(f'Bootstrapped VaR at {alpha*100}% Level')
    plt.xlabel('VaR')
    plt.ylabel('Frequency')
    plt.show()

    return

returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['RET'].dropna().values
output_4b = plot_bootstrapped_var(returns_eix)
print(output_4b)

## c)


def bootstrap_var_confidence_interval(returns, alpha=0.05, n_draws=10000, conf_level=0.95):
    """
    Calculate the 95% confidence interval for the bootstrapped VaR.
    
    Parameters:
    - returns: 1-D array-like, historical return data.
    - alpha: float, confidence level for VaR.
    - n_draws: int, number of bootstrap samples.
    - conf_level: float, confidence level for the interval (e.g., 0.95 for 95%).
    
    Returns:
    - interval: list of floats, formatted as an interval [lower_bound, upper_bound].
    """
    # Generate bootstrap samples
    bootstrap_samples = np.random.choice(returns, size=(n_draws, len(returns)), replace=True)
    # Calculate the VaR for each bootstrap sample
    bootstrap_vars = np.percentile(bootstrap_samples, alpha * 100, axis=1)
    # Compute the lower and upper bounds of the confidence interval
    lower_bound = np.percentile(bootstrap_vars, (1 - conf_level) / 2 * 100)
    upper_bound = np.percentile(bootstrap_vars, (1 + conf_level) / 2 * 100)
    return [round(float(lower_bound), 6), round(float(upper_bound), 6)]

# Filter returns for ticker 'EIX' and drop missing values
returns_eix = stock_df[stock_df['TICKER'] == 'EIX']['RET'].dropna().values

# Calculate the 95% confidence interval for the bootstrapped VaR
output_4c = bootstrap_var_confidence_interval(returns_eix)
print(f"95% Confidence Interval for VaR: {output_4c}")


