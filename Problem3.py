
import numpy as np
from numpy.random import PCG64 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest


## a)

def draw_student_t(BG, degrees:int):
    # Create a Generator object from the provided BitGenerator
    rng = np.random.Generator(BG)
    
    # Draw 200 i.i.d. samples from the Student's t-distribution
    samples = rng.standard_t(df=degrees, size=200)
    
    return(samples)

# Example 

BG = PCG64()
output_a = draw_student_t(BG, 10)
print(output_a)


## b)

def random_draw(bg, a:range, r:bool):
    """
    Draws random values from the input array, with or without replacement.

    Parameters:
    - bg: BitGenerator instance, used to initialize randomness
    - arr: 1-D array from which values are drawn
    - r: Boolean, if True draw with replacement, otherwise without replacement

    Returns:
    - A 1-D array of the same size as arr with randomly drawn values
    """
    # Create a Generator object from the provided BitGenerator
    random = np.random.Generator(bg)
    
    # Draw samples from arr with or without replacement based on r
    samples = random.choice(a, size=len(a), replace=r)
    
    return samples

# Example: with replacement
bg=PCG64()
output_b1 = random_draw(bg,output_a,True)
print(output_b1)

# Example: without replacement

output_b2 = random_draw(bg,output_a,False)
print(output_b2)

## c)

def probability_plot(s):
    """
    Creates a Q-Q plot comparing the values in the 1-D array `s` to a standard normal distribution (N(0,1)).

    Parameters:
    - s: 1-D array of data points to compare with N(0,1) distribution.
    
    Returns:
    - None. Displays a Q-Q plot.
    """
    # Generate the Q-Q plot
    stats.probplot(s, dist="norm", plot=plt)
    
    # Add labels and a title for clarity
    plt.title("Probability Plot of Sample Data vs. N(0,1)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    # Show the plot
    plt.show()

# Example 

output_c=probability_plot(output_a)
print(output_c)

## d)

import numpy as np
from scipy.stats import kstest

def bootstrap_ks_test(bg, a, T):
    """
    Performs T bootstrap samples from array a, applies the K-S test for N(0,1),
    and returns the fraction of tests where H0 is rejected at the 5% level.

    Parameters:
    - bg: BitGenerator instance, used to initialize randomness
    - a: 1-D array of real values to bootstrap
    - T: Positive integer, number of bootstrap samples to generate

    Returns:
    - p: Fraction of the T simulations where we reject H0 at the 5% significance level
    """
    rng = np.random.Generator(bg)
    reject_count = 0  # Counter for the number of times we reject H0

    for _ in range(T):
        # Bootstrap sample from a with replacement
        bootstrapped_sample = rng.choice(a, size=len(a), replace=True)
        
        # Perform the Kolmogorov-Smirnov test against the N(0,1) distribution
        ks_stat, p_value = kstest(bootstrapped_sample, 'norm')
        
        # Check if we reject H0 at the 5% significance level
        if p_value < 0.05:
            reject_count += 1
    
    # Calculate the fraction of times we rejected H0
    p = reject_count / T
    return p

# Example

output_d=bootstrap_ks_test(bg, output_a, 200)
print(output_d)
