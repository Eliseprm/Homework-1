
import numpy as np
from numpy.random import PCG64 
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kstest


## a)

def student_t_array(BG, degrees:int):
    # We create a Generator object from the BitGenerator provided in the inputs. 
    # Acts like a bridge between the BitGenerator BG and the array 
    array = np.random.Generator(BG)
    
    # Creates 200 random samples from a Student's t-distribution with a specified degrees of freedom
    t_dist_samples = array.standard_t(df=degrees, size=200)
    
    return(t_dist_samples)

# Example 1: 15 Degrees of freedom 

BG = PCG64()
output_a1 = student_t_array(BG, 15)
standard_dev=np.std(output_a1)

print(output_a1)
print(standard_dev)

# Example 2: 30 Degrees of freedom 

BG = PCG64()
output_a2 = student_t_array(BG, 30)
standard_dev=np.std(output_a2)

print(output_a2)
print(standard_dev)


## b)

def bootstrap_draw(bg, a:range, r:bool):
   
    # We create a Generator object from the BitGenerator provided in the inputs. 
    # Acts like a bridge between the BitGenerator BG and the array 
    random_variable = np.random.Generator(bg)
    
    # We draw a sample from the provided array a, with or without replacement depending on r being True or False 
    bootstrap_sample = random_variable.choice(a, size=len(a), replace=r)
    
    return bootstrap_sample

# Example 1: with replacement (r = True)
bg=PCG64()
a = [2,4,6,8,10,12,14,16,18,20]

output_b1 = bootstrap_draw(bg,a,True)
print(output_b1)

# Example 2: without replacement (r = False)

output_b2 = bootstrap_draw(bg,a,False)
print(output_b2)

## c)

def QQ_plot(s):
    
    # We create a Q-Q plot plotting the probabilities of the sample student-t distribution against the normal distribution N(0,1)
    stats.probplot(s, dist="norm", plot=plt)
    
    # We add labels for the title, x-axis and y-axis
    plt.title("Q-Q Plot of Sample student-t vs. N(0,1)")
    plt.xlabel("Theoretical Quantiles")
    plt.ylabel("Sample Quantiles")
    
    # Shows the QQ plot obtained
    plt.show()

# Example 

output_c=QQ_plot(output_a1)


## d)

def bootstrap_ks_test_loop(bg, a, T):

    array = np.random.Generator(bg)
    null_rejected = 0  # Counts the number of times we rejected the null hypothesis
    # we initialize it to 0 before doing the loop

    for _ in range(T):
        # We produce a bootstrap sample with or without replacement depending on r
        bootstrapped_sample = array.choice(a, size=len(a), replace=True)
        
        # We do the Kolmogorov-Smirnov test, with H0: bootstrapped sample normally distributed
        ks_stat, p_value = kstest(bootstrapped_sample, 'norm')
        
        # If p-value < 0.05, we reject H0 and add 1 to the reject_count
        if p_value < 0.05:
            null_rejected += 1
    
    # Calculate the rejection rate
    rejection_rate = null_rejected / T
    return rejection_rate

# Example 1

output_d1=bootstrap_ks_test_loop(bg, output_a1, 100)
print(output_d1)

# Example 2 - larger number of iterations

output_d2=bootstrap_ks_test_loop(bg, output_a1, 1000)
print(output_d2)

