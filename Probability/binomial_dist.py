# Binomial Distribution
def binomial_distribution(n, p, k):
    from math import comb
    return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# Poisson Distribution
def poisson_distribution(lmbda, k):
    from math import exp, factorial
    return (lmbda ** k) * exp(-lmbda) / factorial(k)

import numpy as np
import matplotlib.pyplot as plt

def plot_distribution(distribution_func, params, x_range, title):
    x = np.arange(x_range[0], x_range[1] + 1)
    y = [distribution_func(*params, k) for k in x]
    
    plt.bar(x, y, alpha=0.7)
    plt.title(title)
    plt.xlabel('k')
    plt.ylabel('Probability')
    plt.show()
    
# Example plot
plot_distribution(binomial_distribution, (10, 0.5), (0, 10), 'Binomial Distribution')
plot_distribution(poisson_distribution, (5,), (0, 15), 'Poisson Distribution')