import numpy as np
from scipy.stats import norm

#Crude Monte Carlo Estimator
def crude_mc(a, n):
    z = np.random.normal(0, 1, n)
    estimates = (z > a).astype(float)
    mean = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(n)
    return mean, mean - 1.96 * std_error, mean + 1.96 * std_error

#Importance Sampling
def importance_sampling(a, sigma2, n):
    sigma = np.sqrt(sigma2)
    x = np.random.normal(loc=a, scale=sigma, size=n)

    # Weight function: φ(x)/q(x)
    weights = np.exp( ((x - a)**2 - x**2) / 2 ) * sigma
    estimates = weights * (x > a)
    
    mean = np.mean(estimates)
    std_error = np.std(estimates, ddof=1) / np.sqrt(n)
    return mean, mean - 1.96 * std_error, mean + 1.96 * std_error

#Testing for different values
a_vals = [2, 4]
n_vals = [100, 1000, 10000]
sigma2 = 1

for a in a_vals:
    for n in n_vals:
        crude_est, crude_low, crude_up = crude_mc(a, n)
        is_est, is_low, is_up = importance_sampling(a, sigma2, n)

        print(f"\n--- a = {a}, n = {n} ---")
        print(f"Crude MC:     {crude_est:.6e} (95% CI: {crude_low:.6e}, {crude_up:.6e})")
        print(f"Importance:   {is_est:.6e} (95% CI: {is_low:.6e}, {is_up:.6e})")
        print(f"Exact value:  {1 - norm.cdf(a):.6e}")
