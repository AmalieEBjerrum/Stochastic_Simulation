import matplotlib.pyplot as plt
from scipy.stats import expon, norm, pareto
import numpy as np

def pareto_sample(n, k, beta=1.0):
    U = np.random.uniform(0, 1, size=n)
    return beta * (U ** (-1 / k))

# Parameters
beta = 1.0
ks = [2.05, 2.5, 3, 4]
n = 10000

for k in ks:
    #Empirical mean and variance
    samples = pareto_sample(n, k, beta)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples)

    # Analytical mean and variance
    if k > 1:
        mean_analytical = k * beta / (k - 1)
    else:
        mean_analytical = np.inf

    if k > 2:
        var_analytical = (k * beta**2) / ((k - 1)**2 * (k - 2))
    else:
        var_analytical = np.inf

    print(f"\nk = {k}")
    print(f"Sample mean:      {sample_mean:.4f}")
    print(f"Theoretical mean: {mean_analytical:.4f}")
    print(f"Sample variance:      {sample_var:.4f}")
    print(f"Theoretical variance: {var_analytical:.4f}")