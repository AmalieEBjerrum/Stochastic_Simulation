import numpy as np
from scipy.stats import pareto

# Set random seed for reproducibility
np.random.seed(42)

# Parameters for Pareto distribution
beta = 1      # scale
k = 1.05      # shape
N = 200       # sample size

# Generate N Pareto-distributed random variates
data = pareto.rvs(b=k, scale=beta, size=N)

# Compute sample mean and median
sample_mean = np.mean(data)
sample_median = np.median(data)

print(f"Sample mean ≈ {sample_mean:.4f}")
print(f"Sample median ≈ {sample_median:.4f}")
