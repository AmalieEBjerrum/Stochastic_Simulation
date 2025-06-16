import numpy as np
from scipy.stats import pareto

#a)
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

#b
# Bootstrap parameters
r = 100  # number of bootstrap replicates
N = len(data)

# Compute bootstrap sample means
bootstrap_means = np.array([
    np.mean(np.random.choice(data, size=N, replace=True))
    for _ in range(r)
])

# Estimate the variance of the sample mean
bootstrap_var_mean = np.var(bootstrap_means, ddof=1)

print(f"Bootstrap estimate of Var(sample mean) ≈ {bootstrap_var_mean:.4f}")

#c
# Compute bootstrap sample medians
bootstrap_medians = np.array([
    np.median(np.random.choice(data, size=N, replace=True))
    for _ in range(r)
])

# Estimate the variance of the sample median
bootstrap_var_median = np.var(bootstrap_medians, ddof=1)

print(f"Bootstrap estimate of Var(sample median) ≈ {bootstrap_var_median:.6f}")

#d
# Compute ratio of variances (mean variance divided by median variance)
variance_ratio = bootstrap_var_mean / bootstrap_var_median
print(f"\nRatio (Var(mean) / Var(median)) = {variance_ratio:.2f}")

if variance_ratio > 1:
    print("Median is more precise (has smaller variance).")
else:
    print("Mean is more precise (has smaller variance).")
