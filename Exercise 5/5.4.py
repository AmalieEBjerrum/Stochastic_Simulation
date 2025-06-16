import numpy as np
from scipy.stats import norm

# Parameters
n = 100                  # total function evaluations
alpha = 0.05
m = n                    # one sample per stratum

# Stratified sampling
u = np.random.uniform(0, 1, m)
x_strat = (np.arange(m) + u) / m
f_x_strat = np.exp(x_strat)

# Point estimate
X_hat = np.mean(f_x_strat)

# Standard error
std_error = np.std(f_x_strat, ddof=1) / np.sqrt(m)

# Confidence interval
z = norm.ppf(1 - alpha / 2)
ci_lower = X_hat - z * std_error
ci_upper = X_hat + z * std_error

# Output
print(f"Stratified sampling estimate: {X_hat:.6f}")
print(f"95% confidence interval: ({ci_lower:.6f}, {ci_upper:.6f})")
