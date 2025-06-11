import numpy as np
from scipy.stats import norm

# Parameters
n = 100  # total number of function evaluations
alpha = 0.05
half_n = n // 2  # since we use pairs

# Generate uniform samples in [0, 1]
u = np.random.uniform(0, 1, half_n)
u_antithetic = 1 - u

# Evaluate the integrand at both u and 1 - u
fu = np.exp(u)
fu_antithetic = np.exp(u_antithetic)

# Compute antithetic average
fu_avg = 0.5 * (fu + fu_antithetic)

# Estimate of the integral
X_hat = np.mean(fu_avg)

# Standard error
std_error = np.std(fu_avg, ddof=1) / np.sqrt(half_n)

# 95% confidence interval
z = norm.ppf(1 - alpha / 2)
ci_lower = X_hat - z * std_error
ci_upper = X_hat + z * std_error

# Output
print(f"Antithetic Monte Carlo estimate: {X_hat:.6f}")
print(f"95% confidence interval: ({ci_lower:.6f}, {ci_upper:.6f})")
