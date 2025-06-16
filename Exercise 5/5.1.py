import numpy as np
from scipy.stats import norm

# Parameters
n = 100  # number of samples
alpha = 0.05  # significance level for 95% confidence

#Generate uniform random samples from [0, 1]
u_samples = np.random.uniform(0, 1, n)

#Evaluate the integrand e^x at these samples
fu_samples = np.exp(u_samples)

#Monte Carlo estimate of the integral
X_hat = np.mean(fu_samples)

#Standard error of the estimator
std_error = np.std(fu_samples, ddof=1) / np.sqrt(n)

#95% confidence interval using normal approximation
z_score = norm.ppf(1 - alpha / 2)
ci_lower = X_hat - z_score * std_error
ci_upper = X_hat + z_score * std_error

# Output results
print(f"Monte Carlo estimate of the integral: {X_hat:.6f}")
print(f"95% confidence interval: ({ci_lower:.6f}, {ci_upper:.6f})")
