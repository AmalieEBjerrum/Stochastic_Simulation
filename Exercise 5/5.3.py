import numpy as np
from scipy.stats import norm

# Parameters
n = 100
alpha = 0.05
expected_h = 0.5  # E[x] over [0, 1]

#Sample U ~ U(0,1)
u = np.random.uniform(0, 1, n)
f_u = np.exp(u)       # target function
h_u = u               # control variate

#Estimate optimal coefficient c
cov_fh = np.cov(f_u, h_u, ddof=1)[0, 1]
var_h = np.var(h_u, ddof=1)
c = cov_fh / var_h

#Construct control variate estimator
z = f_u - c * (h_u - expected_h)
X_hat = np.mean(z)

# Step 4: Standard error and confidence interval
std_error = np.std(z, ddof=1) / np.sqrt(n)
s = norm.ppf(1 - alpha / 2)
ci_lower = X_hat - s * std_error
ci_upper = X_hat + s * std_error

# Output
print(f"Control variate estimate: {X_hat:.6f}")
print(f"95% confidence interval: ({ci_lower:.6f}, {ci_upper:.6f})")
