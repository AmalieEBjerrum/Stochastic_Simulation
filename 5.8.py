import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.integrate import quad
from scipy.optimize import minimize_scalar

# Define the true value of the integral
true_val = np.exp(1) - 1  # ∫₀¹ e^x dx

# Define the variance function V(λ)
def importance_variance(lambda_):
    if lambda_ <= 0:
        return np.inf
    c = (1 - np.exp(-lambda_)) / lambda_ #constant outside of integral
    integral = (np.exp(2 + lambda_) - 1) / (2 + lambda_) #solution to integral
    return c * integral

# Plot variance vs lambda
lambdas = np.linspace(0.1, 10, 100)
variances = [importance_variance(l) for l in lambdas]

plt.plot(lambdas, variances)
plt.xlabel("λ")
plt.ylabel("Variance of IS estimator")
plt.title("Importance Sampling Variance vs λ")
plt.grid(True)
plt.savefig('5.8 Variance.png')
plt.show()

# Find optimal lambda
res = minimize_scalar(importance_variance, bounds=(0.1, 10), method='bounded')
print(f"Optimal lambda: {res.x:.4f}, Min variance: {res.fun:.6f}")



# Try simulation for a few λ
def is_estimate(lambda_, n=10000):
    # Sample from truncated exponential
    u = np.random.uniform(0, 1, size=n)
    x = -np.log(1 - u * (1 - np.exp(-lambda_))) / lambda_
    weights = np.exp(x + lambda_ * x) * (1 - np.exp(-lambda_)) / lambda_
    estimate = np.mean(weights)
    std = np.std(weights) / np.sqrt(n)
    return estimate, std

for lam in [0.5, 1, 2, 4, 8]:
    est, stderr = is_estimate(lam)
    print(f"λ = {lam:.1f}: estimate = {est:.5f}, stderr = {stderr:.5f}, true = {true_val:.5f}")
