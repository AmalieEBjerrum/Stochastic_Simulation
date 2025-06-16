import numpy as np
import matplotlib.pyplot as plt

#Input data
data = np.array([5, 4, 9, 6, 21, 17, 11, 20, 7, 10, 21, 15, 13, 16, 8])
n = len(data)
B = 10000

#Compute sample variance
sample_variance = np.var(data, ddof=1)

#Bootstrap resampling to compute sample variances
bootstrap_variances = np.array([
    np.var(np.random.choice(data, size=n, replace=True), ddof=1)
    for _ in range(B)
])

#Estimate Var(S²)
bootstrap_var_of_variance = np.var(bootstrap_variances, ddof=1)
print(f"Bootstrap estimate of Var(S²) ≈ {bootstrap_var_of_variance:.4f}")
print(f"Original sample variance S² ≈ {sample_variance:.4f}")

#Plot the bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(bootstrap_variances, bins=50, density=True, color='lightblue', edgecolor='black')
plt.axvline(sample_variance, color='red', linestyle='--', linewidth=2, label=r'Sample $S^2$')
plt.title(r"Bootstrap Distribution of Sample Variance $S^{2*}$")
plt.xlabel(r"$S^{2*}$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig('8.2.png')
plt.show()
