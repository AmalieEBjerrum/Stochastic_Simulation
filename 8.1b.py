import numpy as np
import matplotlib.pyplot as plt

#Input data
data = np.array([56, 101, 78, 67, 93, 87, 64, 72, 80, 69])
n = len(data)
m = 10000  # number of bootstrap samples
a, b = -5, 5

#Compute sample mean
sample_mean = np.mean(data)

#Bootstrap resampling and computing deviations
bootstrap_means = np.array([
    np.mean(np.random.choice(data, size=n, replace=True))
    for _ in range(m)
])
deviations = bootstrap_means - sample_mean

#Estimate p
p_hat = np.mean((deviations > a) & (deviations < b))
print(f"Estimated p ≈ {p_hat:.4f}")

#Plot the bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(deviations, bins=50, density=True, color='skyblue', edgecolor='black')
plt.axvline(a, color='red', linestyle='--', label=f'a = {a}')
plt.axvline(b, color='green', linestyle='--', label=f'b = {b}')
plt.title(r"Bootstrap Distribution of $\bar{X}_n^* - \bar{X}_n$")
plt.xlabel(r"$\bar{X}_n^* - \bar{X}_n$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.savefig('8.1b.png')
plt.show()