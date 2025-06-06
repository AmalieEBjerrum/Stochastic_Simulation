import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, pearsonr

#Parameters
n = 10000
x0 = 1
M = 2**16  # More realistic modulus
a = 1103515245
c = 12345

#Generate numbers
sys_vals = np.random.rand(n)

#Histogram
plt.figure(figsize=(12, 5))
plt.hist(sys_vals, bins=10, edgecolor='black')
plt.title("System Randon Number Generator Histogram")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig('Exercise 1.2 Histogram.png')
plt.show()

#Lag scatter plot (lag = 1)
plt.figure(figsize=(12, 5))
plt.scatter(sys_vals[:-1], sys_vals[1:], alpha=0.5, s=1)
plt.title("System Random Number Generator Lag-1 Scatter Plot")
plt.xlabel("uₙ")
plt.ylabel("uₙ₊₁")
plt.savefig('Exercise 1.2 Scatter Plot.png')
plt.show()

#Chi-square test
def chi2_uniform_test(data, bins=10):
    counts, _ = np.histogram(data, bins=bins)
    expected = [len(data) / bins] * bins
    stat, p = chisquare(counts, expected)
    return stat, p

chi_sys = chi2_uniform_test(sys_vals)
print("\n--- Chi² Test ---")
print(f"System RNG: χ² = {chi_sys[0]:.4f}, p = {chi_sys[1]:.4f}")

#Kolmogorov-Smirnov test
ks_sys = kstest(sys_vals, 'uniform')
print("\n--- Kolmogorov-Smirnov Test ---")
print(f"System RNG: KS statistic = {ks_sys.statistic:.4f}, p = {ks_sys.pvalue:.4f}")

#Run test
def run_test(data):
    median = np.median(data)
    runs = 1
    for i in range(1, len(data)):
        if (data[i] > median) != (data[i-1] > median):
            runs += 1
    expected_runs = (2 * len(data) - 1) / 3
    std_runs = np.sqrt((16 * len(data) - 29) / 90)
    z = (runs - expected_runs) / std_runs
    return runs, z

run_sys = run_test(sys_vals)

print("\n--- Run Test ---")
print(f"System RNG: Runs = {run_sys[0]}, z = {run_sys[1]:.4f}")

#Autocorrelation test (lag-h)
def autocorrelation(data, h):
    return pearsonr(data[:-h], data[h:])[0]

print("\n--- Autocorrelations ---")
for h in [1, 2, 5, 10, 50, 100]:
    ac_sys = autocorrelation(sys_vals, h)
    print(f"Lag {h:>3}: Autocorrelation = {ac_sys:.4f}")
