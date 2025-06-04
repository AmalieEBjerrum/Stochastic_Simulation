import matplotlib.pyplot as plt

#Exercise 1: LCG

#Defining the LCG as a function
def LCG(x0, M, a, c, numberofrandomnumbers):
    #Initialise the seed state and define list for storing random numbers
    x = [x0]
    u = []  # Normalize to get a uniform distribution in (0,1)
    #Generate required numbers of random numbers
    for i in range (1,numberofrandomnumbers):
        next_x = (a * x[i-1] + c) % M
        x.append(next_x)
        u_new = next_x / M  # Normalize to get a uniform distribution in (0,1)
        u.append(u_new)
    return u
    

#Defining the parameter values
numberofrandomnumbers = 10000
x0 = 1
M = 2**32  # Commonly used modulus in LCG
a = 64525
c = 451390

#Generate random numbers by calling the function
randomnumbers = LCG(x0, M, a, c, numberofrandomnumbers)


#Plotting histogram with 10 classes
plt.hist(randomnumbers, bins=10, edgecolor='black')
plt.title('Histogram of LCG Random Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import matplotlib.pyplot as plt

u = randomnumbers
print(len(u))

plt.figure()
plt.scatter(u[:-1], u[1:], s=1)
plt.title('Scatter Plot of Successive Pairs (u_i, u_{i+1})')
plt.xlabel('u[i]')
plt.ylabel('u[i+1]')
plt.grid(True)
plt.show()

def chi_square_test(u, bins=10):
    n = len(u)
    expected = n / bins
    observed = [0] * bins
    for value in u:
        index = min(int(value * bins), bins - 1)
        observed[index] += 1
    chi2 = sum((o - expected) ** 2 / expected for o in observed)
    return chi2

chi2_stat = chi_square_test(randomnumbers, bins=10)
print(f"Chi-squared statistic: {chi2_stat:.2f}")

def ks_test(u):
    n = len(u)
    u_sorted = sorted(u)
    D_plus = max((i + 1)/n - val for i, val in enumerate(u_sorted))
    D_minus = max(val - i/n for i, val in enumerate(u_sorted))
    D = max(D_plus, D_minus)
    return D

ks_stat = ks_test(randomnumbers)
print(f"K-S statistic: {ks_stat:.4f}")

def runs_test(u):
    runs = 1
    for i in range(1, len(u) - 1):
        if (u[i] > u[i-1]) != (u[i+1] > u[i]):
            runs += 1
    expected_runs = (2 * len(u) - 1) / 3
    variance_runs = (16 * len(u) - 29) / 90
    z = (runs - expected_runs) / (variance_runs ** 0.5)
    return runs, z

runs, z_val = runs_test(randomnumbers)
print(f"Number of runs: {runs}")
print(f"Z-score for runs test: {z_val:.3f}")

import numpy as np
from math import sqrt
from scipy.stats import norm  # For optional p-value

def wald_wolfowitz_runs_test(u):
    median = np.median(u)
    
    # Convert sequence to signs: 1 for above median, 0 for below
    signs = [1 if val > median else 0 for val in u if val != median]

    # Count number of runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            runs += 1

    # Count n1 and n2
    n1 = signs.count(1)
    n2 = signs.count(0)

    # Mean and variance under H0
    mean = (2 * n1 * n2) / (n1 + n2) + 1
    var = (2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) / ((n1 + n2)**2 * (n1 + n2 - 1))

    # Z-score
    z = (runs - mean) / sqrt(var) if var > 0 else 0

    # Optional p-value
    p_value = 2 * (1 - norm.cdf(abs(z))) if var > 0 else 1.0

    return {
        "runs": runs,
        "n1": n1,
        "n2": n2,
        "expected_runs": mean,
        "variance_runs": var,
        "z_score": z,
        "p_value": p_value
    }

result = wald_wolfowitz_runs_test(randomnumbers)
print(f"Wald-Wolfowitz Runs Test Result:")
for key, value in result.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")



