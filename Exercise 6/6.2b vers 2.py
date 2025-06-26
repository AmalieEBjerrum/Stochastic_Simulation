import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from collections import Counter
from math import factorial
import random

np.random.seed(42)

# Parameters
A1 = A2 = 4
m = 10
num_samples = 100000
burn_in = 1000

# Target distribution (unnormalized)
def g(i, j):
    if i < 0 or j < 0 or i + j > m:
        return 0
    return (A1**i / factorial(i)) * (A2**j / factorial(j))

# Coordinate-wise Metropolis-Hastings sampler
def metropolis_hastings_coordinatewise(num_samples, burn_in):
    samples = []
    i, j = 0, 0  # start state

    for step in range(num_samples + burn_in):
        # Pick coordinate: 0 = update i, 1 = update j
        coord = random.choice([0, 1])

        # Propose change
        if coord == 0:  # update i
            di = random.choice([-1, 1])
            i_new, j_new = i + di, j
        else:           # update j
            dj = random.choice([-1, 1])
            i_new, j_new = i, j + dj

        # Check bounds
        if i_new < 0 or j_new < 0 or i_new + j_new > m:
            continue

        # Metropolis acceptance
        p_old = g(i, j)
        p_new = g(i_new, j_new)
        alpha = min(1, p_new / p_old)

        if np.random.rand() < alpha:
            i, j = i_new, j_new  # accept

        if step >= burn_in:
            samples.append((i, j))

    return samples

# Generate samples
samples = metropolis_hastings_coordinatewise(num_samples=num_samples, burn_in=burn_in)

# Show first few samples
print("First 10 samples from the coordinate-wise MH chain:")
for s in samples[:10]:
    print(s)

# Frequency counts for plotting
counts = np.zeros((m+1, m+1))
for i, j in samples:
    counts[i, j] += 1

# Mask invalid (i+j > m)
for i in range(m+1):
    for j in range(m+1):
        if i + j > m:
            counts[i, j] = np.nan

# Plot
plt.figure(figsize=(8, 6))
plt.imshow(counts, origin='lower', cmap='viridis')
plt.colorbar(label='Frequency')
plt.xlabel('Call type 1 (i)')
plt.ylabel('Call type 2 (j)')
plt.title('Empirical distribution from coordinate-wise Metropolis-Hastings')
plt.savefig('6.2b.png')
plt.show()


# Chi-squared test
# Get frequencies for legal (i,j) states
counts_dict = Counter(samples)
all_states = [(i, j) for i in range(m+1) for j in range(m+1 - i)]
empirical_counts_raw = np.array([counts_dict.get((i, j), 0) for (i, j) in all_states])

# Expected probabilities (normalized)
unnormalized = np.array([g(i, j) for (i, j) in all_states])
expected_probs = unnormalized / unnormalized.sum()
expected_counts_raw = expected_probs * empirical_counts_raw.sum()

# Combine bins with low expected counts
min_expected_count = 5 # Standard threshold for chi-squared test
combined_empirical = []
combined_expected = []
dof_reduction = 0 # Degrees of freedom reduction due to combining bins

current_empirical_sum = 0
current_expected_sum = 0

for i in range(len(all_states)):
    if expected_counts_raw[i] < min_expected_count:
        current_empirical_sum += empirical_counts_raw[i]
        current_expected_sum += expected_counts_raw[i]
        dof_reduction += 1 # Each combined bin effectively reduces a degree of freedom
    else:
        # If there are accumulated low-expected counts, add them before adding the current one
        if current_expected_sum > 0:
            combined_empirical.append(current_empirical_sum)
            combined_expected.append(current_expected_sum)
            current_empirical_sum = 0
            current_expected_sum = 0

        combined_empirical.append(empirical_counts_raw[i])
        combined_expected.append(expected_counts_raw[i])

# Add any remaining accumulated low-expected counts
if current_expected_sum > 0:
    combined_empirical.append(current_empirical_sum)
    combined_expected.append(current_expected_sum)

empirical_counts = np.array(combined_empirical)
expected_counts = np.array(combined_expected)
degrees_of_freedom = len(empirical_counts) - 1

# Run test
chi2_stat, p_value = chisquare(empirical_counts, f_exp=expected_counts)

print(f"Chi-squared statistic: {chi2_stat:.2f}, p-value: {p_value:.4f}")
