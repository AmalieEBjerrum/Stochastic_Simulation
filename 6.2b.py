import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from collections import Counter
from math import factorial
import random

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
empirical_counts = np.array([counts_dict.get((i, j), 0) for (i, j) in all_states])

# Expected probabilities (normalized)
unnormalized = np.array([g(i, j) for (i, j) in all_states])
expected_probs = unnormalized / unnormalized.sum()
expected_counts = expected_probs * empirical_counts.sum()

# Combine low-expected bins if needed (optional)
# Run test
chi2_stat, p_value = chisquare(empirical_counts, f_exp=expected_counts)

print(f"Chi-squared statistic: {chi2_stat:.2f}, p-value: {p_value:.4f}")
