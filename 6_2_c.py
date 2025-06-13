import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from collections import defaultdict

# Parameters
A1, A2, m = 4, 4, 10
n_samples = 100000

# Helper: normalized truncated Poisson-like pmf
def sample_conditional_poisson_trunc(A, upper):
    support = np.arange(upper + 1)
    weights = A**support / [np.math.factorial(k) for k in support]
    probs = weights / np.sum(weights)
    return np.random.choice(support, p=probs)

# Gibbs Sampling
samples = []
i, j = 0, 0  # initial state

for _ in range(n_samples):
    i = sample_conditional_poisson_trunc(A1, m - j)
    j = sample_conditional_poisson_trunc(A2, m - i)
    samples.append((i, j))

samples =samples[1000:]  # Burn-in period
# Count occurrences
counts = defaultdict(int)
for pair in samples:
    counts[pair] += 1

# All valid states
valid_states = [(i, j) for i in range(m + 1) for j in range(m + 1) if i + j <= m]

# Observed and expected frequencies
observed = np.array([counts[(i, j)] for (i, j) in valid_states])
unnormalized_probs = np.array([(A1**i / np.math.factorial(i)) * (A2**j / np.math.factorial(j)) for (i, j) in valid_states])
expected = len(samples)* (unnormalized_probs / unnormalized_probs.sum())


# Chi-squared test
chi2_stat, p_value = chisquare(observed, expected)
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Flatten (i, j) pairs to labels for x-axis
labels = [f"({i},{j})" for (i, j) in valid_states]
x = np.arange(len(labels))

# Plotting
plt.figure(figsize=(18, 6))
bar_width = 0.4

# Observed bars
plt.bar(x - bar_width/2, observed, width=bar_width, label='Observed', alpha=0.7, color='blue')

# Expected bars
plt.bar(x + bar_width/2, expected, width=bar_width, label='Expected', alpha=0.7, color='red')

plt.xticks(x, labels, rotation=90)
plt.xlabel("(i, j) Pair")
plt.ylabel("Count")
plt.title("Observed vs Expected Counts for (Type 1, Type 2) Calls")
plt.legend()
plt.tight_layout()
plt.show()
