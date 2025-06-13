import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from collections import defaultdict

# Parameters
A1, A2, m = 4, 4, 10
n_samples = 100000

# Unnormalized joint PMF
def joint_pmf(i, j):
    if i < 0 or j < 0 or i + j > m:
        return 0
    return (A1 ** i / np.math.factorial(i)) * (A2 ** j / np.math.factorial(j))

# Proposal: random neighbor (±1 to i or j)
def propose(i, j):
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    np.random.shuffle(moves)
    for di, dj in moves:
        ni, nj = i + di, j + dj
        if 0 <= ni <= m and 0 <= nj <= m and ni + nj <= m:
            return ni, nj
    return i, j  # stay in place if no valid move

n_samples = 100000
burn_in = 5000  # for example, discard first 5000 samples

# Metropolis-Hastings sampling
samples = []
i, j = 0, 0  # initial state

for _ in range(n_samples):
    ni, nj = propose(i, j)
    alpha = min(1, joint_pmf(ni, nj) / joint_pmf(i, j))
    if np.random.rand() < alpha:
        i, j = ni, nj
    samples.append((i, j))

# Discard burn-in samples
samples = samples[burn_in:]

# Count occurrences
counts = defaultdict(int)
for pair in samples:
    counts[pair] += 1

# All valid states
valid_states = [(i, j) for i in range(m + 1) for j in range(m + 1) if i + j <= m]

# Observed and expected frequencies
observed = np.array([counts[(i, j)] for (i, j) in valid_states])
unnormalized_probs = np.array([joint_pmf(i, j) for (i, j) in valid_states])
expected = len(samples) * (unnormalized_probs / unnormalized_probs.sum())
print(min(expected))

# Chi-squared test
chi2_stat, p_value = chisquare(observed, expected)
print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Optional: Plot
from mpl_toolkits.mplot3d import Axes3D

x = [i for (i, j) in valid_states]
y = [j for (i, j) in valid_states]
z = observed

# Extract expected values as well
expected_z = expected

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

# Offsets to show side-by-side bars
bar_width = 0.4
x = np.array([i for (i, j) in valid_states])
y = np.array([j for (i, j) in valid_states])
z_observed = np.zeros_like(x)
z_expected = np.zeros_like(x)

# Plot observed counts
ax.bar3d(x - bar_width/2, y, z_observed, bar_width, bar_width, observed, shade=True, color='blue', alpha=0.6, label='Observed')

# Plot expected counts
ax.bar3d(x + bar_width/2, y, z_expected, bar_width, bar_width, expected_z, shade=True, color='red', alpha=0.6, label='Expected')

# Axis labels and title
ax.set_xlabel('i (Type 1 calls)')
ax.set_ylabel('j (Type 2 calls)')
ax.set_zlabel('Count')
plt.title('Observed vs Expected Counts (Metropolis-Hastings)')

# Manual legend workaround
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='blue', edgecolor='k', label='Observed'),
    Patch(facecolor='red', edgecolor='k', label='Expected')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.show()
