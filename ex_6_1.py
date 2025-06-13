import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare

def truncated_poisson_pmf(i, A):
    return A**i / np.math.factorial(i)

def metropolis_hastings_truncated_poisson(A, m, n_samples):
    samples = []
    i = np.random.randint(0, m + 1)  # initial state
    for _ in range(n_samples):
        # Propose i ± 1
        proposal = i + np.random.choice([-1, 1])
        if proposal < 0 or proposal > m:
            samples.append(i)
            continue

        # Compute acceptance probability
        pi_current = truncated_poisson_pmf(i, A)
        pi_proposal = truncated_poisson_pmf(proposal, A)
        alpha = min(1, pi_proposal / pi_current)

        if np.random.rand() < alpha:
            i = proposal
        samples.append(i)

    return np.array(samples)

# Parameters (e.g. from exercise 4)
A = 8
m = 10
n_samples = 10000
samples = metropolis_hastings_truncated_poisson(A, m, n_samples)
burn_in = 1000
samples = samples[burn_in:]  # Discard burn-in samples

# Normalize the expected distribution
unnorm_probs = np.array([A**i / np.math.factorial(i) for i in range(m+1)])
expected_probs = unnorm_probs / unnorm_probs.sum()
expected_counts = len(samples) * expected_probs
print(expected_counts)

observed_counts = np.bincount(samples, minlength=m+1)

# Chi-squared test
chi2_stat, p_value = chisquare(observed_counts, expected_counts)

print(f"Chi-squared statistic: {chi2_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Visualization
plt.bar(range(m+1), observed_counts, alpha=0.6, label='Observed')
plt.plot(range(m+1), expected_counts, 'ro-', label='Expected')
plt.xlabel('Number of busy lines')
plt.ylabel('Counts')
plt.legend()
plt.title('Metropolis-Hastings sampling of truncated Poisson')
plt.show()


import seaborn as sns
import statsmodels.api as sm
sns.lineplot(x=range(500), y=samples[:500])
sm.graphics.tsa.plot_acf(samples, lags=40)
