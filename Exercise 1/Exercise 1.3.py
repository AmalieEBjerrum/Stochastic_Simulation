import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, pearsonr, chi2
from statsmodels.sandbox.stats.runs import runstest_1samp

# --- Parameters ---
num_samples = 100           # number of independent RNG tests
sample_size = 10000         # size of each individual RNG sample
num_bins = 10               # number of histogram bins
bin_edges = np.linspace(0, 1, num_bins + 1)
rng = np.random.default_rng(seed=123)

# --- Storage for test results ---
results = {
    'chi2': [],
    'ks': [],
    'runs_z': [],
    'lag1_corr': []
}

# --- Critical values (95% level) ---
chi2_crit = chi2.ppf(0.95, df=num_bins - 1)
ks_crit = 1.36 / np.sqrt(sample_size)  # Approximation for KS test
runs_crit = 1.96                       # 95% two-sided z-threshold
corr_crit = 1.96 / np.sqrt(sample_size)  # Approx. 95% CI for correlation under H0

# --- Fail counters ---
fails = {
    'chi2': 0,
    'ks': 0,
    'runs_z': 0,
    'lag1_corr': 0
}

# --- Function to run all tests on a sample ---
def run_tests(sample):
    observed, _ = np.histogram(sample, bins=bin_edges)
    expected = np.full(num_bins, sample_size / num_bins)
    chi2_stat, _ = chisquare(observed, expected)
    
    ks_stat, _ = kstest(sample, 'uniform')
    
    signs = np.where(sample > np.median(sample), 1, 0)
    _, runs_z = runstest_1samp(signs)
    
    corr, _ = pearsonr(sample[:-1], sample[1:])
    
    return chi2_stat, ks_stat, runs_z, corr

# --- Main Simulation Loop ---
for _ in range(num_samples):
    sample = rng.random(sample_size)
    chi2_stat, ks_stat, runs_z, corr = run_tests(sample)
    
    results['chi2'].append(chi2_stat)
    results['ks'].append(ks_stat)
    results['runs_z'].append(runs_z)
    results['lag1_corr'].append(corr)
    
    if chi2_stat > chi2_crit:
        fails['chi2'] += 1
    if ks_stat > ks_crit:
        fails['ks'] += 1
    if abs(runs_z) > runs_crit:
        fails['runs_z'] += 1
    if abs(corr) > corr_crit:
        fails['lag1_corr'] += 1

# --- Visualization ---
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.ravel()

test_labels = [
    ("Chi² Statistic", "chi2", chi2_crit),
    ("Kolmogorov–Smirnov Statistic", "ks", ks_crit),
    ("Runs Test Z-Statistic", "runs_z", runs_crit),
    ("Lag-1 Correlation", "lag1_corr", corr_crit)
]

for i, (title, key, crit) in enumerate(test_labels):
    axs[i].hist(results[key], bins=20, color='steelblue', edgecolor='black')
    axs[i].axvline(crit, color='red', linestyle='--', label='95% Threshold')
    if key in ['runs_z', 'lag1_corr']:
        axs[i].axvline(-crit, color='red', linestyle='--')
    axs[i].set_title(title)
    axs[i].set_ylabel("Frequency")
    axs[i].set_xlabel("Test Statistic")
    axs[i].legend()

plt.tight_layout()
plt.savefig('1.3.png')
plt.show()

# --- Print Fail Summary ---
print("\nTest Failures (out of", num_samples, "samples):")
for test, count in fails.items():
    print(f"{test:10s}: {count:3d} failures")
