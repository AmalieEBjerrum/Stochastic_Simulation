import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, pearsonr, chi2
from statsmodels.sandbox.stats.runs import runstest_1samp

# --- Parameters ---
num_simulations = 100           # number of independent RNG tests (renamed for clarity)
sample_size = 10000             # size of each individual RNG sample
num_bins = 10                   # number of histogram bins for chi-squared test
bin_edges = np.linspace(0, 1, num_bins + 1)
rng = np.random.default_rng(seed=123)

# --- Storage for p-values ---
p_value_results = {
    'chi2_pvalue': [],
    'ks_pvalue': [],
    'runs_pvalue': [],
    'lag1_corr_pvalue': []
}

# --- Function to run all tests on a sample and return p-values ---
def run_tests_return_pvalues(sample):
    # Chi-squared test
    observed, _ = np.histogram(sample, bins=bin_edges)
    expected = np.full(num_bins, sample_size / num_bins)
    _, chi2_pvalue = chisquare(observed, expected) # Capture p-value
    
    # Kolmogorov-Smirnov test
    _, ks_pvalue = kstest(sample, 'uniform') # Capture p-value
    
    # Runs Test
    # The 'runstest_1samp' function returns (z_statistic, p_value)
    # Be careful: statsmodels runstest_1samp often gives a two-sided p-value by default.
    # We want to check for non-randomness, so the two-sided p-value is generally appropriate.
    _, runs_pvalue = runstest_1samp(sample > np.median(sample)) # Use boolean array for signs
    
    # Lag-1 Autocorrelation test (Pearson correlation)
    # The 'pearsonr' function returns (correlation_coefficient, p_value)
    corr_coeff, lag1_corr_pvalue = pearsonr(sample[:-1], sample[1:])
    
    return chi2_pvalue, ks_pvalue, runs_pvalue, lag1_corr_pvalue

# --- Main Simulation Loop ---
for _ in range(num_simulations):
    sample = rng.random(sample_size)
    chi2_p, ks_p, runs_p, lag1_p = run_tests_return_pvalues(sample)
    
    p_value_results['chi2_pvalue'].append(chi2_p)
    p_value_results['ks_pvalue'].append(ks_p)
    p_value_results['runs_pvalue'].append(runs_p)
    p_value_results['lag1_corr_pvalue'].append(lag1_p)

# --- Visualization for P-value Distributions ---
fig_p, axs_p = plt.subplots(2, 2, figsize=(12, 10))
axs_p = axs_p.ravel()

p_value_labels = [
    ("Chi² P-value Distribution", "chi2_pvalue"),
    ("Kolmogorov–Smirnov P-value Distribution", "ks_pvalue"),
    ("Runs Test P-value Distribution", "runs_pvalue"),
    ("Lag-1 Correlation P-value Distribution", "lag1_corr_pvalue")
]

for i, (title, key) in enumerate(p_value_labels):
    axs_p[i].hist(p_value_results[key], bins=20, range=(0, 1), color='skyblue', edgecolor='black')
    axs_p[i].axhline(num_simulations / 20, color='red', linestyle='--', label='Expected Uniform Density') # For uniform, bins * (height / total_sims) = 1
    axs_p[i].set_title(title)
    axs_p[i].set_ylabel("Frequency")
    axs_p[i].set_xlabel("P-value")
    axs_p[i].set_ylim(bottom=0) # Ensure y-axis starts at 0
    axs_p[i].legend()

plt.tight_layout()
plt.suptitle("P-value Distributions from Multiple RNG Test Runs", y=1.03)
plt.savefig('1.3_pvalue_dist.png') # New filename for p-value plot
plt.show()

# --- Optional: Print summary of p-values if needed (e.g., how many below 0.05) ---
print("\n--- P-value Analysis (Proportion below 0.05 significance) ---")
alpha = 0.05
for key, p_values in p_value_results.items():
    failures = np.sum(np.array(p_values) < alpha)
    proportion = failures / num_simulations
    print(f"{key:20s}: {failures:3d}/{num_simulations} failures ({proportion:.2f}%)")