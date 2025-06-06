import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare, kstest, pearsonr
from statsmodels.sandbox.stats.runs import runstest_1samp
    
#Defining the parameter values
n = 10000
x0 = 1
M = 2**32
a = 64525
c = 451390

#Generate random numbers by calling the function
randomnumbers = LCG(x0, M, a, c, n)

#Settings
num_samples = 100
sample_size = 10000
num_bins = 10
bin_edges = np.linspace(0, 1, num_bins + 1)

#Generate numbers
sys_rng = np.random.rand(n)

#Storage
tests = ['chi2', 'ks', 'runs', 'corr']
results_lcg = {t: [] for t in tests}
results_sys = {t: [] for t in tests}

#Test Runner
def run_tests(sample, results):
    observed, _ = np.histogram(sample, bins=bin_edges)
    expected = np.full(num_bins, sample_size / num_bins)
    chi2, _ = chisquare(observed, expected)
    results['chi2'].append(chi2)

    ks, _ = kstest(sample, 'uniform')
    results['ks'].append(ks)

    signs = np.where(sample > np.median(sample), 1, 0)
    _, z = runstest_1samp(signs)
    results['runs'].append(z)

    corr, _ = pearsonr(sample[:-1], sample[1:])
    results['corr'].append(corr)

#Main Loop
for _ in range(num_samples):
    lcg_sample = LCG(x0, M, a, c, sample_size)
    sys_sample = sys_rng.random(sample_size)

    run_tests(lcg_sample, results_lcg)
    run_tests(sys_sample, results_sys)

#Plotting
fig, axs = plt.subplots(4, 2, figsize=(14, 14))
titles = ['Chi²', 'KS', 'Runs Z', 'Corr h=1']

for i, t in enumerate(tests):
    axs[i, 0].hist(results_lcg[t], bins=20, color='cornflowerblue', edgecolor='black')
    axs[i, 0].set_title(f'{titles[i]} - LCG')
    axs[i, 1].hist(results_sys[t], bins=20, color='seagreen', edgecolor='black')
    axs[i, 1].set_title(f'{titles[i]} - System RNG')

for ax in axs.flat:
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
