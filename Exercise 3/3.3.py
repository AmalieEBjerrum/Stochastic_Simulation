import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from scipy.stats import chi2

def normal_sample(n):
    U1 = np.random.uniform(0, 1, size=n//2)
    U2 = np.random.uniform(0, 1, size=n//2)
    R = np.sqrt(-2 * np.log(U1))
    theta = 2 * np.pi * U2
    Z1 = R * np.cos(theta)
    Z2 = R * np.sin(theta)
    return np.concatenate([Z1, Z2])


#Confidence Intervals for Mean
# Parameters
num_intervals = 100
sample_size = 10
alpha = 0.05
df = sample_size - 1
t_crit = t.ppf(1 - alpha / 2, df)

intervals = []
contains_true_mean = []

for _ in range(num_intervals):
    sample = normal_sample(sample_size)
    sample_mean = np.mean(sample)
    sample_std = np.std(sample, ddof=1)

    margin = t_crit * sample_std / np.sqrt(sample_size)
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin

    intervals.append((ci_lower, ci_upper))
    contains_true_mean.append(ci_lower <= 0 <= ci_upper)

# Plot
plt.figure(figsize=(10, 6))
for i, ((low, high), covers) in enumerate(zip(intervals, contains_true_mean)):
    color = 'green' if covers else 'red'
    plt.plot([low, high], [i, i], color=color, linewidth=2)
    plt.plot(0, i, 'k|')  # true mean

plt.axvline(0, color='black', linestyle='--', label='True mean = 0')
plt.xlabel("Confidence Interval")
plt.ylabel("Sample index")
plt.title("95% Confidence Intervals for the Mean (n = 10)")
plt.legend()
plt.tight_layout()
plt.savefig('3.3 Mean CI.png')
plt.show()

#Confidence Intervals for Variance
# Parameters
chi2_low = chi2.ppf(0.025, df)
chi2_high = chi2.ppf(0.975, df)

variance_intervals = []
contains_true_var = []
true_variance = 1  # For standard normal

for _ in range(num_intervals):
    sample = normal_sample(sample_size)
    sample_var = np.var(sample, ddof=1)

    lower = (df * sample_var) / chi2_high
    upper = (df * sample_var) / chi2_low

    variance_intervals.append((lower, upper))
    contains_true_var.append(lower <= true_variance <= upper)

# Plot
plt.figure(figsize=(10, 6))
for i, ((low, high), covers) in enumerate(zip(variance_intervals, contains_true_var)):
    color = 'green' if covers else 'red'
    plt.plot([low, high], [i, i], color=color, linewidth=2)
    plt.plot(true_variance, i, 'k|')  # true variance

plt.axvline(true_variance, color='black', linestyle='--', label='True variance = 1')
plt.xlabel("Confidence Interval for Variance")
plt.ylabel("Sample index")
plt.title("95% Confidence Intervals for the Variance (n = 10)")
plt.legend()
plt.tight_layout()
plt.savefig('3.3 Variance CI.png')
plt.show()

# Count how many intervals contain the true value
mean_coverage = sum(contains_true_mean)
var_coverage = sum(contains_true_var)

print(f"Mean CI coverage: {mean_coverage}/100 ({mean_coverage}%)")
print(f"Variance CI coverage: {var_coverage}/100 ({var_coverage}%)")
