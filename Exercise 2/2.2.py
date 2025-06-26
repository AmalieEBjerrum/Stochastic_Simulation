import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import stats

np.random.seed(42)

#Defining 6 Point Distribution
X = [1,2,3,4,5,6]
Probabilities = [7/48, 5/48, 1/8, 1/16, 1/4, 5/16]

#a) Direct Crude Method

#Computing the CDF
cdf = np.cumsum(Probabilities)

def crude_method(size):
    samples = []
    for _ in range(size):
        u = np.random.uniform()
        idx = np.searchsorted(cdf, u)
        samples.append(X[idx]) # Append the actual X value, not a list
    return np.array(samples)
        
#Simulating
#Simulating
size = 10000
samples = crude_method(size)

 # Plotting
plt.hist(samples, bins=np.arange(0.5, 6.6, 1), density=True, edgecolor='black', alpha=0.7)
plt.xticks(X)
plt.title('Direct Crude Sampling from 6-Point Distribution')
plt.xlabel('X')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('2.2a Direct Crude.png')
plt.show()     

#Chi-squared test
#Counts for values 1 through 6
observed_counts = np.bincount(samples.astype(int))[1:7]
# Calculate expected frequencies
expected_counts = np.array(Probabilities) * size

chi2_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"\n--- Chi-squared Goodness-of-Fit Test ---")
print(f"Chi-squared Statistic: {chi2_statistic:.4f}")
print(f"P-value:               {p_value:.4f}")
if p_value > 0.05:
    print("The samples are consistent with the theoretical distribution (fail to reject H0).")
else:
    print("The samples are NOT consistent with the theoretical distribution (reject H0).")



#b) Rejection Method

#Definitions
q = np.array([1/6]*6) #Proposed uniform distribution from X
C = np.max(Probabilities / q)  #Rejection constant 

def rejection_method(size):
    samples = []
    trials = 0
    while len(samples) < size:
        # Sample from proposal (uniform over indices 0–5)
        i = np.random.randint(0, 6)
        u = np.random.uniform()
        trials +=1
        accept = Probabilities[i] / (C * q[i])  
        if u <= accept:
            samples.append(X[i])
    return np.array(samples), trials

# Simulation
size = 10000
samples, num_trials = rejection_method(size)

# Plot
plt.hist(samples, bins=np.arange(0.5, 6.6, 1), density=True, edgecolor='black', alpha=0.7)
plt.xticks(X)
plt.title('Rejection Sampling from 6-Point Distribution')
plt.xlabel('X')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('2.2b Rejection Method.png')
plt.show()


#Chi-squared test
#Counts for values 1 through 6
observed_counts = np.bincount(samples.astype(int))[1:7]
# Calculate expected frequencies
expected_counts = np.array(Probabilities) * size

# Perform Chi-squared test
chi2_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"\n--- Chi-squared Goodness-of-Fit Test (Rejection Method) ---")
print(f"Number of samples generated: {len(samples)}")
print(f"Total trials to get {len(samples)} samples: {num_trials}")
print(f"Efficiency (samples / trials): {len(samples)/num_trials:.4f}")
print(f"Chi-squared Statistic: {chi2_statistic:.4f}")
print(f"P-value:               {p_value:.4f}")
if p_value > 0.05:
    print("The samples are consistent with the theoretical distribution (fail to reject H0).")
else:
    print("The samples are NOT consistent with the theoretical distribution (reject H0).")





#c) Alias method

#Setup function that returns the two tables
def alias_setup(probabilities):
    n = len(probabilities)
    scaled_probs = np.array(probabilities) * n  # Scale to sum to n
    F = np.zeros(n)
    L = np.zeros(n, dtype=int)
    
    # Split into small and large stacks
    small = []
    large = []
    
    for i, sp in enumerate(scaled_probs):
        if sp < 1:
            small.append(i)
        else:
            large.append(i)
    
    # Main loop
    while small and large:
        s = small.pop()
        l = large.pop()
        
        F[s] = scaled_probs[s]
        L[s] = l
        
        scaled_probs[l] -= (1- scaled_probs[s])
        if scaled_probs[l] < 1:
            small.append(l)
        else:
            large.append(l)

    # For any leftover values
    for i in small + large:
        F[i] = 1

    return F, L

#Function for carrying out the steps using samples from the tables
def alias_method(F, L, X,size):
    n = len(F)
    idx = np.random.randint(0, n, size)
    coin = np.random.uniform(0, 1, size)
    result = np.where(coin < F[idx], idx, L[idx])
    return np.array(X)[result]

# Preprocess
F, L = alias_setup(Probabilities)

# Simulation
size=10000
samples = alias_method(F, L, X,size)

# Plot
plt.hist(samples, bins=np.arange(0.5, 6.6, 1), density=True, edgecolor='black', alpha=0.7)
plt.xticks(X)
plt.title('Alias Method Sampling from 6-Point Distribution')
plt.xlabel('X')
plt.ylabel('Probability')
plt.grid(True)
plt.savefig('2.2c Alias Method.png')
plt.show()

#Chi-squared test
#Counts for values 1 through 6
observed_counts = np.bincount(samples.astype(int))[1:7]
# Calculate expected frequencies
expected_counts = np.array(Probabilities) * size

# Perform Chi-squared test
chi2_statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

print(f"\n--- Chi-squared Goodness-of-Fit Test (Alias Method) ---")
print(f"Number of samples generated: {len(samples)}")
print(f"Total trials to get {len(samples)} samples: {num_trials}")
print(f"Efficiency (samples / trials): {len(samples)/num_trials:.4f}")
print(f"Chi-squared Statistic: {chi2_statistic:.4f}")
print(f"P-value:               {p_value:.4f}")
if p_value > 0.05:
    print("The samples are consistent with the theoretical distribution (fail to reject H0).")
else:
    print("The samples are NOT consistent with the theoretical distribution (reject H0).")




#3. Measurements

#---Timing---
results = {}
timings = {}

# Crude Method
start = time.perf_counter()
samples_crude = crude_method()
timings['crude'] = time.perf_counter() - start
results['crude'] = samples_crude

# Rejection Method
start = time.perf_counter()
samples_reject, trials_reject = rejection_method()
timings['rejection'] = time.perf_counter() - start
results['rejection'] = samples_reject
acceptance_rate = 10000/ trials_reject

# Alias Method
start_setup = time.perf_counter()
F_table, L_table = alias_setup(Probabilities)
preprocessing_time = time.perf_counter()-start_setup

start = time.perf_counter()
samples_alias = alias_method(F_table, L_table,X)
timings['alias'] = time.perf_counter() - start
results['alias'] = samples_alias

#---Analyse and Compare---
def empirical_error(samples, name):
    counts = np.array([(samples == xi).sum() for xi in X]) / len(samples)
    mae = np.abs(counts - Probabilities).mean()
    print(f"{name.capitalize()} method:")
    print(f"  Empirical Probabilities: {counts.round(4)}")
    print(f"  Mean Absolute Error (MAE): {mae:.5f}")
    print(f"  Time: {timings[name]:.4f} seconds\n")
    if name == 'rejection':
        print(f"  Acceptance rate: {acceptance_rate:.4f}")
    if name == 'alias':
        print(f"  Alias setup time: {preprocessing_time:.4f} seconds")
    print()
    return counts

# Run analysis
plt.figure(figsize=(12, 4))
for i, method in enumerate(['crude', 'rejection', 'alias']):
    plt.subplot(1, 3, i+1)
    empirical_error(results[method], method)
    plt.hist(results[method], bins=np.arange(0.5, 7.5, 1), density=True,
             alpha=0.7, edgecolor='black')
    plt.title(f"{method.capitalize()} Method")
    plt.xticks(X)
    plt.xlabel("X")
    plt.ylabel("Probability")
plt.tight_layout()
plt.savefig('2.3 Comparison.png')
plt.show()