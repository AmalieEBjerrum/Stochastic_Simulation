import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chisquare
from collections import Counter
from math import factorial
import random
import time # Import time for measuring execution

# Parameters
A1 = A2 = 4
m = 10
num_samples = 500000 # Increased number of samples for better statistics
burn_in = 10000      # Increased burn-in period for better chain convergence

# Target distribution (unnormalized)
# This function calculates the unnormalized probability density for a given state (i, j).
# It returns 0 for invalid states (i.e., i or j are negative, or i+j exceeds m).
def g(i, j):
    if i < 0 or j < 0 or i + j > m:
        return 0
    # The term 4^i/i! * 4^j/j! represents the unnormalized probability.
    return (A1**i / factorial(i)) * (A2**j / factorial(j))

# Metropolis-Hastings sampler
def metropolis_hastings(num_samples, burn_in):
    samples = []
    # Start at a valid initial state. (0,0) is a safe and common choice.
    current_i, current_j = 0, 0

    # For monitoring acceptance rate
    accepted_moves = 0
    total_proposals = 0

    print(f"Starting Metropolis-Hastings simulation with {num_samples} samples and {burn_in} burn-in steps...")
    start_time = time.time()

    for step in range(num_samples + burn_in):
        # Define possible symmetric random walk steps: up, down, left, right.
        # This makes the proposal distribution q(Y|X) symmetric, i.e., q(Y|X) = q(X|Y).
        di, dj = random.choice([(1,0), (-1,0), (0,1), (0,-1)])
        proposed_i, proposed_j = current_i + di, current_j + dj

        # Calculate the unnormalized probabilities for the current and proposed states.
        # The g() function already handles boundary conditions by returning 0 for invalid states.
        p_current = g(current_i, current_j)
        p_proposed = g(proposed_i, proposed_j)

        # Calculate the acceptance ratio.
        # If p_current is 0 (which ideally shouldn't happen if starting from a valid state
        # and g handles invalid proposals as 0), we avoid division by zero.
        if p_current == 0:
            # If current state has zero probability, it's an invalid state (shouldn't be reached
            # unless initialization is wrong or proposal leads to an impossible state).
            # In such a rare case, force rejection or accept to move out.
            # For this problem, (0,0) has non-zero probability, so this branch is mostly a safeguard.
            alpha = 0 # Effectively reject if somehow current state has zero probability
        else:
            alpha = min(1, p_proposed / p_current)

        # Decide whether to accept the proposed state.
        if np.random.rand() < alpha:
            current_i, current_j = proposed_i, proposed_j
            accepted_moves += 1 # Increment accepted moves only when a move is truly accepted

        total_proposals += 1 # Increment total proposals regardless of acceptance

        # After burn-in, start collecting samples.
        if step >= burn_in:
            samples.append((current_i, current_j))

        # Periodically print progress for long simulations
        if (step + 1) % (num_samples + burn_in) // 10 == 0: # Print at 10%, 20%, ...
            print(f"  {((step + 1) / (num_samples + burn_in) * 100):.1f}% complete...")

    end_time = time.time()
    print(f"Metropolis-Hastings simulation finished in {end_time - start_time:.2f} seconds.")
    print(f"Acceptance Rate: {(accepted_moves / total_proposals * 100):.2f}%")

    return samples

# Run the simulation
samples = metropolis_hastings(num_samples=num_samples, burn_in=burn_in)

# --- Output and Visualization ---

print("\nFirst 10 samples from the chain:")
for s in samples[:10]:
    print(s)

# Plotting how often each (i,j) pair appears (Empirical Distribution)
# This helps visually assess if the distribution is converging to what's expected.
counts_matrix = np.zeros((m+1, m+1))
for i, j in samples:
    if 0 <= i <= m and 0 <= j <= m and i + j <= m: # Ensure valid indices for counts_matrix
        counts_matrix[i, j] += 1

# Mask out illegal (i, j) pairs for plotting where i + j > m, displaying them as NaN.
for i in range(m+1):
    for j in range(m+1):
        if i + j > m:
            counts_matrix[i, j] = np.nan

plt.figure(figsize=(10, 8))
plt.imshow(counts_matrix, origin='lower', cmap='viridis', extent=[-0.5, m+0.5, -0.5, m+0.5])
plt.colorbar(label='Frequency')
plt.xlabel('Call type 1 (i)')
plt.ylabel('Call type 2 (j)')
plt.title('Empirical Distribution from Metropolis-Hastings Samples')
plt.xticks(np.arange(0, m + 1, 1))
plt.yticks(np.arange(0, m + 1, 1))
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('empirical_distribution_6.2a.png')
plt.show()

# --- Chi-squared Goodness-of-Fit Test ---

print("\n--- Chi-squared Goodness-of-Fit Test ---")

# 1. Get all possible valid states (i, j) where i + j <= m.
# This ensures we cover all states defined by the problem.
all_states = [(i, j) for i in range(m + 1) for j in range(m + 1) if i + j <= m]

# 2. Calculate empirical counts for each valid state from the collected samples.
empirical_counts_dict = Counter(samples)
empirical_counts = np.array([empirical_counts_dict.get(state, 0) for state in all_states])

# 3. Calculate expected probabilities (unnormalized) for each valid state using the g function.
unnormalized_expected_probs = np.array([g(i, j) for (i, j) in all_states])

# 4. Normalize the expected probabilities to sum to 1.
# This gives us the true theoretical probabilities for each state.
sum_unnormalized_probs = unnormalized_expected_probs.sum()
if sum_unnormalized_probs == 0:
    raise ValueError("Sum of unnormalized probabilities is zero. Check g(i,j) function or parameters.")
expected_probs = unnormalized_expected_probs / sum_unnormalized_probs

# 5. Calculate expected counts based on the total number of samples collected.
total_samples_collected = empirical_counts.sum()
expected_counts = expected_probs * total_samples_collected

# 6. Filter for valid bins (expected count >= 5).
# This is crucial for the validity of the chi-squared test. If many bins have
# expected counts less than 5, the test's assumptions are violated.
valid_bins_mask = expected_counts >= 5

f_obs_filtered = empirical_counts[valid_bins_mask]
f_exp_filtered = expected_counts[valid_bins_mask]

# Perform chi-squared test only if there are enough valid bins.
# At least two bins are typically required for a meaningful chi-squared test.
if len(f_obs_filtered) > 1 and len(f_obs_filtered) == len(f_exp_filtered):
    chi2_stat, p_value = chisquare(
        f_obs=f_obs_filtered,
        f_exp=f_exp_filtered,
        ddof=0 # Degrees of freedom adjustment if parameters were estimated from data. Here, we know the parameters.
    )
    print(f"Chi-squared statistic: {chi2_stat:.2f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Total possible states (bins): {len(all_states)}")
    print(f"Valid bins (expected count >= 5) included in test: {valid_bins_mask.sum()}")
    print(f"Minimum expected count in valid bins: {f_exp_filtered.min():.4f}")
    print(f"Maximum expected count in valid bins: {f_exp_filtered.max():.4f}")

    # Interpret the p-value
    alpha_level = 0.05 # Common significance level
    if p_value < alpha_level:
        print(f"Interpretation: With a p-value ({p_value:.4f}) less than {alpha_level}, we reject the null hypothesis.")
        print("This suggests that the empirical distribution from the M-H sampler is significantly different from the theoretical target distribution.")
        print("Possible reasons: insufficient burn-in, insufficient samples, poor mixing of the chain.")
    else:
        print(f"Interpretation: With a p-value ({p_value:.4f}) greater than or equal to {alpha_level}, we fail to reject the null hypothesis.")
        print("This suggests that the empirical distribution is a good fit for the theoretical target distribution.")

else:
    print("\nNot enough valid bins (expected count >= 5) to perform a reliable chi-squared test.")
    print(f"Total possible states: {len(all_states)}")
    print(f"Valid bins (expected count >= 5) for chi-squared test: {valid_bins_mask.sum()}")
    if valid_bins_mask.sum() > 0:
        print(f"Minimum expected count across valid bins: {f_exp_filtered.min():.4f}")
    else:
        print("No bins met the expected count >= 5 criterion.")