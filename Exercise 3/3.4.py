import numpy as np
import matplotlib.pyplot as plt


#Pareto using Composition
def pareto_by_composition(n, k, beta=1.0):
    X = np.random.exponential(scale=1.0, size=n)  # X ~ Exp(1)
    return beta * np.exp(X / k)

#To compare with the analytical solution
def analytical_pdf(x, k, beta):
    return (k * beta**k) / (x ** (k + 1)) * (x >= beta)

def analytical_mean(k, beta):
    return (k * beta) / (k - 1) if k > 1 else np.inf

def analytical_variance(k, beta):
    if k > 2:
        return (k * beta**2) / ((k - 1)**2 * (k - 2))
    else:
        return np.inf

# Parameters
n = 10000
beta = 1.0
k_values = [2.05, 2.5, 3, 4]

plt.figure(figsize=(12, 8))

x_vals = np.linspace(1, 10, 1000)

for i, k in enumerate(k_values, 1):
    samples = pareto_by_composition(n, k, beta)
    sample_mean = np.mean(samples)
    sample_var = np.var(samples, ddof=1)
    theo_mean = analytical_mean(k, beta)
    theo_var = analytical_variance(k, beta)

    # Plot histogram
    plt.subplot(2, 2, i)
    plt.hist(samples, bins=100, density=True, alpha=0.6, range=(1, 10), label='Simulated')
    
    # Plot analytical PDF
    pdf_vals = analytical_pdf(x_vals, k, beta)
    plt.plot(x_vals, pdf_vals, 'r--', label='Analytical PDF')

    plt.title(f"k = {k}")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.grid(True)
    plt.legend()

    # Print stats
    print(f"\nk = {k}")
    print(f"Empirical Mean     = {sample_mean:.4f}")
    print(f"Analytical Mean    = {theo_mean:.4f}")
    print(f"Empirical Variance = {sample_var:.4f}")
    print(f"Analytical Variance= {theo_var:.4f}")

plt.tight_layout()
plt.suptitle("Pareto Distribution — Simulated vs Analytical PDF", y=1.03)
plt.savefig('3.4 Composition.png')
plt.show()