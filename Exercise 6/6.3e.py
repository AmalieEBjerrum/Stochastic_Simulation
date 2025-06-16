import numpy as np
import matplotlib.pyplot as plt

# ----- Prior from 3(a) -----
np.random.seed(42)
rho = 0.5
mean = [0, 0]
cov = [[1, rho], [rho, 1]]
xi_prior, gamma_prior = np.random.multivariate_normal(mean, cov)
theta_prior = np.exp(xi_prior)
psi_prior = np.exp(gamma_prior)
print(f"Prior sample: θ = {theta_prior:.4f}, ψ = {psi_prior:.4f}")

# ----- Log-posterior function -----
def log_posterior(xi, gamma, x_bar, s2, n, rho=0.5):
    theta = np.exp(xi)
    psi = np.exp(gamma)
    if psi <= 0 or theta <= 0:
        return -np.inf
    log_likelihood = -n / 2 * np.log(psi) - n / (2 * psi) * ((x_bar - theta) ** 2 + s2)
    log_prior = -(1 / (2 * (1 - rho ** 2))) * (xi ** 2 - 2 * rho * xi * gamma + gamma ** 2)
    return log_likelihood + log_prior

# ----- MCMC -----
def run_mcmc(x, xi0, gamma0, n_samples=10000, burn_in=1000, proposal_sd=(0.2, 0.2)):
    n = len(x)
    x_bar = np.mean(x)
    s2 = np.mean((x - x_bar) ** 2)
    xi_chain = np.zeros(n_samples)
    gamma_chain = np.zeros(n_samples)
    xi_chain[0] = xi0
    gamma_chain[0] = gamma0

    for t in range(1, n_samples):
        xi_star = np.random.normal(xi_chain[t - 1], proposal_sd[0])
        gamma_star = np.random.normal(gamma_chain[t - 1], proposal_sd[1])
        log_alpha = log_posterior(xi_star, gamma_star, x_bar, s2, n) - \
                    log_posterior(xi_chain[t - 1], gamma_chain[t - 1], x_bar, s2, n)
        if np.log(np.random.rand()) < log_alpha:
            xi_chain[t] = xi_star
            gamma_chain[t] = gamma_star
        else:
            xi_chain[t] = xi_chain[t - 1]
            gamma_chain[t] = gamma_chain[t - 1]

    return np.exp(xi_chain[burn_in:]), np.exp(gamma_chain[burn_in:])

# ----- Run for n = 100 and n = 1000 -----
np.random.seed(0)
for n_current in [100, 1000]:
    print(f"\n--- Results for n = {n_current} ---")
    X = np.random.normal(loc=theta_prior, scale=np.sqrt(psi_prior), size=n_current)
    theta_chain, psi_chain = run_mcmc(X, np.log(1.0), np.log(1.0))

    print(f"Posterior mean θ: {np.mean(theta_chain):.4f}, sd: {np.std(theta_chain):.4f}")
    print(f"Posterior mean ψ: {np.mean(psi_chain):.4f}, sd: {np.std(psi_chain):.4f}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(theta_chain, color='blue')
    plt.title(f"θ trace plot (n={n_current})")
    plt.ylabel("θ")

    plt.subplot(1, 2, 2)
    plt.plot(psi_chain, color='green')
    plt.title(f"ψ trace plot (n={n_current})")
    plt.ylabel("ψ")

    plt.tight_layout()
    plt.show()
