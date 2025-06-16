import numpy as np
import matplotlib.pyplot as plt

#a Prior sampling
np.random.seed(42)
rho = 0.5
mean = [0, 0]
cov = [[1, rho], [rho, 1]]

xi, gamma = np.random.multivariate_normal(mean, cov)
theta = np.exp(xi)
psi = np.exp(gamma)

print(f"Sampled prior (theta, psi) = ({theta:.4f}, {psi:.4f})")

#b Generate observations
n = 10
np.random.seed(0)
theta_true = 1.25
psi_true = 0.8
X = np.random.normal(loc=theta_true, scale=np.sqrt(psi_true), size=n)
print("Generated observations X_i:")
print(X)

# Sample statistics
x_bar = np.mean(X)
s2 = np.mean((X - x_bar) ** 2)

#d Log-posterior function
def log_posterior(xi, gamma, rho=0.5):
    theta = np.exp(xi)
    psi = np.exp(gamma)
    if psi <= 0 or theta <= 0:
        return -np.inf
    log_likelihood = -n / 2 * np.log(psi) - n / (2 * psi) * ((x_bar - theta) ** 2 + s2)
    log_prior = -(1 / (2 * (1 - rho ** 2))) * (xi ** 2 - 2 * rho * xi * gamma + gamma ** 2)
    return log_likelihood + log_prior

#MCMC function
def run_mcmc(xi0, gamma0, n_samples=10000, proposal_sd=(0.2, 0.2), burn_in=1000):
    xi_chain = np.zeros(n_samples)
    gamma_chain = np.zeros(n_samples)
    xi_chain[0] = xi0
    gamma_chain[0] = gamma0

    for t in range(1, n_samples):
        xi_star = np.random.normal(xi_chain[t - 1], proposal_sd[0])
        gamma_star = np.random.normal(gamma_chain[t - 1], proposal_sd[1])

        log_alpha = log_posterior(xi_star, gamma_star) - log_posterior(xi_chain[t - 1], gamma_chain[t - 1])
        if np.log(np.random.rand()) < log_alpha:
            xi_chain[t] = xi_star
            gamma_chain[t] = gamma_star
        else:
            xi_chain[t] = xi_chain[t - 1]
            gamma_chain[t] = gamma_chain[t - 1]

    theta_chain = np.exp(xi_chain[burn_in:])
    psi_chain = np.exp(gamma_chain[burn_in:])
    return theta_chain, psi_chain

#Run two chains with different initial values
theta1, psi1 = run_mcmc(np.log(1.0), np.log(1.0))  # Chain 1
theta2, psi2 = run_mcmc(np.log(2.0), np.log(2.0))  # Chain 2 (different start)

#Plot trace plots for both chains
plt.figure(figsize=(12, 6))

# θ trace plots
plt.subplot(2, 1, 1)
plt.plot(theta1, label='Chain 1 (θ)', color='blue')
plt.plot(theta2, label='Chain 2 (θ)', color='orange')
plt.ylabel("θ")
plt.title("Trace plots for θ")
plt.legend()

# ψ trace plots
plt.subplot(2, 1, 2)
plt.plot(psi1, label='Chain 1 (ψ)', color='green')
plt.plot(psi2, label='Chain 2 (ψ)', color='red')
plt.ylabel("ψ")
plt.xlabel("Iteration")
plt.title("Trace plots for ψ")
plt.legend()

plt.tight_layout()
plt.savefig('6.3d.png')
plt.show()
