import numpy as np
import matplotlib.pyplot as plt

def geometric(p, size=10000):
    #Generate uniform random numbers
    U=np.random.uniform(0,1,size)

    #Inverse transformation
    X = np.floor(np.log(1-U)/np.log(1-p)).astype(int)+1
    return X


#Simulating
ps = [0.3, 0.5, 0.7] #Investigating different values of p
fig, axes = plt.subplots(1, len(ps), figsize=(15, 4), sharey=True)

for i, p in enumerate(ps):
    samples = geometric(p)
    ax = axes[i]
    ax.hist(samples, bins=range(1, max(samples)+1), density=True,
            alpha=0.7, edgecolor='black')
    ax.set_title(f'p = {p}')
    ax.set_xlabel('Value')
    if i == 0:
        ax.set_ylabel('Probability')
    ax.grid(True)

fig.suptitle('Geometric Distribution Simulations for Different p-values')
plt.tight_layout()
plt.savefig('2.1 Histogram.png')
plt.show()