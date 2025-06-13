import numpy as np
import matplotlib.pyplot as plt


#Defining the cost matrix
cost_matrix_str= """
0,225,110,8,257,22,83,231,277,243,94,30,4,265,274,250,87,83,271,86
255,0,265,248,103,280,236,91,3,87,274,265,236,8,24,95,247,259,28,259
87,236,0,95,248,110,25,274,250,271,9,244,83,250,248,280,29,26,239,7
8,280,83,0,236,28,91,239,280,259,103,23,6,280,244,259,95,87,230,84
268,87,239,271,0,244,275,9,84,25,244,239,275,83,110,24,274,280,84,274
21,265,99,29,259,0,99,230,265,271,87,5,22,239,236,250,87,95,271,91
95,236,28,91,247,93,0,247,259,244,27,91,87,268,275,280,7,8,240,27
280,83,250,261,4,239,230,0,103,24,239,261,271,95,87,21,274,255,110,280
247,9,280,274,84,255,259,99,0,87,255,274,280,3,27,83,259,244,28,274
230,103,268,275,23,244,264,28,83,0,268,275,261,91,95,8,277,261,84,247
87,239,9,103,261,110,29,255,239,261,0,259,84,239,261,242,24,25,242,5
30,255,95,30,247,4,87,274,242,255,99,0,24,280,274,259,91,83,247,91
8,261,83,6,255,29,103,261,247,242,110,29,0,261,244,230,87,84,280,100
242,8,259,280,99,242,244,99,3,84,280,236,259,0,27,95,274,261,24,268
274,22,250,236,83,261,247,103,22,91,250,236,261,25,0,103,255,261,5,247
244,91,261,255,28,236,261,29,103,9,242,261,244,87,110,0,242,236,95,259
84,236,27,99,230,83,7,259,230,230,22,87,93,250,255,247,0,9,259,24
91,242,28,87,250,110,6,271,271,255,27,103,84,250,271,244,5,0,271,29
261,24,250,271,84,255,261,87,28,110,250,248,248,22,3,103,271,248,0,236
103,271,8,91,255,91,21,271,236,271,7,250,83,247,250,271,22,27,248,0
"""

cost_matrix = np.array([[int(x) for x in line.split(",")] for line in cost_matrix_str.strip().split("\n")])


#New Route Cost Function
def total_cost(route, cost_matrix):
    cost = 0
    for i in range(len(route) - 1):
        cost += cost_matrix[route[i], route[i+1]]
    # Add cost to return to start
    cost += cost_matrix[route[-1], route[0]]
    return cost


#Route Proposal
def propose(route):
    """Swap two randomly chosen cities."""
    i, j = np.random.choice(len(route), 2, replace=False)
    new_route = route.copy()
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route

#Cooling Schedule
def cooling_schedule(k, method="sqrt"):
    if method == "sqrt":
        return 1 / np.sqrt(1 + k)
    elif method == "log":
        return 1 / np.log(2 + k)
    else:
        raise ValueError("Unknown cooling schedule.")

#New Simulated Annealing
def simulated_annealing(cost_matrix, max_iter=10000, cooling='sqrt'):
    n = cost_matrix.shape[0]
    current_route = list(np.random.permutation(n))
    current_cost = total_cost(current_route, cost_matrix)
    best_route = current_route[:]
    best_cost = current_cost

    for k in range(max_iter):
        # Propose new route by swapping two cities
        new_route = current_route[:]
        i, j = np.random.choice(n, 2, replace=False)
        new_route[i], new_route[j] = new_route[j], new_route[i]

        new_cost = total_cost(new_route, cost_matrix)

        # Compute temperature
        if cooling == 'sqrt':
            T = 1 / np.sqrt(k + 1)
        elif cooling == 'log':
            T = -np.log(k + 1) if k + 1 > 1 else 1
        else:
            T = 1 / np.sqrt(k + 1)  # default

        # Acceptance probability
        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_route = new_route
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = current_route

    return best_route, best_cost


#Visualisation
def plot_route(route, station_names):
    n = len(route)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Coordinates on a circle
    x = np.cos(angles)
    y = np.sin(angles)
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_aspect('equal')
    
    # Plot stations
    for i, (xi, yi) in enumerate(zip(x, y)):
        ax.plot(xi, yi, 'o', color='blue')
        ax.text(xi, yi, station_names[i], fontsize=9,
                ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.6, pad=1))
    
    # Draw arrows for the route
    for i in range(n):
        start_idx = route[i]
        end_idx = route[(i+1) % n]  # wrap around to start
        
        start_x, start_y = x[start_idx], y[start_idx]
        end_x, end_y = x[end_idx], y[end_idx]
        
        # Draw arrow with a small offset so arrows don't overlap the nodes exactly
        ax.annotate("",
                    xy=(end_x, end_y), xycoords='data',
                    xytext=(start_x, start_y), textcoords='data',
                    arrowprops=dict(arrowstyle="->", color='red', lw=1.5),
                    )

    ax.set_title("Route visualization on circular layout")
    ax.axis('off')
    plt.savefig('7.1b.png')
    plt.show()


if __name__ == "__main__":
    best_route, best_cost = simulated_annealing(cost_matrix, max_iter=10000, cooling='sqrt')
    station_names = [f"Station {i}" for i in range(cost_matrix.shape[0])]
    named_route = [station_names[i] for i in best_route]
    print("Best route found:")
    print(" -> ".join(named_route))
    print(f"Total cost: {best_cost:.2f}")
    plot_route(best_route, station_names)
