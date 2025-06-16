import numpy as np
import matplotlib.pyplot as plt

#Route Cost Function
def route_distance(route, coords):
    """Compute total length of route, returning to the start."""
    dist = 0.0
    for i in range(len(route)):
        a = coords[route[i]]
        b = coords[route[(i + 1) % len(route)]]  # wrap around
        dist += np.linalg.norm(a - b)
    return dist

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

#Simulated Annealing
def simulated_annealing(coords, max_iter=10000, cooling='sqrt'):
    n = len(coords)
    route = list(np.random.permutation(n))  # random initial route
    best_route = route.copy()
    best_cost = route_distance(route, coords)
    current_cost = best_cost

    for k in range(1, max_iter + 1):
        T = cooling_schedule(k, cooling)
        candidate = propose(route)
        candidate_cost = route_distance(candidate, coords)
        delta = candidate_cost - current_cost

        # Accept better or probabilistically worse solution
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            route = candidate
            current_cost = candidate_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = route.copy()

        # Debug print every 1000 iterations
        if k % 1000 == 0:
            print(f"Iteration {k}: Current cost = {current_cost:.2f}, Best = {best_cost:.2f}")

    return best_route, best_cost

#Route Plotting
def plot_route(route, coords, title="TSP Route", save_path=None):
    coords = np.array(coords)
    full_route = route + [route[0]]  # close the loop
    route_coords = coords[full_route]
    plt.figure(figsize=(6,6))
    plt.plot(route_coords[:,0], route_coords[:,1], '-o', markersize=8)
    for i, (x, y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


#Optional Debugging Circle
def generate_circle_stations(n, radius=1.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.column_stack((x, y))

#Run the Program
if __name__ == "__main__":
    n = 10

    # === First: Random coordinates ===
    coords_random = np.random.rand(n, 2)
    route_random, cost_random = simulated_annealing(coords_random, max_iter=10000, cooling='sqrt')
    plot_route(route_random, coords_random,
               title=f"Random TSP (Cost: {cost_random:.2f})",
               save_path="7.1a Random TSP.png")

    # === Second: Circle coordinates ===
    coords_circle = generate_circle_stations(n)
    route_circle, cost_circle = simulated_annealing(coords_circle, max_iter=10000, cooling='sqrt')
    plot_route(route_circle, coords_circle,
               title=f"Circle TSP (Cost: {cost_circle:.2f})",
               save_path="7.1a Circle TSP.png")

