# Vicsek-Model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Parameters
N_fish = 1000  # Number of fish
N_sharks = 3  # Number of sharks
L = 100  # Size of the sea (square arena)
v_fish = 1.0  # Speed of a single fish
v_shark = 1.1 * v_fish  # Sharks swim 10% faster
R_flock = 2.0  # Flocking radius for fish
eta = 0.1  # Noise in fish movement
R_predator_avoid = 5.0  # Radius within which fish avoid sharks
dt = 1.0  # Time step

# Initialize positions and velocities
fish_positions = np.random.rand(N_fish, 2) * L  # Random positions in the arena
fish_velocities = np.random.rand(N_fish, 2) - 0.5  # Random initial velocities
fish_velocities = (fish_velocities.T / np.linalg.norm(fish_velocities, axis=1)).T * v_fish

shark_positions = np.random.rand(N_sharks, 2) * L
shark_velocities = np.zeros((N_sharks, 2))  # Sharks start stationary

# Periodic boundary condition function
def apply_periodic_boundary(positions, L):
    return positions % L

# Fish clustering behavior (Vicsek model)
def update_fish(fish_positions, fish_velocities, shark_positions):
    new_velocities = np.zeros_like(fish_velocities)
    for i in range(N_fish):
        # Find neighbors within R_flock
        distances = np.linalg.norm(fish_positions - fish_positions[i], axis=1)
        neighbors = distances < R_flock

        # Calculate average direction of neighbors
        avg_velocity = np.mean(fish_velocities[neighbors], axis=0)
        avg_velocity /= np.linalg.norm(avg_velocity)  # Normalize

        # Add random noise
        noise = np.random.uniform(-eta, eta, size=2)
        direction = avg_velocity + noise

        # Avoid predators within R_predator_avoid
        for shark in shark_positions:
            if np.linalg.norm(fish_positions[i] - shark) < R_predator_avoid:
                direction += (fish_positions[i] - shark)

        new_velocities[i] = direction / np.linalg.norm(direction) * v_fish

    # Update positions
    fish_positions += new_velocities * dt
    fish_positions = apply_periodic_boundary(fish_positions, L)

    return fish_positions, new_velocities

# Shark chasing behavior
def update_sharks(shark_positions, shark_velocities, fish_positions):
    new_velocities = np.zeros_like(shark_velocities)
    for i in range(N_sharks):
        # Find the nearest fish
        distances = np.linalg.norm(fish_positions - shark_positions[i], axis=1)
        nearest_fish_idx = np.argmin(distances)
        direction = fish_positions[nearest_fish_idx] - shark_positions[i]

        # Normalize and move toward fish
        direction /= np.linalg.norm(direction)
        new_velocities[i] = direction * v_shark

    # Update positions
    shark_positions += new_velocities * dt
    shark_positions = apply_periodic_boundary(shark_positions, L)

    return shark_positions, new_velocities

# Visualization function
def animate(frame):
    global fish_positions, fish_velocities, shark_positions, shark_velocities

    # Update fish and sharks
    fish_positions, fish_velocities = update_fish(fish_positions, fish_velocities, shark_positions)
    shark_positions, shark_velocities = update_sharks(shark_positions, shark_velocities, fish_positions)

    # Update scatter plot
    fish_scatter.set_offsets(fish_positions)
    shark_scatter.set_offsets(shark_positions)

    return fish_scatter, shark_scatter

# Setup visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, L)
ax.set_ylim(0, L)

fish_scatter = ax.scatter(fish_positions[:, 0], fish_positions[:, 1], s=10, c='blue', label='Fish')
shark_scatter = ax.scatter(shark_positions[:, 0], shark_positions[:, 1], s=50, c='red', label='Sharks')
ax.legend()

# Run animation
ani = animation.FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
plt.show()
