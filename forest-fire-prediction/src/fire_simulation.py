import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random # For initial fire points and probabilities

# --- 1. Define Cell States ---
UNBURNED = 0
BURNING = 1
BURNED = 2
OBSTACLE = 3 # Optional: for water, rocks, etc.

# --- 2. Define Colors for Visualization ---
# Custom colormap for visualization
colors = ['#1a9850', '#d73027', '#a6a6a6', '#4575b4'] # Green (unburned), Red (burning), Gray (burned), Blue (obstacle)
cmap = ListedColormap(colors)
norm = plt.Normalize(vmin=0, vmax=3) # Normalize for our 0-3 states

class FireCA:
    def __init__(self, grid_size=(50, 50), initial_fire_points=1, obstacle_density=0.05,
                 burning_duration=1, prob_spread=0.8, prob_self_extinguish=0.0):
        self.rows, self.cols = grid_size
        self.grid = np.full(grid_size, UNBURNED, dtype=int)
        self.burning_timers = np.zeros(grid_size, dtype=int) # To track how long a cell has been burning

        self.burning_duration = burning_duration # How many steps a cell stays BURNING before becoming BURNED
        self.prob_spread = prob_spread # Probability an unburned cell catches fire from a burning neighbor
        self.prob_self_extinguish = prob_self_extinguish # Probability a burning cell extinguishes on its own

        self._initialize_grid(initial_fire_points, obstacle_density)

        # Initialize wind_data correctly as a 3D array for 2D vectors per cell
        # Each cell (r, c) will store a vector [x_component, y_component]
        # Example: [0, 1] could mean wind blowing from South to North
        # You would later define how these components translate to direction and speed
        self.wind_data = np.full((self.rows, self.cols, 2), [0, 1], dtype=int)
        # Note: In a real scenario, you'd load or generate a more complex wind map
        # where each cell might have a different wind vector.

        # Example of how you might add a basic fuel map (optional, for future enhancement)
        # self.fuel_map = np.full(grid_size, 'forest', dtype='U10') # e.g., 'grass', 'forest', 'shrub'
        # self.fuel_map[self.grid == OBSTACLE] = 'none' # Obstacles have no fuel

    def _initialize_grid(self, initial_fire_points, obstacle_density):
        # Add obstacles
        num_obstacles = int(self.rows * self.cols * obstacle_density)
        # Create a list of all possible (row, col) indices
        all_indices = [(r, c) for r in range(self.rows) for c in range(self.cols)]
        # Randomly sample unique indices for obstacles
        obstacle_coords = random.sample(all_indices, num_obstacles)

        for r, c in obstacle_coords:
            self.grid[r, c] = OBSTACLE

        # Set initial fire points (ensure they are not obstacles)
        fire_set = 0
        while fire_set < initial_fire_points:
            r, c = random.randint(0, self.rows - 1), random.randint(0, self.cols - 1)
            if self.grid[r, c] == UNBURNED:
                self.grid[r, c] = BURNING
                self.burning_timers[r, c] = self.burning_duration
                fire_set += 1

    def _get_neighbors(self, r, c):
        """Returns the states of the Moore neighborhood (8 neighbors) as a list of (state, nr, nc) tuples."""
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue # Skip the cell itself
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    neighbors.append((self.grid[nr, nc], nr, nc))
        return neighbors

    def update(self):
        """Applies the CA rules for one time step."""
        new_grid = np.copy(self.grid) # Create a copy to store new states (synchronous update)
        new_burning_timers = np.copy(self.burning_timers)

        for r in range(self.rows):
            for c in range(self.cols):
                current_state = self.grid[r, c]

                if current_state == UNBURNED:
                    neighbors_info = self._get_neighbors(r, c)
                    # Check if any neighbor is BURNING
                    burning_neighbors_present = any(state == BURNING for state, _, _ in neighbors_info)

                    if burning_neighbors_present:
                        # Rule: Unburned to Burning
                        # You would add complex environmental factors here, e.g.:
                        # - wind_vector_at_cell = self.wind_data[r, c]
                        # - fuel_type_at_cell = self.fuel_map[r, c] (if you added it)
                        # - Adjust self.prob_spread based on these factors and neighbor direction
                        effective_prob_spread = self.prob_spread # Starting with base probability

                        # Example of simple wind influence (wind from South to North, favors North spread)
                        # This would need to be much more sophisticated for true realism
                        # For now, just a placeholder to show where you'd integrate it
                        # if self.wind_data[r, c][1] > 0: # If there's a positive y-component (blowing north)
                        #     # If a neighbor directly to the north is burning, increase prob
                        #     for state, nr, nc in neighbors_info:
                        #         if state == BURNING and nr < r: # Neighbor is above (North)
                        #             effective_prob_spread *= 1.2 # Boost
                        #         elif state == BURNING and nr > r: # Neighbor is below (South)
                        #             effective_prob_spread *= 0.8 # Reduce


                        if random.random() < effective_prob_spread:
                            new_grid[r, c] = BURNING
                            new_burning_timers[r, c] = self.burning_duration # Start timer

                elif current_state == BURNING:
                    # Rule: Burning to Burned (based on timer)
                    new_burning_timers[r, c] -= 1
                    if new_burning_timers[r, c] <= 0:
                        new_grid[r, c] = BURNED
                    # Optional: Self-extinguish probability (e.g., if conditions are unfavorable)
                    elif random.random() < self.prob_self_extinguish:
                        new_grid[r, c] = BURNED # Changed to BURNED directly if self-extinguishing
                        new_burning_timers[r, c] = 0

                # Rule: Burned remains Burned
                # Rule: Obstacle remains Obstacle
                # No change needed for BURNED or OBSTACLE cells unless specific rules apply

        self.grid = new_grid
        self.burning_timers = new_burning_timers
        return np.any(self.grid == BURNING) # Return True if any cell is still burning

    def visualize(self, ax, title="Fire Simulation"):
        ax.imshow(self.grid, cmap=cmap, norm=norm)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

# --- Main Simulation Loop ---
if __name__ == "__main__":
    grid_size = (100, 100)
    initial_fire_points = 3
    obstacle_density = 0.08 # Percentage of land that is obstacles
    burning_duration = 2 # Cells burn for 2 time steps
    prob_spread = 0.7 # 70% chance to spread
    prob_self_extinguish = 0.01 # Small chance a burning cell self-extinguishes

    fire_sim = FireCA(grid_size, initial_fire_points, obstacle_density,
                      burning_duration, prob_spread, prob_self_extinguish)

    # Setup plot for animation
    fig, ax = plt.subplots(figsize=(8, 8))
    fire_sim.visualize(ax, title="Initial State")
    plt.show(block=False) # Non-blocking show to allow updates

    print("Starting simulation...")
    time_step = 0
    while True:
        time_step += 1
        burning_active = fire_sim.update()
        fire_sim.visualize(ax, title=f"Time Step {time_step}")
        fig.canvas.draw_idle() # Update the plot efficiently
        plt.pause(0.1) # Pause to see the animation

        if not burning_active:
            print(f"Fire extinguished at time step {time_step}.")
            break
        if time_step > 200: # Max time steps to prevent infinite loop
            print("Maximum time steps reached.")
            break

    plt.show() # Keep the final plot open