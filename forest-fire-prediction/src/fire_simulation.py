import numpy as np
import random

class ForestFireCA:
    def __init__(self, grid_size, initial_ignition_points, fuel_map, wind_data, slope_data,
                 base_flammability_factor=0.5, wind_effect_factor=0.1, slope_effect_factor=0.05,
                 ignition_threshold=0.6):
        """
        Initializes the Forest Fire Cellular Automata simulation.

        Args:
            grid_size (tuple): (rows, cols) representing the dimensions of the grid.
            initial_ignition_points (list): A list of (row, col) tuples for initial burning cells.
            fuel_map (np.ndarray): A 2D array representing fuel types/vegetation (0-1, higher is more flammable).
            wind_data (np.ndarray): A 2D array representing wind direction and strength.
                                     Can be complex (e.g., tuples for (magnitude, direction_radians))
                                     or simplified (e.g., integer codes for N, NE, E, etc.).
                                     For this example, let's assume it's a 2D array where each element
                                     is an integer representing a simplified wind direction (0: no wind, 1: N, 2: NE, etc.).
                                     Or even simpler, a constant global wind direction for now.
            slope_data (np.ndarray): A 2D array representing the slope (e.g., angle in degrees or a classification).
                                     Positive values indicate uphill, negative downhill.
            base_flammability_factor (float): Multiplier for base flammability.
            wind_effect_factor (float): Multiplier for the wind's influence on ignition probability.
            slope_effect_factor (float): Multiplier for the slope's influence on ignition probability.
            ignition_threshold (float): The probability threshold for an unburned cell to ignite.
        """
        self.rows, self.cols = grid_size
        self.grid = np.zeros(grid_size, dtype=int)  # 0: Unburned, 1: Burning, 2: Burned

        # Initialize ignition points
        for r, c in initial_ignition_points:
            if 0 <= r < self.rows and 0 <= c < self.cols:
                self.grid[r, c] = 1  # Set to burning

        self.fuel_map = fuel_map
        self.wind_data = wind_data
        self.slope_data = slope_data

        self.base_flammability_factor = base_flammability_factor
        self.wind_effect_factor = wind_effect_factor
        self.slope_effect_factor = slope_effect_factor
        self.ignition_threshold = ignition_threshold

        # For simplified wind: Let's assume a global wind direction for now.
        # Example: (dr, dc) for wind vector. (0, 1) for East, (-1, 0) for North.
        # This will need to be refined based on actual wind data format.
        self.global_wind_direction = (0, 1)  # Example: Wind blowing towards East

    def _calculate_ignition_probability(self, r, c, neighbor_r, neighbor_c):
        """
        Calculates the ignition probability for an unburned cell (r, c)
        based on a burning neighbor (neighbor_r, neighbor_c).
        """
        probability = 0.0

        # 1. Base Flammability
        probability += self.fuel_map[r, c] * self.base_flammability_factor

        # 2. Wind Effect
        # Simplified wind effect: If wind blows from burning neighbor towards current cell
        # (This is a very basic implementation and would need more sophisticated wind modeling)
        wind_dr, wind_dc = self.global_wind_direction # Assuming global wind for now

        # Vector from neighbor to current cell
        vec_to_current_dr = r - neighbor_r
        vec_to_current_dc = c - neighbor_c

        # If wind is blowing roughly in the same direction as the vector from neighbor to current cell
        # This is a very rough approximation. A more accurate model would use dot products with wind vectors.
        if (wind_dr == vec_to_current_dr and wind_dc == vec_to_current_dc) or \
           (wind_dr == -vec_to_current_dr and wind_dc == -vec_to_current_dc): # This means wind is blowing towards the neighbor
            probability += self.wind_effect_factor
        
        # More sophisticated wind: check if wind is blowing *from* neighbor *to* (r,c)
        # If wind vector points roughly from (neighbor_r, neighbor_c) to (r, c)
        # This requires more complex wind data (magnitude and direction for each cell)
        # For simplicity, if wind is generally aligned with the direction of spread from burning neighbor, increase probability.
        # For a truly accurate wind model, you'd convert wind_data (e.g., angle) to a vector
        # and calculate the dot product with the vector from (neighbor_r, neighbor_c) to (r, c).
        # Dot product > 0 suggests wind is aiding spread in that direction.

        # 3. Slope Effect
        # If current cell (r, c) is uphill from burning neighbor (neighbor_r, neighbor_c)
        # This assumes slope_data provides elevation or slope direction.
        # For simplicity, let's assume slope_data[r,c] > slope_data[neighbor_r, neighbor_c] means uphill.
        if self.slope_data[r, c] > self.slope_data[neighbor_r, neighbor_c]:
            probability += self.slope_effect_factor * abs(self.slope_data[r, c] - self.slope_data[neighbor_r, neighbor_c])
        elif self.slope_data[r, c] < self.slope_data[neighbor_r, neighbor_c]:
            # Fire spreads slower downhill, so we could subtract a smaller factor or do nothing
            probability -= self.slope_effect_factor * 0.5 * abs(self.slope_data[r, c] - self.slope_data[neighbor_r, neighbor_c])


        return min(1.0, max(0.0, probability)) # Ensure probability is between 0 and 1

    def step(self):
        """
        Executes one time step of the simulation.
        Returns True if there are still burning cells, False otherwise.
        """
        new_grid = self.grid.copy()
        burning_cells_count = 0

        for r in range(self.rows):
            for c in range(self.cols):
                current_state = self.grid[r, c]

                if current_state == 1:  # If cell is burning
                    new_grid[r, c] = 2  # It transitions to burned
                elif current_state == 0:  # If cell is unburned
                    # Check neighbors for burning cells
                    burning_neighbor_found = False
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue  # Skip self

                            neighbor_r, neighbor_c = r + dr, c + dc

                            # Boundary Conditions: Ensure neighbor is within grid bounds
                            if 0 <= neighbor_r < self.rows and 0 <= neighbor_c < self.cols:
                                if self.grid[neighbor_r, neighbor_c] == 1:  # If neighbor is burning
                                    burning_neighbor_found = True
                                    # Calculate ignition probability
                                    prob = self._calculate_ignition_probability(r, c, neighbor_r, neighbor_c)

                                    if random.random() < prob:  # If random check passes
                                        new_grid[r, c] = 1  # Cell ignites
                                        burning_cells_count += 1
                                        break  # No need to check other neighbors if already ignited
                        if burning_neighbor_found:
                            break

        self.grid = new_grid
        return burning_cells_count > 0

    def get_grid(self):
        """Returns the current state of the simulation grid."""
        return self.grid

# --- Example Usage ---
if __name__ == "__main__":
    grid_rows = 50
    grid_cols = 50
    grid_size = (grid_rows, grid_cols)

    # Example: Create dummy data for fuel, wind, and slope
    fuel_map = np.random.rand(grid_rows, grid_cols)  # Random flammability between 0 and 1
    
    # Simplified wind: Assume a constant global wind blowing from West to East (towards positive column direction)
    # This is (dr, dc) = (0, 1)
    wind_data = np.full(grid_size, (0, 1), dtype=object) # Not really used per cell in current simplified model
    
    # Slope data: Create a synthetic slope, e.g., higher values towards the top-right
    slope_data = np.zeros(grid_size)
    for r in range(grid_rows):
        for c in range(grid_cols):
            slope_data[r, c] = (r + c) / (grid_rows + grid_cols) # Slope increasing diagonally

    # Initial ignition points (from a hypothetical prediction map)
    # Let's say top 3 predicted points
    initial_ignition_points = [(25, 25), (20, 30), (30, 20)]

    # Initialize the simulation
    fire_sim = ForestFireCA(grid_size, initial_ignition_points, fuel_map, wind_data, slope_data,
                            base_flammability_factor=0.3, # Lower base flammability
                            wind_effect_factor=0.2,      # Moderate wind effect
                            slope_effect_factor=0.1,     # Moderate slope effect
                            ignition_threshold=0.5)      # Probability threshold for ignition

    print("Initial Grid State:")
    print(fire_sim.get_grid())

    # Run the simulation for a few steps
    num_steps = 20
    for i in range(num_steps):
        print(f"\n--- Step {i+1} ---")
        still_burning = fire_sim.step()
        print(f"Current Grid State (Showing only a portion):")
        print(fire_sim.get_grid()[20:35, 20:35]) # Print a central portion for readability

        if not still_burning:
            print("No more burning cells. Fire extinguished.")
            break

    print("\nFinal Grid State:")
    print(fire_sim.get_grid())