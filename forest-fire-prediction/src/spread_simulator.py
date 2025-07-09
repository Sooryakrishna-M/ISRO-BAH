# src/spread_simulator.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def simulate_spread(center_lat, center_lon, radius_km=10):
    fire_points = []
    for _ in range(100):
        lat_offset = np.random.uniform(-0.1, 0.1)
        lon_offset = np.random.uniform(-0.1, 0.1)
        fire_points.append((center_lat + lat_offset, center_lon + lon_offset))
    
    df = pd.DataFrame(fire_points, columns=['lat', 'lon'])
    df.to_csv('data/processed/simulation.csv', index=False)
    return df

def plot_simulation():
    df = pd.read_csv('data/processed/simulation.csv')
    plt.scatter(df['lon'], df['lat'], color='red', alpha=0.5)
    plt.title('🔥 Simulated Fire Spread')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.savefig('data/processed/spread_map.png')
    plt.show()

if __name__ == "__main__":
    simulate_spread(21.5, 78.9)
    plot_simulation()
