import numpy as np
import matplotlib.pyplot as plt

def visualize_data(file_path: str, t: int, save_path: str, cmap: str = 'viridis'):
    """
    Visualize data from a numpy file at a specific time step using a simple rectangular projection.
    
    Args:
        file_path (str): Path to the numpy file
        t (int): Time step to visualize
        cmap (str): Colormap to use for visualization (default: 'viridis')
        save_path (str): If provided, save the plot to this path instead of displaying
    """
    # Load the data
    data = np.load(file_path)
    
    # Check if time index is valid
    if t >= data.shape[0]:
        raise ValueError(f"Time index {t} is out of bounds. Max time index is {data.shape[0]-1}")
    
    # Extract the time slice
    time_slice = data[t]
    
    # Create latitude and longitude arrays (note the -90 to 90 is reversed to fix orientation)
    lats = np.linspace(90, -90, 721)  # Reversed to fix orientation
    lons = np.linspace(-180, 180, 1440)
    
    # Convert from Kelvin to Celsius
    data_celsius = time_slice - 273.15
    
    # Shift longitudes to center on Africa (0°)
    data_shifted = np.roll(data_celsius, lons.size // 2, axis=1)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Create the plot using pcolormesh with shifted data
    mesh = plt.pcolormesh(lons, lats, data_shifted, cmap=cmap)
    
    # Add colorbar and labels
    plt.colorbar(mesh, orientation='horizontal', pad=0.05, label='Temperature (°C)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Data visualization for time step {t}')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    
visualize_data('/weather/WeatherExtremes/experiments/2m_temperature_1960_1961_AGG.MAX_aggrwindow_1_percboost_15/percentiles_0_9.npy', 0, 'percentiles_plot.png')
