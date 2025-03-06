import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os

def calculate_seasonal_trends(exceedances):
    """
    Calculate seasonal trend lines for temperature exceedances over 60 years.
    
    Args:
        exceedances: numpy array of shape (N_YEARS * 365, 721, 1440) containing boolean values
                    where
    
    Returns:
        numpy array of shape (4, 721, 1440, 2) containing slope and intercept 
        for each season at each lat/long point
    """
    N_YEARS = exceedances.shape[0] // 365
    
    # Initialize output array for all seasons
    # Dimensions: (4 seasons, 721 lats, 1440 longs, 2 parameters)
    trends = np.zeros((4, 721, 1440, 2))
    
    # Define season boundaries (days within year)
    season_bounds = [
        (-31, 59),   # DJF: Dec(-31 to -1), Jan(0-30), Feb(31-58)
        (59, 151),   # MAM: Mar-May
        (151, 243),  # JJA: Jun-Aug
        (243, 334)   # SON: Sep-Nov
    ]
    
    years_array = np.arange(N_YEARS)
    
    # Reshape exceedances to (N_YEARS, 365, lat, lon)
    exceedances_by_year = exceedances.reshape(N_YEARS, 365, 721, 1440)
    
    season_exceedances = []
    
    for season_idx, (start, end) in enumerate(season_bounds):
        print(f"Season: {season_idx}")
        if season_idx == 0:  # DJF
            # Handle December separately
            dec_data = np.sum(exceedances_by_year[:, 334:, :, :], axis=1)
            # Shift December data one year forward and pad first year with zeros
            dec_data = np.roll(dec_data, 1, axis=0)
            dec_data[0] = 0
            
            # Handle January-February
            janfeb_data = np.sum(exceedances_by_year[:, :59, :, :], axis=1)
            # Zero out last year's Jan-Feb data since we only want December
            janfeb_data[-1] = 0
            
            # Combine December and Jan-Feb data
            season_data = dec_data + janfeb_data
            
        else:
            # For other seasons, simply sum over the season's days
            season_data = np.sum(exceedances_by_year[:, start:end, :, :], axis=1)
        
        # Prepare arrays for vectorized regression
        x = years_array[:, np.newaxis, np.newaxis]  # Shape: (N_YEARS, 1, 1)
        y = season_data  # Shape: (N_YEARS, 721, 1440)
        season_exceedances.append(y)
        
        # Calculate means
        x_mean = np.mean(x)
        y_mean = np.mean(y, axis=0)
        
        # Calculate slope and intercept
        numerator = np.sum((x - x_mean) * (y - y_mean[np.newaxis, :, :]), axis=0)
        denominator = np.sum((x - x_mean) ** 2)
        
        slopes = numerator / denominator
        intercepts = y_mean - slopes * x_mean
        
        # Store results
        trends[season_idx, :, :, 0] = slopes
        trends[season_idx, :, :, 1] = intercepts
    
    return trends, season_exceedances

def create_seasonal_plots(trends, intercepts=False):
    """
    Create a figure with 4 subplots showing seasonal slope trends.
    
    Args:
        trends: numpy array of shape (4, 721, 1440, 2) containing slope and intercept
               values for each season at each lat/long point
    """
    
    data_type = 0 #trends
    if intercepts:
        data_type = 1 # intercepts
    
    season_names = ['DJF', 'MAM', 'JJA', 'SON']
    
    # trend fig
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    for idx, (ax, season) in enumerate(zip(axes, season_names)):
        plot_single_season(ax, trends[idx, :, :, data_type], season)
    plt.tight_layout()

    return fig

def plot_single_season(ax, data, season_name):
    """
    Create a single subplot for one season's slope data.
    
    Args:
        ax: matplotlib axis object
        data: numpy array of shape (721, 1440) containing one value per (lat, lon)
        season_name: string indicating the season (DJF, MAM, etc.)
    """
    
    # Create mesh grid for plotting
    lats = np.linspace(-90, 90, 721)
    lons = np.linspace(-180, 180, 1440)
    lon_mesh, lat_mesh = np.meshgrid(lons, lats)
    
    # Plot the data
    im = ax.pcolormesh(lon_mesh, lat_mesh, data, 
                       cmap='RdBu_r', 
                       shading='auto')
    
    # Add colorbar and labels
    plt.colorbar(im, ax=ax, label='Trend (exceedances/year)')
    ax.set_title(f'{season_name} Seasonal Trend')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    

def process_exceedances_file(agg_window = 1, percboost = 1, seasonality = 0):
    fpath = f'experiments/2m_temperature_1960_1989_AGG.MEAN_aggrwindow_{agg_window}_percboost_{percboost}_seasonality_{seasonality}/exceedances_0_9.npy'
    
    if not os.path.exists(fpath):
        print(f"Warning: File {fpath} does not exist, skipping...")
        return
    
    out_file = f'trends_aggrwindow_{agg_window}_percboost_{percboost}_seasonality_{seasonality}'
    trend_fig_file = f'trends_plot_aggrwindow_{agg_window}_percboost_{percboost}_seasonality_{seasonality}'
    intercept_fig_file = f'intercept_plot_aggrwindow_{agg_window}_percboost_{percboost}_seasonality_{seasonality}'

    exceedances = np.load(fpath)
    trends, _season_exceedances = calculate_seasonal_trends(exceedances)
    np.save(out_file, trends)

    # After calculating trends, create and display the plot
    # fig = create_seasonal_plots(trends)
    # plt.savefig(trend_fig_file)
    
    fig = create_seasonal_plots(trends, intercepts=True)
    plt.savefig(intercept_fig_file)
    

# for agg_window in [1,7,15]:
#     for percboost in [1,7,15]:
#         for seasonality in [0,1]:
#             print(('agg', agg_window, 'perc', percboost, 'seas', seasonality))
#             process_exceedances_file(agg_window, percboost, seasonality)


agg_window = 1
percboost = 1
seasonality = 0


fpath = f'experiments/2m_temperature_1960_1989_AGG.MEAN_aggrwindow_{agg_window}_percboost_{percboost}_seasonality_{seasonality}/exceedances_0_9.npy'
arr = np.load(fpath)

trends, seasons_data = calculate_seasonal_trends(arr)
print(seasons_data.shape)