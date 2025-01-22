import numpy as np
import xarray as xr
import os

from scipy.stats import linregress
import matplotlib.pyplot as plt

from optimised_solution.Experiment import Experiment
from constants import LAT_LON

# TODO: replace monthly 1M with any bucket size
def read_or_compute_monthly_exceedances(monthly_exceedances_path, an_start, an_end, percentiles):
    # CALCULATE MONTHLY EXCEEDANCES 
    if os.path.exists(monthly_exceedances_path):
        print(f"The monthly exceedances file exists. Loading from the file {monthly_exceedances_path}")
        return xr.open_zarr(monthly_exceedances_path)    
    print("Monthly exceedances file not found. Calculating exceedances")
    
    exceedances_doy = calcluate_exceedances(an_start, an_end, percentiles)
    
    monthly_exceedances = exceedances_doy.resample(time="1M").sum(dim="time")
    
    print(f"Saving monthly exceedances to {monthly_exceedances_path}")
    monthly_exceedances.to_zarr(monthly_exceedances_path)
    return monthly_exceedances

def calculate_slopes(monthly_exceedances):
    # Calculate the trend slope for a single time series
    def calculate_slope(time_series):
        # Remove NaNs
        time = np.arange(len(time_series))
        mask = ~np.isnan(time_series)
        if np.sum(mask) > 1:  # Ensure there are enough valid points to calculate slope
            slope, _, _, _, _ = linregress(time[mask], time_series[mask])
            return slope
        else:
            return np.nan

    # Apply this function across latitude and longitude for each time series
    slopes = xr.apply_ufunc(
        calculate_slope,
        monthly_exceedances,
        input_core_dims=[["time"]],
        dask="parallelized",
        vectorize=True
    )

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(slopes['longitude'], slopes['latitude'], slopes, shading='auto')
    plt.colorbar(label="Trend Slope")
    plt.title("Trend of Exceedance Count per Location")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

def optimised(params, raw_data_path, input_zarr_path):
    # for percentile in params['percentiles']:
    #     experiment = Experiment(params, input_zarr_path, percentile)
    #     experiment.calculate_percentile_scores()
        
    for percentile in params['percentiles']:
        experiment = Experiment(params, raw_data_path, input_zarr_path, percentile)
        percentiles = experiment.calculate_percentiles()
        exceedances = experiment.calcluate_exceedances()
        
        # for _cityname, (lat, lon) in LAT_LON.items():
        #     print(lat,lon)
        #     print(exceedances.isel(latitude=lat, longitude=lon).values.tolist())
            
        # monthly_exceedances = read_or_compute_monthly_exceedances(monthly_exceedances_path, an_start, an_end, percentiles)
        # calculate_slopes(monthly_exceedances)
    
