from datetime import timedelta
import numpy as np
from enums import AGG
from tqdm import tqdm
import time
import dask.array as da

import zarr
import xarray as xr
import dask.array as da
import os
from exceedences import generate_temperature_exceedance_mask

from scipy.stats import linregress
import matplotlib.pyplot as plt

def aggregate(data, aggregation, agg_wind, start, end):
    print("Converting to no-leap calendar")
    data = data.convert_calendar('noleap')
    
    data = data.sel(time=slice(start, end))
    
    assert data['time'][0] == start, f"Missing data at the beginning of the reference period. Data: {data['time'][0]}, Start: {start}"
    assert data['time'][-1] == end, f"Missing data at the end of the reference period. Data: {data['time'][-1]}, Start: {start}"
    
    print("Calculating Aggregations...")
    rolling_data = data.rolling(time=agg_wind, center=True)

    if aggregation == AGG.MEAN:
        aggregated_data = rolling_data.mean()
    elif aggregation == AGG.SUM:
        aggregated_data = rolling_data.sum()
    elif aggregation == AGG.MIN:
        aggregated_data = rolling_data.min()
    elif aggregation == AGG.MAX:
        aggregated_data = rolling_data.max()
    else:
        raise Exception("Wrong type of aggregation provided: params['aggregation'] = ", aggregation)
    
    # Make sure we use the correct dimension names
    if 'lat' in aggregated_data.dims:
        aggregated_data = aggregated_data.rename({"lat": "latitude"})
    if 'lon' in aggregated_data.dims:
        aggregated_data = aggregated_data.rename({"lon": "longitude"})
    assert 'latitude' in aggregated_data.dims
    assert 'longitude' in aggregated_data.dims
    
    return aggregated_data
    
def parallel_pre_percentile_arrange(agg_data, n_years, ref_start, ref_end, perc_boost, start_doy=1, end_doy=365):        
    half_perc_boost = perc_boost // 2
    perc_start = ref_start - timedelta(days=half_perc_boost)
    perc_end = ref_end + timedelta(days=half_perc_boost)
    dec31doy = 365
    jan1doy = 1
    prefix_to_append = dec31doy + half_perc_boost - 365 # How many doy_groups from the beginning should be appended
    suffix_to_preppend = -(jan1doy - half_perc_boost - 1) # How many doy_groups from the end should be prepended
    
    # Rearrange the grouped by DOY data and save into a zar suitable for faster percentile calculation
    print("Rearranging data into pre-percentile format")
    
    assert prefix_to_append >= 0
    assert suffix_to_preppend >= 0
    
    # Read the data and group by Day Of Year
    agg_data = agg_data.sel(time=slice(perc_start, perc_end))
    doy_grouped = agg_data.groupby('time.dayofyear')
    
    prefix_to_append_index=prefix_to_append
    suffix_to_preppend_index=len(doy_grouped)-suffix_to_preppend
    
    prefix_arrays = []
    main_arrays = []
    suffix_arrays = []
    for day_of_year in range(start_doy,end_doy+1):
        # Select the group corresponding to this day of year
        doy_groups_np = doy_grouped[day_of_year].data  # Get the underlying dask array

        if day_of_year-1 < prefix_to_append_index:
            suffix_arrays.append(doy_groups_np[1:])
            doy_groups_np = doy_groups_np[:-1]
        elif day_of_year-1 >= suffix_to_preppend_index:
            prefix_arrays.append(doy_groups_np[:-1])
            doy_groups_np = doy_groups_np[1:]
            
        assert doy_groups_np.shape[0] == n_years, f"Wrong array shape! Shape: {doy_groups_np.shape}"
        main_arrays.append(doy_groups_np)
    arrays = prefix_arrays + main_arrays + suffix_arrays
    return arrays

def save_pre_percentile_to_zarr(arrays, zarr_path, n_years, batch_size=1, lat_size=721, lon_size=1440):
    start = time.time()
    
    pre_percentile_zarr_store = zarr.open(zarr_path, mode='w', shape=(len(arrays) * n_years, lat_size, lon_size), chunks=(batch_size * n_years, 1, lon_size), dtype=arrays[0].dtype)
    
    for i in tqdm(list(range(0, len(arrays), batch_size))):
        batch = da.concatenate(arrays[i:i + batch_size])
        start_index = i * n_years
        end_index = start_index + batch.shape[0]
        da.to_zarr(batch, pre_percentile_zarr_store, region=(slice(start_index, end_index), slice(None), slice(None)))

    end = time.time()
    print(f"Rearranging: {end - start} seconds")
    
    return pre_percentile_zarr_store
    

def calculate_percentile(perc_boost, n_years, pre_perc_zarr, perc, lat_size=721, lon_size=1440):
    # To reduce the memory footprint and enable easy parallelization we will calculate the percentiles band, by band
    # Since the original weather state array is 721 x 1440, we will split the first dimension into bands 
    max_mem = 1000000000 # ~1 GB
    max_floats = max_mem // 4 # 4 bytes per float
    band_size = max_floats // (n_years * (365 + perc_boost - 1) * lon_size)
    band_size = min(lat_size, band_size)
    
    print("BAND_SIZE", band_size)
    
    bands_indices = list(range(0, lat_size, band_size))
    if bands_indices[-1] < lat_size:
        bands_indices.append(lat_size)
    
    bands = list(zip(bands_indices, bands_indices[1:]))
    all_percentiles = []
    for (start, end) in tqdm(bands):
        band = pre_perc_zarr[:,start:end,:]
        assert np.any(band > 0)
        percentiles = []
        for doy in tqdm(range(365), leave=False):
            assert len(band) == n_years * (365 + perc_boost-1), f'Wrong band shape: {band.shape}. The expected length is {n_years * (365 + perc_boost-1)}'
            pre_percentile_window = band[n_years * doy: n_years * (perc_boost + doy)]
            try:
                assert not np.isnan(pre_percentile_window).any()
            except:
                nan_indices = np.argwhere(np.isnan(pre_percentile_window))
                raise Exception(f"The array contains NaN values in band {band} for day-of-year {doy} (indexed from 0)! {nan_indices.shape}")
            
            percentile =np.quantile(pre_percentile_window, perc, axis=0) 
            percentiles.append(percentile)
    
        percentiles_band = np.stack(percentiles)
        print('Percentiles Band Shape', percentiles_band.shape)
        assert np.any(percentiles_band > 0)
        all_percentiles.append(percentiles_band)

    res_percentiles = np.concatenate(all_percentiles, axis=1)
    return res_percentiles

def optimised(params, input_zarr_path):
    # PARAMETERS INITIALIZATION
    var = params['var']
    aggregation = params['aggregation']
    ref_start, ref_end = params['reference_period']
    an_start, an_end = params['analysis_period']
    perc_boost = params['perc_boosting_window']
    agg_window = params['aggregation_window']
    half_perc_boost = perc_boost // 2
    n_years = ref_end.year - ref_start.year + 1
    print("NUM YEARS: ", n_years)

    half_agg_window = timedelta(days=agg_window//2)
    half_perc_boost_window = timedelta(days=half_perc_boost)
    
    agg_start = min(ref_start, an_start) - (half_agg_window + half_perc_boost_window)
    agg_end = max(ref_end, an_end) + (half_agg_window + half_perc_boost_window)
    
    pre_percentile_zarr_path = f"{var}_{ref_start.year}_{ref_end.year}_{str(aggregation)}_aggrwindow_{params['aggregation_window']}_percboost_{perc_boost}.zarr"
    
    # LOAD AND AGGREGATE DATA
    data = xr.open_zarr(input_zarr_path)[var]
    aggregated_data = aggregate(data, var, aggregation, agg_window, agg_start, agg_end)
    
    if os.path.exists(pre_percentile_zarr_path):
        print(f"PrePercentile Zarr exists, reading from {pre_percentile_zarr_path}")
        pre_perc_zarr = zarr.open(pre_percentile_zarr_path)
    else:
        print("PrePercentile Zarr not found. Calcluating and saving pre-percentiles")

        # REARRANGE DATA BEFORE CALCULATING THE PERCENTILES
        pre_perc_array = parallel_pre_percentile_arrange(aggregated_data, n_years, ref_start, ref_end, perc_boost)
        pre_perc_zarr = save_pre_percentile_to_zarr(pre_perc_array, pre_percentile_zarr_path, n_years)
        
    # CALCULATE PERCENTILES
    print("Calculating percentiles...")
    exceedances_dir = f"exceedances/var_{var}_ref_{ref_start.year}_{ref_end.year}_{str(aggregation).replace('.','_')}_agg_wind_{agg_window}_perc_boost_{perc_boost}"
    os.makedirs(exceedances_dir, exist_ok=True)
    
    for percentile in params['percentiles']:
        perc_string = str(percentile).replace('.','_')
        threshholds_path = f"{exceedances_dir}/thresholds_{perc_string}"

        if os.path.exists(threshholds_path + '.npy'):
            path = threshholds_path + '.npy'
            print(f"Threshholds file exists. Loading from {path}")
            threshholds = np.load(path)
        else:
            print("Threshholds file not found. Calculating percentiles")
            threshholds = calculate_percentile(perc_boost, n_years, pre_perc_zarr, percentile)
            print("Saving threshholds")
            np.save(threshholds_path, threshholds)
        
        monthly_exceedances_path = f'{exceedances_dir}/monthly_exceedances_{perc_string}.zarr'
        if os.path.exists(monthly_exceedances_path):
            print(f"Monthly exceedances file exists. Loading them from file from {monthly_exceedances_path}")
            monthly_exceedances = xr.open_zarr(monthly_exceedances_path)
        else:
            print("Monthly exceedances file not found. Calculating exceedances")
            aggregated_data = aggregated_data.sel(time=slice(an_start, an_end))
            print('Aggregated Data', aggregated_data)
            
            assert aggregated_data.shape[0] % threshholds.shape[0] == 0, f"{aggregated_data.shape}, {threshholds.shape}"
            aggregated_data_doy = aggregated_data.groupby('time.dayofyear')
            print('Aggregated Data DOY', aggregated_data_doy)
            
            
            threshhold_da = xr.DataArray(threshholds, dims=["dayofyear", 'latitude', 'longitude'])
            exceedances_doy = (aggregated_data_doy > threshhold_da)
            exceedances_doy = exceedances_doy.chunk({"time": -1})
            monthly_exceedances = exceedances_doy.resample(time="1M").sum()
            print(f"Saving monthly exceedances to {monthly_exceedances_path}")
            monthly_exceedances.to_zarr(monthly_exceedances_path)
        
        aggregated_data = aggregated_data.sel(time=slice(an_start, an_end))
        
        aggregated_data_doy = aggregated_data.groupby('time.dayofyear')
        threshhold_da = xr.DataArray(threshholds, dims=["dayofyear", "latitude", "longitude"])
        exceedances_doy = (aggregated_data_doy > threshhold_da)
        exceedances_doy = exceedances_doy.chunk({"time": -1})
        monthly_exceedances = exceedances_doy.resample(time="1M")
        monthly_exceedances.sum(dim="time")
        
        
        # Define a function to calculate the trend slope for a single time series
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
    