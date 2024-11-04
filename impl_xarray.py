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


def load_and_aggregate(params, input_zarr_path, agg_start, agg_end):
    print("Loading the Data...")
    data = xr.open_zarr(input_zarr_path)
    print(data)

    print("Converting to no-leap calendar")
    data = data.convert_calendar('noleap')

    print("Calculating Aggregations...")
    agg_data = data[params['var']].sel(time=slice(agg_start, agg_end))
    
    assert agg_data['time'][0] == agg_start, "Missing data at the beginning of the reference period"
    assert agg_data['time'][-1] == agg_end, "Missing data at the end of the reference period"
    
    rolling_data = agg_data.rolling(time=params['aggregation_window'], center=True)

    if params['aggregation'] == AGG.MEAN:
        aggregated_data = rolling_data.mean()
    elif params['aggregation'] == AGG.SUM:
        aggregated_data = rolling_data.sum()
    elif params['aggregation'] == AGG.MIN:
        aggregated_data = rolling_data.min()
    elif params['aggregation'] == AGG.MAX:
        aggregated_data = rolling_data.max()
    else:
        raise Exception("Wrong type of aggregation provided: params['aggregation'] = ", params['aggregation'])
    
    return aggregated_data
    
def parallel_pre_percentile_arrange(agg_data, n_years, ref_start, ref_end, pre_percentile_zarr_path, perc_boost):        
    half_perc_boost = perc_boost // 2
    perc_start = ref_start - timedelta(days=half_perc_boost)
    perc_end = ref_end + timedelta(days=half_perc_boost)
    
    dec31doy = 365
    jan1doy = 1
    prefix_to_append = dec31doy + half_perc_boost - 365 # How many doy_groups from the beginning should be appended
    suffix_to_preppend = -(jan1doy - half_perc_boost - 1) # How many doy_groups from the end should be prepended
    
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
    for day_of_year in range(1,366):
        # Select the group corresponding to this day of year
        day_group = doy_grouped[day_of_year]
        
        array = day_group.data  # Get the underlying dask array

        if day_of_year-1 < prefix_to_append_index:
            suffix_arrays.append(array[1:])
            array = array[:-1]
        elif day_of_year-1 >= suffix_to_preppend_index:
            prefix_arrays.append(array[:-1])
            array = array[1:]
        assert array.shape[0] == n_years, f"Wrong array shape! Shape: {array.shape}"
        main_arrays.append(array)
    arrays = prefix_arrays + main_arrays + suffix_arrays
    
    # Rearrange the grouped by DOY data and save into a zar suitable for faster percentile calculation
    print("Rearranging data into pre-percentile format")
    start = time.time()
    
    batch_size = 100
    
    pre_percentile_zarr_store = zarr.open(pre_percentile_zarr_path, mode='w', shape=(len(arrays) * n_years, 721, 1440), chunks=(batch_size * n_years, 1, 1440), dtype=arrays[0].dtype)
    
    for i in tqdm(list(range(0, len(arrays), batch_size))):
        batch = da.concatenate(arrays[i:i + batch_size])
        start_index = i * n_years
        end_index = start_index + batch.shape[0]
        da.to_zarr(batch, pre_percentile_zarr_store, region=(slice(start_index, end_index), slice(None), slice(None)))
        
    end = time.time()
    print(f"Rearranging: {end - start} seconds")
    
    return pre_percentile_zarr_store

def calculate_percentile(perc_boost, n_years, pre_perc_zarr, perc):

    # To reduce the memory footprint and enable easy parallelization we will calculate the percentiles band, by band
    # Since the original weather state array is 721 x 1440, we will split the first dimension into bands 
    dim = 721
    max_mem = 1000000000 # ~1 GB
    max_floats = max_mem // 4 # 4 bytes per float
    band_size = max_floats // (n_years * (365 + perc_boost - 1) * 1440) 
    band_size = min(721, band_size)
    
    print("BAND_SIZE", band_size)
    
    bands_indices = list(range(0, dim, band_size))
    if bands_indices[-1] < dim:
        bands_indices.append(dim)
    
    bands = list(zip(bands_indices, bands_indices[1:]))
    all_percentiles = []
    for (start, end) in tqdm(bands):
        band = pre_perc_zarr[:,start:end,:]
        assert np.any(band > 0)
        percentiles = []
        for doy in tqdm(range(365), leave=False):
            assert len(band) == n_years * (365 + perc_boost-1), f'Wrong band shape: {band.shape} when expecte length is {n_years * (365 + perc_boost-1)}'
            pre_percentile_window = band[n_years * doy: n_years * (perc_boost + doy)]
            try:
                assert not np.isnan(pre_percentile_window).any()
            except:
                nan_indices = np.argwhere(np.isnan(pre_percentile_window))
                raise Exception(f"The array contains NaN values in band {band} for day {doy} (indexed from 0)! {nan_indices.shape}")
            
            percentile =np.quantile(pre_percentile_window, perc, axis=0) 
            percentiles.append(percentile)
    
        percentiles_band = np.stack(percentiles)
        print('Percentiles Band Shape', percentiles_band.shape)
        assert np.any(percentiles_band > 0)
        all_percentiles.append(percentiles_band)

    res_percentiles = np.concat(all_percentiles, axis=1)
    return res_percentiles

def optimised(params, input_zarr_path):
    # PARAMETERS INITIALIZATION
    var = params['var']
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
    
    pre_percentile_zarr_path = f"{var}_{ref_start.year}_{ref_end.year}_{str(params['aggregation'])}_aggrwindow_{params['aggregation_window']}_percboost_{perc_boost}.zarr"
    
    # LOAD AND AGGREGATE DATA
    aggregated_data = load_and_aggregate(params, input_zarr_path, agg_start, agg_end)
    
    if os.path.exists(pre_percentile_zarr_path):
        print("PrePercentile Zarr exists, reading from disk")
        pre_perc_zarr = zarr.open(pre_percentile_zarr_path)
    else:
        print("PrePercentile Zarr not found. Calcluating and saving pre-percentiles")

        # REARRANGE DATA BEFORE CALCULATING THE PERCENTILES
        pre_perc_zarr = parallel_pre_percentile_arrange(aggregated_data, n_years, ref_start, ref_end, pre_percentile_zarr_path, perc_boost)

    # CALCULATE PERCENTILES
    print("Calculating percentiles...")
    exceedances_dir = f"exceedances/var_{var}_ref_{ref_start.year}_{ref_end.year}_{str(params['aggregation']).replace('.','_')}_agg_wind_{agg_window}_perc_boost_{perc_boost}"
    os.makedirs(exceedances_dir, exist_ok=True)
    
    for percentile in params['percentiles']:
        threshholds_path = f"{exceedances_dir}/thresholds_{str(percentile).replace('.','_')}"

        if os.path.exists(threshholds_path + '.npy'):
            print("Threshholds file exists. Loading threshholds")
            threshholds = np.load(threshholds_path + '.npy')
        else:
            print("Threshholds file not found. Calculating percentiles")
            threshholds = calculate_percentile(perc_boost, n_years, pre_perc_zarr, percentile)
            print("Saving threshholds")
            np.save(threshholds_path)
        
        
        aggregated_data = aggregated_data.sel(time=slice(an_start, an_end))
        
        # assert aggregated_data.shape[0] % threshholds.shape[0] == 0
        # mult = aggregated_data.shape[0] // threshholds.shape[0]
        # expanded_threshold = np.tile(threshholds, (mult, 1, 1))
        # monthly_exceedances = (aggregated_data > expanded_threshold).resample(time="1M").sum(dim="time")
        
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
        
        # threshholds = np.zeros((365,721,1440))
        # generate_temperature_exceedance_mask(aggregated_data, var, an_start, an_end, threshholds, percentile, exceedances_dir)
    