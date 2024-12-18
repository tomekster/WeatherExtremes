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
    print('Input Zarr:', data)
    print(list(data.data_vars))

    print("Resample to daily values")    
    resampled = data[params['var']].resample(time='1D').max()
    # resampled = resampled.rename({"temperature": "temperature_mean"})

    
    print("Converting to no-leap calendar")
    resampled = resampled.convert_calendar('noleap')
    print("Resampled:", resampled)
    
    agg_data = resampled.sel(time=slice(agg_start, agg_end))
    
    print(agg_data)

    print("Calculating Aggregations...")
    assert agg_data['time'][0] == agg_start, f"Missing data at the beginning of the reference period, {agg_data['time'][0]} != {agg_start}"
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
    
    # Make sure we use the correct dimension names
    if 'lat' in aggregated_data.dims:
        aggregated_data = aggregated_data.rename({"lat": "latitude"})
    if 'lon' in aggregated_data.dims:
        aggregated_data = aggregated_data.rename({"lon": "longitude"})
    assert 'latitude' in aggregated_data.dims
    assert 'longitude' in aggregated_data.dims
    
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
        assert array.shape[0] == n_years, f"Wrong array shape! Shape: {array.shape}, Nyears: {n_years}"
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

    res_percentiles = np.concatenate(all_percentiles, axis=1)
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
    
    identifier = f"{var}_{ref_start.year}_{ref_end.year}_{str(params['aggregation'])}_aggrwindow_{params['aggregation_window']}_percboost_{perc_boost}"
    
    pre_percentile_zarr_path = f"{identifier}.zarr"
    
    # LOAD AND AGGREGATE DATA
    agg_path = f"{identifier}_agg.zarr"
    
    if os.path.exists(agg_path):
        print(f"Aggregation Zarr exists, reading from {agg_path}")
        aggregated_data = xr.open_zarr(agg_path)[var]
        print(aggregated_data)
    else:
        print("Aggregation Zarr not found. Calculating and saving aggregations")
        print("Calculating aggregations... ")
        aggregated_data = load_and_aggregate(params, input_zarr_path, agg_start, agg_end)
        print("Saving aggregations... ")
        aggregated_data.to_zarr(agg_path, mode='w', compute=True)
    
    if os.path.exists(pre_percentile_zarr_path):
        print(f"PrePercentile Zarr exists, reading from {pre_percentile_zarr_path}")
        pre_perc_zarr = zarr.open(pre_percentile_zarr_path)
    else:
        print("PrePercentile Zarr not found. Calcluating and saving pre-percentiles")

        # REARRANGE DATA BEFORE CALCULATING THE PERCENTILES
        pre_perc_zarr = parallel_pre_percentile_arrange(aggregated_data, n_years, ref_start, ref_end, pre_percentile_zarr_path, perc_boost)

    # CALCULATE PERCENTILES
    print("Calculating percentiles...")
    exceedances_dir = f"exceedances/{identifier}"
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
        
        # {var}_{ref_start.year}_{ref_end.year}_{str(params['aggregation'])}_aggrwindow_{params['aggregation_window']}_percboost_{perc_boost}
        trends_path = f'{exceedances_dir}/trends/{perc_string}_trends.zarr'
        if os.path.exists(trends_path):
            print(f"Trends file exists, loading zarr from {trends_path}")
            trends = xr.open_zarr(trends_path)
            print("TRENDS", trends)
        else:
            # Create a new DataArray to store the trends
            trends = xr.DataArray(
                np.nan, 
                dims=["month", "latitude", "longitude"],
                coords={"month": np.arange(1, 13), "latitude": monthly_exceedances.latitude, "longitude": monthly_exceedances.longitude},
                name="trend"
            )
            
            # Group by month
            for month in range(1, 13):
                print(f"Calculating trend for month {month}")
                # Select data for the current month
                monthly_data = monthly_exceedances.sel(time=monthly_exceedances['time.month'] == month)
                
                # Get the number of time points for regression (60 years, assuming one point per year)
                time_index = np.arange(monthly_data.time.size)
                
                # Define a function to calculate the trend (slope) for each lat-lon pair
                def calculate_slope(y):
                    # Perform linear regression on the time dimension
                    if np.all(np.isnan(y)):
                        return np.nan
                    slope, _, _, _, _ = linregress(time_index, y)
                    return slope
                
                # Apply the function across the latitude and longitude dimensions
                trend = xr.apply_ufunc(
                    calculate_slope,
                    monthly_data,
                    input_core_dims=[["time"]],
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                    dask_gufunc_kwargs={"allow_rechunk": True}
                )
                
                # Store the trend for the current month
                print(trend)
                trends.loc[month] = trend['__xarray_dataarray_variable__']
            print("Saving trends to {trends_path}")
            trends.to_zarr(trends_path, mode='w', compute=True)

        trends = trends['trend']
        
        for month in range(1,13):
            plt.figure(figsize=(10, 6))
            trend_month = trends.sel(month=month)
            # trend_month.plot(
            #     x='longitude', y='latitude', cmap='viridis', 
            #     cbar_kwargs={'label': 'Trend'}
            # )
            trend_month.plot(
                x='longitude', y='latitude', cmap='RdBu_r', 
                cbar_kwargs={'label': 'Trend'}
            )
            plt.title(f'Trend Values for Month {month}')
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.savefig(f"{exceedances_dir}/trends/{month}.png")
            plt.show()
        
        # threshholds = np.zeros((365,721,1440))
        # generate_temperature_exceedance_mask(aggregated_data, var, an_start, an_end, threshholds, percentile, exceedances_dir)
    