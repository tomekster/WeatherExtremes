from datetime import timedelta
import numpy as np
from enums import AGG
from tqdm import tqdm
from dask.distributed import Client
import time
import dask.array as da

import zarr
import xarray as xr
import dask.array as da

def load_and_aggregate(params, input_zarr_path, agg_start, agg_end, perc_start, perc_end):
    print("Loading the Data...")
    data = xr.open_zarr(input_zarr_path)

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

    reference_period_aggregations = aggregated_data.sel(time=slice(perc_start, perc_end))
    return reference_period_aggregations
    
def parallel_pre_percentile_arrange(doy_grouped, n_years, pre_percentile_zarr_path, prefix_to_append_index, suffix_to_preppend_index):        
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
        assert array.shape[0] == n_years
        main_arrays.append(array)
    arrays = prefix_arrays + main_arrays + suffix_arrays
    
    
    print("Rearranging data into pre-percentile format")
    start = time.time()
    combined = da.concatenate(arrays).compute()
    pre_percentile_zarr_store = zarr.open(pre_percentile_zarr_path, mode='w', shape=combined.shape, chunks=(365 * n_years, 1, 1440), dtype=combined.dtype)
    pre_percentile_zarr_store[:] = combined
    end = time.time()
    print(f"Rearranging: {end - start} seconds")
    
    return pre_percentile_zarr_store

def optimised(params):
    # PARAMETERS INITIALIZATION
    ref_start, ref_end = params['reference_period']
    an_start, an_end = params['analysis_period'] # TODO
    n_years = ref_end.year - ref_start.year + 1
    print("NUM YEARS: ", n_years)

    half_agg_window = timedelta(days=params['aggregation_window']//2)
    half_perc_boost_window = timedelta(days=params['perc_boosting_window']//2)
    
    half_perc_boost_int = params['perc_boosting_window'] // 2
    
    agg_start = ref_start - (half_agg_window + half_perc_boost_window)
    agg_end = ref_end + (half_agg_window + half_perc_boost_window)
    
    perc_start = ref_start - half_perc_boost_window
    perc_end = ref_end + half_perc_boost_window
    
    input_zarr_path = 'data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr'
    
    prefix_to_append = ref_end.dayofyr + half_perc_boost_int - 365 # How many doy_groups from the beginning should be appended
    suffix_to_preppend = -(ref_start.dayofyr - half_perc_boost_int - 1) # How many doy_groups from the end should be prepended
    
    assert prefix_to_append > 0
    assert suffix_to_preppend > 0
    
    pre_percentile_zarr_path = f"{params['var']}_{ref_start}_{ref_end}_{str(params['aggregation'])}_aggrwindow_{params['aggregation_window']}_percboost_{params['perc_boosting_window']}_perc_{params['percentile']}.zarr"
    
    # LOAD AND AGGREGATE DATA
    start = time.time()
    agg_data = load_and_aggregate(params, input_zarr_path, agg_start, agg_end, perc_start, perc_end)
    end = time.time()
    print(f"Loading and aggregating: {end - start} seconds")
    
    # zm = ZarrManager(params, output_zarr)
    doy_grouped = agg_data.groupby('time.dayofyear')

    # REARRANGE DATA BEFORE CALCULATING THE PERCENTILES
    pre_perc_zarr = parallel_pre_percentile_arrange(doy_grouped, n_years, pre_percentile_zarr_path, prefix_to_append_index=prefix_to_append, suffix_to_preppend_index=len(doy_grouped)-suffix_to_preppend)

    ###
    # Percentile Calculation
    ###
    perc_boost = params['perc_boosting_window']
    percentile_parts = []

    # To reduce the memory footprint and enable easy parallelization we will calculate the percentiles band, by band
    # Since the original weather state array is 721 x 1440, we will split the first dimension into bands 
    dim = 721
    max_mem = 16000000000 # ~16 GB
    max_floats = max_mem // 4 # 4 bytes per float
    band_size = max_floats // (n_years * (365 + perc_boost - 1) * 1440) 
    band_size = min(721, band_size)
    
    print("BAND_SIZE", band_size)
    
    bands_indices = list(range(0, dim, band_size))
    if bands_indices[-1] < dim:
        bands_indices.append(dim)
    
    bands = list(zip(bands_indices, bands_indices[1:]))
    for (start, end) in bands:
        print(f"Retrieving the band ({start, end})...")
        band = pre_perc_zarr[:,start:end,:]
        print("BAND_SHAPE:", band.shape)
        percentiles = []
        print(f"Calculating percentiles for the band...")
        for doy in tqdm(range(365)):
            assert len(band) == n_years * (365 + perc_boost-1)
            pre_percentile_subset = band[n_years * doy: n_years * (perc_boost + doy)]
            try:
                assert not np.isnan(pre_percentile_subset).any()
            except:
                nan_indices = np.argwhere(np.isnan(pre_percentile_subset))
                raise Exception(f"The array contains NaN values in band {band} for day {doy} (indexed from 0)! {nan_indices.shape}")
            
            percentile =np.quantile(pre_percentile_subset, params['percentile'], axis=0) 
            percentiles.append(percentile)
    
        percentiles = np.stack(percentiles)
        print("Percentiles calculated!")
        percentile_parts.append(percentiles)
    
    return agg_data, band
    
    # print("Calculating Mask...")
    # an_agg_data = aggregated_data.sel(time=slice(an_start, an_end))
    # doy = an_agg_data.time.dt.dayofyear
    # percentile_array = xr.DataArray(percentiles, dims=["dayofyear"],  coords={"dayofyear": np.arange(1, len(percentiles) + 1)})
    # percentile_for_time = percentile_array.sel(dayofyear=doy)
    # binary_mask = an_agg_data > percentile_for_time
    # print(binary_mask)
    