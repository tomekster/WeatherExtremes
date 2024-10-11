from datetime import timedelta
import numpy as np

from dataloader import ZarrLoader
from enums import AGG
from tqdm import tqdm
from zarrmanager import ZarrManager

def optimised(params):
    # Set dates
    ref_start, ref_end = params['reference_period']
    an_start, an_end = params['analysis_period']
    n_years = ref_end.year - ref_start.year + 1
    print("NUM YEARS: ", n_years)

    half_agg_window = timedelta(days=params['aggregation_window']//2)
    half_perc_boost_window = timedelta(days=params['perc_boosting_window']//2)
    
    agg_start = ref_start - (half_agg_window + half_perc_boost_window)
    agg_end = ref_end + (half_agg_window + half_perc_boost_window)
    
    print("Loading Data...")
    # Manually Averaged Weatherbench2 data
    # dl = ZarrLoader('data/daily_mean_2m_temperature_1959_1980.zarr')
    
    # Raw Weatherbench2 data
    # dl = GCSDataLoader()
    # data = dl.load_weatherbench()
    
    # print("Temp2m daily mean from weatherbench")
    # print(data[params['var']].sel(time=slice('1962-08-30', '1962-09-03')).to_numpy()[:,0,:3])
    
    
    # Michaels T2MEAN .nc files
    dl = ZarrLoader('data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr')
    data = dl.load()
    
    print("Converting to no-leap calendar")
    data = data.convert_calendar('noleap')

    print("Calculating Aggregations...")
    data = data[params['var']].sel(time=slice(agg_start, agg_end))
    
    assert data['time'][0] == agg_start, "Missing data at the beginning of the reference period"
    assert data['time'][-1] == agg_end, "Missing data at the end of the reference period"
    
    rolling_data = data.rolling(time=params['aggregation_window'], center=True)

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

    reference_period_aggregations = aggregated_data.sel(time=slice(ref_start - half_perc_boost_window, ref_end + half_perc_boost_window))

    half_perc_boost = params['perc_boosting_window'] // 2
    
    # For calulating percentile boosting we need to prepend the days from the end of the year and append the days from the beginning of the year
    prefix_len = -(ref_start.dayofyr - half_perc_boost - 1) # How many groups from the end should be prepended
    suffix_len = ref_end.dayofyr + half_perc_boost - 365 # How many groups from the beginning should be appended
    
    zm = ZarrManager(params)
    doy_grouped = reference_period_aggregations.groupby('time.dayofyear')
    
    # Process prefixes
    print(f"Processing the prefix ({prefix_len})...")
    for doy, group in tqdm(list(doy_grouped)[-prefix_len:]):
        group = group.to_numpy()[:-1]
        assert group.shape[0] == n_years
        zm.add(group)
    
    # Process main
    print("Processing the days of the year...")
    for i, (doy, group) in tqdm(enumerate(doy_grouped)):
        group = group.to_numpy()
        print(i, doy, group.shape)
        if i < suffix_len:
            group = group[:-1]
        elif i >= len(doy_grouped) - prefix_len:
            group = group[1:]
        assert group.shape[0] == n_years
        zm.add(group)
    
    #Process suffixes
    print(f"Processing the suffix ({suffix_len})...")
    for doy, group in tqdm(list(doy_grouped)[:suffix_len]):
        group = group.to_numpy()[1:]
        assert group.shape[0] == n_years
        zm.add(group)

    # Store remaining
    if zm.elements_list:
        zm.store()

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
        band_parts = []
        print(f"Retrieving the band ({start, end})...")
        for i in tqdm(range(zm.num_arrays)):
            band_part = zm.zarr_store[f'array_{i}'][:, start:end, :]
            band_parts.append(band_part)
        band = np.concat(band_parts)
        
        print("BAND_SHAPE:", band.shape)

        percentiles = []
        print(f"Calculating percentiles for the band...")
        for doy in tqdm(range(365)):
            assert len(band) == n_years * (365 + perc_boost-1)
            print(f"from index: {n_years * doy} to index: {n_years * (perc_boost + doy)}, total_len: {len(band)} = {n_years} * (365 + {perc_boost-1})")
            pre_percentile_subset = band[n_years * doy: n_years * (perc_boost + doy)]
            try:
                assert not np.isnan(pre_percentile_subset).any()
            except:
                nan_indices = np.argwhere(np.isnan(pre_percentile_subset))
                print(f"The array contains NaN values in band {band} for day {doy} (indexed from 0)!", nan_indices.shape)
                continue
            
            percentile =np.quantile(pre_percentile_subset, params['percentile'], axis=0) 
            percentiles.append(percentile)
    
        percentiles = np.stack(percentiles)
        print("Percentiles calculated!")
        percentile_parts.append(percentiles)
    
    return reference_period_aggregations, band
    
    # print("Calculating Mask...")
    # an_agg_data = aggregated_data.sel(time=slice(an_start, an_end))
    # doy = an_agg_data.time.dt.dayofyear
    # percentile_array = xr.DataArray(percentiles, dims=["dayofyear"],  coords={"dayofyear": np.arange(1, len(percentiles) + 1)})
    # percentile_for_time = percentile_array.sel(dayofyear=doy)
    # binary_mask = an_agg_data > percentile_for_time
    # print(binary_mask)
    