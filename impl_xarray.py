from datetime import timedelta

import cftime
import numpy as np

from dataloader import ZarrLoader
from enums import AGG
import timeit
from tqdm import tqdm
from zarrmanager import ZarrManager

PARAMS = {
    # 'var': '2m_temperature',
    'var': 'daily_mean_2m_temperature',
    'reference_period': (cftime.DatetimeNoLeap(1960, 1, 1), cftime.DatetimeNoLeap(1965, 12, 31)), # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
    'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
    'aggregation': AGG.MAX,
    'aggregation_window': 5,
    'perc_boosting_window': 5,
    'percentile': 0.99,
}

def main(params):
    # Set dates
    ref_start, ref_end = params['reference_period']
    an_start, an_end = params['analysis_period']
    n_years = ref_end.year - ref_start.year + 1

    half_agg_window = timedelta(days=params['aggregation_window']//2)
    half_perc_boost_window = timedelta(days=params['perc_boosting_window']//2)
    
    agg_start = ref_start - (half_agg_window + half_perc_boost_window)
    agg_end = ref_end + (half_agg_window + half_perc_boost_window)
    
    print("Loading Data...")
    # dl = ZarrLoader('data/daily_mean_2m_temperature_1959_1980.zarr')
    
    # dl = GCSDataLoader()
    # data = dl.load_weatherbench()
    
    # print("Temp2m daily mean from weatherbench")
    # print(data[params['var']].sel(time=slice('1962-08-30', '1962-09-03')).to_numpy()[:,0,:3])
    
    dl = ZarrLoader('data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr')
    data = dl.load()
    
    print("Converting to no-leap calendar")
    data = data.convert_calendar('noleap')

    print("Calculating Aggregations...")
    data = data[params['var']].sel(time=slice(agg_start, agg_end))
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
    
    # 1st Jan = 1 and 31st Dec = 31
    # For calulating percentile boosting we need to prepend the days from the end of the year and append the days from the beginning of the year
    prefix_len = -(ref_start.dayofyr - half_perc_boost - 1) # How many groups from the end should be prepended
    suffix_len = ref_end.dayofyr + half_perc_boost - 365 # How many groups from the beginning should be appended
    
    zm = ZarrManager(params)
    doy_grouped = reference_period_aggregations.groupby('time.dayofyear')
    
    # Process prefixes
    print(f"Processing the prefix ({prefix_len})...")
    for doy, group in tqdm(list(doy_grouped)[-prefix_len:]):
        group = group.to_numpy()
        zm.add(group[:-1])
            
    # Process main
    print("Processing the days of the year...")
    for i, (doy, group) in tqdm(enumerate(doy_grouped)):
        group = group.to_numpy()
        if i < suffix_len:
            group = group[:-1]
        elif i >= len(doy_grouped) - prefix_len:
            group = group[1:]
        zm.add(group)
    
    #Process suffixes
    print(f"Processing the suffix ({suffix_len})...")
    for doy, group in tqdm(list(doy_grouped)[:suffix_len]):
        group = group.to_numpy()
        zm.add(group[1:])

    # Store remaining
    if zm.elements_list:
        zm.store()

    perc_boost = params['perc_boosting_window']
    
    percentile_parts = []
    
    # To reduce the memory footprint and enable easy parallelization we will calculate the percentiles band, by band
    # Since the original weather state array is 721 x 1440, we will split the first dimension into bands 
    dim = 721
    # step = 241
    step=721
    steps = list(range(0, dim, step))
    if steps[-1] < dim:
        steps.append(dim)
    
    bands = list(zip(steps, steps[1:]))
    for band in bands:
        combined = []
        print(f"Retrieving the band {band}...")
        for i in tqdm(range(zm.num_arrays)):
            subarray = zm.zarr_store[f'array_{i}'][:,band[0]:band[1],:]
            combined.append(subarray)
        combined_groups = np.concat(combined)

        percentiles = []
        print(f"Calculating percentiles for the band...")
        for doy in tqdm(range(365)):
            percentile =np.quantile(combined_groups[n_years * doy: (perc_boost + doy) * n_years], params['percentile'], axis=0) 
            percentiles.append(percentile)
    
        percentiles = np.stack(percentiles)
        print("Percentiles calculated!")
        percentile_parts.append(percentiles)
    
    return reference_period_aggregations, combined_groups
    
    # print("Calculating Mask...")
    # an_agg_data = aggregated_data.sel(time=slice(an_start, an_end))
    # doy = an_agg_data.time.dt.dayofyear
    # percentile_array = xr.DataArray(percentiles, dims=["dayofyear"],  coords={"dayofyear": np.arange(1, len(percentiles) + 1)})
    # percentile_for_time = percentile_array.sel(dayofyear=doy)
    # binary_mask = an_agg_data > percentile_for_time
    # print(binary_mask)


if __name__ == '__main__':
    runtime = timeit.timeit(lambda: main(PARAMS), number=1)
    print(f"Total runtime: {runtime:.4f} seconds")