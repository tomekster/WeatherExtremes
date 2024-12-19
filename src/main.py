from impl_xarray import optimised
from enums import AGG
import cftime
from tqdm import tqdm
import itertools

if __name__ == '__main__':
    # zarr_path = 'data/daily_max_wind_speed_1959_2021.zarr'
    # zarr_path = '/ERA5/weatherbench2_original'
    
    # Fixed across runs
    # ref_per_start = cftime.DatetimeNoLeap(1995, 1, 1) # Must be 1st Jan
    # ref_per_end = cftime.DatetimeNoLeap(2004, 12, 31) # Must be 31st Dec
    # an_start = cftime.DatetimeNoLeap(2005, 1, 1) # Must be 1st Jan
    # an_end = cftime.DatetimeNoLeap(2014, 12, 31) # Must be 31st Dec

    ref_per_start = cftime.DatetimeNoLeap(1960, 1, 1) # Must be 1st Jan
    ref_per_end = cftime.DatetimeNoLeap(1989, 12, 31) # Must be 31st Dec
    an_start = cftime.DatetimeNoLeap(1960, 1, 1) # Must be 1st Jan
    an_end = cftime.DatetimeNoLeap(2019, 12, 31) # Must be 31st Dec
    
    # Parameter configurations
    zarr_path = 'data/michaels_t2_single_arr_mean_zarr_1959-11-01_2021-02-01.zarr'
    var = 'daily_mean_2m_temperature'
    aggregations = [AGG.MAX]
    agg_windows = [1]
    perc_boosting_windows = [11]
    percentiles = [0.90]
    
    # zarr_path = '/weather/data/10m_wind_speed.zarr'
    # var = '10m_wind_speed'
    # aggregations = [AGG.MAX]
    # agg_windows = [1]
    # perc_boosting_windows = [5]
    # percentiles = [0.99]
    
    cartesian_product = itertools.product(aggregations, agg_windows, perc_boosting_windows)

    for aggregation, agg_window, perc_boosting_window in tqdm(list(cartesian_product)):
        params = {
            'var': var,
            'reference_period': (ref_per_start, ref_per_end), 
            'analysis_period': (an_start, an_end),
            'aggregation': aggregation,
            'aggregation_window': agg_window,
            'perc_boosting_window': perc_boosting_window,
            'percentiles': percentiles,
        }
        print(params)
        
        optimised(params, zarr_path)
