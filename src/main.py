from enums import AGG
import cftime
from tqdm import tqdm
import itertools
from collections import namedtuple
from experiment import Experiment
from recordtype import recordtype

Params = recordtype('Params',['ref_start', 'ref_end', 'an_start', 
                              'an_end', 'input_zarr_path', 'var', 'aggregation', 'agg_window',
                              'perc_boosting_window', 'percentile', 'lat_size', 'lon_size','output_dir'])

full_run_params = Params(ref_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1989, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         an_end=cftime.DatetimeNoLeap(2019, 12, 31),
                         input_zarr_path='data/michaels_t2_single_arr_mean_zarr_1959-11-01_2021-02-01.zarr',
                         var='daily_mean_2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721, 
                         lon_size=1440)

local_run_params = Params(ref_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1994, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1995, 1, 1),
                         an_end=cftime.DatetimeNoLeap(1999, 12, 31),
                         input_zarr_path='data/michaels_t2_single_arr_mean_zarr_1990_2006.zarr',
                         var='daily_mean_2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721, 
                         lon_size=1440)

test_run_params = Params(ref_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1990, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         an_end=cftime.DatetimeNoLeap(1990, 12, 31),
                         input_zarr_path='/net/litho/atmosdyn2/svoigt/project_extremes/t2mean/zarr_files/michaels_t2_single_arr_mean_zarr_1989_1992.zarr',
                         var='daily_mean_2m_temperature',
                         aggregation=AGG.MEAN,
                         agg_window=3,
                         perc_boosting_window=3,
                         percentile=0.9,
                         lat_size=721, 
                         lon_size=1440,
                        output_dir='/net/litho/atmosdyn2/svoigt/project_extremes/percentiles/')

# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2012GL053361

year_start = 1990
year_end = 1991
ref_end = 1991

# year_start = 1960
# year_end = 1962
# ref_end = 1961

compare_perkins_2012 = Params(ref_start=cftime.DatetimeNoLeap(year_start, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(ref_end, 12, 31), # Match the paper
                         an_start=cftime.DatetimeNoLeap(year_start, 1, 1),
                         an_end=cftime.DatetimeNoLeap(year_end, 12, 31),
                         input_zarr_path=f'data/preprocessed/weatherbench2_2m_temperature_daily_max.zarr',
                         var='2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721,
                         lon_size=1440)

if __name__ == '__main__':
    cfg = test_run_params
    
    # aggregations=[AGG.MEAN]
    # agg_windows=[1,3,5]
    # perc_boosting_windows=[1,3,5]
    # percentiles=[0.9, 0.95, 0.97, 0.99]
    
    aggregations=[AGG.MAX]
    agg_windows=[1]
    perc_boosting_windows=[15]
    percentiles=[0.90]
    
    cartesian_product = itertools.product(aggregations, agg_windows, perc_boosting_windows, percentiles)
    for aggregation, agg_window, perc_boosting_window, percentile in tqdm(list(cartesian_product)):
        cfg.aggregation = aggregation
        cfg.agg_window = agg_window
        cfg.perc_boosting_window = perc_boosting_window
        cfg.percentile = percentile
        Experiment(cfg).run()
