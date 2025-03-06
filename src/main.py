from enums import AGG
import cftime
from tqdm import tqdm
import itertools
from experiment import Experiment
from recordtype import recordtype
import os

Params = recordtype('Params',['ref_start', 'ref_end', 'an_start', 
                              'an_end', 'input_zarr_path', 'var', 'aggregation', 'agg_window', 'perc_boosting_window', 'percentile', 'lat_size', 'lon_size', 'seasonality_window', 'output_dir'])

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
                         lon_size=1440,
                         seasonality_window=0,
                         output_dir=os.getenv('OUT_DIR'))

local_run_params = Params(ref_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1991, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         an_end=cftime.DatetimeNoLeap(1991, 12, 31),
                         input_zarr_path='data/michaels_t2_single_arr_mean_zarr_1990_2006.zarr',
                         var='daily_mean_2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721, 
                         lon_size=1440,
                         seasonality_window=0,
                         output_dir=os.getenv('OUT_DIR'))



# https://agupubs.onlinelibrary.wiley.com/doi/epdf/10.1029/2012GL053361

year_start = 1990
year_end = 1991
ref_end = 1991

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
                         lon_size=1440,
                         seasonality_window=1,
                         output_dir=os.getenv('OUT_DIR'))


test_run_params = Params(ref_start=cftime.DatetimeNoLeap(1991, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1992, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1991, 1, 1),
                         an_end=cftime.DatetimeNoLeap(1992, 12, 31),
                         #input_zarr_path='../WeatherExtremes2/data/michaels_t2_single_arr_mean_zarr_1990_2006.zarr',
                         #var='daily_mean_2m_temperature',
                         input_zarr_path='data/preprocessed/weatherbench2_2m_temperature_daily_mean.zarr/',
                         var='2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721, 
                         lon_size=1440,
                         seasonality_window=1,
                         output_dir=os.getenv('OUT_DIR'))

# The actual test we designed with SVEN
with_seasonality = Params(ref_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1989, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         an_end=cftime.DatetimeNoLeap(2019, 12, 31),
                         #input_zarr_path='data/michaels_t2_single_arr_mean_zarr_1990_2006.zarr',
                         #var='daily_mean_2m_temperature',
                         input_zarr_path='data/preprocessed/weatherbench2_2m_temperature_daily_mean.zarr/',
                         var='2m_temperature',
                         aggregation=None,
                         agg_window=None,
                         perc_boosting_window=None,
                         percentile=None,
                         lat_size=721, 
                         lon_size=1440,
                         seasonality_window=None,
                         output_dir=os.getenv('OUT_DIR'))


if __name__ == '__main__':
    #cfg = test_run_params
    cfg = with_seasonality 
    
    aggregations=[AGG.MEAN]
    # agg_windows=[1,7,15]
    # perc_boosting_windows=[7,15]
    agg_windows=[1]
    perc_boosting_windows=[15]
    percentiles=[0.90]
    seasonality_window=[0,1]
    
    cartesian_product = itertools.product(aggregations, agg_windows, perc_boosting_windows, percentiles, seasonality_window)
    for aggregation, agg_window, perc_boosting_window, percentile, seasonality_window in tqdm(list(cartesian_product)):
        cfg.aggregation = aggregation
        cfg.agg_window = agg_window
        cfg.perc_boosting_window = perc_boosting_window
        cfg.percentile = percentile
        cfg.seasonality_window = seasonality_window
        Experiment(cfg).run()
