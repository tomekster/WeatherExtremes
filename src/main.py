from optimised_solution.optimised import optimised
from enums import AGG
import cftime
from tqdm import tqdm
import itertools
from collections import namedtuple

Params = namedtuple('Params',['ref_start', 'ref_end', 'an_start', 
                              'an_end', 'zarr_path', 'var', 'aggregations', 'agg_windows',
                              'perc_boosting_windows', 'percentiles'])

full_run_params = Params(ref_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1989, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1960, 1, 1),
                         an_end=cftime.DatetimeNoLeap(2019, 12, 31),
                         zarr_path='data/michaels_t2_single_arr_mean_zarr_1959-11-01_2021-02-01.zarr',
                         var='daily_mean_2m_temperature',
                         aggregations=[AGG.MAX],
                         agg_windows=[1],
                         perc_boosting_windows=[11],
                         percentiles=[0.95])

local_run_params = Params(ref_start=cftime.DatetimeNoLeap(1990, 1, 1),
                         ref_end=cftime.DatetimeNoLeap(1994, 12, 31),
                         an_start=cftime.DatetimeNoLeap(1995, 1, 1),
                         an_end=cftime.DatetimeNoLeap(1999, 12, 31),
                         zarr_path='data/michaels_t2_single_arr_mean_zarr_1990_2006.zarr',
                         var='daily_mean_2m_temperature',
                         aggregations=[AGG.MEAN],
                         agg_windows=[1],
                         perc_boosting_windows=[1],
                         percentiles=[0.95])

if __name__ == '__main__':
    ### Full 1960-1989 reference period and 1990-2019 ananlysis period
    # cfg = full_run_params
    cfg =  local_run_params
    
    cartesian_product = itertools.product(cfg.aggregations, cfg.agg_windows, cfg.perc_boosting_windows)

    for aggregation, agg_window, perc_boosting_window in tqdm(list(cartesian_product)):
        params = {
            'var': cfg.var,
            'reference_period': (cfg.ref_start, cfg.ref_end), 
            'analysis_period': (cfg.an_start, cfg.an_end),
            'aggregation': aggregation,
            'aggregation_window': agg_window,
            'perc_boosting_window': perc_boosting_window,
            'percentiles': cfg.percentiles,
        }
        print(params)
        
        optimised(params, cfg.zarr_path)
