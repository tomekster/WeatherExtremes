import timeit
from impl_xarray import optimised
from enums import AGG
import cftime

if __name__ == '__main__':
    
    params = {
        'var': 'daily_mean_2m_temperature',
        # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'reference_period': (cftime.DatetimeNoLeap(1965, 1, 1), cftime.DatetimeNoLeap(1969, 12, 31)),
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 5,
        'perc_boosting_window': 5,
        'percentile': 0.99,
    }
    
    runtime = timeit.timeit(lambda: optimised(params), number=1)
    print(f"Total runtime: {runtime:.4f} seconds")