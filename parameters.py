
import cftime
import itertools
import numpy as np
from enums import AGG
from impl_xarray import optimised

ref_period = [
    (cftime.DatetimeNoLeap(1961, 1, 1), cftime.DatetimeNoLeap(1990, 12, 31)),
    (cftime.DatetimeNoLeap(1971, 1, 1), cftime.DatetimeNoLeap(2000, 12, 31)),
    (cftime.DatetimeNoLeap(1990, 1, 1), cftime.DatetimeNoLeap(2000, 12, 31)),
    (cftime.DatetimeNoLeap(1950, 1, 1), cftime.DatetimeNoLeap(1970, 12, 31)),
    (cftime.DatetimeNoLeap(1990, 1, 1), cftime.DatetimeNoLeap(2020, 12, 31)),
]
percentiles = [0.995, 0.99, 0.95, 0.90]
analysis_period = [(cftime.DatetimeNoLeap(1940, 1, 1), cftime.DatetimeNoLeap(2020, 12, 31))]


# Define the parameter values for each variable
params_2m_temp = {
    'var': ['2m_temperature'],
    'aggregation': [AGG.MEAN, AGG.MAX],
    'aggregation_window': {
        AGG.MEAN: [30, 14, 7, 3],  # in days (1 month, 2 weeks, 1 week, 3 days)
        AGG.MAX: [3, 2, 1],  # in days
    },
    'percentile': percentiles,
    'reference_period': ref_period,
    'analysis_period': analysis_period,
}

params_wind_speed = {
    'var': ['wind_speed'],
    'aggregation': [AGG.MEAN, AGG.MAX],
    'aggregation_window': {
        AGG.MEAN: [7, 3, 0.25, 1/24],  # in days (1 week, 3 days, 6 hours, 1 hour)
        AGG.MAX: [7, 3, 0.25, 1/24],  # same for max
    },
    'percentile': percentiles,
    'reference_period': ref_period,
    'analysis_period': analysis_period,
}

params_precipitation = {
    'var': ['precipitation'],
    'aggregation': [AGG.SUM],
    'aggregation_window': {
        AGG.SUM: [30, 14, 7, 3, 2, 1, 0.25, 1/24],  # in days (1 month, 2 weeks, 1 week, etc.)
    },
    'percentile': percentiles,
    'reference_period': ref_period,
    'analysis_period': analysis_period,
}




def calc_boosting_window(percentile, aggregation_window, reference_period):
    # Calculate how many years are in reference_period
    nr_years = reference_period[1].year - reference_period[0].year + 1
    nr_values = nr_years * aggregation_window

    # Calculate number of values needed: using this code, there will always be at least 10 values above the threshold
    nr_values_needed = int(np.ceil((1 / (1 - percentile)) * 10))

    boosting_window = int(np.ceil(nr_values_needed / nr_values))

    # make sure boosting window is a odd number
    if boosting_window % 2 == 0:
        boosting_window += 1

    return boosting_window



def generate_combinations(variable_params):
    """Generates all combinations of parameters for a given variable."""
    # Unpack the parameters
    var = variable_params['var'][0]
    aggregations = variable_params['aggregation']
    reference_periods = variable_params['reference_period']
    analysis_periods = variable_params['analysis_period']
    percentiles = variable_params['percentile']
    
    all_combinations = []
    
    for agg in aggregations:
        agg_windows = variable_params['aggregation_window'][agg]
        combinations = itertools.product(
            [var], [agg], agg_windows, percentiles, reference_periods, analysis_periods
        )
        for comb in combinations:
            comb_dict = {
                'var': comb[0],
                'aggregation': comb[1],
                'aggregation_window': comb[2],
                'percentile': comb[3],
                'reference_period': comb[4],
                'analysis_period': comb[5],
                'perc_boosting_window': calc_boosting_window(comb[3],comb[2],comb[4])  # calculate boosting window depending on percentile, aggregation window, and reference period
            }
            all_combinations.append(comb_dict)
    
    return all_combinations

# Generate combinations for each variable
combinations_2m_temp = generate_combinations(params_2m_temp)
combinations_wind_speed = generate_combinations(params_wind_speed)
combinations_precipitation = generate_combinations(params_precipitation)

# Combine all into one list
all_combinations = combinations_2m_temp + combinations_wind_speed + combinations_precipitation