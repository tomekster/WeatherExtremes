from datetime import timedelta

import cftime
import numpy as np
import xarray as xr

from dataloader import GCSDataLoader
from enums import AGG

params = {
    'var': '2m_temperature',
    # TODO: Should we assume that the start of the analysis period is always the same as the start of the reference period?
    'analysis_period': (cftime.DatetimeNoLeap(1960, 1, 1), cftime.DatetimeNoLeap(1961, 1, 1)),
    'reference_period': (cftime.DatetimeNoLeap(1960, 1, 1), cftime.DatetimeNoLeap(1961, 1, 1)),
    # 'analysis_period': (datetime(1960, 1, 1), datetime(1962, 1, 1)),
    # 'reference_period': (datetime(1960, 1, 1), datetime(1961, 1, 1)),
    'aggregation': AGG.SUM,
    'aggregation_window': 4, #TODO: currently timestep=6h in our data. Will have to either replace the input dataset in data loader or perform preporocessing 
    'perc_boosting_window': 5,
    'percentile': 0.99,
}


print("Loading Data...")
dl = GCSDataLoader()
data = dl.load().convert_calendar('noleap')

an_start, an_end = params['analysis_period']
ref_start, ref_end = params['reference_period']

boosting_prefix = timedelta(days=params['aggregation_window'])
half_boosting = timedelta(days=params['perc_boosting_window'] // 2)

agg_start = min(ref_start - boosting_prefix - half_boosting, an_start - boosting_prefix + -half_boosting) 
agg_end = max(ref_end + half_boosting, an_end + half_boosting)

print("Aggregating...")
data = data[params['var']].sel(time=slice(agg_start, agg_end)) 
rolling_data = data.rolling(time=params['aggregation_window'], center=False)

if params['aggregation'] == AGG.MEAN:
    aggregated_data = rolling_data.mean()
elif params['aggregation'] == AGG.SUM:
    aggregated_data = rolling_data.sum()
elif params['aggregation'] == AGG.MIN:
    aggregated_data = rolling_data.min()
elif params['aggregation'] == AGG.MAX:
    aggregated_data = rolling_data.MAX()
else:
    raise Exception("Wrong type of aggregation provided: params['aggregation'] = ", params['aggregation'])

day_of_year = aggregated_data.time.dt.dayofyear

print("Calculating Percentiles...")
percentiles = []
# For every unique day of a year
for day in np.unique(day_of_year):
    print("Day: ", day)
    # Define the boosting window
    start_day = (day - params['perc_boosting_window']//2 + 365) % 365
    end_day = (day + params['perc_boosting_window']//2) % 365
    
    # Account for the fact that we might need data from two consecutive years if (start date > end date)
    if start_day <= end_day:
        selected_data = data.sel(time=(day_of_year >= start_day) & (day_of_year <= end_day))
    else:
        selection1 = data.sel(time=(day_of_year >= start_day))
        selection2 = data.sel(time=(day_of_year <= end_day))
        selected_data = xr.concat([selection1, selection2], dim="time")

    selected_data = selected_data.chunk({'time': -1})
    percentile = selected_data.quantile(params['percentile'])
    percentiles.append(percentile.compute().item())
print("Precentiles calculated!")
print(percentiles)

percentiles = np.array(percentiles)

print("Calculating Mask...")
an_agg_data = aggregated_data.sel(time=slice(an_start, an_end))
doy = an_agg_data.time.dt.dayofyear
percentile_array = xr.DataArray(percentiles, dims=["dayofyear"],  coords={"dayofyear": np.arange(1, len(percentiles) + 1)})
percentile_for_time = percentile_array.sel(dayofyear=doy)
binary_mask = an_agg_data > percentile_for_time
print(binary_mask)




    
