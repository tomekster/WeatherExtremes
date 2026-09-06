import cftime 
from scripts.load_vienna_data import load_vienna_subset
from src.synthetic_vienna.exceedances import compute_exceedances

vienna_subset = load_vienna_subset()

# 1. Set periods and constants
# reference_start = vienna_subset['time'].values[0]
reference_start = cftime.DatetimeNoLeap(1960,1,1)
reference_end = cftime.DatetimeNoLeap(1989,12,31)
analysis_start = cftime.DatetimeNoLeap(1990,1,1)
# reference_start = vienna_subset['time'].values[-1]
analysis_end = cftime.DatetimeNoLeap(2019,12,31)

aggregation_window = 3
boosting_window = 3
percentile = 0.95
aggregation_method = "mean"

exceedances = compute_exceedances(
    data=vienna_subset, 
    percentile=percentile, 
    boosting_window=boosting_window,
    aggregation_window=aggregation_window,
    agg_method=aggregation_method,
    reference_window=(reference_start, reference_end),
    analysis_window=(analysis_start, analysis_end)
    )

print(exceedances)
