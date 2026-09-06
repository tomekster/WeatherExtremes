# c) beta as a function of PERC for different AGG and REF —> this goes to the core of our paper and relates to the examples shown with real data

from src.synthetic_vienna.temperature_model import TemperatureModel
from src.synthetic_vienna.exceedances import ExceedancesCalculator, compute_trendline_per_year
from scripts.load_vienna_data import load_vienna_subset
import numpy as np
import cftime
import pickle

VIENNA_STD = 4.564433741768569
VIENNA_AR_INNOVATION_STD = 2.9136834469177537
VIENNA_AR_RHO = 0.8289066046135836
VIENNA_TREND = 0.027

data_path = 'scripts/synthetic_plots/data/c_ex_count_per_perc_for_agg_and_ref/c_data.pkl'

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=True)

first_thirty = (cftime.DatetimeNoLeap(1960,1,1), cftime.DatetimeNoLeap(1989,12,31))
second_thirty = (cftime.DatetimeNoLeap(1990,1,1), cftime.DatetimeNoLeap(2019,12,31))
all_sixty = (cftime.DatetimeNoLeap(1960,1,1), cftime.DatetimeNoLeap(2019,12,31))

aggregation_window = 1
boosting_window = 1
percentile = 0.95
aggregation_method = "mean"

ex_lst = {}

ec = ExceedancesCalculator()

n_samples = 10
for sample_id in range(n_samples):
    for perc in [0.9, 0.92, 0.95, 0.97, 0.99]:
        for aggregation_window in [1, 5, 15, 31]:
            for ref, ref_name in [(first_thirty, "1960-1989"), (second_thirty, "1990-2019")]:
                print(f"Percentile: {perc}, AGG: {aggregation_window}, ref: {ref_name}")
                y_synth = model.generate(with_autocorrelation=True)
                data = vienna_subset.copy(deep=True)
                data['daily_max_2m_temperature'].values = y_synth
            
                exceedances = ec.compute_exceedances(
                    data=data, 
                    percentile=percentile, 
                    boosting_window=boosting_window,
                    aggregation_window=aggregation_window,
                    agg_method=aggregation_method,
                    reference_window=first_thirty,
                    analysis_window=all_sixty
                    )
                slope, intercept = compute_trendline_per_year(exceedances)
                ex_lst[(perc, aggregation_window, ref_name, sample_id) ] = (slope,intercept)

with open(data_path, "wb") as f:
    pickle.dump(ex_lst, f)
print(f"Saved ex_lst to {data_path}")

