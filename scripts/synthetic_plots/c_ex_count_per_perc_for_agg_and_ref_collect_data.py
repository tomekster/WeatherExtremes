# c) beta as a function of PERC for different AGG and REF —> this goes to the core of our paper and relates to the examples shown with real data

from src.synthetic_vienna.temperature_model import TemperatureModel
from src.synthetic_vienna.exceedances import ExceedancesCalculator, compute_trendline_per_year_linear, compute_trendline_per_year_theil_sen
from scripts.load_vienna_data import load_vienna_subset
import numpy as np
import cftime
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trend", choices=["ols", "theil-sen"], default="ols",
                    help="theil-sen stores (ts_slope, ts_intercept, ols_slope, ols_intercept) in <name>_ts.pkl")
parser.add_argument("--in-base-bootstrap", action="store_true",
                    help="Zhang et al. (2005) bootstrap for analysis years inside the reference period")
parser.add_argument("--boost", type=int, default=1, help="boosting (pooling) window in days, positive odd integer")
args = parser.parse_args()
suffix = (("_zb" if args.in_base_bootstrap else "") + (f"_boost{args.boost}" if args.boost != 1 else "") + ("_ts" if args.trend == "theil-sen" else ""))

VIENNA_STD = 4.564433741768569
VIENNA_AR_INNOVATION_STD = 2.9136834469177537
VIENNA_AR_RHO = 0.8289066046135836
VIENNA_TREND = 0.027

data_path = f'scripts/synthetic_plots/data/c_ex_count_per_perc_for_agg_and_ref/c_data{suffix}.pkl'

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=True)

first_thirty = (cftime.DatetimeNoLeap(1961,1,1), cftime.DatetimeNoLeap(1990,12,31))
second_thirty = (cftime.DatetimeNoLeap(1991,1,1), cftime.DatetimeNoLeap(2020,12,31))
all_sixty = (cftime.DatetimeNoLeap(1961,1,1), cftime.DatetimeNoLeap(2020,12,31))

aggregation_window = 1
boosting_window = args.boost
percentile = 0.95
aggregation_method = "mean"

ex_lst = {}

ec = ExceedancesCalculator()

n_samples = 10
for sample_id in range(n_samples):
    for perc in [0.9, 0.92, 0.95, 0.97, 0.99]:
        for aggregation_window in [1, 5, 15, 31]:
            for ref, ref_name in [(first_thirty, "1961-1990"), (second_thirty, "1991-2020")]:
                print(f"Percentile: {perc}, AGG: {aggregation_window}, ref: {ref_name}")
                y_synth = model.generate(with_autocorrelation=True)
                data = vienna_subset.copy(deep=True)
                data['daily_max_2m_temperature'].values = y_synth
            
                exceedances = ec.compute_exceedances(
                    data=data, 
                    percentile=perc,
                    boosting_window=boosting_window,
                    aggregation_window=aggregation_window,
                    agg_method=aggregation_method,
                    reference_window=ref,
                    analysis_window=all_sixty,
                    in_base_bootstrap=args.in_base_bootstrap
                    )
                slope, intercept = compute_trendline_per_year_linear(exceedances)
                if args.trend == "theil-sen":
                    ts_slope, ts_intercept = compute_trendline_per_year_theil_sen(exceedances)
                    ex_lst[(perc, aggregation_window, ref_name, sample_id) ] = (ts_slope, ts_intercept, slope, intercept)
                else:
                    ex_lst[(perc, aggregation_window, ref_name, sample_id) ] = (slope,intercept)

with open(data_path, "wb") as f:
    pickle.dump(ex_lst, f)
print(f"Saved ex_lst to {data_path}")

