# b) beta as a function of autocorrelation for different alpha and AGG —> this should show whether beta depends on the autocorrelation and AGG (my assumption is that beta reacts differently on autocorrelation for different values of AGG, but let’s see)

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

data_path = f'scripts/synthetic_plots/data/b_ex_count_per_ar_for_alpha_and_agg/b_data{suffix}.pkl'

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
    for warming_trend in np.linspace(0.0, 0.05, 3):
        for aggregation_window in [1, 5, 15, 31]:
            for ar_rho in np.linspace(0.66, 0.86, 5):  # centred on the fitted Vienna AR(1) coefficient (0.763)
                print(f"Warming Trend: {warming_trend}, AGG: {aggregation_window}, ar_rho: {ar_rho}")
                y_synth = model.generate(warming_rate=warming_trend, ar_coeffs=[ar_rho], with_autocorrelation=True)
                data = vienna_subset.copy(deep=True)
                data['daily_max_2m_temperature'].values = y_synth
            
                exceedances = ec.compute_exceedances(
                    data=data, 
                    percentile=percentile, 
                    boosting_window=boosting_window,
                    aggregation_window=aggregation_window,
                    agg_method=aggregation_method,
                    reference_window=first_thirty,
                    analysis_window=all_sixty,
                    in_base_bootstrap=args.in_base_bootstrap
                    )
                slope, intercept = compute_trendline_per_year_linear(exceedances)
                if args.trend == "theil-sen":
                    ts_slope, ts_intercept = compute_trendline_per_year_theil_sen(exceedances)
                    ex_lst[(warming_trend, aggregation_window, ar_rho, sample_id) ] = (ts_slope, ts_intercept, slope, intercept)
                else:
                    ex_lst[(warming_trend, aggregation_window, ar_rho, sample_id) ] = (slope,intercept)

with open(data_path, "wb") as f:
    pickle.dump(ex_lst, f)
print(f"Saved ex_lst to {data_path}")

