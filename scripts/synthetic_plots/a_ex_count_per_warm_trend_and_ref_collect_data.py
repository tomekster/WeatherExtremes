# a) similar to the figure below, beta as a function of sigma for different alpha and REF —> this mainly shows how strongly beta depends on the combination of (alpha, sigma), and of course on REF

from src.synthetic_vienna.temperature_model import TemperatureModel
from src.synthetic_vienna.exceedances import ExceedancesCalculator, compute_trendline_per_year_linear, compute_trendline_per_year_theil_sen
from scripts.load_vienna_data import load_vienna_subset
import numpy as np
import cftime
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--noise", choices=["white", "ar"], default="white",
                    help="white: independent Gaussian noise with std sigma; "
                         "ar: AR(1) noise (fitted rho) with stationary std sigma")
parser.add_argument("--in-base-bootstrap", action="store_true",
                    help="Zhang et al. (2005) bootstrap for analysis years inside the reference period")
parser.add_argument("--boost", type=int, default=1, help="boosting (pooling) window in days, positive odd integer")
parser.add_argument("--trend", choices=["ols", "theil-sen"], default="ols",
                    help="theil-sen stores (ts_slope, ts_intercept, ols_slope, ols_intercept) in <name>_ts.pkl")
args = parser.parse_args()
with_autocorrelation = args.noise == "ar"
suffix = (("_ar" if with_autocorrelation else "") + ("_zb" if args.in_base_bootstrap else "")
          + (f"_boost{args.boost}" if args.boost != 1 else "") + ("_ts" if args.trend == "theil-sen" else ""))

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y, with_autocorrelation=with_autocorrelation)

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
    for warming_trend in np.linspace(0.0, 0.05, 6):
        for sigma in np.linspace(1.5, 7.5, 5): # standard deviation s not variance s^2
            print(f"Warming Trend: {warming_trend}")
            y_synth = model.generate(warming_rate=warming_trend, std=sigma, with_autocorrelation=with_autocorrelation)
            data = vienna_subset.copy(deep=True)
            data['daily_max_2m_temperature'].values = y_synth

            for ref, ref_label in [(first_thirty, '1961-1990'), (second_thirty, '1991-2020')]:
                exceedances = ec.compute_exceedances(
                    data=data, 
                    percentile=percentile, 
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
                    ex_lst[(warming_trend, ref_label, sigma, sample_id) ] = (ts_slope, ts_intercept, slope, intercept)
                else:
                    ex_lst[(warming_trend, ref_label, sigma, sample_id) ] = (slope,intercept)

output_path = f"scripts/synthetic_plots/data/a_ex_count_per_warm_trend_and_ref/a_data{suffix}.pkl"
with open(output_path, "wb") as f:
    pickle.dump(ex_lst, f)
print(f"Saved ex_lst to {output_path}")

