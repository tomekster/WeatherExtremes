# a) similar to the figure below, beta as a function of sigma for different alpha and REF —> this mainly shows how strongly beta depends on the combination of (alpha, sigma), and of course on REF

from src.synthetic_vienna.temperature_model import TemperatureModel
from src.synthetic_vienna.exceedances import ExceedancesCalculator, compute_trendline_per_year
from scripts.load_vienna_data import load_vienna_subset
import numpy as np
import cftime
import pickle

vienna_subset = load_vienna_subset()
y = vienna_subset['daily_max_2m_temperature'].values

model = TemperatureModel()
model.fit(y)

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
    for warming_trend in np.linspace(0.0, 0.05, 6):
        for sigma in np.linspace(1.5, 7.5, 5): # standard deviation s not variance s^2
            print(f"Warming Trend: {warming_trend}")
            y_synth = model.generate(warming_rate=warming_trend, std=sigma)
            data = vienna_subset.copy(deep=True)
            data['daily_max_2m_temperature'].values = y_synth

            for ref, ref_label in [(first_thirty, '1960-1989'), (second_thirty, '1990-2019')]:
                exceedances = ec.compute_exceedances(
                    data=data, 
                    percentile=percentile, 
                    boosting_window=boosting_window,
                    aggregation_window=aggregation_window,
                    agg_method=aggregation_method,
                    reference_window=ref,
                    analysis_window=all_sixty
                    )
                slope, intercept = compute_trendline_per_year(exceedances)
                ex_lst[(warming_trend, ref_label, sigma, sample_id) ] = (slope,intercept)

output_path = "scripts/synthetic_plots/data/a_ex_count_per_warm_trend_and_ref/a_data.pkl"
with open(output_path, "wb") as f:
    pickle.dump(ex_lst, f)
print(f"Saved ex_lst to {output_path}")

