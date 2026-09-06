from scripts.load_vienna_data import load_vienna_subset
from src.synthetic_vienna.exceedances import ExceedancesCalculator, compute_trendline_per_year, count_exceedances_per_year
from src.synthetic_vienna.temperature_model import TemperatureModel
import numpy as np 
import matplotlib.pyplot as plt
import cftime

N = 365 * 5
ec = ExceedancesCalculator()
vienna_subset = load_vienna_subset()

all_days = list(vienna_subset['time'].values)
max2mtemp = vienna_subset['daily_max_2m_temperature'].values

plt.figure(figsize=(12, 6))
plt.plot(max2mtemp[:N], label="Observed max2mtemp", alpha=0.7)

tm  = TemperatureModel()
tm.fit(max2mtemp)

for std in np.linspace(0, 6, 3):
    gen_data = tm.generate(std=std)
    plt.plot(gen_data[:N], label=f"Model fit (synthetic, std={std})", alpha=0.7, linewidth=1)
    plt.xlabel("Time (days since start)")
    plt.ylabel("Daily Max 2m Temp")
    plt.title("Vienna Daily Max 2m Temperature: Observed vs Model Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("scripts/synthetic_plots/vis_inspect_intermediate_results/vienna_model_fit_std.png")


plt.figure(figsize=(12, 6))
plt.plot(max2mtemp, label="Observed max2mtemp", alpha=0.7)
std = 4
for warming_rate in [0.02, 0.04, 0.1, 1]:
    gen_data = tm.generate(std=std, warming_rate=warming_rate)
    plt.plot(gen_data, label=f"Model fit (synthetic, std={std}, warming_rate={warming_rate})", alpha=0.7, linewidth=1)
    plt.xlabel("Time (days since start)")
    plt.ylabel("Daily Max 2m Temp")
    plt.title("Vienna Daily Max 2m Temperature: Observed vs Model Fit")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("scripts/synthetic_plots/vis_inspect_intermediate_results/vienna_model_fit_warming_rate.png")
    

def plot_fitted_trendline(temp_data, path="scripts/synthetic_plots/vis_inspect_intermediate_results/exceedance_counts_and_fitted_trendline.png"):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    axs[0].plot(max2mtemp, label="Raw Data", alpha=0.7, linewidth=1)
    ec = ExceedancesCalculator()
    
    first_thirty = (cftime.DatetimeNoLeap(1960,1,1), cftime.DatetimeNoLeap(1989,12,31))
    second_thirty = (cftime.DatetimeNoLeap(1990,1,1), cftime.DatetimeNoLeap(2019,12,31))
    all_sixty = (cftime.DatetimeNoLeap(1960,1,1), cftime.DatetimeNoLeap(2019,12,31))

    aggregation_window = 1
    boosting_window = 1
    percentile = 0.95
    aggregation_method = "mean"

    data = vienna_subset.copy(deep=True)
    data['daily_max_2m_temperature'].values = temp_data
    ec.compute_exceedances(
        data=data, 
        percentile=percentile, 
        boosting_window=boosting_window,
        aggregation_window=aggregation_window,
        agg_method=aggregation_method,
        reference_window=first_thirty,
        analysis_window=all_sixty
        )
    
    exceedances_per_year = count_exceedances_per_year(ec.exceedances)
    slope, intercept = compute_trendline_per_year(ec.exceedances)

    axs[1].plot(exceedances_per_year, label="Exceedances per year", marker='o')
    # Plot the fitted trendline
    years = np.arange(len(exceedances_per_year))
    trendline = slope * years + intercept
    axs[1].plot(years, trendline, label=f"Trendline (slope={slope:.2f})", color='red', linestyle='--')

    axs[1].set_xlabel("Year")
    axs[1].set_ylabel("Exceedances count")
    axs[1].set_title("Exceedances per Year")
    axs[1].legend()
    plt.savefig(path)

plot_fitted_trendline(max2mtemp)