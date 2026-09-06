import cftime 
from scripts.load_vienna_data import load_vienna_subset
from src.synthetic_vienna.exceedances import ExceedancesCalculator
import matplotlib.pyplot as plt
import numpy as np

# Visualise thresholds
reference_start = cftime.DatetimeNoLeap(1960,1,1)
reference_end = cftime.DatetimeNoLeap(1989,12,31)
analysis_start = cftime.DatetimeNoLeap(1990,1,1)
analysis_end = cftime.DatetimeNoLeap(2019,12,31)

aggregation_window = 1
boosting_window = 1
aggregation_method = "mean"

ec = ExceedancesCalculator()
vienna_subset = load_vienna_subset()

all_days = list(vienna_subset['time'].values)
max2mtemp = vienna_subset['daily_max_2m_temperature'].values
max2mtemp_reshaped = max2mtemp.reshape(-1, 365)

dates = np.array([str(d)[:10] for d in all_days])

# Use gridspec to control subplot layout, with an extra column for per-year totals

import matplotlib.gridspec as gridspec

# -- Setup initial grid with 4 rows, 2 columns --
fig = plt.figure(figsize=(15, 16))
gs = gridspec.GridSpec(
    4, 2, 
    width_ratios=[12, 2],  # wide main plots, skinnier side bar
    height_ratios=[1, 0.2, 2, 0.7],  # main exceeded grid slightly less tall
    hspace=0.25, wspace=0.35
)

# First row: main time series plot (ax1) spans both columns
ax1 = fig.add_subplot(gs[0, 0:2])
ax1.plot(dates, max2mtemp, label="Max daily 2m temperature")
tick_indices = np.arange(0, len(dates), 365)
ax1.set_xticks(tick_indices)
ax1.set_xticklabels([dates[i] for i in tick_indices], rotation=45, ha='right')
ax1.set_xlabel("Date")
ax1.set_ylabel("Daily Max 2m Temp")
ax1.set_title("Daily Max 2m Temp Data for Vienna (All Years)")
ax1.legend()

# Second row: aggregations
ax2 = fig.add_subplot(gs[1, 0])
ec.compute_exceedances(
    data=vienna_subset, 
    percentile=0.95, 
    boosting_window=boosting_window,
    # aggregation_window=aggregation_window,
    aggregation_window=5,
    agg_method=aggregation_method,
    reference_window=(reference_start, reference_end),
    analysis_window=(analysis_start, analysis_end)
)
dates = np.array([str(d)[:10] for d in all_days])
dates_reshaped = dates.reshape(-1, 365)

for i in range(3):
    ax2.plot(dates_reshaped[i], max2mtemp_reshaped[i], label=f"RawData {i}")
    ax2.plot(dates_reshaped[i], ec.reference_agg[i], label=f"Aggregated {i}")
tick_indices = np.arange(0, len(dates_reshaped[0]), 365)
ax2.set_xticks(tick_indices)
ax2.set_xticklabels([dates[i] for i in tick_indices], rotation=45, ha='right')
ax2.set_xlabel("Date")
ax2.set_ylabel("Daily Max 2m Temp")
ax2.set_title("Aggregations for Vienna")
ax2.legend()

# Third row: reference period per-year plot
ax3 = fig.add_subplot(gs[2, 0])
dates = np.array([str(d)[5:10] for d in all_days])
dates_reshaped = dates.reshape(-1, 365)
for i in range(30):
    dates_365 = dates_reshaped[0]
    max2mtemp_365 = max2mtemp_reshaped[i]
    ax3.plot(dates_365, max2mtemp_365, label=f"Year {i} Max daily 2m temp", color='tab:orange', linewidth=0.7)
tick_indices_2 = np.arange(0, len(dates[0]), 30)
ax3.set_xticks(tick_indices_2)
ax3.set_xticklabels([dates[0][i] for i in tick_indices_2], rotation=45, ha='right')
ax3.set_xlabel("Date")
ax3.set_ylabel("Daily Max 2m Temp")
ax3.set_title("Reference Period")
ax3.legend()

for percentile in [0.25, 0.5, 0.75, 0.95, 1]:
    ec.compute_exceedances(
        data=vienna_subset, 
        percentile=percentile, 
        boosting_window=boosting_window,
        aggregation_window=aggregation_window,
        agg_method=aggregation_method,
        reference_window=(reference_start, reference_end),
        analysis_window=(analysis_start, analysis_end)
        )
    ax3.plot(dates_reshaped[0], ec.thresholds, label=f"Threshold (p={percentile})", linewidth=2.5)
    ax3.legend()

# Run full detect for grid visualization
ec.compute_exceedances(
    data=vienna_subset, 
    percentile=0.95, 
    # boosting_window=boosting_window,
    boosting_window=3,
    aggregation_window=aggregation_window,
    agg_method=aggregation_method,
    reference_window=(reference_start, reference_end),
    # analysis_window=(analysis_start, analysis_end)
    analysis_window=(reference_start, reference_end)
)

# Collect exceedance totals
exceedances_per_day = ec.exceedances.sum(axis=0)   # shape (365,)
exceedances_per_year = ec.exceedances.sum(axis=1)  # shape (n_years,)

plt.tight_layout()
plt.savefig("scripts/synthetic_plots/vis_inspect_intermediate_results/vienna_data.png")
plt.show()
