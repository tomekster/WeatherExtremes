# d) beta as a function of POOL for different AGG and PERC —> should show that the sensitivity to POOL is not that dramatic

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D

from src.core import aggregation

data_path = 'scripts/synthetic_plots/data/d_ex_count_per_pool_for_agg_and_perc/d_data.pkl'
with open(data_path, "rb") as f:
    ex_lst = pickle.load(f)

boosts = sorted({k[0] for k in ex_lst.keys()})
aggregation_windows = sorted({k[1] for k in ex_lst.keys()})
percs = sorted({k[2] for k in ex_lst.keys()})

# Increased figure height for better visibility
plt.figure(figsize=(12, 10))

# Assign a unique color to each aggregation window
color_map = plt.get_cmap('tab10')

perc_to_color = {
    perc: color_map(i % 10) for i, perc in enumerate(percs)
}

# Define unique line styles for each aggregation window
# Define a list of line styles to cycle through for aggregation windows
line_styles_list = ['solid', 'dashed', 'dashdot', 'dotted', (0, (3, 1, 1, 1)), (0, (5, 2)), (0, (1, 1)), (0, (5, 5))]
agg_linestyles = {agg: line_styles_list[i % len(line_styles_list)] for i, agg in enumerate(aggregation_windows)}

# Determine n_samples as the maximum sample_id + 1 found in ex_lst keys
sample_ids = [k[3] for k in ex_lst.keys()]
n_samples = max(sample_ids) + 1 if sample_ids else 0

# Plotting the errorbar curves
for agg in aggregation_windows:
    for perc in percs:
        mean_trend_values = []
        std_trend_values = []
        boost_values = []
        for boost in boosts:
            # Collect all sample trend values for this (perc, agg, ref, sample_id)
            trends_for_boost = [
                ex_lst[(boost, agg, perc, sample_id)][0]
                for sample_id in range(n_samples)
                if (boost, agg, perc, sample_id) in ex_lst
            ]
            if trends_for_boost:
                mean_trend = np.mean(trends_for_boost)
                std_trend = np.std(trends_for_boost)
                mean_trend_values.append(mean_trend)
                std_trend_values.append(std_trend)
                boost_values.append(boost)
        if mean_trend_values:
            linestyle = agg_linestyles.get(agg, 'solid')
            color = perc_to_color[perc]
            # Don't supply label here, we'll do legend manually
            (errlines, caplines, barlines) = plt.errorbar(
                boost_values,
                mean_trend_values,
                yerr=std_trend_values,
                marker='o',
                linestyle=linestyle,
                color=color,
                capsize=3
            )

plt.xlabel("Percentile")
plt.ylabel("Slope a (trendline slope, mean ± std across samples)")
plt.title(f"Trendline Slope vs POOL (boosting window) for various Aggregation Windows and Percentiles  \n(mean ± std across {n_samples} samples)")

# Legend entries: one per aggregation window (color only), plus one per ref (black with correct linestyle)
agg_handles = [
    Line2D([0], [0], color=perc_to_color[perc], lw=2, marker='o', linestyle='solid', label=f"PERC: {perc}")
    for perc in percs
]
ref_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle=agg_linestyles[agg], label=f"AGG={agg} ({agg_linestyles[agg]})")
    for agg in aggregation_windows
]

legend_handles = agg_handles + ref_handles

plt.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    title="Percentile (color), Aggregation Window (line style)"
)

plt.tight_layout()
plt.show()
path = 'scripts/synthetic_plots/figures/d_ex_count_per_pool_for_agg_and_perc.png'
plt.savefig(path)