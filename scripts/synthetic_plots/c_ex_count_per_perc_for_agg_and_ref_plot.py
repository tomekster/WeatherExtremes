# c) beta as a function of PERC for different AGG and REF —> this goes to the core of our paper and relates to the examples shown with real data

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D

data_path = 'scripts/synthetic_plots/data/c_ex_count_per_perc_for_agg_and_ref/c_data.pkl'
with open(data_path, "rb") as f:
    ex_lst = pickle.load(f)

percs = sorted({k[0] for k in ex_lst.keys()})
aggregation_windows = sorted({k[1] for k in ex_lst.keys()})
refs = sorted({k[2] for k in ex_lst.keys()})

# Increased figure height for better visibility
plt.figure(figsize=(12, 10))

# Assign a unique color to each aggregation window
color_map = plt.get_cmap('tab10')

agg_to_color = {
    agg: color_map(i % 10) for i, agg in enumerate(aggregation_windows)
}

ref_linestyles = {
    '1960-1989': 'solid',
    '1990-2019': 'dashed',
}

# Determine n_samples as the maximum sample_id + 1 found in ex_lst keys
sample_ids = [k[3] for k in ex_lst.keys()]
n_samples = max(sample_ids) + 1 if sample_ids else 0

# Plotting the errorbar curves
for ref in refs:
    for agg in aggregation_windows:
        mean_trend_values = []
        std_trend_values = []
        perc_values = []
        for perc in percs:
            # Collect all sample trend values for this (perc, agg, ref, sample_id)
            trends_for_perc = [
                ex_lst[(perc, agg, ref, sample_id)][0]
                for sample_id in range(n_samples)
                if (perc, agg, ref, sample_id) in ex_lst
            ]
            if trends_for_perc:
                mean_trend = np.mean(trends_for_perc)
                std_trend = np.std(trends_for_perc)
                mean_trend_values.append(mean_trend)
                std_trend_values.append(std_trend)
                perc_values.append(perc)
        if mean_trend_values:
            linestyle = ref_linestyles.get(ref, 'solid')
            color = agg_to_color[agg]
            # Don't supply label here, we'll do legend manually
            (errlines, caplines, barlines) = plt.errorbar(
                perc_values,
                mean_trend_values,
                yerr=std_trend_values,
                marker='o',
                linestyle=linestyle,
                color=color,
                capsize=3
            )

plt.xlabel("Percentile")
plt.ylabel("Slope a (trendline slope, mean ± std across samples)")
plt.title(f"Trendline Slope vs Percentile for various Aggregation Windows and Reference Periods  \n(mean ± std across {n_samples} samples)")

# Legend entries: one per aggregation window (color only), plus one per ref (black with correct linestyle)
agg_handles = [
    Line2D([0], [0], color=agg_to_color[agg], lw=2, marker='o', linestyle='solid', label=f"Agg: {agg}")
    for agg in aggregation_windows
]
ref_handles = [
    Line2D([0], [0], color='black', lw=2, linestyle=ref_linestyles[ref], label=f"REF={ref} ({ref_linestyles[ref]})")
    for ref in refs
]

legend_handles = agg_handles + ref_handles

plt.legend(
    handles=legend_handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    title="Aggregation window (color), Reference period (line style)"
)

plt.tight_layout()
plt.show()
path = 'scripts/synthetic_plots/figures/c_ex_count_per_perc_for_agg_and_ref.png'
plt.savefig(path)