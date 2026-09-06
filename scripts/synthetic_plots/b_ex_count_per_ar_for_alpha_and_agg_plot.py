# b) beta as a function of autocorrelation for different alpha and AGG —> this should show whether beta depends on the autocorrelation and AGG (my assumption is that beta reacts differently on autocorrelation for different values of AGG, but let’s see)

import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.lines import Line2D

data_path = 'scripts/synthetic_plots/data/b_ex_count_per_ar_for_alpha_and_agg/b_data.pkl'
with open(data_path, "rb") as f:
    ex_lst = pickle.load(f)


warming_trends = sorted({k[0] for k in ex_lst.keys()})
aggregation_windows = sorted({k[1] for k in ex_lst.keys()})
ar_rhos = sorted({k[2] for k in ex_lst.keys()})

# Increased figure height for better visibility
plt.figure(figsize=(12, 10))

# Assign a unique color to each warming trend
color_map = plt.get_cmap('tab10')
warming_trend_to_color = {
    w: color_map(i % 10) for i, w in enumerate(warming_trends)
}

# Populate aggregation_window_styles with a style for each aggregation window (for later use)
aggregation_window_linestyles = {
    1: 'solid',
    3: 'dashed',
    5: 'dashdot',
    9: (0, (3, 5, 1, 5)),    # custom dash pattern
    15: (0, (1, 1)),         # densely dotted
    31: 'dotted',
}

n_samples = -1 

# Determine n_samples as the maximum sample_id + 1 found in ex_lst keys
sample_ids = [k[3] for k in ex_lst.keys()]
n_samples = max(sample_ids) + 1 if sample_ids else 0

# Plot the errorbars (no need to collect handles for each line, legend will be handled separately)
for warming_trend in warming_trends:
    for agg in aggregation_windows:
        mean_trend_values = []
        std_trend_values = []
        ar_rho_values = []
        for ar_rho in ar_rhos:
            trends_for_ar_rho = [
                ex_lst[(warming_trend, agg, ar_rho, sample_id)][0]
                for sample_id in range(n_samples)
                if (warming_trend, agg, ar_rho, sample_id) in ex_lst
            ]
            if trends_for_ar_rho:
                mean_trend = np.mean(trends_for_ar_rho)
                std_trend = np.std(trends_for_ar_rho)
                mean_trend_values.append(mean_trend)
                std_trend_values.append(std_trend)
                ar_rho_values.append(ar_rho)
        if mean_trend_values:
            linestyle = aggregation_window_linestyles.get(agg, 'solid')
            color = warming_trend_to_color[warming_trend]
            plt.errorbar(
                ar_rho_values,
                mean_trend_values,
                yerr=std_trend_values,
                marker='o',
                linestyle=linestyle,
                color=color,
                capsize=3
            )

plt.xlabel("rho (autocorrelation coefficient for AR(1))")
plt.ylabel("Slope a (trendline slope, mean ± std across samples)")
plt.title(f"Trendline Slope vs Rho for Various Warming Trends and Aggregation Windows \n(mean ± std across {n_samples} samples)")

# Legend handles: one per warming trend (for color), plus aggregation window style handles (black)
warming_trend_handles = [
    Line2D([0], [0], color=warming_trend_to_color[w], linestyle='solid', marker='o', lw=2, label=f"Warming Trend: {w:.2f}")
    for w in warming_trends
]
style_handles = [
    Line2D([0], [0], color='black', linestyle=aggregation_window_linestyles[agg], lw=2, label=f"AGG={agg} ({aggregation_window_linestyles[agg]})")
    for agg in aggregation_windows
]

# Combine all handles: first the warming trends, then the style explainer lines
all_handles = warming_trend_handles + style_handles

plt.legend(
    handles=all_handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    title="Warming trend (color), AGG window (line style)"
)

plt.tight_layout()
plt.show()
path = 'scripts/synthetic_plots/figures/b_ex_count_per_ar_for_alpha_and_agg.png'
plt.savefig(path)