# a) similar to the figure below, beta as a function of sigma for different alpha and REF —> this mainly shows how strongly beta depends on the combination of (alpha, sigma), and of course on REF

import matplotlib.pyplot as plt
import pickle
import numpy as np
import argparse
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument("--noise", choices=["white", "ar"], default="white")
parser.add_argument("--in-base-bootstrap", action="store_true")
parser.add_argument("--boost", type=int, default=1)
parser.add_argument("--trend", choices=["ols", "theil-sen"], default="ols")
args = parser.parse_args()
suffix = (("_ar" if args.noise == "ar" else "") + ("_zb" if args.in_base_bootstrap else "")
          + (f"_boost{args.boost}" if args.boost != 1 else "") + ("_ts" if args.trend == "theil-sen" else ""))
noise_label = "AR(1) noise, sigma = stationary std" if args.noise == "ar" else "white noise"
if args.in_base_bootstrap:
    noise_label += ", in-base bootstrap"
noise_label += f"\nBOOST={args.boost}"

data_path = f"scripts/synthetic_plots/data/a_ex_count_per_warm_trend_and_ref/a_data{suffix}.pkl"
with open(data_path, "rb") as f:
    ex_lst = pickle.load(f)

# Infer the sets of warming trends, ref periods, and sigmas from ex_lst keys
warming_trends = sorted({k[0] for k in ex_lst.keys()})
ref_periods = sorted({k[1] for k in ex_lst.keys()})
sigmas = sorted({k[2] for k in ex_lst.keys()})

# PLOTTING
plt.figure(figsize=(8, 6))

# Map REF periods to line styles
ref_period_linestyles = {
    '1961-1990': 'solid',
    '1991-2020': 'dashed',
}

# Assign a unique color to each warming trend
color_map = plt.get_cmap('tab10')
warming_trend_to_color = {
    w: color_map(i % 10) for i, w in enumerate(warming_trends)
}

# Determine n_samples as the maximum sample_id + 1 found in ex_lst keys
sample_ids = [k[3] for k in ex_lst.keys()]
n_samples = max(sample_ids) + 1 if sample_ids else 0

# Plot all the lines, but don't use these for the legend
for warming_trend in warming_trends:
    for ref_period in ref_periods:
        mean_trend_values = []
        std_trend_values = []
        sigma_values = []
        for sigma in sigmas:
            # Collect all sample trend values for this (warming_trend, ref_period, sigma)
            trends_for_sigma = [
                ex_lst[(warming_trend, ref_period, sigma, sample_id)][0]
                for sample_id in range(n_samples)
                if (warming_trend, ref_period, sigma, sample_id) in ex_lst
            ]
            if trends_for_sigma:
                mean_trend = np.mean(trends_for_sigma)
                std_trend = np.std(trends_for_sigma)
                mean_trend_values.append(mean_trend)
                std_trend_values.append(std_trend)
                sigma_values.append(sigma)
        if mean_trend_values:
            linestyle = ref_period_linestyles.get(ref_period, 'solid')
            color = warming_trend_to_color[warming_trend]
            plt.errorbar(
                sigma_values,
                mean_trend_values,
                yerr=std_trend_values,
                marker='o',
                linestyle=linestyle,
                color=color,
                capsize=3
            )

plt.xlabel("Sigma")
plt.ylabel(f"Slope a ({'Theil–Sen' if args.trend == 'theil-sen' else 'OLS'} trendline slope, mean ± std across samples)")
plt.title(f"Trendline Slope vs Sigma for Various Warming Trends and REF Periods\n(mean ± std across {n_samples} samples, {noise_label})", fontsize=10)

# Legend handles: one per warming trend (for color), plus one per ref (black with correct linestyle)
warming_trend_handles = [
    Line2D([0], [0], color=warming_trend_to_color[w], linestyle='solid', marker='o', lw=2, label=f"Warming Trend: {w:.2f}")
    for w in warming_trends
]
style_handles = [
    Line2D([0], [0], color='black', linestyle='solid', lw=2, label="REF 1961-1990 (solid)"),
    Line2D([0], [0], color='black', linestyle='dashed', lw=2, label="REF 1991-2020 (dashed)")
]

all_handles = warming_trend_handles + style_handles

plt.legend(
    handles=all_handles,
    loc="upper right",
    fontsize=9,
    frameon=True,
    title="Warming Trend (color), Reference period (line style)"
)

plt.tight_layout()
path = f'scripts/synthetic_plots/figures/a_ex_count_per_warm_trend_and_ref{suffix}.png'
plt.savefig(path)
plt.show()