#!/usr/bin/env python3
"""
Overlay figure: all 4 sine-wave experiment variants on the same panels.

For each season one figure is produced.  The panel grid is identical to the
per-variant figures (rows = agg_window × slope, cols = variance), but each
panel now shows four overlaid time series — one per variant A/B/C/D — so the
effect of autocorrelation and increasing variance can be compared directly.

Variant colour key
------------------
  A  φ=0,   const σ²   → blue
  B  φ=0.7, const σ²   → red
  C  φ=0,   incr σ²    → green
  D  φ=0.7, incr σ²    → amber

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/vis/visualise_sine_overlay.py

    python src/vis/visualise_sine_overlay.py \\
        --sine-dir data/synthetic/experiments/sine \\
        --out-dir  figures/sine_experiment/overlay
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import zarr

SEASON_NAMES = ["DJF", "MAM", "JJA", "SON"]

# Ordered variant definitions: (directory prefix, short label, color)
VARIANTS = [
    ("A_no_ar_const_var", "A: φ=0, const σ²",    "#2563eb"),  # blue
    ("B_ar0.7_const_var", "B: φ=0.7, const σ²",  "#dc2626"),  # red
    ("C_no_ar_var_trend", "C: φ=0, incr σ²",      "#16a34a"),  # green
    ("D_ar0.7_var_trend", "D: φ=0.7, incr σ²",   "#d97706"),  # amber
]

ZARR_NAME = "synthetic_extremes_sine_wave.zarr"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zarr(path: str) -> dict:
    g = zarr.open_group(path, mode="r")
    return {
        "agg_windows":     g["agg_window"][:].tolist(),
        "perc_boosts":     g["perc_boost"][:].tolist(),
        "slopes":          g["slope"][:],
        "variances":       g["variance"][:],
        "years":           g["year"][:],
        "ref_periods":     g["ref_period"][:].tolist(),
        "annual_counts":   g["annual_counts"][:],    # (n_agg,n_boost,n_sl,n_var,n_ref,n_loc,n_yr)
        "annual_trend":    g["annual_trend"][:],     # (n_agg,n_boost,n_sl,n_var,n_ref,n_loc,2)
        "seasonal_counts": g["seasonal_counts"][:],  # (n_agg,n_boost,n_sl,n_var,n_ref,4,n_loc,n_yr)
        "seasonal_trend":  g["seasonal_trend"][:],   # (n_agg,n_boost,n_sl,n_var,n_ref,4,n_loc,2)
    }


def _get_counts_trend(d: dict, season: str):
    """Return (counts, trend) for location 0 sliced to shape
    (n_agg, n_boost, n_sl, n_var, n_ref, n_yr/2)."""
    if season == "annual":
        counts = d["annual_counts"][:, :, :, :, :, 0, :]
        trend  = d["annual_trend"][:, :, :, :, :, 0, :]
    else:
        si     = SEASON_NAMES.index(season)
        counts = d["seasonal_counts"][:, :, :, :, :, si, 0, :]
        trend  = d["seasonal_trend"][:, :, :, :, :, si, 0, :]
    return counts, trend


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_overlay_figure(datasets: list[tuple], season: str, out_path: str) -> None:
    """
    datasets : list of (label, color, counts, trend, d) tuples, one per variant.
               counts/trend already sliced to (n_agg,n_boost,n_sl,n_var,n_ref,n_yr/2).
    """
    # Use geometry from the first dataset (all variants share the same grid)
    _, _, _, _, d0 = datasets[0]
    agg_windows = d0["agg_windows"]
    perc_boosts = d0["perc_boosts"]
    slopes      = d0["slopes"]
    variances   = d0["variances"]
    years       = d0["years"]
    ref_periods = d0["ref_periods"]

    n_agg  = len(agg_windows)
    n_boost = len(perc_boosts)
    n_sl   = len(slopes)
    n_var  = len(variances)
    n_rows = n_agg * n_boost * n_sl
    n_cols = n_var

    slabel = "Annual" if season == "annual" else season

    # y-axis ceiling
    all_counts = np.concatenate([c for _, _, c, _, _ in datasets], axis=-1)
    y_max = 100 if season != "annual" else int(np.max(all_counts)) + 1

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 1.6 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for ai, agg in enumerate(agg_windows):
        for bi, boost in enumerate(perc_boosts):
            for si in range(n_sl):
                for vi in range(n_var):
                    row = ai * (n_boost * n_sl) + bi * n_sl + si
                    col = vi
                    ax  = axes[row, col]

                    legend_lines = []

                    for label, color, counts, trend, _ in datasets:
                        # Only one ref period per run
                        for ri, (ref_s, ref_e) in enumerate(ref_periods):
                            y         = counts[ai, bi, si, vi, ri, :]
                            slope_det = trend[ai, bi, si, vi, ri, 0]
                            intercept = trend[ai, bi, si, vi, ri, 1]
                            fitted    = slope_det * (years - years[0]) + intercept

                            ax.plot(years, y, linewidth=0.7, color=color, alpha=0.35)
                            line, = ax.plot(years, fitted, linewidth=1.1,
                                            linestyle="--", color=color)
                            legend_lines.append((line, f"{label}: {slope_det:.3f} d/yr"))

                    ax.set_ylim(0, y_max)
                    ax.spines["top"].set_linewidth(1.8 if si == 0 else 0.4)
                    ax.spines["left"].set_linewidth(0.4)

                    if row == 0:
                        ax.set_title(f"σ²={variances[vi]:.2g}°C²",
                                     fontsize=6, fontweight="bold")

                    if si == 0 and vi == n_var // 2:
                        ax.annotate(
                            f"agg_window={agg}d  |  boost_window={boost}doy",
                            xy=(0.5, 1.0), xycoords="axes fraction",
                            xytext=(0, 22), textcoords="offset points",
                            ha="center", va="bottom",
                            fontsize=7, fontweight="bold",
                            annotation_clip=False,
                        )

                    ax.tick_params(labelsize=5)
                    ax.legend(
                        [h for h, _ in legend_lines],
                        [t for _, t in legend_lines],
                        fontsize=4, loc="upper left",
                        handlelength=1.2, borderpad=0.3, labelspacing=0.15,
                    )

                    if row == n_rows - 1:
                        ax.set_xlabel("Year", fontsize=6)
                    if col == 0:
                        ax.set_ylabel(f"Exc. days [{slabel}]", fontsize=6)
                        ax.annotate(
                            f"slope={slopes[si]:.2f}°C/yr",
                            xy=(0, 0.5), xycoords="axes fraction",
                            xytext=(-48, 0), textcoords="offset points",
                            ha="center", va="center", rotation=90,
                            fontsize=6, fontweight="bold",
                            annotation_clip=False,
                        )

    # Figure-level colour legend
    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color, _, _, _ in datasets
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(datasets),
        fontsize=8,
        frameon=True,
        title="Variants  (dashed = OLS trend, faint = annual counts)",
        title_fontsize=8,
    )

    fig.suptitle(
        f"Season: {slabel}  |  All 4 variants overlaid\n"
        f"Rows = agg_window × slope  |  Columns = variance",
        fontsize=10,
        y=1.02,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--sine-dir", default="data/synthetic/experiments/sine",
        help="Directory containing the 4 variant sub-directories "
             "(default: data/synthetic/experiments/sine)",
    )
    p.add_argument(
        "--out-dir", default="figures/sine_experiment/overlay",
        help="Output directory for PDF files "
             "(default: figures/sine_experiment/overlay)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading variants from {args.sine_dir} ...")
    datasets = []
    for dir_name, label, color in VARIANTS:
        path = os.path.join(args.sine_dir, dir_name, ZARR_NAME)
        if not os.path.exists(path):
            raise SystemExit(f"Zarr not found: {path}")
        print(f"  {dir_name}")
        datasets.append((label, color, path))

    # Load all zarrs once, then produce figures per season
    loaded = []
    for label, color, path in datasets:
        d = load_zarr(path)
        loaded.append((label, color, d))

    os.makedirs(args.out_dir, exist_ok=True)

    for season in ["annual"] + SEASON_NAMES:
        seasfx = season.lower()
        out_path = os.path.join(args.out_dir, f"fig_overlay_sine_wave_{seasfx}.pdf")
        print(f"Plotting season: {season}")

        season_data = []
        for label, color, d in loaded:
            counts, trend = _get_counts_trend(d, season)
            season_data.append((label, color, counts, trend, d))

        make_overlay_figure(season_data, season, out_path)

    print(f"\nDone. 5 figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
