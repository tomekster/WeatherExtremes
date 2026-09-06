#!/usr/bin/env python3
"""
Ensemble-averaged overlay figure: all 4 sine-wave variants, 10 runs each.

For each season one figure is produced with the same panel grid as the single-run
overlay (rows = agg_window × slope, cols = variance).  Each variant is shown as:
  - A solid mean line (exceedance counts averaged over runs)
  - A shaded 95 % confidence interval around the mean
  - A dashed mean OLS-trend line

The CI uses the t-distribution with (n_runs - 1) degrees of freedom so it is
valid even for small ensembles.

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/vis/visualise_sine_overlay_ensemble.py

    python src/vis/visualise_sine_overlay_ensemble.py \\
        --ensemble-dir data/synthetic/experiments/sine_ensemble \\
        --out-dir      figures/sine_experiment/overlay_ensemble
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import zarr
from scipy import stats

SEASON_NAMES = ["DJF", "MAM", "JJA", "SON"]
ZARR_NAME    = "synthetic_extremes_sine_wave.zarr"

VARIANTS = [
    ("A_no_ar_const_var", "A: φ=0, const σ²",   "#2563eb"),  # blue
    ("B_ar0.7_const_var", "B: φ=0.7, const σ²", "#dc2626"),  # red
    ("C_no_ar_var_trend", "C: φ=0, incr σ²",     "#16a34a"),  # green
    ("D_ar0.7_var_trend", "D: φ=0.7, incr σ²",  "#d97706"),  # amber
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_run(path: str) -> dict:
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


def discover_runs(variant_dir: str) -> list[str]:
    """Return sorted list of per-run zarr paths found inside variant_dir."""
    paths = []
    for entry in sorted(os.listdir(variant_dir)):
        zarr_path = os.path.join(variant_dir, entry, ZARR_NAME)
        if os.path.isdir(os.path.join(variant_dir, entry)) and os.path.exists(zarr_path):
            paths.append(zarr_path)
    return paths


def _get_counts(d: dict, season: str) -> np.ndarray:
    """Slice to (n_agg, n_boost, n_sl, n_var, n_ref, n_yr) for location 0."""
    if season == "annual":
        return d["annual_counts"][:, :, :, :, :, 0, :]
    si = SEASON_NAMES.index(season)
    return d["seasonal_counts"][:, :, :, :, :, si, 0, :]


def _get_trend(d: dict, season: str) -> np.ndarray:
    """Slice to (n_agg, n_boost, n_sl, n_var, n_ref, 2) for location 0."""
    if season == "annual":
        return d["annual_trend"][:, :, :, :, :, 0, :]
    si = SEASON_NAMES.index(season)
    return d["seasonal_trend"][:, :, :, :, :, si, 0, :]


def ensemble_stats(runs_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean and 95 % CI from an ensemble of count arrays.

    Parameters
    ----------
    runs_counts : shape (n_runs, n_agg, n_boost, n_sl, n_var, n_ref, n_yr)

    Returns
    -------
    mean_c  : (n_agg, n_boost, n_sl, n_var, n_ref, n_yr)
    lo_c    : lower CI bound, same shape
    hi_c    : upper CI bound, same shape
    """
    n = runs_counts.shape[0]
    mean_c = runs_counts.mean(axis=0)
    std_c  = runs_counts.std(axis=0, ddof=1)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    sem    = std_c / np.sqrt(n)
    return mean_c, mean_c - t_crit * sem, mean_c + t_crit * sem


def ensemble_trend_mean(runs_trend: np.ndarray) -> np.ndarray:
    """Mean OLS trend across runs. Shape (n_agg,n_boost,n_sl,n_var,n_ref,2)."""
    return runs_trend.mean(axis=0)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_overlay_figure(variant_data: list[tuple], season: str, out_path: str) -> None:
    """
    variant_data : list of (label, color, mean_counts, lo, hi, mean_trend, d0) per variant
    """
    _, _, _, _, _, _, d0 = variant_data[0]
    agg_windows = d0["agg_windows"]
    perc_boosts = d0["perc_boosts"]
    slopes      = d0["slopes"]
    variances   = d0["variances"]
    years       = d0["years"]
    ref_periods = d0["ref_periods"]

    n_agg   = len(agg_windows)
    n_boost = len(perc_boosts)
    n_sl    = len(slopes)
    n_var   = len(variances)
    n_rows  = n_agg * n_boost * n_sl
    n_cols  = n_var

    slabel = "Annual" if season == "annual" else season

    all_hi = np.concatenate([hi for _, _, _, _, hi, _, _ in variant_data], axis=-1)
    y_max  = 100 if season != "annual" else int(np.max(all_hi)) + 5

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

                    for label, color, mean_c, lo_c, hi_c, mean_t, _ in variant_data:
                        for ri in range(len(ref_periods)):
                            y      = mean_c[ai, bi, si, vi, ri, :]
                            lo     = lo_c[ai,   bi, si, vi, ri, :]
                            hi     = hi_c[ai,   bi, si, vi, ri, :]
                            s_det  = mean_t[ai,  bi, si, vi, ri, 0]
                            interc = mean_t[ai,  bi, si, vi, ri, 1]
                            fitted = s_det * (years - years[0]) + interc

                            ax.fill_between(years, lo, hi, alpha=0.15, color=color)
                            ax.plot(years, y, linewidth=0.9, color=color, alpha=0.7)
                            line, = ax.plot(years, fitted, linewidth=1.1,
                                            linestyle="--", color=color)
                            legend_lines.append(
                                (line, f"{label}: {s_det:.3f} d/yr")
                            )

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

    handles = [
        mpatches.Patch(color=color, label=label)
        for label, color, *_ in variant_data
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(variant_data),
        fontsize=8,
        frameon=True,
        title="Variants  (dashed = mean OLS trend | shading = 95 % CI across runs)",
        title_fontsize=8,
    )
    fig.suptitle(
        f"Season: {slabel}  |  All 4 variants overlaid  |  "
        f"n_runs={variant_data[0][2].shape[-1] if False else _n_runs_label}\n"
        "Rows = agg_window × slope  |  Columns = variance",
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

_n_runs_label = "?"   # filled in main() before figures are drawn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ensemble-dir", default="data/synthetic/experiments/sine_ensemble",
        help="Root directory containing per-variant / per-run result zarrs "
             "(default: data/synthetic/experiments/sine_ensemble)",
    )
    p.add_argument(
        "--out-dir", default="figures/sine_experiment/overlay_ensemble",
        help="Output directory for PDF files "
             "(default: figures/sine_experiment/overlay_ensemble)",
    )
    return p.parse_args()


def main() -> None:
    global _n_runs_label
    args = parse_args()

    print(f"Loading ensemble from {args.ensemble_dir} ...")

    # Load all runs for each variant
    variant_runs = []   # list of (label, color, list_of_run_dicts)
    for dir_name, label, color in VARIANTS:
        variant_dir = os.path.join(args.ensemble_dir, dir_name)
        if not os.path.isdir(variant_dir):
            raise SystemExit(f"Variant directory not found: {variant_dir}")
        run_paths = discover_runs(variant_dir)
        if not run_paths:
            raise SystemExit(f"No run zarrs found in {variant_dir}")
        print(f"  {dir_name}: {len(run_paths)} runs")
        run_dicts = [load_run(p) for p in run_paths]
        variant_runs.append((label, color, run_dicts))

    n_runs = len(variant_runs[0][2])
    _n_runs_label = str(n_runs)
    os.makedirs(args.out_dir, exist_ok=True)

    for season in ["annual"] + SEASON_NAMES:
        seasfx   = season.lower()
        out_path = os.path.join(args.out_dir, f"fig_overlay_ensemble_{seasfx}.pdf")
        print(f"Plotting season: {season}")

        variant_data = []
        for label, color, run_dicts in variant_runs:
            # Stack counts and trends across runs
            runs_counts = np.stack([_get_counts(d, season) for d in run_dicts], axis=0)
            runs_trend  = np.stack([_get_trend(d,  season) for d in run_dicts], axis=0)

            mean_c, lo_c, hi_c = ensemble_stats(runs_counts)
            mean_t              = ensemble_trend_mean(runs_trend)
            d0                  = run_dicts[0]

            variant_data.append((label, color, mean_c, lo_c, hi_c, mean_t, d0))

        make_overlay_figure(variant_data, season, out_path)

    print(f"\nDone. 5 figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
