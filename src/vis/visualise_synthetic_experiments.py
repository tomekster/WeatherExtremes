#!/usr/bin/env python3
"""
Visualise synthetic extremes experiments.

Takes a single unified zarr produced by run_synthetic_extremes.py and writes
one figure per location.  Each figure is a grid of time-series panels — one
panel per (agg_window, perc_boost, slope, variance) combination — showing the
annual or seasonal exceedance count over time with an OLS trendline and a
shaded reference period.

Panel grid layout
-----------------
  Rows    : agg_window (outer) × slope (inner)
  Columns : perc_boost (outer) × variance (inner)

  Thicker spines mark the boundaries between agg-window groups (row) and
  boost-window groups (column) to aid visual grouping.

Output files
------------
  <out_dir>/fig_exceedances_<location>_<season>.pdf   — one per location

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/vis/visualise_synthetic_experiments.py \\
        data/synthetic/experiments/synthetic_extremes_zurich.zarr

    python src/vis/visualise_synthetic_experiments.py \\
        data/synthetic/experiments/synthetic_extremes_zurich.zarr --season DJF

    python src/vis/visualise_synthetic_experiments.py \\
        data/synthetic/experiments/synthetic_extremes_zurich.zarr \\
        --season MAM --out-dir figures/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

SEASON_NAMES = ["DJF", "MAM", "JJA", "SON"]

# One entry per reference period; extend if more are needed
_TREND_COLORS = ["#dc2626", "#92400e", "#7c3aed", "#065f46", "#b45309"]  # red, brown, …
_COUNT_COLORS = ["#2563eb", "#0369a1", "#0891b2", "#0d9488", "#4f46e5"]  # blues
_SPAN_COLORS  = ["green",   "orange",  "purple",  "teal",    "goldenrod"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_zarr(path: str) -> dict:
    """Load the unified experiment zarr into a flat dict of arrays."""
    g = zarr.open_group(path, mode="r")
    return {
        "agg_windows":     g["agg_window"][:].tolist(),        # [int, ...]
        "perc_boosts":     g["perc_boost"][:].tolist(),        # [int, ...]
        "slopes":          g["slope"][:],                      # (n_sl,)
        "variances":       g["variance"][:],                   # (n_var,)
        "locations":       [b.decode() for b in g["location"][:]],
        "years":           g["year"][:],                       # (n_yr,)
        "ref_periods":     g["ref_period"][:].tolist(),         # [[start, end], ...]
        "annual_counts":   g["annual_counts"][:],              # (n_agg, n_boost, n_sl, n_var, n_ref, n_loc, n_yr)
        "annual_trend":    g["annual_trend"][:],               # (n_agg, n_boost, n_sl, n_var, n_ref, n_loc, 2)
        "seasonal_counts": g["seasonal_counts"][:],            # (n_agg, n_boost, n_sl, n_var, n_ref, 4, n_loc, n_yr)
        "seasonal_trend":  g["seasonal_trend"][:],             # (n_agg, n_boost, n_sl, n_var, n_ref, 4, n_loc, 2)
        "attrs":           dict(g.attrs),
    }


def _slice_location(d: dict, loc_idx: int, season: str):
    """
    Return (counts, trend) for one location and season.

    Both arrays have shape:
      counts : (n_agg, n_boost, n_sl, n_var, n_ref, n_yr)
      trend  : (n_agg, n_boost, n_sl, n_var, n_ref, 2)
    """
    if season == "annual":
        counts = d["annual_counts"][:, :, :, :, :, loc_idx, :]
        trend  = d["annual_trend"][:, :, :, :, :, loc_idx, :]
    else:
        si = SEASON_NAMES.index(season)
        counts = d["seasonal_counts"][:, :, :, :, :, si, loc_idx, :]
        trend  = d["seasonal_trend"][:, :, :, :, :, si, loc_idx, :]
    return counts, trend


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def fig_location(inupt_zarr_path: str, d: dict, loc_idx: int, season: str, out_path: str) -> None:
    """
    Produce one figure for a single location.

    Panel grid:
      rows = agg_window (outer) × slope (inner)
      cols = perc_boost (outer) × variance (inner)

    Each panel: exceedance count time series + OLS trendline +
                shaded reference period.
    """
    location    = d["locations"][loc_idx]
    agg_windows = d["agg_windows"]
    perc_boosts = d["perc_boosts"]
    ref_periods = d["ref_periods"]          # [[start, end], ...]
    slopes      = d["slopes"]
    variances   = d["variances"]
    years       = d["years"]

    n_agg   = len(agg_windows)
    n_boost = len(perc_boosts)
    n_ref   = len(ref_periods)
    n_sl    = len(slopes)
    n_var   = len(variances)
    n_rows  = n_agg * n_boost * n_sl
    n_cols  = n_var

    slabel = "Annual" if season == "annual" else season

    counts, trend = _slice_location(d, loc_idx, season)
    # counts : (n_agg, n_boost, n_sl, n_var, n_ref, n_yr)
    # trend  : (n_agg, n_boost, n_sl, n_var, n_ref, 2)

    if season == "annual":
        y_max = int(np.max(counts)) + 1
    else:
        y_max = 90

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 1.4 * n_rows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)

    for ai, agg in enumerate(agg_windows):
        for bi, boost in enumerate(perc_boosts):
            for si in range(n_sl):
                for vi in range(n_var):
                    # INSERT_YOUR_CODE
                    print(f"ai={ai}, agg={agg}, bi={bi}, boost={boost}, si={si}, vi={vi}")
             
                    row = ai * (n_boost * n_sl) + bi * n_sl + si
                    col = vi
                    ax  = axes[row, col]

                    # Overlay one series per reference period
                    for ri, (ref_s, ref_e) in enumerate(ref_periods):
                        tc = _TREND_COLORS[ri % len(_TREND_COLORS)]
                        cc = _COUNT_COLORS[ri % len(_COUNT_COLORS)]
                        sc = _SPAN_COLORS[ri  % len(_SPAN_COLORS)]

                        y         = counts[ai, bi, si, vi, ri, :]
                        slope_det = trend[ai, bi, si, vi, ri, 0]
                        intercept = trend[ai, bi, si, vi, ri, 1]
                        fitted    = slope_det * (years - years[0]) + intercept

                        ax.plot(years, y, linewidth=1.0, color=cc, alpha=0.7)
                        ax.plot(years, fitted, linewidth=0.9, linestyle="--",
                                color=tc,
                                label=f"{ref_s}–{ref_e}: {slope_det:.3f} d/yr")
                        ax.axvspan(ref_s, ref_e, alpha=0.08, color=sc)

                    ax.set_ylim(0, y_max)

                    ax.spines["top"].set_linewidth(1.8 if si == 0 else 0.4)
                    ax.spines["left"].set_linewidth(0.4)

                    if row == 0:
                        ax.set_title(
                            f"σ²={variances[vi]:.2g}°C²",
                            fontsize=6, fontweight="bold",
                        )

                    # (agg, boost) group header
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
                    ax.legend(fontsize=5, loc="upper left", handlelength=1,
                              borderpad=0.3, labelspacing=0.2)

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

    fig.suptitle(
        f"Location: {location}  |  Season: {slabel}\n"
        f"ref_periods: {ref_periods}  |  agg_window: {agg_windows}  |  boost_window: {perc_boosts}\n"
        f"Input Zarr: {inupt_zarr_path}\n"
        "Row groups = ref_period × agg_window × boost_window × slope  |  "
        "Columns = variance  |  "
        "dashed = OLS trend  |  green shading = reference period",
        fontsize=10,
    )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "zarr",
        help="Path to the unified synthetic_extremes zarr "
             "(output of run_synthetic_extremes.py)",
    )
    p.add_argument(
        "--season", default="annual",
        choices=["annual"] + SEASON_NAMES,
        help="Season to visualise (default: annual)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Output directory for PDF files "
             "(default: same directory as the input zarr)",
    )
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    zarr_path = args.zarr
    season    = args.season
    seasfx    = season.lower()
    out_dir   = args.out_dir or os.path.dirname(os.path.abspath(zarr_path))

    if not os.path.exists(zarr_path):
        raise SystemExit(f"Zarr not found: {zarr_path}")

    print(f"Loading {zarr_path} ...")
    d = load_zarr(zarr_path)

    print(
        f"  Locations    : {d['locations']}\n"
        f"  Agg windows  : {d['agg_windows']}\n"
        f"  Boost windows: {d['perc_boosts']}\n"
        f"  Slopes       : {d['slopes'].tolist()}\n"
        f"  Variances    : {d['variances'].tolist()}\n"
        f"  Season       : {season}\n"
        f"  Output dir   : {out_dir}\n"
    )

    os.makedirs(out_dir, exist_ok=True)

    for li, location in enumerate(d["locations"]):
        loc_tag  = location.lower().replace(" ", "_")
        out_path = os.path.join(out_dir, f"fig_exceedances_{loc_tag}_{seasfx}.pdf")
        print(f"Plotting location {li + 1}/{len(d['locations'])}: {location}")
        fig_location(zarr_path, d, li, season, out_path)

    print(f"\nDone. {len(d['locations'])} figure(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
