#!/usr/bin/env python3
"""
Plot global-mean exceedances per year from an exceedances zarr store.

For each year in the analysis period, computes the fraction of (lat, lon, day)
grid-cells that exceed the threshold, then plots it as a time series.

Example
-------
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/vis/plot_exceedances_per_year.py \
        --input experiments/daily_max_2m_temperature_ref1960-1989_an1960-2019_agg3max_boost3/exceedances_0_9.zarr \
        --output exceedances_per_year.png

    # Overlay multiple experiments (e.g. 90th vs 95th percentile):
    python src/vis/plot_exceedances_per_year.py \
        --input \
            experiments/.../exceedances_0_9.zarr \
            experiments/.../exceedances_0_95.zarr \
        --labels "p=0.90" "p=0.95" \
        --output exceedances_per_year_comparison.png
"""
import argparse
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def exceedances_per_year(exc_path: str, area_weighted: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (years, mean_fraction_per_year) for one exceedances zarr.

    The fraction is the global-mean exceedance rate per year.
    If area_weighted=True, grid cells are weighted by cos(lat); otherwise
    all cells contribute equally.
    """
    ds = xr.open_zarr(exc_path, consolidated=False)
    exc = ds["data"]  # (time, latitude, longitude) bool

    # Area weights: cos(lat) normalised to sum to 1, or uniform
    lat_rad = np.deg2rad(exc.latitude.values)
    if area_weighted:
        weights = np.cos(lat_rad)
        weights /= weights.sum()
    else:
        weights = np.ones(len(lat_rad)) / len(lat_rad)

    # Group by year; for each year compute weighted-mean exceedance rate
    # Convert int32 days-since-1900 back to pandas-friendly timestamps for groupby
    time_vals = exc.time.values
    if np.issubdtype(time_vals.dtype, np.integer):
        from datetime import timedelta
        import cftime
        epoch = cftime.DatetimeNoLeap(1900, 1, 1)
        time_cf = [epoch + timedelta(days=int(d)) for d in time_vals]
        years = np.array([t.year for t in time_cf])
    else:
        import pandas as pd
        years = np.array([pd.Timestamp(t).year for t in time_vals])

    unique_years = np.unique(years)
    annual_rates = np.empty(len(unique_years))

    for i, yr in enumerate(unique_years):
        mask = years == yr
        # Load this year's data: (n_days, nlat, nlon) bool → float32
        yr_data = exc.isel(time=mask).values.astype(np.float32)
        # Weighted spatial mean, then time mean
        # weights shape: (nlat,) → broadcast over (n_days, nlat, nlon)
        weighted = yr_data * weights[np.newaxis, :, np.newaxis]
        annual_rates[i] = weighted.mean(axis=(0, 2)).sum()  # sum over lat after weighting

    return unique_years, annual_rates


def plot(inputs: list[str], labels: list[str], output: str, area_weighted: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))

    for path, label in zip(inputs, labels):
        years, rates = exceedances_per_year(path, area_weighted=area_weighted)
        ax.plot(years, rates * 100, marker="o", markersize=3, linewidth=1.5, label=label)

    ax.set_xlabel("Year")
    ax.set_ylabel("Exceedance rate (%)")
    weight_label = "area-weighted" if area_weighted else "unweighted"
    ax.set_title(f"Global-mean exceedances per year ({weight_label})")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved: {output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--input",  required=True, nargs="+",
                   help="One or more exceedances zarr paths")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels (default: basename of each --input path)")
    p.add_argument("--output", required=True,
                   help="Output PNG path")
    p.add_argument("--area-weighted", action="store_true", default=False,
                   help="Weight grid cells by cos(lat) when computing the "
                        "global mean (default: all cells contribute equally)")
    return p


def main() -> None:
    args = build_parser().parse_args()
    labels = args.labels or [os.path.basename(p.rstrip("/")) for p in args.input]
    if len(labels) != len(args.input):
        raise ValueError("--labels must have the same number of entries as --input")
    plot(args.input, labels, args.output, area_weighted=args.area_weighted)


if __name__ == "__main__":
    main()
