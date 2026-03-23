#!/usr/bin/env python3
"""
Compute weather-extreme exceedances from a zarr dataset.

Example
-------
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    python src/main.py \\
        --input   t2max.zarr \\
        --var     daily_max_2m_temperature \\
        --ref-start 1960-01-01  --ref-end 1989-12-31 \\
        --an-start  1960-01-01  --an-end  2019-12-31 \\
        --agg-window  3 \\
        --agg-method  max \\
        --perc-boost  3 \\
        --percentile  0.90 \\
        --output  experiments/

Outputs (written inside --output/<run-name>/)
---------------------------------------------
    thresholds.zarr          per-DOY threshold array  (365, nlat, nlon) float32
    exceedances_0_90.zarr    binary exceedance mask   (n_days, nlat, nlon) bool

Both stores are xarray-compatible: open with xr.open_zarr(path)["data"].
No large intermediate files are written – peak disk usage equals the
two outputs above (~1.5 GB + ~5 GB compressed for a 60-year global run).
"""
import argparse
import os
import sys

import cftime

# Allow running from the repo root or from src/
sys.path.insert(0, os.path.dirname(__file__))

from experiment import Config, Experiment


def _parse_date(s: str) -> cftime.DatetimeNoLeap:
    """Convert 'YYYY-MM-DD' to cftime.DatetimeNoLeap."""
    try:
        y, m, d = map(int, s.split("-"))
        return cftime.DatetimeNoLeap(y, m, d)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{s}': expected YYYY-MM-DD"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- required ----
    p.add_argument("--input",       required=True,
                   help="Path to input zarr store")
    p.add_argument("--var",         required=True,
                   help="Variable name inside the zarr store")
    p.add_argument("--ref-start",   required=True, type=_parse_date,
                   metavar="YYYY-MM-DD",
                   help="Reference period start (inclusive)")
    p.add_argument("--ref-end",     required=True, type=_parse_date,
                   metavar="YYYY-MM-DD",
                   help="Reference period end (inclusive)")
    p.add_argument("--an-start",    required=True, type=_parse_date,
                   metavar="YYYY-MM-DD",
                   help="Analysis period start (inclusive)")
    p.add_argument("--an-end",      required=True, type=_parse_date,
                   metavar="YYYY-MM-DD",
                   help="Analysis period end (inclusive)")
    p.add_argument("--agg-window",  required=True, type=int,
                   help="Rolling-window size in days (positive odd integer, "
                        "e.g. 3 = current day ± 1)")
    p.add_argument("--perc-boost",  required=True, type=int,
                   help="Percentile boosting-window size in DOYs "
                        "(positive odd integer, e.g. 31 = current DOY ± 15)")
    p.add_argument("--percentile",  required=True, type=float,
                   help="Percentile threshold, e.g. 0.90 for the 90th percentile")
    p.add_argument("--output",      required=True,
                   help="Parent directory for experiment outputs")

    # ---- optional ----
    p.add_argument("--agg-method",  default="max",
                   choices=["mean", "max", "min", "sum"],
                   help="Aggregation method (default: max)")
    p.add_argument("--lat-band-size", type=int, default=None,
                   metavar="N",
                   help="Latitude rows per memory chunk during threshold "
                        "computation.  Default: auto-sized to fit in ~1 GB.")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config(
        input_zarr_path=args.input,
        var=args.var,
        ref_start=args.ref_start,
        ref_end=args.ref_end,
        an_start=args.an_start,
        an_end=args.an_end,
        agg_window=args.agg_window,
        agg_method=args.agg_method,
        perc_boost=args.perc_boost,
        percentile=args.percentile,
        output_dir=args.output,
        lat_band_size=args.lat_band_size,
    )

    Experiment(cfg).run()


if __name__ == "__main__":
    main()
