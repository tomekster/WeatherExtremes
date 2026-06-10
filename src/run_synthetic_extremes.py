#!/usr/bin/env python3
"""
Compute extremes and trendlines for the synthetic t2max zarr.

Reads   data/synthetic/synthetic_t2max.zarr
Writes  data/synthetic/synthetic_extremes.zarr

Pipeline (mirrors main.py / experiment.py):
  1. Reconstruct no-leap daily time axis (drop Feb-29 days).
  2. For every (slope_idx, variance_idx) pair process all 8 locations at once
     using the same core functions as the ERA5 pipeline:
       - rolling_aggregate      (src/core/aggregation.py)
       - compute_thresholds     (src/core/thresholds.py)
       - detect_exceedances     (src/core/exceedances.py)
  3. Compute annual + seasonal exceedance counts.
  4. Fit linear trends (vectorised OLS, same approach as analyse_exceedances.py).

Output zarr arrays
------------------
  thresholds      (n_sl, n_var, n_loc, 365)        float32
  exceedances     (n_sl, n_var, n_loc, n_an_days)  bool
  annual_counts   (n_sl, n_var, n_loc, n_years)    int32
  seasonal_counts (n_sl, n_var, 4, n_loc, n_years) int32
  annual_trend    (n_sl, n_var, n_loc, 2)          float32  [slope, intercept]
  seasonal_trend  (n_sl, n_var, 4, n_loc, 2)       float32  [slope, intercept]

Coordinates also written: slope, variance, location, time (noleap), year,
dayofyear, season_name.

Parameters (hardcoded constants at top of file, easy to change)
----------------------------------------------------------------
  AGG_WINDOW   = 3      days (centred rolling window)
  AGG_METHOD   = MAX
  PERC_BOOST   = 31     DOY-boost window width (±15 DOYs)
  PERCENTILE   = 0.90   (90th percentile)
  REF_START    = 1950
  REF_END      = 1979   (30-year baseline)

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate
    python src/run_synthetic_extremes.py
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

# Add src/ to path so core imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from core.aggregation import rolling_aggregate, AggMethod
from core.thresholds import compute_thresholds
from core.exceedances import detect_exceedances

# ---------------------------------------------------------------------------
# Defaults (overridable via CLI)
# ---------------------------------------------------------------------------

_DEFAULT_INPUT    = "data/synthetic/synthetic_t2max.zarr"
_DEFAULT_OUTPUT   = "data/synthetic/synthetic_extremes.zarr"
_DEFAULT_AGG_WINS = [3]
_DEFAULT_BOOSTS   = [31]
_DEFAULT_PERC     = 0.90
_DEFAULT_REF_S    = 1950
_DEFAULT_REF_E    = 1979

# Season names and DOY bounds (0-indexed, no-leap calendar, 365 days)
# Matches SEASON_BOUNDS in src/utils/utils.py
SEASON_NAMES  = ["DJF", "MAM", "JJA", "SON"]
SEASON_BOUNDS = [
    (-31, 59),   # DJF  Dec(334-364) + Jan(0-30) + Feb(31-58)
    (59, 151),   # MAM  Mar-May
    (151, 243),  # JJA  Jun-Aug
    (243, 334),  # SON  Sep-Nov
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def noleap_time_index(full_times: pd.DatetimeIndex):
    """Return (noleap_times, mask) dropping Feb-29 from full_times."""
    mask = ~((full_times.month == 2) & (full_times.day == 29))
    return pd.DatetimeIndex(full_times[mask]), mask


def noleap_doy(times: pd.DatetimeIndex) -> np.ndarray:
    """
    Day-of-year in a no-leap calendar (1..365) for dates that have already
    had Feb-29 removed.  For leap years, dates from Mar-1 onwards are
    shifted back by 1.
    """
    year  = np.array(times.year)
    month = np.array(times.month)
    doys  = np.array(times.day_of_year, dtype=np.int32).copy()
    is_leap    = (year % 4 == 0) & ((year % 100 != 0) | (year % 400 == 0))
    after_feb28 = month > 2
    doys[is_leap & after_feb28] -= 1
    return doys


def vectorised_ols(y: np.ndarray) -> np.ndarray:
    """
    Fit y = slope * t + intercept along axis 0 (t = 0, 1, …, T-1).
    y: (T, ...)  →  returns (2, ...) with [slope, intercept] along axis 0.
    """
    T = y.shape[0]
    t = np.arange(T, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y.mean(axis=0)
    denom  = np.sum((t - t_mean) ** 2)  # scalar
    numer  = np.sum((t[:, *([np.newaxis] * (y.ndim - 1))] - t_mean) * (y - y_mean), axis=0)
    slope  = numer / denom
    intercept = y_mean - slope * t_mean
    return np.stack([slope, intercept], axis=0).astype(np.float32)


def load_existing_data(path: str) -> dict | None:
    """Return arrays from an existing output zarr, or None if the path does not exist.

    Raises SystemExit if the zarr exists but was written in the old single-ref-period
    format (no 'ref_period' coordinate array) — those zarrs must be recomputed.
    """
    if not os.path.exists(path):
        return None
    g = zarr.open_group(path, mode="r")
    if "ref_period" not in g:
        raise SystemExit(
            f"'{path}' was written by an older version of this script that stored "
            "only a single reference period in attrs. Delete it and rerun to "
            "generate a zarr that supports multiple reference periods."
        )
    ref_periods = [tuple(row) for row in g["ref_period"][:].tolist()]
    return {
        "agg_windows":     g["agg_window"][:].tolist(),
        "perc_boosts":     g["perc_boost"][:].tolist(),
        "ref_periods":     ref_periods,             # list of (start, end) tuples
        "thresholds":      g["thresholds"][:],
        "annual_counts":   g["annual_counts"][:],
        "seasonal_counts": g["seasonal_counts"][:],
        "annual_trend":    g["annual_trend"][:],
        "seasonal_trend":  g["seasonal_trend"][:],
    }


def seasonal_annual_counts(
    exc: np.ndarray,        # (n_an_days, n_loc)
    an_years: np.ndarray,   # (n_an_days,)
    an_doys: np.ndarray,    # (n_an_days,) 1-indexed no-leap DOY
    years: np.ndarray,      # (n_years,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    annual   : (n_years, n_loc)  – total exceedance days per year
    seasonal : (4, n_years, n_loc) – per-season counts
               DJF for year Y = Dec of Y-1 (DOY 335-365) + Jan-Feb of Y (DOY 1-59)
               MAM = DOY 60-151, JJA = DOY 152-243, SON = DOY 244-334
    """
    n_years = len(years)
    n_loc   = exc.shape[1]

    annual = np.zeros((n_years, n_loc), dtype=np.int32)
    for yi, y in enumerate(years):
        annual[yi] = exc[an_years == y].sum(axis=0)

    # 1-indexed DOY season bounds (inclusive on both ends):
    #   DJF: Dec = DOY 335-365 (year Y-1) + Jan-Feb = DOY 1-59 (year Y)
    #   MAM: DOY 60-151, JJA: DOY 152-243, SON: DOY 244-334
    season_doy_bounds = [
        None,           # DJF handled separately
        (60,  151),     # MAM
        (152, 243),     # JJA
        (244, 334),     # SON
    ]
    seasonal = np.zeros((4, n_years, n_loc), dtype=np.int32)
    for yi, y in enumerate(years):
        # DJF: Dec of previous year + Jan-Feb of this year
        dec_mask = (an_years == y - 1) & (an_doys >= 335)
        jf_mask  = (an_years == y)     & (an_doys <= 59)
        seasonal[0, yi] = exc[dec_mask].sum(axis=0) + exc[jf_mask].sum(axis=0)
        # Other seasons
        for si, (d0, d1) in enumerate(season_doy_bounds[1:], start=1):
            mask = (an_years == y) & (an_doys >= d0) & (an_doys <= d1)
            seasonal[si, yi] = exc[mask].sum(axis=0)

    return annual, seasonal  # (n_years, n_loc), (4, n_years, n_loc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute extremes and trendlines for a synthetic t2max zarr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",        default=_DEFAULT_INPUT,
                   help="Input synthetic zarr path")
    p.add_argument("--output",       default=_DEFAULT_OUTPUT,
                   help="Output zarr path")
    p.add_argument("--agg-windows",  type=int, nargs="+", default=_DEFAULT_AGG_WINS,
                   help="One or more rolling-window sizes in days (odd integers)")
    p.add_argument("--perc-boosts",  type=int, nargs="+", default=_DEFAULT_BOOSTS,
                   help="One or more percentile boosting-window widths in DOYs (odd integers)")
    p.add_argument("--percentile",   type=float, default=_DEFAULT_PERC,
                   help="Percentile threshold, e.g. 0.90")
    p.add_argument("--ref-periods",  type=int, nargs="+",
                   default=[_DEFAULT_REF_S, _DEFAULT_REF_E],
                   help="One or more reference periods as pairs: start1 end1 start2 end2 …")
    p.add_argument("--location",     type=str,   default=None,
                   help="Process only this location name (default: all locations)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    INPUT_ZARR  = args.input
    OUTPUT_ZARR = args.output
    AGG_WINDOWS = args.agg_windows   # list[int]
    PERC_BOOSTS = args.perc_boosts   # list[int]
    AGG_METHOD  = AggMethod.MAX
    PERCENTILE  = args.percentile

    raw_rp = args.ref_periods
    if len(raw_rp) % 2 != 0:
        raise SystemExit("--ref-periods requires an even number of integers (start end pairs).")
    REF_PERIODS = [(raw_rp[i], raw_rp[i + 1]) for i in range(0, len(raw_rp), 2)]

    # ------------------------------------------------------------------
    # Append mode: merge requested values with any already in the output zarr
    # ------------------------------------------------------------------
    existing = load_existing_data(OUTPUT_ZARR)
    if existing:
        print("Output zarr exists — will append new combinations only.")
        ex_aggs = existing["agg_windows"]
        ex_boosts = existing["perc_boosts"]
        ex_rp   = existing["ref_periods"]
    else:
        ex_aggs = []; ex_boosts = []; ex_rp = []

    skip_combos     = {(a, b, rs, re)
                       for a in ex_aggs for b in ex_boosts for rs, re in ex_rp}
    all_agg_windows = sorted(set(ex_aggs)   | set(AGG_WINDOWS))
    all_perc_boosts = sorted(set(ex_boosts) | set(PERC_BOOSTS))
    all_ref_periods = sorted(set(ex_rp)     | set(REF_PERIODS))
    n_agg_all       = len(all_agg_windows)
    n_boost_all     = len(all_perc_boosts)
    n_ref_all       = len(all_ref_periods)

    # ------------------------------------------------------------------
    # Load input zarr
    # ------------------------------------------------------------------
    print(f"Opening {INPUT_ZARR} ...")
    grp_in = zarr.open_group(INPUT_ZARR, mode="r")

    slopes    = grp_in["slope"][:]
    variances = grp_in["variance"][:]
    locations = [b.decode() for b in grp_in["location"][:]]
    time_ns   = grp_in["time"][:]

    n_sl  = len(slopes)
    n_var = len(variances)

    # Optional single-location filter
    if args.location is not None:
        if args.location not in locations:
            raise SystemExit(
                f"Location '{args.location}' not found in {INPUT_ZARR}.\n"
                f"Available: {locations}"
            )
        _loc_idx   = locations.index(args.location)
        _loc_slice = slice(_loc_idx, _loc_idx + 1)
        locations  = [args.location]
    else:
        _loc_slice = slice(None)

    n_loc = len(locations)

    # Time axis (shared across all agg windows)
    full_times   = pd.to_datetime(time_ns)
    noleap_times, noleap_mask = noleap_time_index(full_times)
    doys_all     = noleap_doy(noleap_times)
    n_all        = len(noleap_times)
    noleap_years = np.array(noleap_times.year)

    # Analysis period derived from the data itself — independent of ref period
    years   = np.arange(int(noleap_times.year[0]), int(noleap_times.year[-1]) + 1)
    n_years = len(years)

    n_new = sum(
        1 for a, b, (rs, re) in itertools.product(
            all_agg_windows, all_perc_boosts, all_ref_periods)
        if (a, b, rs, re) not in skip_combos
    )
    print(
        f"No-leap days total : {n_all}\n"
        f"Location(s)        : {locations}\n"
        f"Agg windows [days] : {all_agg_windows}  (requested: {AGG_WINDOWS})\n"
        f"Boost windows [DOY]: {all_perc_boosts}  (requested: {PERC_BOOSTS})\n"
        f"Ref periods        : {all_ref_periods}  (requested: {REF_PERIODS})\n"
        f"Slopes [°C/yr]     : {slopes.tolist()}\n"
        f"Variances [°C²]    : {variances.tolist()}\n"
        f"New (agg,boost,ref) combos to compute: {n_new} of "
        f"{n_agg_all * n_boost_all * n_ref_all}\n"
    )

    # ------------------------------------------------------------------
    # Pre-allocate output arrays  (n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, n_loc, …)
    # ------------------------------------------------------------------
    thresholds_all      = np.empty((n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, n_loc, 365),        dtype=np.float32)
    annual_counts_all   = np.empty((n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, n_loc, n_years),    dtype=np.int32)
    seasonal_counts_all = np.empty((n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, 4, n_loc, n_years), dtype=np.int32)
    annual_trend_all    = np.empty((n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, n_loc, 2),          dtype=np.float32)
    seasonal_trend_all  = np.empty((n_agg_all, n_boost_all, n_sl, n_var, n_ref_all, 4, n_loc, 2),       dtype=np.float32)

    # Copy existing data into the pre-allocated arrays at correct index positions
    if existing:
        for ai_old, agg in enumerate(ex_aggs):
            ai_new = all_agg_windows.index(agg)
            for bi_old, boost in enumerate(ex_boosts):
                bi_new = all_perc_boosts.index(boost)
                for ri_old, rp in enumerate(ex_rp):
                    ri_new = all_ref_periods.index(rp)
                    thresholds_all[ai_new, bi_new, :, :, ri_new]      = existing["thresholds"][ai_old, bi_old, :, :, ri_old]
                    annual_counts_all[ai_new, bi_new, :, :, ri_new]   = existing["annual_counts"][ai_old, bi_old, :, :, ri_old]
                    seasonal_counts_all[ai_new, bi_new, :, :, ri_new] = existing["seasonal_counts"][ai_old, bi_old, :, :, ri_old]
                    annual_trend_all[ai_new, bi_new, :, :, ri_new]    = existing["annual_trend"][ai_old, bi_old, :, :, ri_old]
                    seasonal_trend_all[ai_new, bi_new, :, :, ri_new]  = existing["seasonal_trend"][ai_old, bi_old, :, :, ri_old]

    # ------------------------------------------------------------------
    # Main sweep
    # Rolling aggregation depends only on agg_window, not on perc_boost,
    # so we aggregate once per (agg_window, slope, variance) and then
    # compute thresholds/exceedances for each boost value.
    # ------------------------------------------------------------------
    for ai, AGG_WINDOW in enumerate(all_agg_windows):
        # Skip entire agg level if all (boost, ref) combos already in zarr
        if all((AGG_WINDOW, b, rs, re) in skip_combos
               for b in all_perc_boosts for rs, re in all_ref_periods):
            print(f"\n--- agg_window={AGG_WINDOW}d: all combinations already in zarr, skipping ---")
            continue

        half          = AGG_WINDOW // 2
        trim_mask     = np.zeros(n_all, dtype=bool)
        trim_mask[half: n_all - half] = True
        analysis_mask = trim_mask

        an_times = noleap_times[analysis_mask]
        an_doys  = doys_all[analysis_mask]
        an_years = np.array(an_times.year)

        print(f"\n--- agg_window={AGG_WINDOW}d  "
              f"analysis={an_times[0].date()} → {an_times[-1].date()} ---")

        for si in range(n_sl):
            for vi in range(n_var):
                # Load + aggregate once for this (agg, slope, variance)
                raw_noleap = grp_in["t2max"][si, vi, _loc_slice, :].T[noleap_mask]
                agg_data   = rolling_aggregate(raw_noleap, AGG_WINDOW, AGG_METHOD)
                an_agg     = agg_data[analysis_mask]

                for ri, (REF_S, REF_E) in enumerate(all_ref_periods):
                    ref_mask    = (noleap_years >= REF_S) & (noleap_years <= REF_E) & trim_mask
                    n_ref_years = int(ref_mask.sum() // 365)
                    ref_agg_365 = agg_data[ref_mask][: n_ref_years * 365]

                    for bi, PERC_BOOST in enumerate(all_perc_boosts):
                        if (AGG_WINDOW, PERC_BOOST, REF_S, REF_E) in skip_combos:
                            continue

                        print(
                            f"  ref={REF_S}-{REF_E}  boost={PERC_BOOST:3d}  "
                            f"sl={slopes[si]:.3f}  var={variances[vi]:.2f}  ...",
                            end=" ", flush=True,
                        )

                        thr = compute_thresholds(ref_agg_365, PERC_BOOST, PERCENTILE)
                        exc = detect_exceedances(an_agg, an_doys, thr)

                        ann_cnt, seas_cnt = seasonal_annual_counts(
                            exc, an_years, an_doys, years
                        )
                        ann_trend  = vectorised_ols(ann_cnt.astype(np.float64)).T
                        seas_trend = np.stack(
                            [vectorised_ols(seas_cnt[s].astype(np.float64)).T
                             for s in range(4)], axis=0,
                        )

                        thresholds_all[ai, bi, si, vi, ri]      = thr.T
                        annual_counts_all[ai, bi, si, vi, ri]   = ann_cnt.T
                        seasonal_counts_all[ai, bi, si, vi, ri] = np.transpose(seas_cnt, (0, 2, 1))
                        annual_trend_all[ai, bi, si, vi, ri]    = ann_trend
                        seasonal_trend_all[ai, bi, si, vi, ri]  = seas_trend

                        print("done")

    # ------------------------------------------------------------------
    # Write output zarr
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(OUTPUT_ZARR)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"\nWriting {OUTPUT_ZARR} ...")

    grp   = zarr.open_group(OUTPUT_ZARR, mode="w")
    codec = [BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.shuffle)]

    _DIMS = ["agg_window", "perc_boost", "slope_idx", "variance_idx", "ref_period"]

    # Data arrays — leading dims are (agg_window, perc_boost, slope_idx, variance_idx, ref_period)
    grp.create_array(
        "thresholds", data=thresholds_all,
        chunks=(1, 1, 1, 1, 1, n_loc, 365), compressors=codec,
        dimension_names=_DIMS + ["location", "dayofyear"],
    )
    grp.create_array(
        "annual_counts", data=annual_counts_all,
        chunks=(1, 1, 1, 1, 1, n_loc, n_years), compressors=codec,
        dimension_names=_DIMS + ["location", "year"],
    )
    grp.create_array(
        "seasonal_counts", data=seasonal_counts_all,
        chunks=(1, 1, 1, 1, 1, 4, n_loc, n_years), compressors=codec,
        dimension_names=_DIMS + ["season", "location", "year"],
    )
    grp.create_array(
        "annual_trend", data=annual_trend_all,
        chunks=(1, 1, n_sl, n_var, 1, n_loc, 2), compressors=codec,
        dimension_names=_DIMS + ["location", "param"],
    )
    grp.create_array(
        "seasonal_trend", data=seasonal_trend_all,
        chunks=(1, 1, n_sl, n_var, 1, 4, n_loc, 2), compressors=codec,
        dimension_names=_DIMS + ["season", "location", "param"],
    )

    # Coordinate arrays
    grp.create_array("agg_window", data=np.array(all_agg_windows, dtype=np.int32),
                     chunks=(n_agg_all,),   dimension_names=["agg_window"])
    grp.create_array("perc_boost", data=np.array(all_perc_boosts, dtype=np.int32),
                     chunks=(n_boost_all,), dimension_names=["perc_boost"])
    grp.create_array("ref_period", data=np.array(all_ref_periods, dtype=np.int32),
                     chunks=(n_ref_all, 2), dimension_names=["ref_period", "bound"])
    grp.create_array("slope",      data=slopes,    chunks=slopes.shape,
                     dimension_names=["slope_idx"])
    grp.create_array("variance",   data=variances, chunks=variances.shape,
                     dimension_names=["variance_idx"])
    grp.create_array("location",   data=np.array(locations, dtype="S30"),
                     chunks=(n_loc,),   dimension_names=["location"])
    grp.create_array("year",       data=years.astype(np.int32), chunks=(n_years,),
                     dimension_names=["year"])
    grp.create_array("dayofyear",  data=np.arange(1, 366, dtype=np.int32),
                     chunks=(365,),     dimension_names=["dayofyear"])
    grp.create_array("season",     data=np.array(SEASON_NAMES, dtype="S3"),
                     chunks=(4,),       dimension_names=["season"])
    grp.create_array("param",      data=np.array(["slope", "intercept"], dtype="S9"),
                     chunks=(2,),       dimension_names=["param"])

    grp.attrs.update({
        "description": (
            "Extremes and trendlines for synthetic t2max dataset. "
            f"agg_windows={all_agg_windows}, agg_method={AGG_METHOD.value}, "
            f"perc_boosts={all_perc_boosts}, percentile={PERCENTILE}, "
            f"ref_periods={all_ref_periods}."
        ),
        "source":       INPUT_ZARR,
        "agg_method":   AGG_METHOD.value,
        "percentile":   PERCENTILE,
        "ref_periods":  all_ref_periods,
    })

    zarr.consolidate_metadata(OUTPUT_ZARR)
    print("Done.")
    print(f"  thresholds      : {thresholds_all.shape}")
    print(f"  annual_counts   : {annual_counts_all.shape}")
    print(f"  seasonal_counts : {seasonal_counts_all.shape}")
    print(f"  annual_trend    : {annual_trend_all.shape}  [slope, intercept]")
    print(f"  seasonal_trend  : {seasonal_trend_all.shape}  [slope, intercept]")


if __name__ == "__main__":
    main()
