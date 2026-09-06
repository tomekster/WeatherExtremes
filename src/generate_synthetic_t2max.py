#!/usr/bin/env python3
"""
Generate synthetic daily maximum 2m temperature dataset.

Two modes are available (select with --mode):

  era5  (default)
    The reference daily climatological profile for each of the 8 fixed locations
    is derived by averaging daily_max_2m_temperature from the rechunked ERA5
    zarr over the decade 1950–1959 (mean for each calendar day-of-year).

  sine
    The reference profile is a pure sine wave with the coldest day at the middle
    of winter (doy 15, ~Jan 15) and the warmest day at the middle of summer
    (doy 197, ~Jul 16):

        T_ref(doy) = mean_temp + amplitude * cos(2π * (doy - 197) / 365)

    No ERA5 input is needed.  A single location "sine_wave" is produced.

In both modes synthetic daily series for 1950–2020 are produced as:

    T(Y, doy) = T_ref(doy) + slope * Y + ε

where Y = year − 1950 (Y=0 for 1950) and ε is the noise term.

By default (--autocorr 0.0, --variance-trend 0.0) ε ~ i.i.d. N(0, sigma²).

With --autocorr φ (φ ∈ (-1, 1), φ ≠ 0) ε follows an AR(1) process with
unit variance, then scaled by sigma:

    z_t = φ · z_{t-1} + √(1 − φ²) · w_t,   w_t ~ i.i.d. N(0,1)
    ε_t = sigma · z_t

The marginal variance of ε is sigma² regardless of φ.

With --variance-trend k > 0 the noise variance increases linearly with year:

    σ²(Y) = sigma² + k · Y

so sigma(t) = sqrt(max(0, sigma² + k·Y(t))).  This is applied on top of
either noise model:  ε_t = sigma(Y(t)) · z_t where z_t is either i.i.d.
N(0,1) or unit-variance AR(1).

5 slope values and 5 variance values are combined (25 realisations per location).
A fixed random seed ensures reproducibility.

Usage
-----
    cd /home/tsternal/phd/WeatherExtremes2
    source venv/bin/activate

    # ERA5 mode (original behaviour)
    python src/generate_synthetic_t2max.py --mode era5

    # Sine-wave mode
    python src/generate_synthetic_t2max.py --mode sine --amplitude 15 --mean-temp 10

    # With AR(1) autocorrelation and increasing variance
    python src/generate_synthetic_t2max.py --mode sine --amplitude 15 --mean-temp 10 \\
        --autocorr 0.7 --variance-trend 0.02

    # Override output path
    python src/generate_synthetic_t2max.py --mode sine --amplitude 15 --mean-temp 10 \\
        --output data/synthetic/my_sine_experiment.zarr
"""
import argparse
import os

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_ZARR  = "data/preprocessed/rechunked/t2max_rechunked.zarr"
VAR_NAME    = "daily_max_2m_temperature"

# Day-of-year of the seasonal extremes (1-based, no-leap calendar):
#   coldest: middle of winter ~Jan 15
#   warmest: middle of summer ~Jul 16
DOY_WINTER_MIN = 15
DOY_SUMMER_MAX = 197

# Locations used in era5 mode: (lat, lon)
LOCATIONS = {
    "Zurich":         (47.37,   8.54),
    "Vienna":         (48.21,  16.37),
    "Beijing":        (39.91, 116.39),
    "New York City":  (40.71,  -74.01),
    "San Francisco":  (37.77, -122.42),
    "Rio de Janeiro": (-22.91, -43.17),
    "Johannesburg":   (-26.20,  28.04),
    "Sydney":         (-33.87, 151.21),
}

# 5 slope values [°C / year]
SLOPES    = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=np.float32)

# 5 variance values [°C²]  → sigma = sqrt(variance)
VARIANCES = np.array([0.25, 0.5, 1.0, 2.0, 4.0],    dtype=np.float32)

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Helpers – shared
# ---------------------------------------------------------------------------

def generate_series(
    clim: np.ndarray,
    times: pd.DatetimeIndex,
    slope: float,
    sigma: float,
    rng: np.random.Generator,
    autocorr: float = 0.0,
    variance_trend: float = 0.0,
) -> np.ndarray:
    """
    Build a synthetic daily series for a single (slope, sigma) combination.

    T(t) = clim[doy(t)-1]  +  slope * (year(t) - 1950)  +  ε(t)

    Noise model
    -----------
    A unit-variance base process z(t) is generated first:
      - autocorr == 0.0: z(t) ~ i.i.d. N(0, 1)
      - autocorr != 0.0: z(t) = autocorr*z(t-1) + sqrt(1-autocorr²)*w(t),
                                 w(t) ~ i.i.d. N(0,1)  [unit-variance AR(1)]

    The noise is then scaled by a (possibly time-varying) sigma:
      - variance_trend == 0.0: σ(t) = sigma  (constant)
      - variance_trend != 0.0: σ²(Y) = sigma² + variance_trend * Y
                                where Y = year(t) - 1950

    ε(t) = sigma(t) * z(t)
    """
    n     = len(times)
    years = times.year.to_numpy()
    doys  = times.dayofyear.to_numpy()   # 1-based
    trend = slope * (years - 1950).astype(np.float32)

    # Per-day sigma: constant or linearly increasing variance
    if variance_trend != 0.0:
        Y = (years - 1950).astype(np.float64)
        sigma_t = np.sqrt(np.maximum(0.0, sigma ** 2 + variance_trend * Y)).astype(np.float32)
    else:
        sigma_t = np.full(n, sigma, dtype=np.float32)

    # Unit-variance base process
    if autocorr == 0.0:
        z = rng.normal(0.0, 1.0, size=n)
    else:
        innov_std   = np.sqrt(max(0.0, 1.0 - autocorr ** 2))
        innovations = rng.normal(0.0, innov_std, size=n)
        z = np.empty(n, dtype=np.float64)
        z[0] = rng.normal(0.0, 1.0)   # initial condition from stationary dist
        for i in range(1, n):
            z[i] = autocorr * z[i - 1] + innovations[i]

    noise = (z * sigma_t).astype(np.float32)
    return clim[doys - 1] + trend + noise


def write_zarr(
    output_path: str,
    data: np.ndarray,
    clims: np.ndarray,
    loc_names: list[str],
    times: pd.DatetimeIndex,
    extra_attrs: dict,
) -> None:
    """Write the standard output zarr layout."""
    n_sl, n_var, n_loc, n_t = data.shape

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"\nWriting {output_path} ...")
    grp = zarr.open_group(output_path, mode="w")

    codec = [BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.shuffle)]

    grp.create_array(
        "t2max", data=data,
        chunks=(1, 1, n_loc, n_t),
        compressors=codec,
        dimension_names=["slope_idx", "variance_idx", "location", "time"],
    )
    grp.create_array(
        "t2max_ref", data=clims,
        chunks=clims.shape,
        compressors=codec,
        dimension_names=["location", "dayofyear"],
    )

    grp.create_array("slope",     data=SLOPES,    chunks=SLOPES.shape,
                     dimension_names=["slope_idx"])
    grp.create_array("variance",  data=VARIANCES, chunks=VARIANCES.shape,
                     dimension_names=["variance_idx"])
    grp.create_array("location",  data=np.array(loc_names, dtype="S30"),
                     chunks=(n_loc,), dimension_names=["location"])
    grp.create_array("time",      data=times.astype(np.int64).to_numpy(),
                     chunks=(n_t,),  dimension_names=["time"])
    grp.create_array("dayofyear", data=np.arange(1, 367, dtype=np.int32),
                     chunks=(366,),  dimension_names=["dayofyear"])

    attrs = {
        "locations":      loc_names,
        "slopes_unit":    "degC/year",
        "variance_unit":  "degC^2",
        "random_seed":    RANDOM_SEED,
    }
    attrs.update(extra_attrs)
    grp.attrs.update(attrs)

    zarr.consolidate_metadata(output_path)
    print(f"Done. Output: {output_path}")
    print(f"Shape: (slope={n_sl}, variance={n_var}, location={n_loc}, time={n_t})")
    print(f"Slopes    [°C/yr]  : {SLOPES.tolist()}")
    print(f"Variances [°C²]    : {VARIANCES.tolist()}")
    print(f"Locations          : {loc_names}")


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _noise_desc(autocorr: float, variance_trend: float) -> str:
    """Human-readable description of the noise model for logging / zarr attrs."""
    parts = []
    parts.append(f"AR(1) noise, φ={autocorr}" if autocorr != 0.0 else "i.i.d. Gaussian noise")
    if variance_trend != 0.0:
        parts.append(f"increasing variance (k={variance_trend} °C²/yr)")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# ERA5 mode
# ---------------------------------------------------------------------------

def nearest_point(ds: xr.Dataset, lat: float, lon: float) -> xr.DataArray:
    """Return the DataArray for the grid point nearest to (lat, lon)."""
    return ds[VAR_NAME].sel(latitude=lat, longitude=lon, method="nearest")


def compute_climatology(da: xr.DataArray) -> np.ndarray:
    """
    Average da over the 1950–1959 decade, grouping by day-of-year.
    Returns a float32 array of shape (366,) indexed 1..366.
    Leap day (doy=366) uses the mean of all available Feb-29 values;
    if none exist it is filled by interpolating between doy=365 and doy=1.
    """
    decade = da.sel(time=slice("1950-01-01", "1959-12-31"))
    clim   = decade.groupby("time.dayofyear").mean(dim="time").compute()
    clim_vals = clim.values.astype(np.float32)
    doys      = clim.dayofyear.values

    full = np.empty(366, dtype=np.float32)
    full[:] = np.nan
    for i, d in enumerate(doys):
        full[d - 1] = clim_vals[i]

    if np.isnan(full[365]):
        full[365] = (full[364] + full[0]) / 2.0

    return full   # shape (366,), 0-indexed by doy-1


def run_era5(output_path: str, autocorr: float = 0.0,
             variance_trend: float = 0.0, seed: int = RANDOM_SEED) -> None:
    print(f"Opening {INPUT_ZARR} ...")
    ds = xr.open_zarr(INPUT_ZARR, consolidated=False)

    loc_names = list(LOCATIONS.keys())
    n_loc = len(loc_names)
    n_sl  = len(SLOPES)
    n_var = len(VARIANCES)

    times = pd.date_range("1950-01-01", "2020-12-31", freq="D")
    n_t   = len(times)

    # Step 1 – extract climatologies
    print("Computing 1950–1959 climatologies ...")
    clims = np.empty((n_loc, 366), dtype=np.float32)
    for i, (name, (lat, lon)) in enumerate(LOCATIONS.items()):
        print(f"  {name:20s}  lat={lat:7.2f}  lon={lon:8.2f}")
        da = nearest_point(ds, lat, lon)
        clims[i] = compute_climatology(da)

    # Step 2 – generate synthetic series
    noise_desc = _noise_desc(autocorr, variance_trend)
    print(f"Generating synthetic time series ({noise_desc}) ...")
    data = np.empty((n_sl, n_var, n_loc, n_t), dtype=np.float32)
    rng  = np.random.default_rng(seed)

    for si, slope in enumerate(SLOPES):
        for vi, variance in enumerate(VARIANCES):
            sigma = float(np.sqrt(variance))
            for li in range(n_loc):
                data[si, vi, li, :] = generate_series(
                    clims[li], times, slope, sigma, rng,
                    autocorr=autocorr, variance_trend=variance_trend,
                )
            print(f"  slope={slope:.2f} °C/yr  variance={variance:.2f} °C²  done")

    write_zarr(
        output_path, data, clims, loc_names, times,
        extra_attrs={
            "mode": "era5",
            "description": (
                "Synthetic daily maximum 2 m temperature. "
                "Reference climatology averaged from ERA5 rechunked t2max over 1950-1959. "
                f"Synthetic series: T(t) = clim[doy] + slope*(year-1950) + noise ({noise_desc})."
            ),
            "source_zarr":    INPUT_ZARR,
            "autocorr":       autocorr,
            "variance_trend": variance_trend,
            "random_seed":    seed,
        },
    )


# ---------------------------------------------------------------------------
# Sine mode
# ---------------------------------------------------------------------------

def build_sine_climatology(amplitude: float, mean_temp: float) -> np.ndarray:
    """
    Build a 366-element reference climatology from a pure cosine wave.

        T_ref(doy) = mean_temp + amplitude * cos(2π * (doy - DOY_SUMMER_MAX) / 365)

    This places the maximum at doy=DOY_SUMMER_MAX (mid-summer, ~Jul 16) and the
    minimum at doy=DOY_WINTER_MIN (mid-winter, ~Jan 15), 182 days apart.

    Returns float32 array of shape (366,), 0-indexed (index 0 = doy 1).
    Day 366 (leap day) is set equal to day 365 so the array is always full.
    """
    doys = np.arange(1, 367, dtype=np.float32)
    clim = mean_temp + amplitude * np.cos(
        2 * np.pi * (doys - DOY_SUMMER_MAX) / 365.0
    )
    clim[365] = clim[364]   # doy 366 (leap day) ≈ doy 365
    return clim.astype(np.float32)


def run_sine(amplitude: float, mean_temp: float, output_path: str,
             autocorr: float = 0.0, variance_trend: float = 0.0,
             seed: int = RANDOM_SEED) -> None:
    print(
        f"Sine-wave mode: amplitude={amplitude} °C, mean_temp={mean_temp} °C\n"
        f"  T_ref(doy) = {mean_temp} + {amplitude}*cos(2π*(doy-{DOY_SUMMER_MAX})/365)\n"
        f"  Warmest day: doy {DOY_SUMMER_MAX} (~Jul 16)  "
        f"T = {mean_temp + amplitude:.1f} °C\n"
        f"  Coldest day: doy {DOY_WINTER_MIN} (~Jan 15)  "
        f"T = {mean_temp - amplitude:.1f} °C"
    )

    loc_names = ["sine_wave"]
    n_loc = 1
    n_sl  = len(SLOPES)
    n_var = len(VARIANCES)

    times = pd.date_range("1950-01-01", "2020-12-31", freq="D")
    n_t   = len(times)

    clim  = build_sine_climatology(amplitude, mean_temp)   # (366,)
    clims = clim[np.newaxis, :]                            # (1, 366)

    noise_desc = _noise_desc(autocorr, variance_trend)
    print(f"Generating synthetic time series ({noise_desc}) ...")
    data = np.empty((n_sl, n_var, n_loc, n_t), dtype=np.float32)
    rng  = np.random.default_rng(seed)

    for si, slope in enumerate(SLOPES):
        for vi, variance in enumerate(VARIANCES):
            sigma = float(np.sqrt(variance))
            data[si, vi, 0, :] = generate_series(
                clim, times, slope, sigma, rng,
                autocorr=autocorr, variance_trend=variance_trend,
            )
            print(f"  slope={slope:.2f} °C/yr  variance={variance:.2f} °C²  done")

    write_zarr(
        output_path, data, clims, loc_names, times,
        extra_attrs={
            "mode":        "sine",
            "description": (
                "Synthetic daily maximum 2 m temperature from a sine-wave reference. "
                f"T_ref(doy) = mean_temp + amplitude*cos(2π*(doy-{DOY_SUMMER_MAX})/365). "
                f"Synthetic series: T(t) = T_ref[doy] + slope*(year-1950) + noise ({noise_desc})."
            ),
            "sine_amplitude":  amplitude,
            "sine_mean_temp":  mean_temp,
            "sine_doy_max":    DOY_SUMMER_MAX,
            "sine_doy_min":    DOY_WINTER_MIN,
            "autocorr":        autocorr,
            "variance_trend":  variance_trend,
            "random_seed":     seed,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["era5", "sine"], default="era5",
        help="Reference climatology source (default: era5)",
    )
    p.add_argument(
        "--amplitude", type=float, default=None,
        help="[sine mode] Half peak-to-peak range [°C]. "
             "Peak = mean_temp + amplitude, trough = mean_temp - amplitude.",
    )
    p.add_argument(
        "--mean-temp", type=float, default=None,
        help="[sine mode] Annual mean temperature of the sine wave [°C].",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output zarr path. Defaults to data/synthetic/synthetic_t2max.zarr "
             "(era5) or data/synthetic/synthetic_t2max_sine.zarr (sine).",
    )
    p.add_argument(
        "--autocorr", type=float, default=0.0,
        help="Lag-1 autocorrelation φ for AR(1) noise, in (-1, 1). "
             "0.0 (default) = independent i.i.d. Gaussian noise.",
    )
    p.add_argument(
        "--variance-trend", type=float, default=0.0,
        dest="variance_trend",
        help="Annual increase in noise variance k [°C²/yr] for the model "
             "σ²(Y) = σ_base² + k·Y where Y = year-1950. "
             "0.0 (default) = constant variance.",
    )
    p.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help=f"Random seed for reproducibility (default: {RANDOM_SEED}).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not (-1.0 < args.autocorr < 1.0):
        raise SystemExit(
            f"error: --autocorr must be in (-1, 1), got {args.autocorr}"
        )
    if args.variance_trend < 0.0:
        raise SystemExit(
            f"error: --variance-trend must be >= 0, got {args.variance_trend}"
        )

    if args.mode == "sine":
        if args.amplitude is None or args.mean_temp is None:
            raise SystemExit(
                "error: --amplitude and --mean-temp are required for --mode sine"
            )
        if args.output:
            output = args.output
        else:
            # Embed noise-model parameters in the default filename so successive
            # invocations with different settings don't overwrite each other.
            ar_tag = f"ar{args.autocorr}".replace(".", "")
            vt_tag = f"vt{args.variance_trend}".replace(".", "")
            output = f"data/synthetic/synthetic_t2max_sine_{ar_tag}_{vt_tag}.zarr"
        run_sine(args.amplitude, args.mean_temp, output,
                 autocorr=args.autocorr, variance_trend=args.variance_trend,
                 seed=args.seed)

    else:  # era5
        output = args.output or "data/synthetic/synthetic_t2max.zarr"
        run_era5(output, autocorr=args.autocorr, variance_trend=args.variance_trend,
                 seed=args.seed)


if __name__ == "__main__":
    main()
