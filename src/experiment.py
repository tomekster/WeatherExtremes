"""
Experiment: single-pass band-by-band extremes-detection pipeline.

Design goals
------------
* **No intermediate aggregated zarr.**  The rolling aggregation is computed
  on the fly for each latitude band and discarded after the band's thresholds
  and exceedances have been written to disk.  This saves ~91 GB of disk I/O
  for a 60-year global dataset.

* **Streaming writes.**  Both the thresholds and exceedances zarr stores are
  pre-created (empty) before the main loop and filled band by band, so neither
  output array is ever fully resident in RAM.

* **Tight memory budget.**  The only large in-memory object at any one time is
  the aggregated float32 array for one lat band.  With the default 1 GB
  budget that is 7 lat rows × 21 902 days × 1 440 lons × 4 bytes ≈ 880 MB.

* **Good zarr chunking.**  Exceedances chunks are sized ``(365, band, nlon)``
  (~3.6 MB each for a 7-row band, global grid).  This aligns with the
  band-by-band write pattern and groups full years for time-window reads.
  Blosc/LZ4 compression is applied; for ~10% True (90th percentile) boolean
  data the compressed size is roughly ¼ of the raw size.

Pipeline stages (per band)
--------------------------
1. Load raw data lazily → select this lat band → convert to noleap calendar.
2. Trim to padded time range ``[agg_start, agg_end]`` and validate bounds.
3. Apply lazy rolling aggregation (xarray/dask).
4. Materialise the band (``compute()``): peak RAM ≈ budget.
5. Slice out the reference period; call ``compute_thresholds``; write to the
   thresholds zarr.
6. Slice out the analysis period; call ``detect_exceedances``; write to the
   exceedances zarr.

Outputs
-------
``thresholds.zarr``   – float32, shape ``(365, nlat, nlon)``
``exceedances.zarr``  – bool,    shape ``(n_an_days, nlat, nlon)``, compressed
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

import cftime
import numpy as np
import xarray as xr
import zarr
from zarr.codecs import BloscCodec, BloscShuffle
from tqdm import tqdm

from core import (
    AggMethod,
    rolling_aggregate_xarray,
    compute_thresholds,
    detect_exceedances,
)
from utils.utils import ensure_dimensions


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """All parameters needed for one experiment run.

    Dates must be ``cftime.DatetimeNoLeap`` objects.
    ``agg_method`` is a string: ``'mean'``, ``'max'``, ``'min'``, or ``'sum'``.
    ``lat_band_size``: rows processed at once (``None`` → auto from 1 GB budget).
    """
    input_zarr_path: str
    var:            str
    ref_start:      object    # cftime.DatetimeNoLeap
    ref_end:        object
    an_start:       object
    an_end:         object
    agg_window:     int
    agg_method:     str       # 'mean' | 'max' | 'min' | 'sum'
    perc_boost:     int
    percentile:     float
    output_dir:     str
    lat_band_size:  Optional[int] = None   # None → auto


_STR_TO_AGG = {
    "mean": AggMethod.MEAN,
    "max":  AggMethod.MAX,
    "min":  AggMethod.MIN,
    "sum":  AggMethod.SUM,
}


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

class Experiment:
    """Run the three-stage extremes pipeline for a given Config."""

    def __init__(self, cfg: Config) -> None:
        _validate(cfg)

        self.input_zarr_path = cfg.input_zarr_path
        self.var         = cfg.var
        self.agg_method  = _STR_TO_AGG[cfg.agg_method.lower()]
        self.ref_start   = cfg.ref_start
        self.ref_end     = cfg.ref_end
        self.an_start    = cfg.an_start
        self.an_end      = cfg.an_end
        self.agg_window  = cfg.agg_window
        self.perc_boost  = cfg.perc_boost
        self.percentile  = cfg.percentile
        self.n_years     = cfg.ref_end.year - cfg.ref_start.year + 1
        self._cfg_lat_band_size = cfg.lat_band_size

        # The rolling window needs half_agg days of padding so that no
        # boundary value in the ref or analysis period is NaN.
        half_agg = timedelta(days=self.agg_window // 2)
        self.agg_start = min(self.ref_start, self.an_start) - half_agg
        self.agg_end   = max(self.ref_end,   self.an_end)   + half_agg

        # Precomputed integer offsets (days from agg_start)
        self._ref_offset  = (self.ref_start - self.agg_start).days
        self._an_offset   = (self.an_start  - self.agg_start).days

        perc_str = str(self.percentile).replace(".", "_")
        self.experiment_dir = os.path.join(
            cfg.output_dir,
            f"{self.var}"
            f"_ref{self.ref_start.year}-{self.ref_end.year}"
            f"_an{self.an_start.year}-{self.an_end.year}"
            f"_agg{self.agg_window}{cfg.agg_method}"
            f"_boost{self.perc_boost}",
        )
        os.makedirs(self.experiment_dir, exist_ok=True)

        self.thresholds_zarr_path  = os.path.join(self.experiment_dir,
                                                   "thresholds.zarr")
        self.exceedances_zarr_path = os.path.join(self.experiment_dir,
                                                   f"exceedances_{perc_str}.zarr")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self._outputs_exist():
            print("All outputs already exist – nothing to do.")
            print(f"  Thresholds : {self.thresholds_zarr_path}")
            print(f"  Exceedances: {self.exceedances_zarr_path}")
            return

        # ---- Open raw data (lazy, no data loaded yet) ----------------
        raw = xr.open_zarr(self.input_zarr_path)[self.var]
        raw = ensure_dimensions(raw)
        raw = raw.convert_calendar("noleap")

        nlat = raw.sizes["latitude"]
        nlon = raw.sizes["longitude"]
        n_total_days = (self.agg_end - self.agg_start).days + 1
        n_an_days    = (self.an_end  - self.an_start).days  + 1

        # ---- Validate source data covers the required range ----------
        # Check on a single point to avoid loading the full grid.
        _check_bounds(
            raw.isel(latitude=0, longitude=0)
               .sel(time=slice(self.agg_start, self.agg_end)),
            self.agg_start, self.agg_end,
        )

        # ---- Decide lat_band_size ------------------------------------
        lat_band_size = self._cfg_lat_band_size or _auto_lat_band_size(
            n_total_days, nlon, budget_bytes=1_000_000_000
        )
        lat_band_size = min(lat_band_size, nlat)

        print(
            f"Grid: {nlat} lat × {nlon} lon | "
            f"lat_band_size: {lat_band_size} rows | "
            f"bands: {-(-nlat // lat_band_size)} | "
            f"peak RAM per band: "
            f"~{n_total_days * lat_band_size * nlon * 4 / 1e6:.0f} MB"
        )

        # ---- Pre-create output stores (empty, with schema) -----------
        lat_vals = raw.latitude.values.astype(np.float32)
        lon_vals = raw.longitude.values.astype(np.float32)
        an_time_vals = (
            raw.isel(latitude=0, longitude=0)
               .sel(time=slice(self.an_start, self.an_end))
               .time.values
        )

        thr_store = _create_zarr_store(
            self.thresholds_zarr_path,
            shape=(365, nlat, nlon),
            chunks=(365, lat_band_size, nlon),
            dtype=np.float32,
            dims=["dayofyear", "latitude", "longitude"],
            coords={
                "dayofyear": np.arange(1, 366, dtype=np.int32),
                "latitude":  lat_vals,
                "longitude": lon_vals,
            },
        )

        exc_store = _create_zarr_store(
            self.exceedances_zarr_path,
            shape=(n_an_days, nlat, nlon),
            chunks=(min(365, n_an_days), lat_band_size, nlon),
            dtype=bool,
            dims=["time", "latitude", "longitude"],
            coords={
                "time":      _encode_time(an_time_vals),
                "latitude":  lat_vals,
                "longitude": lon_vals,
            },
            compressors=[BloscCodec(cname="lz4", clevel=5, shuffle=BloscShuffle.bitshuffle)],
        )

        # ---- Main loop: one lat band at a time ----------------------
        for lat_start in tqdm(range(0, nlat, lat_band_size), desc="lat bands"):
            lat_end = min(lat_start + lat_band_size, nlat)

            # --- 1. Load and aggregate (lazy → compute) ---------------
            band_raw = (
                raw.isel(latitude=slice(lat_start, lat_end))
                   .sel(time=slice(self.agg_start, self.agg_end))
            )
            # Materialise: (n_total_days, band_rows, nlon), float32
            agg_band = (
                rolling_aggregate_xarray(band_raw, self.agg_window, self.agg_method)
                .astype(np.float32)
                .compute()
                .values  # numpy array from here on
            )

            # --- 2. Thresholds ----------------------------------------
            ref_data = agg_band[
                self._ref_offset : self._ref_offset + self.n_years * 365
            ]
            band_thr = compute_thresholds(
                ref_data.astype(np.float64),
                self.perc_boost,
                self.percentile,
            ).astype(np.float32)

            thr_store[:, lat_start:lat_end, :] = band_thr

            # --- 3. Exceedances ---------------------------------------
            an_data = agg_band[self._an_offset : self._an_offset + n_an_days]
            doys    = np.tile(np.arange(1, 366), -(-n_an_days // 365))[:n_an_days]

            # If the analysis period doesn't start on Jan 1 we need the
            # correct starting DOY, not 1.
            an_start_doy = int(an_time_vals[0].timetuple().tm_yday)
            if an_start_doy != 1:
                doys = np.tile(np.arange(1, 366), 1 + n_an_days // 365)
                doys = doys[an_start_doy - 1 : an_start_doy - 1 + n_an_days]

            band_exc = detect_exceedances(
                an_data.astype(np.float64),
                doys,
                band_thr.astype(np.float64),
            )

            exc_store[:, lat_start:lat_end, :] = band_exc

        print("\nDone.")
        print(f"  Thresholds : {self.thresholds_zarr_path}")
        print(f"  Exceedances: {self.exceedances_zarr_path}")

    # ------------------------------------------------------------------

    def _outputs_exist(self) -> bool:
        return (
            os.path.exists(self.thresholds_zarr_path)
            and os.path.exists(self.exceedances_zarr_path)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate(cfg: Config) -> None:
    if cfg.agg_window < 1 or cfg.agg_window % 2 == 0:
        raise ValueError(
            f"agg_window must be a positive odd integer, got {cfg.agg_window}"
        )
    if cfg.perc_boost < 1 or cfg.perc_boost % 2 == 0:
        raise ValueError(
            f"perc_boost must be a positive odd integer, got {cfg.perc_boost}"
        )
    if not (0.0 < cfg.percentile < 1.0):
        raise ValueError(
            f"percentile must be in (0, 1), got {cfg.percentile}"
        )
    if cfg.agg_method.lower() not in _STR_TO_AGG:
        raise ValueError(
            f"agg_method must be one of {list(_STR_TO_AGG)}, "
            f"got '{cfg.agg_method}'"
        )
    if cfg.ref_end < cfg.ref_start:
        raise ValueError("ref_end must be >= ref_start")
    if cfg.an_end < cfg.an_start:
        raise ValueError("an_end must be >= an_start")


def _check_bounds(da, expected_start, expected_end) -> None:
    actual_start = da["time"].values[0]
    actual_end   = da["time"].values[-1]
    if actual_start != expected_start:
        raise ValueError(
            f"Data starts at {actual_start} but expected {expected_start}. "
            f"The source zarr does not cover the full padded range required "
            f"by the rolling window (agg_window // 2 days before the first "
            f"day of interest)."
        )
    if actual_end != expected_end:
        raise ValueError(
            f"Data ends at {actual_end} but expected {expected_end}. "
            f"The source zarr does not cover the full padded range."
        )


def _auto_lat_band_size(n_total_days: int, nlon: int,
                        budget_bytes: int = 1_000_000_000) -> int:
    """Largest lat band whose aggregated float32 array fits in *budget_bytes*."""
    bytes_per_row = n_total_days * nlon * 4  # float32
    return max(1, budget_bytes // bytes_per_row)


def _encode_time(cftime_values) -> np.ndarray:
    """Encode cftime.DatetimeNoLeap values as int32 days since 1900-01-01."""
    epoch = cftime.DatetimeNoLeap(1900, 1, 1)
    return np.array([(t - epoch).days for t in cftime_values], dtype=np.int32)


def _create_zarr_store(
    path: str,
    shape: tuple,
    chunks: tuple,
    dtype,
    dims: list[str],
    coords: dict,
    compressors=None,
) -> zarr.Array:
    """Pre-create a zarr group with one data array and coordinate arrays.

    The ``_ARRAY_DIMENSIONS`` attribute written on every array makes the
    store openable with ``xr.open_zarr(path)`` without any extra arguments.
    """
    if os.path.exists(path):
        import shutil
        shutil.rmtree(path)

    grp = zarr.open_group(path, mode="w")

    kw = dict(shape=shape, chunks=chunks, dtype=dtype, dimension_names=dims)
    if compressors is not None:
        kw["compressors"] = compressors
    data_arr = grp.create_array("data", **kw)

    for coord_name, coord_vals in coords.items():
        grp.create_array(coord_name, data=coord_vals,
                         chunks=coord_vals.shape,
                         dimension_names=[coord_name])

    return data_arr
