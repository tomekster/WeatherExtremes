"""
Per-DOY percentile threshold computation.

Terminology
-----------
* **DOY** – day of year, 1-indexed (1 = Jan 1, 365 = Dec 31).  All
  computation assumes a **no-leap** calendar (every year has exactly 365 days).
* **perc_boost** – total boosting-window width in DOYs.  Must be a positive
  odd integer.  ``perc_boost = 1`` means no boosting (only the target DOY
  itself contributes).  ``perc_boost = 2b + 1`` gives a symmetric window of
  *b* DOYs on each side of the target DOY (``half = perc_boost // 2``).  The
  window wraps around the year boundary (DOY 1 is adjacent to DOY 365).

Algorithm (per spatial location / lat band)
-------------------------------------------
For a reference period of *n_years* complete no-leap years:

1. Reshape the flat ``(n_years * 365, ...)`` band to ``(n_years, 365, ...)``.
2. Circularly pad the DOY axis by ``half`` DOYs on each side so that the
   window for DOY 1 can look back into DOY 365 of the same years and
   vice-versa.
3. For each DOY index *d* (0-based), slice
   ``padded[:, d : d + perc_boost, ...]``, flatten the (n_years, perc_boost)
   dims, and compute ``np.quantile(..., percentile, axis=0)``.

Two entry points
----------------
``compute_thresholds``
    Pure numpy function.  Operates on an in-memory array of one lat band (or
    the full grid for small test datasets).  Called by
    ``compute_thresholds_chunked`` for each band.

``compute_thresholds_chunked``
    Production entry point.  Reads a zarr array band-by-band so that at most
    ``lat_band_size`` latitude rows are in RAM at once, keeping peak memory
    usage around ``n_years * 365 * lat_band_size * nlon * 4`` bytes.  For the
    full 721 × 1440 grid with a 30-year reference period and
    ``lat_band_size=10`` that is roughly 630 MB per band.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pure numpy – operates on a single in-memory band
# ---------------------------------------------------------------------------

def compute_thresholds(
    ref_agg: np.ndarray,
    perc_boost: int,
    percentile: float,
) -> np.ndarray:
    """Compute per-DOY percentile thresholds from an in-memory band.

    WARNING – loads all of *ref_agg* into memory.  For the full global grid
    use ``compute_thresholds_chunked``.

    Parameters
    ----------
    ref_agg:
        Aggregated reference-period values, shape ``(n_years * 365, ...)``.
        Must contain no NaN values.  Years must be contiguous and complete
        (no-leap calendar, exactly 365 days each).
    perc_boost:
        Total boosting-window width in DOYs.  Positive odd integer.
        ``1`` = no boosting (target DOY only).
    percentile:
        Quantile to compute, in ``[0.0, 1.0]``.

    Returns
    -------
    np.ndarray
        Shape ``(365, ...)``.  ``thresholds[d]`` is the threshold for
        DOY ``d + 1`` (0-based indexing internally).

    Raises
    ------
    ValueError
        If *perc_boost* is not a positive odd integer, if *ref_agg* has a
        time axis not divisible by 365, or if *ref_agg* contains NaN values.
    """
    if perc_boost < 1 or perc_boost % 2 == 0:
        raise ValueError(
            f"perc_boost must be a positive odd integer, got {perc_boost}"
        )
    if not (0.0 <= percentile <= 1.0):
        raise ValueError(f"percentile must be in [0, 1], got {percentile}")

    total_days = ref_agg.shape[0]
    if total_days % 365 != 0:
        raise ValueError(
            f"ref_agg time axis length ({total_days}) must be a multiple of 365"
        )
    if np.isnan(ref_agg).any():
        raise ValueError("ref_agg contains NaN values")

    n_years = total_days // 365
    half = perc_boost // 2
    rest = ref_agg.shape[1:]

    # Shape: (n_years, 365, *rest)
    data = ref_agg.reshape(n_years, 365, *rest)

    # Circular padding along the DOY axis (axis 1):
    #   padded[:, 0 : half]        → last `half` DOYs of the year (wrap-back)
    #   padded[:, half : half+365] → all 365 DOYs (main body)
    #   padded[:, half+365 :]      → first `half` DOYs of the year (wrap-forward)
    if half > 0:
        prefix = data[:, 365 - half :, ...]  # (n_years, half, *rest)
        suffix = data[:, :half, ...]          # (n_years, half, *rest)
        padded = np.concatenate([prefix, data, suffix], axis=1)
    else:
        padded = data  # perc_boost == 1, no padding needed

    thresholds = np.empty((365, *rest), dtype=np.float64)
    for d in range(365):
        # Window: perc_boost DOY-groups centred on DOY d+1.
        window = padded[:, d : d + perc_boost, ...]  # (n_years, perc_boost, *rest)
        flat = window.reshape(n_years * perc_boost, *rest)
        thresholds[d] = np.quantile(flat, percentile, axis=0)

    return thresholds


# ---------------------------------------------------------------------------
# Chunked production variant – reads from a zarr store band-by-band
# ---------------------------------------------------------------------------

def compute_thresholds_chunked(
    ref_zarr,
    n_years: int,
    perc_boost: int,
    percentile: float,
    lat_band_size: int = 10,
    time_offset: int = 0,
) -> np.ndarray:
    """Compute per-DOY thresholds from a zarr store without loading it fully.

    Iterates over latitude bands of size *lat_band_size*, reading each band
    from *ref_zarr* into memory, calling ``compute_thresholds``, and writing
    the result into the output array.

    Peak memory usage is approximately::

        n_years * 365 * lat_band_size * nlon * 4  bytes

    For the full 721 × 1440 grid with n_years=30 and lat_band_size=10 that
    is ~630 MB, well within typical available RAM.

    Parameters
    ----------
    ref_zarr:
        A zarr array (or any object supporting ``[t_slice, lat_slice, :]``
        slicing) of shape ``(T, nlat, nlon)`` where ``T >= time_offset +
        n_years * 365``.  This may be the full aggregated zarr; only the
        reference-period window is read.
    n_years:
        Number of complete no-leap years in the reference period.
    perc_boost:
        Total boosting-window width in DOYs.  Positive odd integer.
    percentile:
        Quantile to compute, in ``[0.0, 1.0]``.
    lat_band_size:
        Number of latitude rows to process at once.  Decrease if memory is
        tight; increase for better I/O throughput.  Default: 10.
    time_offset:
        First time index of the reference period within *ref_zarr*.  The
        reference period occupies ``ref_zarr[time_offset : time_offset +
        n_years * 365]``.  Default: 0 (ref period starts at the beginning).

    Returns
    -------
    np.ndarray
        Shape ``(365, nlat, nlon)``, dtype ``float64``.

    Raises
    ------
    ValueError
        If the zarr time axis is too short, *lat_band_size* is not a positive
        integer, or *time_offset* is negative.
    """
    if lat_band_size < 1:
        raise ValueError(
            f"lat_band_size must be a positive integer, got {lat_band_size}"
        )
    if time_offset < 0:
        raise ValueError(
            f"time_offset must be non-negative, got {time_offset}"
        )

    total_days, nlat, nlon = ref_zarr.shape
    t0 = time_offset
    t1 = time_offset + n_years * 365

    if total_days < t1:
        raise ValueError(
            f"ref_zarr time axis has {total_days} steps but time_offset={time_offset} "
            f"and n_years={n_years} require at least {t1} steps."
        )

    thresholds = np.empty((365, nlat, nlon), dtype=np.float64)

    for lat_start in range(0, nlat, lat_band_size):
        lat_end = min(lat_start + lat_band_size, nlat)
        # Read one band: materialises (n_years*365, band_rows, nlon) in RAM.
        band = np.asarray(ref_zarr[t0:t1, lat_start:lat_end, :], dtype=np.float64)
        thresholds[:, lat_start:lat_end, :] = compute_thresholds(
            band, perc_boost, percentile
        )

    return thresholds
