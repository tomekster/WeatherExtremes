"""
Exceedance detection: flag days where aggregated values exceed their DOY threshold.

The comparison is strict (``>``), matching the existing pipeline behaviour.

Two entry points
----------------
``detect_exceedances``
    Pure numpy function.  Loads all of *agg_data* into memory.  Use for unit
    tests and small datasets only; the full 60-year analysis period is ~91 GB.

``detect_exceedances_xarray``
    Lazy xarray wrapper.  Uses ``DataArray.groupby("time.dayofyear")`` so
    that xarray/dask computes the comparison chunk-by-chunk and the data is
    never fully materialised.  The result can be written directly to zarr with
    ``.to_zarr()``.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Pure numpy – for tests and single-band processing
# ---------------------------------------------------------------------------

def detect_exceedances(
    agg_data: np.ndarray,
    doys: np.ndarray,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Return a boolean mask where *agg_data* strictly exceeds *thresholds*.

    WARNING – loads all of *agg_data* into memory.  For production datasets
    use ``detect_exceedances_xarray``.

    Parameters
    ----------
    agg_data:
        Aggregated analysis-period values, shape ``(T, ...)``.
        Must contain no NaN values.
    doys:
        Integer day-of-year for each time step, shape ``(T,)``.
        Values must be in ``[1, 365]`` (1-indexed, no-leap calendar).
    thresholds:
        Per-DOY thresholds, shape ``(365, ...)``.
        ``thresholds[d]`` corresponds to DOY ``d + 1``.

    Returns
    -------
    np.ndarray
        Boolean array of shape ``(T, ...)``.  ``True`` where *agg_data*
        strictly exceeds the corresponding DOY threshold.

    Raises
    ------
    ValueError
        If shapes are inconsistent, DOY values are out of range, or inputs
        contain NaN values.
    """
    T = agg_data.shape[0]
    rest = agg_data.shape[1:]

    if doys.shape != (T,):
        raise ValueError(
            f"doys must have shape ({T},), got {doys.shape}"
        )
    if thresholds.shape != (365, *rest):
        raise ValueError(
            f"thresholds must have shape (365, {rest}), got {thresholds.shape}"
        )
    if np.isnan(agg_data).any():
        raise ValueError("agg_data contains NaN values")
    if np.isnan(thresholds).any():
        raise ValueError("thresholds contains NaN values")
    if doys.min() < 1 or doys.max() > 365:
        raise ValueError(
            f"doys values must be in [1, 365], got min={doys.min()} max={doys.max()}"
        )

    # Index into thresholds using 0-based DOY index.
    threshold_per_day = thresholds[doys - 1]  # (T, *rest)
    return agg_data > threshold_per_day


# ---------------------------------------------------------------------------
# Lazy xarray wrapper – for production use with large zarr datasets
# ---------------------------------------------------------------------------

def detect_exceedances_xarray(agg_data, thresholds_np: np.ndarray):
    """Return a lazy boolean DataArray flagging exceedances.

    Uses xarray's ``groupby("time.dayofyear")`` so that the comparison is
    evaluated lazily by dask, one chunk at a time.  The full analysis-period
    dataset is never loaded into RAM.

    Parameters
    ----------
    agg_data:
        ``xr.DataArray`` with dimensions ``(time, latitude, longitude)``
        using a no-leap calendar.  Backed by dask (e.g. from
        ``xr.open_zarr``).  Must cover the full analysis period.
    thresholds_np:
        Per-DOY thresholds as a numpy array, shape ``(365, nlat, nlon)``.
        This is the in-memory output of ``compute_thresholds_chunked``.

    Returns
    -------
    xr.DataArray
        Lazy boolean DataArray of the same shape as *agg_data*.  ``True``
        where *agg_data* strictly exceeds the DOY threshold.  Write to disk
        with ``.to_zarr(path)`` to trigger computation.
    """
    import xarray as xr

    lat_coords = agg_data.latitude
    lon_coords = agg_data.longitude

    threshold_da = xr.DataArray(
        thresholds_np,
        dims=["dayofyear", "latitude", "longitude"],
        coords={
            "dayofyear": np.arange(1, 366),
            "latitude":  lat_coords,
            "longitude": lon_coords,
        },
    )

    # groupby aligns each time step to its DOY threshold before the comparison.
    return agg_data.groupby("time.dayofyear") > threshold_da
