"""
Rolling-window aggregation over the time axis.

Two entry points
----------------
``rolling_aggregate``
    Pure numpy function.  Loads all data into memory.  Use for unit tests
    and small datasets only.

``rolling_aggregate_xarray``
    Lazy xarray/dask wrapper.  For production use with large zarr datasets:
    xarray's ``.rolling()`` chains operations on dask-backed arrays so the
    data is never fully materialised in RAM.  The caller is responsible for
    converting to a no-leap calendar and for loading enough padding data
    around the period of interest (so that the rolling window produces no NaN
    values at the boundaries of the actual period of interest).
"""
from __future__ import annotations

import numpy as np
from enum import Enum


class AggMethod(Enum):
    MEAN = "mean"
    SUM  = "sum"
    MAX  = "max"
    MIN  = "min"


# ---------------------------------------------------------------------------
# Pure numpy – for tests and single-band processing
# ---------------------------------------------------------------------------

def rolling_aggregate(
    data: np.ndarray,
    window: int,
    method: AggMethod,
) -> np.ndarray:
    """Apply a centred rolling window of size *window* along axis 0.

    WARNING – loads all of *data* into memory.  Do NOT call this on the full
    global grid; use ``rolling_aggregate_xarray`` for production datasets.

    Parameters
    ----------
    data:
        Array of shape ``(T, ...)``.  The time axis is axis 0.
    window:
        Number of time steps in the rolling window.  Must be a positive odd
        integer so the window is symmetric around each centre point.
    method:
        Aggregation method (MEAN, SUM, MAX, or MIN).

    Returns
    -------
    np.ndarray
        Same shape as *data*.  The first and last ``window // 2`` positions
        along axis 0 are ``NaN`` because there is insufficient context for a
        full window.

    Raises
    ------
    ValueError
        If *window* is not a positive odd integer.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(
            f"window must be a positive odd integer, got {window}"
        )

    if window == 1:
        return data.copy()

    half = window // 2
    T = data.shape[0]
    out = np.full(data.shape, np.nan, dtype=np.float64)

    dispatch = {
        AggMethod.MEAN: np.mean,
        AggMethod.SUM:  np.sum,
        AggMethod.MAX:  np.max,
        AggMethod.MIN:  np.min,
    }
    fn = dispatch[method]

    for t in range(half, T - half):
        out[t] = fn(data[t - half : t + half + 1], axis=0)

    return out


# ---------------------------------------------------------------------------
# Lazy xarray wrapper – for production use with large zarr datasets
# ---------------------------------------------------------------------------

def rolling_aggregate_xarray(data, window: int, method: AggMethod):
    """Apply a centred rolling window using xarray's lazy dask backend.

    The full dataset is never loaded into RAM; operations are chained on the
    dask computation graph and only materialised when the result is written to
    disk (e.g. via ``.to_zarr()``).

    Parameters
    ----------
    data:
        ``xr.DataArray`` with a ``time`` dimension, backed by dask (e.g.
        opened with ``xr.open_zarr``).  Must already use a no-leap calendar.
    window:
        Positive odd integer rolling-window size.
    method:
        Aggregation method (MEAN, SUM, MAX, or MIN).

    Returns
    -------
    xr.DataArray
        Lazy DataArray of the same shape.  Edge values (first and last
        ``window // 2`` time steps) are ``NaN``.

    Raises
    ------
    ValueError
        If *window* is not a positive odd integer.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(
            f"window must be a positive odd integer, got {window}"
        )

    if window == 1:
        return data

    rolling = data.rolling(time=window, center=True, min_periods=window)

    dispatch = {
        AggMethod.MEAN: rolling.mean,
        AggMethod.SUM:  rolling.sum,
        AggMethod.MAX:  rolling.max,
        AggMethod.MIN:  rolling.min,
    }
    return dispatch[method]()
