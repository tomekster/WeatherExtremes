"""
Core implementations for weather extremes detection.

Three pipeline stages, each with two variants:

  ┌─────────────────────────────┬────────────────────────────────────────┐
  │ numpy (in-memory, tests)    │ lazy/chunked (production, large zarr)  │
  ├─────────────────────────────┼────────────────────────────────────────┤
  │ rolling_aggregate           │ rolling_aggregate_xarray               │
  │ compute_thresholds          │ compute_thresholds_chunked             │
  │ detect_exceedances          │ detect_exceedances_xarray              │
  └─────────────────────────────┴────────────────────────────────────────┘

The numpy variants load all data into memory and are suited for unit tests
and single lat-band processing.  The production variants use xarray/dask
(lazy) or explicit zarr band-by-band iteration so that the 45–90 GB datasets
never need to fit in RAM simultaneously.
"""
from .aggregation import AggMethod, rolling_aggregate, rolling_aggregate_xarray
from .thresholds  import compute_thresholds, compute_thresholds_chunked
from .exceedances import detect_exceedances, detect_exceedances_xarray

__all__ = [
    "AggMethod",
    "rolling_aggregate",
    "rolling_aggregate_xarray",
    "compute_thresholds",
    "compute_thresholds_chunked",
    "detect_exceedances",
    "detect_exceedances_xarray",
]
