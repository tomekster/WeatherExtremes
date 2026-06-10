# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

External system dependencies: `cdo`, `geos`, `proj`, `gdal` (via apt/brew). Python environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

The venv lives at `venv/` (not `.venv/`). Always activate with `source venv/bin/activate` before running scripts.

## Common Commands

**Run the main pipeline** (compute thresholds + exceedances for ERA5 data):
```bash
python src/main.py \
    --input   data/preprocessed/rechunked/t2max_rechunked.zarr \
    --var     daily_max_2m_temperature \
    --ref-start 1960-01-01 --ref-end 1989-12-31 \
    --an-start  1960-01-01 --an-end  2019-12-31 \
    --agg-window 3 --agg-method max --perc-boost 3 \
    --percentile 0.90 --output experiments/
```
See `run_2m_temp.sh` and `run_precip.sh` for preconfigured invocations.

**Launch the Dash visualisation app:**
```bash
bash launch_app.sh
# or directly:
python src/vis/exceedances_app.py --experiments-dir experiments/
```

**Run tests:**
```bash
python -m pytest test/test_core.py -v
python -m pytest test/test_HadGHCND_zarr.py -v
```

**Run a single test file:**
```bash
python -m pytest test/test_core.py::TestClassName::test_method -v
```

**Synthetic experiments:**
```bash
python src/generate_synthetic_t2max.py --mode sine --amplitude 15 --mean-temp 10
python src/run_synthetic_extremes.py
bash run_synthetic_battery.sh --input data/synthetic/synthetic_t2max.zarr
python src/summarise_synthetic_trends.py
```

## Architecture

The project detects weather extremes (exceedances of per-DOY percentile thresholds) in large global climate datasets, operating on ERA5 reanalysis data (~45–91 GB zarr stores).

### Three-stage pipeline

All core logic lives in `src/core/` and has two variants each — a **pure numpy** version (for tests and small data) and a **lazy xarray/dask** production version:

| Stage | numpy variant | production variant |
|---|---|---|
| Rolling aggregation | `rolling_aggregate` | `rolling_aggregate_xarray` |
| Per-DOY threshold computation | `compute_thresholds` | `compute_thresholds_chunked` |
| Exceedance detection | `detect_exceedances` | `detect_exceedances_xarray` |

### Memory management strategy

The pipeline uses **band-by-band processing** (`src/experiment.py`): the full global grid is never loaded into RAM. Instead, latitude bands (auto-sized to ~1 GB) are processed one at a time. Output zarr stores (`thresholds.zarr`, `exceedances_<percentile>.zarr`) are pre-created empty and filled band-by-band with streaming writes.

### Key data constraints

- All time axes use the **no-leap calendar** (every year = exactly 365 days). Data is converted with `xr.convert_calendar("noleap")` before processing.
- `perc_boost` and `agg_window` must be **positive odd integers**.
- Input zarr must have `(time, latitude, longitude)` dimensions.

### Experiment outputs

Each run writes to `experiments/<var>_ref<start>-<end>_an<start>-<end>_agg<N><method>_boost<N>/`:
- `thresholds.zarr` — shape `(365, nlat, nlon)` float32
- `exceedances_<percentile>.zarr` — shape `(n_an_days, nlat, nlon)` bool, Blosc/LZ4 compressed

Both are xarray-compatible: `xr.open_zarr(path)["data"]`.

### Source layout

- `src/core/` — pure pipeline logic (aggregation, thresholds, exceedances)
- `src/experiment.py` + `src/main.py` — CLI entry point and `Config`/`Experiment` orchestration
- `src/data_loading/` — `GCSDataLoader` (WeatherBench2 on GCS), `ZarrLoader`
- `src/preprocess.py` — converts raw ERA5 6-hourly zarr → daily aggregates
- `src/vis/exceedances_app.py` — Dash app for exploring ERA5 results and synthetic experiments
- `src/generate_synthetic_t2max.py` — builds synthetic temperature zarr (ERA5-derived or sine-wave modes) with configurable warming slopes × noise variances
- `src/run_synthetic_extremes.py` — runs the full pipeline over the synthetic zarr
- `src/summarise_synthetic_trends.py` — outputs CSV/PDF of detected trends
- `test/` — pytest tests; tests add `src/` to `sys.path` directly, no package install needed

### Data sources

- ERA5 data originates from WeatherBench2 on GCS (`gs://weatherbench2/datasets/era5/...`)
- Preprocessed daily zarrs live in `data/preprocessed/rechunked/`
- HadGHCND observational data in `data/HadGHCND/` (decade-split NetCDF files)
