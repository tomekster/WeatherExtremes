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

**Launch the point temperature viewer** (Dash; click a grid cell on the map → per-DOY thresholds and exceedances for one year via `ExceedancesCalculator.compute_exceedances`, all its parameters as sliders; a `TemperatureModel` fitted to the clicked point and one seeded synthetic realization of it are shown alongside). Must run from the repo root because the zarr path in `scripts/load_vienna_data.py` is relative:
```bash
python -m src.app --port 8051
```
The server is accessed over SSH, so both Dash apps bind to `127.0.0.1` and are viewed through a forwarded port. Desktop GUI toolkits (Qt etc.) have no display there.

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
- `test/` — pytest tests; tests add `src/` to `sys.path` directly, no package install needed. `test_soruce_data_procesing.py` is not a real test: it calls its function at import time and needs a local `data/era5_wb2_1950-2020_combined.zarr`, so leave it out of normal test runs.
- Legacy/scratch code that nothing imports: `src/exceedences.py`, `src/direct_solution/`, `src/optimised_solution/`, `src/scratch*.py`, `src/bug_report.py` and the root-level `analyse_exceedances.py`/`test_ftp.py`. Make pipeline changes in `src/core/` + `src/experiment.py`, not there.

### Vienna single-point synthetic study (`src/synthetic_vienna/` + `scripts/`)

A separate, self-contained line of work from the gridded pipeline above. It studies how the detected trend in yearly exceedance counts depends on the experiment settings, using one ERA5 grid point (Vienna):

- `scripts/load_vienna_data.py::load_vienna_subset()` pulls the Vienna point from `data/preprocessed/rechunked/t2max_rechunked.zarr`, converts it to °C and the no-leap calendar, and keeps 1961–2020.
- `TemperatureModel` (`src/synthetic_vienna/temperature_model.py`) fits intercept + linear trend + K annual harmonics (`fit(y, L=1, K=3, ...)`, time in no-leap years, i.e. days/365), with optional AR(L) residuals (statsmodels; scripts use AR(1)). Parameters live in the `ModelParams` dataclass (`trend`, `sin_coeffs`/`cos_coeffs` of length K, `ar_coeffs` of length L, `innovation_std`). `generate()` produces synthetic series with an overridden warming rate (`warming_rate`, °C/yr; 0.0 means no trend), white-noise `std`, or `ar_coeffs` (one value per fitted lag; innovation std rescaled to keep the noise variance). `std` is the white-noise std, or with autocorrelation the stationary std of the AR noise (innovation std set to σ·√(1−ρ²) for AR(1)); `ar_coeffs` is only valid with autocorrelation. `a_` takes `--noise white|ar` and `--in-base-bootstrap` (collect and plot); output files get `_ar` / `_zb` suffixes.
- `ExceedancesCalculator` (`src/synthetic_vienna/exceedances.py`) is its **own numpy/xarray implementation** (optional `subtract_seasonal_cycle=True` removes the reference-period per-DOY mean before thresholding, per Brunner & Voigt 2024, Nat. Commun.) working on `(n_years, 365)` arrays. `in_base_bootstrap=True` applies the Zhang et al. (2005) in-base bootstrap (procedure as in `climdex.pcic`) to analysis years inside the reference period; `exceedances` is then a float array of rates, which `trends.pot_trend` (runs declustering) rejects. Percentiles go through `_percentile`, because `np.nanpercentile` with an `axis` loops in Python and makes the bootstrap very slow. It does **not** use `src/core/`, so a change to one does not affect the other.
- `src/synthetic_vienna/trends.py` fits trends to the yearly exceedance frequency of a `(n_years, 365)` exceedance array: OLS, Theil–Sen, NHPP (Poisson GLM with log-linear rate on yearly counts) and POT occurrence (runs-declustered cluster counts → Poisson GLM, converted back to days via extremal index θ). Poisson p-values are quasi-Poisson.
- `scripts/synthetic_plots/` holds paired scripts per figure (`a_`–`d_`). `*_collect_data.py` runs the parameter sweep and pickles a dict keyed by parameter tuples into `scripts/synthetic_plots/data/<name>/`. `*_plot.py` loads that pickle and plots it. `scripts/synthetic_plots/vis_inspect_intermediate_results/` holds sanity-check plots of the model fit and of single exceedance/trend runs.

These scripts import with `from src...` / `from scripts...`, so run them as modules from the repo root:
```bash
python -m scripts.synthetic_plots.a_ex_count_per_warm_trend_and_ref_collect_data
python -m scripts.synthetic_plots.a_ex_count_per_warm_trend_and_ref_plot
```

### Data sources

- ERA5 data originates from WeatherBench2 on GCS (`gs://weatherbench2/datasets/era5/...`)
- Preprocessed daily zarrs live in `data/preprocessed/rechunked/`
- HadGHCND observational data in `data/HadGHCND/` (decade-split NetCDF files)
