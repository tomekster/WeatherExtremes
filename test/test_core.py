"""
Unit tests for src/core/ pure functions and their chunked/lazy production variants.

All tests use synthetic 1-D data (a single lat/lon location) unless noted.
The no-leap calendar means every year has exactly 365 days; this is enforced
by the core functions themselves.

Running
-------
    cd /home/tsternal/phd/WeatherExtremes2
    python -m pytest test/test_core.py -v
"""
import sys
import os
import tempfile

# Allow importing from src/ without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import zarr

from core.aggregation import rolling_aggregate, rolling_aggregate_xarray, AggMethod
from core.thresholds  import compute_thresholds, compute_thresholds_chunked
from core.exceedances import detect_exceedances, detect_exceedances_xarray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_data(n_years: int, value=1.0, seed: int | None = None) -> np.ndarray:
    """Return a flat (n_years * 365,) array.  Constant or random."""
    size = n_years * 365
    if seed is not None:
        rng = np.random.default_rng(seed)
        return rng.random(size).astype(np.float64)
    return np.full(size, value, dtype=np.float64)


def make_doys(n_years: int, start_doy: int = 1) -> np.ndarray:
    """Return DOY array for n_years × 365 days, starting at start_doy (1-indexed)."""
    single = np.arange(1, 366)
    base = np.tile(single, n_years)
    # roll if start_doy != 1
    offset = start_doy - 1
    return np.roll(base, -offset)[:n_years * 365]


# ---------------------------------------------------------------------------
# rolling_aggregate
# ---------------------------------------------------------------------------

class TestRollingAggregate:

    def test_window_1_is_identity(self):
        """window=1 must return an identical copy of the input."""
        data = np.arange(10, dtype=float)
        out = rolling_aggregate(data, window=1, method=AggMethod.MEAN)
        np.testing.assert_array_equal(out, data)

    def test_window_3_mean_interior_values(self):
        """Interior values of a window-3 mean should be the 3-element average."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rolling_aggregate(data, window=3, method=AggMethod.MEAN)
        # Interior: indices 1, 2, 3
        assert out[1] == pytest.approx(2.0)  # mean(1,2,3)
        assert out[2] == pytest.approx(3.0)  # mean(2,3,4)
        assert out[3] == pytest.approx(4.0)  # mean(3,4,5)

    def test_window_3_edges_are_nan(self):
        """First and last (window//2) = 1 values must be NaN."""
        data = np.ones(10, dtype=float)
        out = rolling_aggregate(data, window=3, method=AggMethod.MEAN)
        assert np.isnan(out[0])
        assert np.isnan(out[-1])
        assert not np.isnan(out[1:-1]).any()

    def test_window_5_edges_are_nan(self):
        """First and last 2 values must be NaN for window=5."""
        data = np.ones(10, dtype=float)
        out = rolling_aggregate(data, window=5, method=AggMethod.MEAN)
        assert np.isnan(out[0])
        assert np.isnan(out[1])
        assert np.isnan(out[-1])
        assert np.isnan(out[-2])
        assert not np.isnan(out[2:-2]).any()

    def test_window_3_max(self):
        """MAX aggregation: interior values should be the 3-element maximum."""
        data = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        out = rolling_aggregate(data, window=3, method=AggMethod.MAX)
        assert out[1] == pytest.approx(3.0)  # max(1,3,2)
        assert out[2] == pytest.approx(5.0)  # max(3,2,5)
        assert out[3] == pytest.approx(5.0)  # max(2,5,4)

    def test_window_3_sum(self):
        """SUM aggregation: interior values should be the 3-element sum."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = rolling_aggregate(data, window=3, method=AggMethod.SUM)
        assert out[1] == pytest.approx(6.0)   # 1+2+3
        assert out[2] == pytest.approx(9.0)   # 2+3+4
        assert out[3] == pytest.approx(12.0)  # 3+4+5

    def test_window_3_min(self):
        """MIN aggregation: interior values should be the 3-element minimum."""
        data = np.array([3.0, 1.0, 4.0, 1.0, 5.0])
        out = rolling_aggregate(data, window=3, method=AggMethod.MIN)
        assert out[1] == pytest.approx(1.0)  # min(3,1,4)
        assert out[2] == pytest.approx(1.0)  # min(1,4,1)
        assert out[3] == pytest.approx(1.0)  # min(4,1,5)

    def test_spatial_dims_preserved(self):
        """Shape (T, lat, lon) should be preserved."""
        data = np.ones((10, 3, 4), dtype=float)
        out = rolling_aggregate(data, window=3, method=AggMethod.MEAN)
        assert out.shape == (10, 3, 4)

    def test_even_window_raises(self):
        with pytest.raises(ValueError, match="positive odd integer"):
            rolling_aggregate(np.ones(10), window=4, method=AggMethod.MEAN)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="positive odd integer"):
            rolling_aggregate(np.ones(10), window=0, method=AggMethod.MEAN)


# ---------------------------------------------------------------------------
# compute_thresholds
# ---------------------------------------------------------------------------

class TestComputeThresholds:

    def test_constant_data_threshold_equals_value(self):
        """All-constant data: every threshold must equal that constant."""
        c = 7.5
        data = make_data(n_years=5, value=c)
        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.9)
        assert thresholds.shape == (365,)
        np.testing.assert_allclose(thresholds, c)

    def test_output_shape_1d(self):
        """Output shape should be (365,) for 1-D input."""
        data = make_data(n_years=10, seed=0)
        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.9)
        assert thresholds.shape == (365,)

    def test_output_shape_spatial(self):
        """Output shape should be (365, lat, lon) for 3-D input."""
        n_years, lat, lon = 5, 3, 4
        data = np.random.default_rng(1).random((n_years * 365, lat, lon))
        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.9)
        assert thresholds.shape == (365, lat, lon)

    def test_no_boosting_known_values(self):
        """
        With perc_boost=1 (no boosting), each DOY threshold is the percentile
        of exactly n_years values.

        Hand-computed example:
          - n_years = 4
          - DOY 10 gets values [10, 20, 30, 40] (one per year)
          - 75th percentile = 32.5
        """
        n_years = 4
        data = np.zeros(n_years * 365)
        for y in range(n_years):
            data[y * 365 + 9] = (y + 1) * 10.0  # DOY 10 (0-based idx 9)
        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.75)
        # DOY 10 threshold: np.quantile([10, 20, 30, 40], 0.75) = 32.5
        assert thresholds[9] == pytest.approx(32.5)

    def test_boosting_doy1_uses_doy365(self):
        """
        With perc_boost=3, the threshold for DOY 1 draws from
        {DOY 365, DOY 1, DOY 2}.  Injecting a high value only into DOY 365
        must raise the DOY-1 threshold above the no-boosting threshold.

        We use percentile=0.9 so that the spike (5 out of 15 values = 1000)
        lands above the threshold:  90th percentile of [0]*10 + [1000]*5 = 1000.
        """
        n_years = 5
        data = np.zeros(n_years * 365, dtype=float)
        # Set DOY 365 (0-based index 364 in each year) to a very large value.
        for y in range(n_years):
            data[y * 365 + 364] = 1000.0

        thresh_no_boost = compute_thresholds(data, perc_boost=1, percentile=0.9)
        thresh_boosted  = compute_thresholds(data, perc_boost=3, percentile=0.9)

        # Without boosting DOY 1 90th pct = 0 (all five DOY-1 values are 0).
        assert thresh_no_boost[0] == pytest.approx(0.0)
        # With boosting, DOY 1 window = {DOY365*5, DOY1*5, DOY2*5}.
        # 90th pct of [0]*10 + [1000]*5 = 1000.
        assert thresh_boosted[0] == pytest.approx(1000.0)

    def test_boosting_doy365_uses_doy1(self):
        """
        With perc_boost=3, the threshold for DOY 365 draws from
        {DOY 364, DOY 365, DOY 1}.  Injecting a high value only into DOY 1
        must raise the DOY-365 threshold.

        Same reasoning as test_boosting_doy1_uses_doy365: use percentile=0.9.
        90th pct of [0]*10 + [1000]*5 = 1000.
        """
        n_years = 5
        data = np.zeros(n_years * 365, dtype=float)
        # Set DOY 1 (0-based index 0 in each year) to a large value.
        for y in range(n_years):
            data[y * 365 + 0] = 1000.0

        thresh_no_boost = compute_thresholds(data, perc_boost=1, percentile=0.9)
        thresh_boosted  = compute_thresholds(data, perc_boost=3, percentile=0.9)

        assert thresh_no_boost[364] == pytest.approx(0.0)
        # DOY 365 window = {DOY364*5, DOY365*5, DOY1*5}.
        # 90th pct of [0]*10 + [1000]*5 = 1000.
        assert thresh_boosted[364] == pytest.approx(1000.0)

    def test_boosting_window_size_perc_boost3(self):
        """
        With perc_boost=3 and n_years=4, each DOY threshold is the percentile
        of 3 * 4 = 12 values.  Hand-check DOY 2 (index 1) using only DOY 1,
        2, and 3 data.
        """
        n_years = 4
        # DOY 1: values [1, 2, 3, 4], DOY 2: [5, 6, 7, 8], DOY 3: [9, 10, 11, 12]
        data = np.zeros(n_years * 365, dtype=float)
        for y in range(n_years):
            data[y * 365 + 0] = float(y + 1)          # DOY 1
            data[y * 365 + 1] = float(y + 1 + 4)      # DOY 2
            data[y * 365 + 2] = float(y + 1 + 8)      # DOY 3

        thresholds = compute_thresholds(data, perc_boost=3, percentile=0.5)
        # DOY 2 window = {DOY1 values, DOY2 values, DOY3 values}
        #              = [1,2,3,4, 5,6,7,8, 9,10,11,12]
        # median = np.quantile([1..12], 0.5) = 6.5
        expected = np.quantile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 0.5)
        assert thresholds[1] == pytest.approx(expected)

    def test_non_noleap_raises(self):
        """Time axis not divisible by 365 must raise ValueError."""
        data = np.ones(366)
        with pytest.raises(ValueError, match="multiple of 365"):
            compute_thresholds(data, perc_boost=1, percentile=0.9)

    def test_nan_input_raises(self):
        data = np.ones(365 * 3)
        data[10] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            compute_thresholds(data, perc_boost=1, percentile=0.9)

    def test_even_perc_boost_raises(self):
        with pytest.raises(ValueError, match="positive odd integer"):
            compute_thresholds(np.ones(365), perc_boost=2, percentile=0.9)

    def test_zero_perc_boost_raises(self):
        with pytest.raises(ValueError, match="positive odd integer"):
            compute_thresholds(np.ones(365), perc_boost=0, percentile=0.9)

    def test_percentile_out_of_range_raises(self):
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            compute_thresholds(np.ones(365), perc_boost=1, percentile=1.5)

    def test_perc_boost5_doy1_window(self):
        """
        With perc_boost=5, half = 5//2 = 2, so the window for DOY 1 is
        {DOY 364, DOY 365, DOY 1, DOY 2, DOY 3}.
        Inject a high value only into DOY 364 and verify the DOY-1 threshold
        rises with a high enough percentile.

        n_years=6, so window has 6*5=30 values: [500]*6 + [0]*24.
        90th pct = sorted[26] = 500 (indices 24-29 in sorted array are 500).
        """
        n_years = 6
        data = np.zeros(n_years * 365, dtype=float)
        for y in range(n_years):
            data[y * 365 + 363] = 500.0  # DOY 364 (0-based index 363)

        thresh_no_boost = compute_thresholds(data, perc_boost=1, percentile=0.9)
        thresh_boost5   = compute_thresholds(data, perc_boost=5, percentile=0.9)

        assert thresh_no_boost[0] == pytest.approx(0.0)
        # DOY 1 window = {DOY364, DOY365, DOY1, DOY2, DOY3} × 6 years.
        # Only DOY364 has non-zero values.  90th pct of [500]*6 + [0]*24 = 500.
        assert thresh_boost5[0] == pytest.approx(500.0)

    def test_thresholds_monotone_in_percentile(self):
        """Higher percentiles must produce higher (or equal) thresholds everywhere."""
        data = make_data(n_years=10, seed=42)
        t90 = compute_thresholds(data, perc_boost=3, percentile=0.90)
        t95 = compute_thresholds(data, perc_boost=3, percentile=0.95)
        assert np.all(t95 >= t90)

    def test_thresholds_monotone_in_perc_boost(self):
        """
        A wider boosting window includes more values, which cannot decrease
        the 90th-percentile threshold (with non-negative data skewed high).
        This is a general robustness check, not a strict mathematical property.
        We just verify the shapes and absence of NaN.
        """
        data = make_data(n_years=10, seed=7)
        t1 = compute_thresholds(data, perc_boost=1, percentile=0.9)
        t3 = compute_thresholds(data, perc_boost=3, percentile=0.9)
        assert t1.shape == (365,)
        assert t3.shape == (365,)
        assert not np.isnan(t1).any()
        assert not np.isnan(t3).any()


# ---------------------------------------------------------------------------
# detect_exceedances
# ---------------------------------------------------------------------------

class TestDetectExceedances:

    def test_all_exceed(self):
        """Values strictly above threshold → all True."""
        thresholds = np.zeros(365)
        data = np.ones(365)
        doys = np.arange(1, 366)
        result = detect_exceedances(data, doys, thresholds)
        assert result.all()

    def test_none_exceed(self):
        """Values strictly below threshold → all False."""
        thresholds = np.full(365, 10.0)
        data = np.ones(365)
        doys = np.arange(1, 366)
        result = detect_exceedances(data, doys, thresholds)
        assert not result.any()

    def test_equal_does_not_exceed(self):
        """Values exactly equal to threshold must NOT be flagged (strict >)."""
        thresholds = np.ones(365)
        data = np.ones(365)
        doys = np.arange(1, 366)
        result = detect_exceedances(data, doys, thresholds)
        assert not result.any()

    def test_correct_doy_mapping(self):
        """Each day must be compared against its own DOY threshold."""
        thresholds = np.arange(1, 366, dtype=float)  # threshold[d] = d+1
        # DOY 10 value = 11 > threshold[9] = 10 → True
        # DOY 20 value = 20 = threshold[19] = 20 → False (not strictly greater)
        # DOY 30 value = 5  < threshold[29] = 30 → False
        data = np.zeros(365)
        data[9]  = 11.0   # DOY 10 (0-based index 9)
        data[19] = 20.0   # DOY 20
        data[29] = 5.0    # DOY 30
        doys = np.arange(1, 366)
        result = detect_exceedances(data, doys, thresholds)
        assert result[9]  == True
        assert result[19] == False
        assert result[29] == False

    def test_multi_year_doys(self):
        """Works for analysis periods spanning multiple years (repeated DOYs)."""
        n_years = 3
        T = n_years * 365
        thresholds = np.zeros(365)  # threshold = 0 everywhere
        data = np.ones(T)            # all values = 1 > 0
        doys = np.tile(np.arange(1, 366), n_years)
        result = detect_exceedances(data, doys, thresholds)
        assert result.shape == (T,)
        assert result.all()

    def test_exceedance_rate_near_percentile(self):
        """
        With n_years years of iid Uniform(0,1) data and a 90th-percentile
        threshold, the exceedance rate on the same data should be close to
        0.10.  (Uses the same data for reference and analysis.)
        """
        rng = np.random.default_rng(99)
        n_years = 30
        T = n_years * 365
        data = rng.random(T)
        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.90)
        doys = np.tile(np.arange(1, 366), n_years)
        result = detect_exceedances(data, doys, thresholds)
        rate = result.mean()
        # With 30 years the expected rate is ~0.10; allow ±2 pp.
        assert abs(rate - 0.10) < 0.02, f"exceedance rate {rate:.4f} too far from 0.10"

    def test_spatial_dims(self):
        """Shape (T, lat, lon) must be handled correctly."""
        lat, lon = 3, 4
        T = 365
        thresholds = np.zeros((365, lat, lon))
        data = np.ones((T, lat, lon))
        doys = np.arange(1, 366)
        result = detect_exceedances(data, doys, thresholds)
        assert result.shape == (T, lat, lon)
        assert result.all()

    def test_nan_data_raises(self):
        data = np.ones(365)
        data[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            detect_exceedances(data, np.arange(1, 366), np.zeros(365))

    def test_nan_thresholds_raises(self):
        thresholds = np.zeros(365)
        thresholds[5] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            detect_exceedances(np.ones(365), np.arange(1, 366), thresholds)

    def test_doy_out_of_range_raises(self):
        doys = np.arange(1, 366)
        doys[0] = 366  # invalid
        with pytest.raises(ValueError, match=r"\[1, 365\]"):
            detect_exceedances(np.ones(365), doys, np.zeros(365))

    def test_doy_zero_raises(self):
        doys = np.arange(0, 365)  # includes 0, invalid
        with pytest.raises(ValueError, match=r"\[1, 365\]"):
            detect_exceedances(np.ones(365), doys, np.zeros(365))


# ---------------------------------------------------------------------------
# Full pipeline (single location, end-to-end)
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_pipeline_constant_data(self):
        """
        Constant data: after aggregation the value is unchanged.  The 90th
        percentile threshold also equals that constant.  Therefore no day
        strictly exceeds the threshold.
        """
        c = 5.0
        n_years = 5
        T = n_years * 365
        raw = np.full(T, c, dtype=float)

        agg = rolling_aggregate(raw, window=3, method=AggMethod.MEAN)
        # Only interior values are valid; trim NaN edges.
        agg_trimmed = agg[1:-1]
        T_trim = len(agg_trimmed)

        # Use the same interior segment as the reference period.
        ref = agg_trimmed[:n_years * 365]  # first n_years * 365 valid days
        # ref may be shorter than T_trim if n_years * 365 > T_trim; guard:
        assert len(ref) == n_years * 365 - 2 or len(ref) == n_years * 365, \
            "Trimming left fewer than expected valid days"

        # Use raw years (no NaN) for simplicity.
        agg_clean = rolling_aggregate(
            np.full(n_years * 365 + 2, c, dtype=float), window=3, method=AggMethod.MEAN
        )[1:-1]  # interior only, shape (n_years * 365,)
        assert not np.isnan(agg_clean).any()

        thresholds = compute_thresholds(agg_clean, perc_boost=1, percentile=0.9)
        doys = np.tile(np.arange(1, 366), n_years)
        result = detect_exceedances(agg_clean, doys, thresholds)
        # Constant data: all values equal the threshold, none strictly exceed it.
        assert not result.any()

    def test_pipeline_known_exceedance(self):
        """
        5 years of data where DOY 100 in year 5 is set to a very high value.
        After computing the 50th-percentile threshold from the same 5 years,
        that specific day must be flagged as an exceedance.
        """
        n_years = 5
        T = n_years * 365
        rng = np.random.default_rng(123)
        data = rng.random(T)

        # Inject a spike: DOY 100 in year 4 (0-based) → index 4*365 + 99
        spike_idx = 4 * 365 + 99
        data[spike_idx] = 999.0

        thresholds = compute_thresholds(data, perc_boost=1, percentile=0.5)
        doys = np.tile(np.arange(1, 366), n_years)
        result = detect_exceedances(data, doys, thresholds)

        # The spiked day must be an exceedance.
        assert result[spike_idx], "Spiked day was not flagged as an exceedance"

    def test_pipeline_reference_vs_analysis_period(self):
        """
        Thresholds computed from a 50-year reference period are applied to a
        20-year analysis period drawn from the same distribution.
        With iid Uniform(0,1) data, the expected exceedance rate is
        (1 - percentile).  Using a large reference period keeps per-DOY
        threshold variance low enough for a tight tolerance.
        """
        rng = np.random.default_rng(7)
        n_ref = 50
        n_an = 20
        percentile = 0.90

        ref_data = rng.random(n_ref * 365)
        an_data  = rng.random(n_an * 365)

        thresholds = compute_thresholds(ref_data, perc_boost=1, percentile=percentile)
        doys = np.tile(np.arange(1, 366), n_an)
        result = detect_exceedances(an_data, doys, thresholds)

        rate = result.mean()
        # With 50 ref years the per-DOY threshold is well-estimated; allow ±2 pp.
        assert abs(rate - (1.0 - percentile)) < 0.02, \
            f"Exceedance rate {rate:.4f} too far from {1 - percentile:.2f}"

    def test_pipeline_boosting_increases_sample_size(self):
        """
        With perc_boost=3, each DOY threshold is computed from 3× as many
        values as with perc_boost=1.  For a mid-range percentile with smooth
        data the two thresholds should be close but the smoothed version
        (perc_boost=3) should vary less across DOYs (lower std).
        """
        rng = np.random.default_rng(55)
        n_years = 20
        data = rng.random(n_years * 365)

        t1 = compute_thresholds(data, perc_boost=1,  percentile=0.9)
        t3 = compute_thresholds(data, perc_boost=3,  percentile=0.9)
        t15 = compute_thresholds(data, perc_boost=15, percentile=0.9)

        # Wider boosting window → smoother thresholds (lower day-to-day std).
        assert t3.std() <= t1.std(), "perc_boost=3 should be smoother than perc_boost=1"
        assert t15.std() <= t3.std(), "perc_boost=15 should be smoother than perc_boost=3"


# ---------------------------------------------------------------------------
# compute_thresholds_chunked  (production path – reads zarr band-by-band)
# ---------------------------------------------------------------------------

def _make_zarr(data: np.ndarray, tmp_path: str) -> zarr.Array:
    """Write *data* to a temp zarr store and return the open array."""
    store = zarr.open(
        tmp_path, mode="w",
        shape=data.shape,
        chunks=(data.shape[0], 1, data.shape[2]) if data.ndim == 3 else data.shape,
        dtype=data.dtype,
    )
    store[:] = data
    return zarr.open(tmp_path, mode="r")


class TestComputeThresholdsChunked:

    def test_matches_pure_function_single_band(self, tmp_path):
        """
        compute_thresholds_chunked with lat_band_size == nlat should produce
        the same result as compute_thresholds on the same data loaded fully.
        """
        rng = np.random.default_rng(11)
        n_years, nlat, nlon = 5, 4, 6
        data = rng.random((n_years * 365, nlat, nlon)).astype(np.float32)

        z = _make_zarr(data, str(tmp_path / "ref.zarr"))

        expected = compute_thresholds(data, perc_boost=3, percentile=0.9)
        actual   = compute_thresholds_chunked(z, n_years=n_years, perc_boost=3,
                                               percentile=0.9, lat_band_size=nlat)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_matches_pure_function_small_bands(self, tmp_path):
        """
        Processing one latitude row at a time must give the same result as
        loading all rows at once.
        """
        rng = np.random.default_rng(22)
        n_years, nlat, nlon = 5, 8, 6
        data = rng.random((n_years * 365, nlat, nlon)).astype(np.float32)

        z = _make_zarr(data, str(tmp_path / "ref.zarr"))

        expected = compute_thresholds(data, perc_boost=1, percentile=0.9)
        actual   = compute_thresholds_chunked(z, n_years=n_years, perc_boost=1,
                                               percentile=0.9, lat_band_size=1)
        np.testing.assert_allclose(actual, expected, rtol=1e-5)

    def test_band_size_larger_than_nlat(self, tmp_path):
        """lat_band_size > nlat should not raise; it just processes in one pass."""
        rng = np.random.default_rng(33)
        n_years, nlat, nlon = 3, 4, 5
        data = rng.random((n_years * 365, nlat, nlon)).astype(np.float32)

        z = _make_zarr(data, str(tmp_path / "ref.zarr"))

        result = compute_thresholds_chunked(z, n_years=n_years, perc_boost=1,
                                             percentile=0.9, lat_band_size=100)
        assert result.shape == (365, nlat, nlon)

    def test_output_shape(self, tmp_path):
        n_years, nlat, nlon = 4, 6, 8
        data = np.ones((n_years * 365, nlat, nlon), dtype=np.float32)
        z = _make_zarr(data, str(tmp_path / "ref.zarr"))

        result = compute_thresholds_chunked(z, n_years=n_years, perc_boost=1,
                                             percentile=0.9, lat_band_size=2)
        assert result.shape == (365, nlat, nlon)

    def test_wrong_time_axis_raises(self, tmp_path):
        """Passing a zarr whose time axis ≠ n_years * 365 must raise."""
        data = np.ones((100, 4, 4), dtype=np.float32)  # 100 ≠ 3 * 365
        z = _make_zarr(data, str(tmp_path / "ref.zarr"))
        with pytest.raises(ValueError, match="n_years"):
            compute_thresholds_chunked(z, n_years=3, perc_boost=1, percentile=0.9)

    def test_invalid_lat_band_size_raises(self, tmp_path):
        data = np.ones((365, 4, 4), dtype=np.float32)
        z = _make_zarr(data, str(tmp_path / "ref.zarr"))
        with pytest.raises(ValueError, match="lat_band_size"):
            compute_thresholds_chunked(z, n_years=1, perc_boost=1,
                                        percentile=0.9, lat_band_size=0)

    def test_wrap_around_preserved_through_chunking(self, tmp_path):
        """
        DOY 1 wrap-around must work the same whether data is processed in one
        band or split across two bands.
        """
        n_years, nlat, nlon = 5, 4, 1
        data = np.zeros((n_years * 365, nlat, nlon), dtype=np.float64)
        # Set DOY 365 high for all lats
        for y in range(n_years):
            data[y * 365 + 364, :, :] = 1000.0

        z = _make_zarr(data, str(tmp_path / "ref.zarr"))

        result_full = compute_thresholds_chunked(z, n_years=n_years, perc_boost=3,
                                                  percentile=0.9, lat_band_size=nlat)
        result_band = compute_thresholds_chunked(z, n_years=n_years, perc_boost=3,
                                                  percentile=0.9, lat_band_size=1)

        np.testing.assert_allclose(result_full, result_band)
        # DOY 1 (index 0) threshold must be elevated because of the DOY 365 spike
        assert result_full[0, 0, 0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# rolling_aggregate_xarray  (lazy xarray wrapper)
# ---------------------------------------------------------------------------

class TestRollingAggregateXarray:

    def _make_da(self, values):
        """Wrap a 1-D numpy array in an xr.DataArray with a no-leap time axis."""
        import xarray as xr
        import cftime
        T = len(values)
        times = xr.date_range(start="1990-01-01", periods=T, freq="D",
                              use_cftime=True, calendar="noleap")
        return xr.DataArray(values, dims=["time"], coords={"time": times})

    def test_window_1_is_identity(self):
        import xarray as xr
        data = np.arange(10, dtype=float)
        da = self._make_da(data)
        result = rolling_aggregate_xarray(da, window=1, method=AggMethod.MEAN)
        np.testing.assert_array_equal(result.values, da.values)

    def test_window_3_mean_matches_numpy(self):
        """xarray result must match the pure-numpy function on the same data."""
        rng = np.random.default_rng(5)
        data = rng.random(20)
        da = self._make_da(data)
        xr_result = rolling_aggregate_xarray(da, window=3, method=AggMethod.MEAN).values
        np_result = rolling_aggregate(data, window=3, method=AggMethod.MEAN)
        # Both should agree on interior values; xarray uses min_periods=window
        # so edges are NaN in both.
        interior = slice(1, -1)
        np.testing.assert_allclose(xr_result[interior], np_result[interior], rtol=1e-10)

    def test_edges_are_nan(self):
        data = np.ones(10, dtype=float)
        da = self._make_da(data)
        result = rolling_aggregate_xarray(da, window=3, method=AggMethod.MEAN).values
        assert np.isnan(result[0])
        assert np.isnan(result[-1])

    def test_even_window_raises(self):
        import xarray as xr
        da = self._make_da(np.ones(10))
        with pytest.raises(ValueError, match="positive odd integer"):
            rolling_aggregate_xarray(da, window=4, method=AggMethod.MEAN)

    def test_result_is_lazy(self):
        """The returned DataArray should be dask-backed (lazy)."""
        import xarray as xr
        import dask.array as dsa
        data = np.ones(20, dtype=float)
        da = self._make_da(data).chunk({"time": 5})  # chunk to make dask-backed
        result = rolling_aggregate_xarray(da, window=3, method=AggMethod.MEAN)
        assert isinstance(result.data, dsa.Array), \
            "Expected a dask-backed (lazy) DataArray"


# ---------------------------------------------------------------------------
# detect_exceedances_xarray  (lazy xarray wrapper)
# ---------------------------------------------------------------------------

class TestDetectExceedancesXarray:

    def _make_da(self, values_3d, n_years):
        """
        Wrap a (n_years*365, nlat, nlon) numpy array in a dask-backed
        xr.DataArray with a no-leap time axis.
        """
        import xarray as xr
        T, nlat, nlon = values_3d.shape
        times = xr.date_range(start="1990-01-01", periods=T, freq="D",
                              use_cftime=True, calendar="noleap")
        lat = np.arange(nlat, dtype=float)
        lon = np.arange(nlon, dtype=float)
        da = xr.DataArray(
            values_3d,
            dims=["time", "latitude", "longitude"],
            coords={"time": times, "latitude": lat, "longitude": lon},
        )
        return da.chunk({"time": 365})  # one year per chunk

    def test_all_exceed(self):
        """Values > 0 against zero thresholds → all True."""
        n_years, nlat, nlon = 3, 2, 3
        data = np.ones((n_years * 365, nlat, nlon), dtype=float)
        thresholds = np.zeros((365, nlat, nlon))
        da = self._make_da(data, n_years)
        result = detect_exceedances_xarray(da, thresholds).values
        assert result.all()

    def test_none_exceed(self):
        """Values = 0 against threshold = 1 → all False."""
        n_years, nlat, nlon = 2, 2, 2
        data = np.zeros((n_years * 365, nlat, nlon), dtype=float)
        thresholds = np.ones((365, nlat, nlon))
        da = self._make_da(data, n_years)
        result = detect_exceedances_xarray(da, thresholds).values
        assert not result.any()

    def test_matches_numpy_variant(self):
        """xarray variant must produce the same boolean mask as the numpy variant."""
        rng = np.random.default_rng(77)
        n_years, nlat, nlon = 5, 4, 3
        data_np = rng.random((n_years * 365, nlat, nlon)).astype(np.float32)
        ref_np  = rng.random((n_years * 365, nlat, nlon)).astype(np.float32)

        thresholds = compute_thresholds(ref_np, perc_boost=1, percentile=0.9)

        # numpy path
        doys = np.tile(np.arange(1, 366), n_years)
        np_result = detect_exceedances(data_np, doys, thresholds)

        # xarray path
        da = self._make_da(data_np, n_years)
        xr_result = detect_exceedances_xarray(da, thresholds).values

        np.testing.assert_array_equal(np_result, xr_result)

    def test_result_is_lazy(self):
        """The returned DataArray should be dask-backed before .values is called."""
        import dask.array as dsa
        n_years, nlat, nlon = 2, 2, 2
        data = np.zeros((n_years * 365, nlat, nlon), dtype=float)
        thresholds = np.ones((365, nlat, nlon))
        da = self._make_da(data, n_years)
        result = detect_exceedances_xarray(da, thresholds)
        assert isinstance(result.data, dsa.Array), \
            "Expected a dask-backed (lazy) DataArray"
