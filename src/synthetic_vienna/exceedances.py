import numpy as np
import pandas as pd
import cftime
from scipy import stats

class ExceedancesCalculator:
    def __init__(self):
        self.reference_agg = None
        self.analysis_agg = None
        self.thresholds = None
        self.exceedances = None
        self.seasonal_cycle = None
        self.reference_years = None
        self.analysis_years = None

    def aggregate(self, ds, reference_window, analysis_window, aggregation_window=1, agg_method='mean'):
        
        ref_start, ref_end = reference_window
        an_start, an_end = analysis_window
        
        if agg_method == "max":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True, min_periods=1).max()
        elif agg_method == "min":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True, min_periods=1).min()
        elif agg_method == "mean":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True, min_periods=1).mean()
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")
        
        reference_period = ds.sel(time=slice(ref_start, ref_end))
        reference_agg = roll_func(reference_period)
        analysis_period = ds.sel(time=slice(an_start, an_end))
        analysis_agg = roll_func(analysis_period)
        
        def full_year(agg):
            times = agg['time'].values
            first_day, last_day = times[0], times[-1]
            assert isinstance(first_day, cftime._cftime.DatetimeNoLeap)
            return (first_day.month, first_day.day, last_day.month, last_day.day) == (1, 1, 12, 31)
            
        assert full_year(reference_agg)
        assert full_year(analysis_agg)

        self.reference_years = np.array([t.year for t in reference_agg['time'].values[::365]])
        self.analysis_years = np.array([t.year for t in analysis_agg['time'].values[::365]])
        
        reference_arr = reference_agg['daily_max_2m_temperature'].values.reshape(-1, 365)
        analysis_arr = analysis_agg['daily_max_2m_temperature'].values.reshape(-1, 365)
        return reference_arr, analysis_arr

    def compute_thresholds(self, reference_agg, boosting_window, percentile):
        """
        Compute DOY-based percentile thresholds using a rolling window
        reference_agg: numpy array of shape (num_years, 365)
        boosting_window: integer, number of days for rolling window (should be odd)
        percentile: float between 0 and 1
        Returns: numpy array of length 365 (threshold for each DOY)
        """
        num_years, n_days = reference_agg.shape
        assert n_days == 365, "reference_agg must have 365 days per year (no-leap calendar)"
        # All {years} x {window} values for each DOY, shape (365, num_years * boosting_window)
        vals = pooled_window(reference_agg, boosting_window).transpose(1, 0, 2).reshape(365, -1)
        return _percentile(vals, percentile * 100, axis=1)

    def compute_in_base_exceedance_rates(self, reference_agg, in_base_agg, ref_rows, boosting_window, percentile):
        """
        Exceedance rates for the in-base years following Zhang et al. (2005), J. Climate 18, 1641-1651,
        "Removing the 'jump'" (as implemented in climdex.pcic): for each reference year j, the year is
        removed from the reference sample and replaced by each other reference year k in turn (k counted
        twice), thresholds are computed from that sample, and year j is compared against them. The
        exceedance indicator is averaged over the num_years - 1 replicates.
        reference_agg: numpy array of shape (num_years, 365), the data the thresholds are built from
        in_base_agg: numpy array of shape (n_in_base, 365), the in-base years as compared against the
            thresholds (the analysis data of those reference years)
        ref_rows: integer array of length n_in_base, the row of reference_agg each in-base year corresponds to
        Returns: numpy array of shape (n_in_base, 365) with exceedance rates in [0, 1]
        """
        num_years = reference_agg.shape[0]
        if num_years < 2:
            raise ValueError("The in-base bootstrap needs at least 2 reference years")
        window = pooled_window(reference_agg, boosting_window)  # (num_years, 365, w)
        rates = np.empty(in_base_agg.shape)
        for i, j in enumerate(ref_rows):
            others = np.delete(window, j, axis=0)  # (num_years - 1, 365, w)
            # Replicate k: the other years plus a duplicate of other year k, (num_years - 1, num_years, 365, w)
            samples = np.concatenate([np.broadcast_to(others, (num_years - 1,) + others.shape),
                                      others[:, None]], axis=1)
            vals = samples.transpose(0, 2, 1, 3).reshape(num_years - 1, 365, -1)
            thresholds = _percentile(vals, percentile * 100, axis=2)  # (num_years - 1, 365)
            rates[i] = self.get_exceedances(threshold=thresholds, analysis_agg=in_base_agg[i]).mean(axis=0)
        return rates

    def get_exceedances(self, threshold, analysis_agg):
        return (analysis_agg >= threshold)

    def compute_seasonal_cycle(self, reference_agg):
        """
        Mean seasonal cycle following Brunner & Voigt (2024), Nat. Commun. 15, 2087:
        the per-DOY mean over the reference years, without any running window.
        reference_agg: numpy array of shape (num_years, 365)
        Returns: numpy array of length 365
        """
        return reference_agg.mean(axis=0)

    def compute_exceedances(self, data, percentile, boosting_window, aggregation_window, agg_method, reference_window, analysis_window, subtract_seasonal_cycle=False, in_base_bootstrap=False):
        """
        If in_base_bootstrap is True, analysis years that lie inside the reference period get exceedance rates from
        the Zhang et al. (2005) bootstrap (see compute_in_base_exceedance_rates) instead of a comparison against the
        full-reference thresholds, which removes the jump in exceedance frequency at the reference-period boundary.
        self.exceedances is then a float array (rates in [0, 1], 0/1 outside the reference period) instead of bool.

        If subtract_seasonal_cycle is True, the reference-period mean seasonal cycle is subtracted from both the
        reference and the analysis data before computing thresholds (Brunner & Voigt, 2024), so that the boosting
        window does not mix the seasonal cycle into the thresholds. reference_agg, analysis_agg and thresholds are
        then anomalies; add self.seasonal_cycle to get absolute values.
        """
        self.reference_agg, self.analysis_agg = self.aggregate(ds=data, reference_window=reference_window, analysis_window=analysis_window, aggregation_window=aggregation_window, agg_method=agg_method)
        self.seasonal_cycle = None
        if subtract_seasonal_cycle:
            self.seasonal_cycle = self.compute_seasonal_cycle(self.reference_agg)
            self.reference_agg = self.reference_agg - self.seasonal_cycle
            self.analysis_agg = self.analysis_agg - self.seasonal_cycle
        self.thresholds = self.compute_thresholds(self.reference_agg, boosting_window, percentile)
        self.exceedances = self.get_exceedances(threshold=self.thresholds, analysis_agg=self.analysis_agg)
        if in_base_bootstrap:
            in_base = np.isin(self.analysis_years, self.reference_years)
            ref_rows = np.searchsorted(self.reference_years, self.analysis_years[in_base])
            self.exceedances = self.exceedances.astype(float)
            self.exceedances[in_base] = self.compute_in_base_exceedance_rates(
                self.reference_agg, self.analysis_agg[in_base], ref_rows, boosting_window, percentile)
        return self.exceedances

def _percentile(vals, q, axis):
    """np.percentile when vals has no NaN (vectorised); np.nanpercentile otherwise, which with an axis loops in Python."""
    if np.isnan(vals).any():
        return np.nanpercentile(vals, q, axis=axis)
    return np.percentile(vals, q, axis=axis)

def pooled_window(arr, boosting_window):
    """(num_years, 365) -> (num_years, 365, boosting_window): the days of the boosting window centred on each DOY, wrapping around the year."""
    half_window = boosting_window // 2
    idxs = (np.arange(365)[:, None] + np.arange(-half_window, half_window + 1)) % 365
    return arr[:, idxs]

def count_exceedances_per_year(exceedances):
    return exceedances.sum(axis=1)

def compute_trendline_per_year_linear(exceedances):
    years = np.arange(len(exceedances))
    counts = exceedances.sum(axis=1)
    
    # Fit a linear trend: counts = a * years + b
    slope, intercept = np.polyfit(years, counts, 1)

    # print(f"Linear trend: count = {a:.4f} * year + {b:.2f}")
    return slope, intercept

def compute_trendline_per_year_theil_sen(exceedances):
    """Theil-Sen trend of the yearly exceedance counts: median of all pairwise slopes, robust to outlier years."""
    years = np.arange(len(exceedances))
    counts = exceedances.sum(axis=1)
    slope, intercept, _, _ = stats.theilslopes(counts, years)
    return slope, intercept
