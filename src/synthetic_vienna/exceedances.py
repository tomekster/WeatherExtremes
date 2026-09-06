import numpy as np
import pandas as pd
import cftime

class ExceedancesCalculator:
    def __init__(self):
        self.reference_agg = None
        self.analysis_agg = None
        self.thresholds = None
        self.exceedances = None

    def aggregate(self, ds, reference_window, analysis_window, aggregation_window=1, agg_method='mean'):
        
        ref_start, ref_end = reference_window
        an_start, an_end = analysis_window
        
        if agg_method == "max":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True).max()
        elif agg_method == "min":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True).min()
        elif agg_method == "mean":
            roll_func = lambda da: da.rolling(time=aggregation_window, center=True).mean()
        else:
            raise ValueError(f"Unknown agg_method: {agg_method}")
        
        roll_func = lambda da: da.rolling(time=aggregation_window, center=True, min_periods=1).mean()
        reference_period = ds.sel(time=slice(ref_start, ref_end))
        reference_agg = roll_func(reference_period)
        analysis_period = ds.sel(time=slice(an_start, an_end))
        analysis_agg = roll_func(analysis_period)
        
        def full_year(agg):
            times = agg['time'].values
            first_day, last_day = times[0], times[-1]
            assert isinstance(first_day, cftime._cftime.DatetimeNoLeap)
            return first_day.month, first_day.day, last_day.month, last_day.day == (1, 1, 12, 31)
            
        assert full_year(reference_agg)
        assert full_year(analysis_agg)
        
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
        half_window = boosting_window // 2
        thresholds = []
        # Loop over each DOY (1-based for consistency, but zero-indexed for numpy)
        for day in range(365):
            # Collect all days in boosting window centered on this DOY
            # day idx: 0..364 → DOY: 1..365
            lo = day - half_window
            hi = day + half_window
            # Wrap window around year
            idxs = np.arange(lo, hi+1) % 365  # always get indices in 0..364
            # Take all {years} x {window} values and flatten
            vals = reference_agg[:, idxs].reshape(-1)
            # Compute the threshold percentile (ignore NaNs, if any)
            threshold = np.nanpercentile(vals, percentile*100)
            thresholds.append(threshold)
        # Return as numpy array of shape (365,)
        return np.array(thresholds)

    def get_exceedances(self, threshold, analysis_agg):
        return (analysis_agg >= threshold)

    def compute_exceedances(self, data, percentile, boosting_window, aggregation_window, agg_method, reference_window, analysis_window):
        self.reference_agg, self.analysis_agg = self.aggregate(ds=data, reference_window=reference_window, analysis_window=analysis_window, aggregation_window=aggregation_window, agg_method=agg_method)
        self.thresholds = self.compute_thresholds(self.reference_agg, boosting_window, percentile)
        self.exceedances = self.get_exceedances(threshold=self.thresholds, analysis_agg=self.analysis_agg)
        return self.exceedances

def count_exceedances_per_year(exceedances):
    return exceedances.sum(axis=1)

def compute_trendline_per_year(exceedances):
    years = np.arange(len(exceedances))
    counts = exceedances.sum(axis=1)
    
    # Fit a linear trend: counts = a * years + b
    slope, intercept = np.polyfit(years, counts, 1)

    # print(f"Linear trend: count = {a:.4f} * year + {b:.2f}")
    return slope, intercept