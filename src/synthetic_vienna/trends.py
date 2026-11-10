"""
Trend fits for the yearly frequency of exceedances.

All functions take a boolean exceedance array of shape (num_years, 365), as returned by
ExceedancesCalculator.compute_exceedances, and the matching years. Each returns a TrendFit
whose ``fitted`` gives the modelled percentage of exceedance days for every year, so that
all methods can be drawn in the same units as the observed yearly percentages.

- linear_trend:     OLS of the yearly percentage on year.
- theil_sen_trend:  median of pairwise slopes of the yearly percentage (robust to outliers).
- nhpp_trend:       non-homogeneous Poisson process with log-linear intensity, fitted to the
                    yearly counts of exceedance days (Poisson GLM, log link).
- pot_trend:        occurrence part of peaks-over-threshold: exceedance days are runs-declustered
                    into independent clusters, a log-linear Poisson process is fitted to the yearly
                    cluster counts, and cluster rates are converted back to exceedance days with
                    the extremal index theta = n_clusters / n_exceedance_days (assumed constant).

The Poisson fits use quasi-Poisson standard errors (scale estimated from Pearson chi2), since
exceedance days are clustered and therefore overdispersed relative to a Poisson distribution.
"""
from dataclasses import dataclass

import numpy as np
import statsmodels.api as sm
from scipy import stats

DAYS_PER_YEAR = 365


@dataclass
class TrendFit:
    name: str
    fitted: np.ndarray  # modelled % of exceedance days per year
    summary: str        # one-line description of the trend (slope / rate change, uncertainty)


def yearly_percentage(exceedances):
    return 100 * exceedances.mean(axis=1)


def linear_trend(years, exceedances):
    pct = yearly_percentage(exceedances)
    fit = stats.linregress(years, pct)
    return TrendFit("Linear (OLS)", fit.intercept + fit.slope * years,
                    f"{10 * fit.slope:+.2f} %/decade, p={fit.pvalue:.3g}")


def theil_sen_trend(years, exceedances):
    pct = yearly_percentage(exceedances)
    slope, intercept, lo, hi = stats.theilslopes(pct, years, alpha=0.95)
    return TrendFit("Theil–Sen", intercept + slope * years,
                    f"{10 * slope:+.2f} %/decade (95% CI {10 * lo:+.2f} to {10 * hi:+.2f})")


def _poisson_log_linear(years, counts):
    """Poisson GLM log(E[count]) = a + b * (year - first year); returns (fitted counts, b, p-value of b)."""
    X = sm.add_constant(years - years[0])
    fit = sm.GLM(counts, X, family=sm.families.Poisson()).fit(scale="X2")
    return fit.fittedvalues, fit.params[1], fit.pvalues[1]


def nhpp_trend(years, exceedances):
    counts = exceedances.sum(axis=1)
    fitted_counts, b, p = _poisson_log_linear(years, counts)
    return TrendFit("NHPP", 100 * fitted_counts / DAYS_PER_YEAR,
                    f"rate ×{np.exp(10 * b):.2f}/decade, p={p:.3g}")


def decluster_runs(exceedances, run_length):
    """
    Runs declustering of the concatenated daily series: a cluster ends once run_length consecutive
    non-exceedance days occur. Returns the number of clusters starting in each year, shape (num_years,).
    """
    if exceedances.dtype != bool:
        raise ValueError("Runs declustering needs a boolean exceedance array (not in_base_bootstrap rates)")
    flat = exceedances.reshape(-1)
    idx = np.flatnonzero(flat)
    if idx.size == 0:
        return np.zeros(exceedances.shape[0], dtype=int)
    # A new cluster starts at the first exceedance and wherever the gap to the previous one has
    # at least run_length non-exceedance days in between
    starts = idx[np.concatenate([[True], np.diff(idx) - 1 >= run_length])]
    return np.bincount(starts // exceedances.shape[1], minlength=exceedances.shape[0])


def pot_trend(years, exceedances, run_length=3):
    clusters = decluster_runs(exceedances, run_length)
    theta = clusters.sum() / exceedances.sum()  # extremal index: 1 / mean cluster size in days
    fitted_clusters, b, p = _poisson_log_linear(years, clusters)
    return TrendFit(f"POT (r={run_length}, θ={theta:.2f})", 100 * fitted_clusters / theta / DAYS_PER_YEAR,
                    f"cluster rate ×{np.exp(10 * b):.2f}/decade, p={p:.3g}")


def fit_all_trends(years, exceedances, run_length=3):
    """Fit every method; methods that cannot be fitted (too few years, no exceedances) are skipped."""
    years = np.asarray(years, dtype=float)
    if len(years) < 3:
        return []
    fits = [linear_trend(years, exceedances), theil_sen_trend(years, exceedances)]
    if exceedances.any():
        fits.append(nhpp_trend(years, exceedances))
        if exceedances.dtype == bool:  # in_base_bootstrap rates cannot be runs-declustered
            fits.append(pot_trend(years, exceedances, run_length))
    return fits
