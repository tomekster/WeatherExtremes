from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import lfilter
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess, arma_acovf

DAYS_PER_YEAR = 365  # input is on the no-leap calendar (see scripts/load_vienna_data.py)
AR_BURN_IN = 2000    # days simulated and discarded so the AR noise starts from its stationary distribution


@dataclass
class ModelParams:
    intercept: float
    trend: float                    # °C per year
    sin_coeffs: np.ndarray          # shape (K,), coefficient of sin(2πkt) for harmonic k = 1..K
    cos_coeffs: np.ndarray          # shape (K,), coefficient of cos(2πkt)
    ar_coeffs: np.ndarray | None    # shape (L,), AR coefficients for lags 1..L
    innovation_std: float | None

    @property
    def harmonic_amplitudes(self):
        """Amplitude of each annual harmonic, sqrt(sin_k^2 + cos_k^2), shape (K,)."""
        return np.hypot(self.sin_coeffs, self.cos_coeffs)

    def linear_coeffs(self):
        """Coefficients in the column order of TemperatureModel.X: [intercept, trend, sin_1, cos_1, ..., sin_K, cos_K]."""
        harmonics = np.column_stack([self.sin_coeffs, self.cos_coeffs]).reshape(-1)
        return np.concatenate([[self.intercept, self.trend], harmonics])


def ar_variance_factor(ar_coeffs):
    """Stationary variance of an AR process with these coefficients and unit innovation variance."""
    return arma_acovf(np.r_[1, -np.asarray(ar_coeffs)], np.array([1.0]), nobs=1, sigma2=1.0)[0]


class TemperatureModel:
    """
        Daily max 2m temperature model: intercept + linear trend + K annual harmonics,
        with optional AR(L) residuals.
    """
    def __init__(self):
        self.model_params = None
        self.K = None
        self.L = None

    def fit(self, y, L=1, K=3, with_autocorrelation=False):
        self.K, self.L = K, (L if with_autocorrelation else None)
        t = np.arange(len(y)) / DAYS_PER_YEAR
        # Design matrix
        X = [
            np.ones(len(t)),          # intercept
            t,                        # linear trend
        ]
        for k in range(1, K + 1):     # annual cycle
            X.append(np.sin(2 * np.pi * k * t))
            X.append(np.cos(2 * np.pi * k * t))
        self.X = np.column_stack(X)

        # Least-squares fit
        params = np.linalg.lstsq(self.X, y, rcond=None)[0]

        ar_coeffs = None
        innovation_std = None

        if with_autocorrelation:
            # Compute residuals from the linear model
            residuals = y - self.X @ params

            # Fit the autocorrelation model
            ar_model = AutoReg(
                residuals,
                lags=L,
                trend="n"
            ).fit()

            # Get the parameters
            ar_coeffs = np.asarray(ar_model.params)
            innovation_std = np.sqrt(ar_model.sigma2)

        self.model_params = ModelParams(
            intercept=params[0],
            trend=params[1],
            sin_coeffs=params[2::2],
            cos_coeffs=params[3::2],
            ar_coeffs=ar_coeffs,
            innovation_std=innovation_std,
        )

    def generate(self, warming_rate=None, std=None, ar_coeffs=None, with_autocorrelation=False):
        """
        warming_rate: overrides the fitted trend (°C/year); 0.0 means no trend.
        std: noise std. Without autocorrelation, the white-noise std. With autocorrelation, the stationary
            std of the AR noise: the innovation std is set to std / sqrt(ar_variance_factor(ar_coeffs)),
            i.e. std * sqrt(1 - rho^2) for AR(1).
        ar_coeffs: overrides the fitted AR coefficients (only with autocorrelation). Must have one value per
            fitted lag (length L); a scalar is accepted when L == 1. Unless std is given, the innovation std
            is rescaled so that the stationary variance of the noise is unchanged.
        """
        if ar_coeffs is not None and not with_autocorrelation:
            raise ValueError("ar_coeffs can only be set when with_autocorrelation is True")

        params = self.model_params

        if warming_rate is not None:
            params = replace(params, trend=warming_rate)

        y_hat = self.X @ params.linear_coeffs()

        if with_autocorrelation:
            if params.ar_coeffs is None:
                raise ValueError("ar_coeffs is None, probably the model was fit with with_autocorrelation=False")
            if params.innovation_std is None:
                raise ValueError("innovation_std is None, probably the model was fit with with_autocorrelation=False")

            if ar_coeffs is not None:
                ar_coeffs = np.atleast_1d(np.asarray(ar_coeffs, dtype=float))
                if ar_coeffs.shape != params.ar_coeffs.shape:
                    raise ValueError(f"ar_coeffs must have {len(params.ar_coeffs)} values (one per fitted lag), "
                                     f"got {ar_coeffs.size}")
                if not ArmaProcess(np.r_[1, -ar_coeffs]).isstationary:
                    raise ValueError(f"ar_coeffs {ar_coeffs} define a non-stationary AR process")
                # We need to compute a new innovation_std to keep the total variance unchanged.
                innovation_std_new = params.innovation_std * np.sqrt(
                    ar_variance_factor(params.ar_coeffs) / ar_variance_factor(ar_coeffs)
                )
                params = replace(params, ar_coeffs=ar_coeffs, innovation_std=innovation_std_new)

            if std is not None:
                params = replace(params, innovation_std=std / np.sqrt(ar_variance_factor(params.ar_coeffs)))

            # eps[i] = sum_j ar_coeffs[j] * eps[i - 1 - j] + innovation[i]; the burn-in makes the
            # retained part stationary regardless of the (zero) initial state
            innovations = np.random.normal(0, params.innovation_std, size=AR_BURN_IN + len(y_hat))
            eps = lfilter([1.0], np.r_[1, -params.ar_coeffs], innovations)[AR_BURN_IN:]
            y_hat += eps

        else:
            if std and std > 0:
                noise = np.random.normal(
                    loc=0,
                    scale=std,
                    size=len(y_hat)
                )
                y_hat += noise

        return y_hat
