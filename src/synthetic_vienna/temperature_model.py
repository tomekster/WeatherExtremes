import numpy as np
from collections import namedtuple
from statsmodels.tsa.ar_model import AutoReg

# Retrieve time series at the nearest Vienna point
ModelParams = namedtuple('ModelParams', ['beta0', 'beta1', 'a', 'b', 'ar_rho', 'ar_innovation_std'] )

class TemperatureModel:
    """
        Daily max 2m temperature model
    """
    def __init__(self):        
        self.model_params = None
    
    def fit(self, y, with_autocorrelation=False):
        t = np.arange(len(y)) / 365.2425
        # Design matrix
        self.X = np.column_stack([
            np.ones(len(t)),          # intercept
            t,                        # linear trend
            np.sin(2 * np.pi * t),    # annual cycle
            np.cos(2 * np.pi * t),
        ])

        # Least-squares fit
        params = np.linalg.lstsq(self.X, y, rcond=None)[0]
        
        rho = None
        innovation_std = None
        
        if with_autocorrelation:
            # Compute residuals from the linear model
            y_hat = self.X @ params
            residuals = y - y_hat
            
            # Fit the autocorrelation model
            ar_model = AutoReg(
                residuals,
                lags=2,
                trend="n"
            ).fit()

            # Get the parameters
            rho = ar_model.params[0]
            innovation_std = np.sqrt(ar_model.sigma2)
        
        self.model_params = ModelParams(*(list(params) + [rho, innovation_std] ))

    def generate(self, warming_rate=None, std=None, ar_rho=None, with_autocorrelation=False):
        if std is not None and with_autocorrelation:
            raise ValueError("std can only be set when with_autocorrelation is False.")
   
        if ar_rho is not None and not with_autocorrelation:
            raise ValueError("ar_rho can only be set when autocorrelation is True")
        
        params = self.model_params
        
        if warming_rate:
            params = params._replace(beta1=warming_rate)

        y_hat = self.X @ np.array(list(params[:4]))
        
        if with_autocorrelation:
            if self.model_params.ar_rho is None:
                raise ValueError("ar_rho is None, probably the model was fit with with_autocorrelation=False")
            if self.model_params.ar_innovation_std is None:
                raise ValueError("ar_innovation_std is None, probably the model was fit with with_autocorrelation=False")
            
            if ar_rho:
                # We need to compute a new innovation_std to keep the total variance unchanged.
                innovation_std_new = (
                    params.ar_innovation_std
                    * np.sqrt(
                        (1 - ar_rho**2)
                        / (1 - params.ar_rho**2)
                    )
                )
            
                params = params._replace(ar_rho=ar_rho)
                params = params._replace(ar_innovation_std=innovation_std_new)
                
            eps = np.empty(len(y_hat))
            # stationary initialization
            eps[0] = np.random.normal(
                0,
                params.ar_innovation_std / np.sqrt(1 - params.ar_rho**2)
            )

            for i in range(1, len(y_hat)):
                eps[i] = (
                    params.ar_rho * eps[i - 1]
                    + np.random.normal(0, params.ar_innovation_std)
                )
            y_hat += eps
        
        else:
            if std and std>0:
                noise = np.random.normal(
                    loc=0,
                    scale=std,
                    size=len(y_hat)
                )   
                y_hat += noise
        
        
        return y_hat

