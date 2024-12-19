import os
import xarray as xr
import numpy as np
from tqdm import tqdm

def generate_temperature_exceedance_mask(aggregated_data, var, an_start, an_end, thresholds, perc, res_dir):
    """
    Generate a 3D tensor of binary masks for the entire multi-year timespan where each (721, 1440) slice 
    is a binary mask indicating whether the daily mean temperature exceeds the threshold 
    for the corresponding day and location.

    Args:
        aggregated_data (xarray dataset): Xarray dataset for a given variable with appropriately aggregated values for the appropriate analysis timespan
        thresholds (numpy.ndarray): A 3D array of shape (365, 721, 1440) with 
                                    thresholds for a given variable for each day of the year and each grid point.

    Returns:
        numpy.ndarray: A 3D array of shape (n_years * 365, 721, 1440) containing binary masks 
                       (1 if the variable exceeds the threshold, 0 otherwise).
    """
    
    aggregated_data = aggregated_data.sel(time=slice(an_start, an_end))
    
    n_days, lat, long = aggregated_data.shape
    assert n_days % 365 == 0, f"{n_days}_{lat}_{long}"
    assert lat == 721, f"{n_days}_{lat}_{long}"
    assert long == 1440, f"{n_days}_{lat}_{long}"
    assert thresholds.shape == (365, 721, 1440), f"{thresholds.shape}"
    
    all_years_per_day  = []
    all_years_per_month = []
    
    import cftime
    start = cftime.DatetimeNoLeap(1960, 1, 1) # Must be 1st Jan
    end = cftime.DatetimeNoLeap(1961, 12, 31) # Must be 31st Dec
    aggregated_data = aggregated_data.sel(time=slice(start, end))
    
    ds_by_year = aggregated_data.groupby("time.year")
    
    for year, ds_year in tqdm(list(ds_by_year)):
        exceedances = ds_year > thresholds
        
        # DOY
        # by_day = exceedances.groupby("time.dayofyear")
        # for day, ds_day in tqdm(list(by_day)):
        #     all_years_per_day.append([(ds_day == 1).sum().compute().item()])
            
        # MONTH
        by_month = exceedances.groupby("time.month")
        for month, ds_month in by_month:
            # all_years_per_month.append([(ds_month == 1).sum().compute().item() ])
            monthly_exceedances_per_lat_long = (ds_month == 1).sum(dim='time').compute().to_numpy()
            
            all_years_per_month.append(monthly_exceedances_per_lat_long)
        
    print(len(all_years_per_month))
    print(all_years_per_month)
    a = all_years_per_month[0]
    print(a)
    print(len(a))
        
    exit()
        
    # np.save(res_dir + '/' + str(an_start.year) + '_' + str(an_end.year) + '_' + str(perc).replace('.','_') + '_per_day', np.array(all_years_per_day, dtype=np.int32))
    np.save(res_dir + '/' + str(an_start.year) + '_' + str(an_end.year) + '_' + str(perc).replace('.','_') + '_per_month', np.array(all_years_per_month, dtype=np.int32))
