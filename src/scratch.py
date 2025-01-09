"""LOAD DATA"""

import cftime
from datetime import timedelta 
import xarray as xr

YEARS = [2000, 2001, 2002, 2003, 2004]
zarr_path = f"data/michaels_t2_single_arr_mean_zarr_1959-11-01_2021-02-01.zarr"
var = "daily_mean_2m_temperature"

daily_mean_2m_temperature = xr.open_zarr(zarr_path)[var]
print(daily_mean_2m_temperature)


def grid_index_to_lat_long(x, y):
    return (x*0.25 - 90, y*0.25-180)

def lat_lon_to_grid_index(x, y):
    return ((x+90) * 4, (y+180) * 4)

print(grid_index_to_lat_long(334,576))
print(grid_index_to_lat_long(390,1030))


# Zurich (549, 754), (47.25, 8.5)
# San Francisco (511, 230), (37.75, -122.5)
# Cape Town (224, 794), (-34.0, 18.5)