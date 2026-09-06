from src.utils.utils import lat_lon_to_grid_index
from collections import namedtuple
import xarray as xr

def load_vienna_subset():
    Coord = namedtuple('Coord', ['lat', 'lon'])
    input_zarr = 'data/preprocessed/rechunked/t2max_rechunked.zarr/'
    ds = xr.open_zarr(input_zarr)
    ds["daily_max_2m_temperature"] = ds["daily_max_2m_temperature"] - 273.15 # Convert from Kelvin to Celsius

    # Get statistics for Vienna
    vienna = Coord(lat=48.2082, lon=16.3738)
    idx_lat, idx_lon = lat_lon_to_grid_index(vienna.lat, vienna.lon)
    vienna_subset = ds.sel(latitude=idx_lat, longitude=idx_lon)
    vienna_subset = vienna_subset.convert_calendar("noleap")  # Ensure 365 days per year
    # Select the following range of dates
    # 1960-01-01 to 2019-12-31, inclusive (using no-leap calendar)
    vienna_subset = vienna_subset.sel(time=slice("1960-01-01", "2019-12-31"))
    print(vienna_subset.time)
    return vienna_subset