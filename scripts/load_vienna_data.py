from src.utils.utils import lat_lon_to_grid_index
from collections import namedtuple
import xarray as xr

Coord = namedtuple('Coord', ['lat', 'lon'])
VIENNA = Coord(lat=48.2082, lon=16.3738)


def open_t2max_celsius():
    input_zarr = 'data/preprocessed/rechunked/t2max_rechunked.zarr/'
    ds = xr.open_zarr(input_zarr)
    ds["daily_max_2m_temperature"] = ds["daily_max_2m_temperature"] - 273.15 # Convert from Kelvin to Celsius
    return ds


def select_point_subset(ds, lat, lon):
    idx_lat, idx_lon = lat_lon_to_grid_index(lat, lon)
    subset = ds.sel(latitude=idx_lat, longitude=idx_lon)
    subset = subset.convert_calendar("noleap")  # Ensure 365 days per year
    # Select the following range of dates
    # 1961-01-01 to 2020-12-31, inclusive (using no-leap calendar)
    subset = subset.sel(time=slice("1961-01-01", "2020-12-31"))
    return subset


def load_vienna_subset():
    ds = open_t2max_celsius()

    # Get statistics for Vienna
    vienna_subset = select_point_subset(ds, VIENNA.lat, VIENNA.lon)
    print(vienna_subset.time)
    return vienna_subset
