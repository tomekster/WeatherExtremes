import xarray as xr
path = '/home/tsternal/phd/WeatherExtremes/data/HadGHCND/HadGHCND_TXTN_acts_1950-2014_padded.zarr'
ds = xr.open_zarr(path)
print(ds['temperature_2m_max'].isel(time=slice(29,33)))
