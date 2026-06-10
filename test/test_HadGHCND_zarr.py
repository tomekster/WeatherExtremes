import xarray as xr
path = '/home/tsternal/phd/WeatherExtremes2/data/HadGHCND/HadGHCND_TXTN_acts_1950-2014_padded.zarr'
ds = xr.open_zarr(path)

print(ds)
