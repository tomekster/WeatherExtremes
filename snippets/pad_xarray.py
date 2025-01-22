import xarray as xr
import numpy as np

zarr_path = '/home/tsternal/phd/WeatherExtremes/data/HadGHCND/HadGHCND_TXTN_acts_1950-2014_combined.zarr'
out_path = '/home/tsternal/phd/WeatherExtremes/data/HadGHCND/HadGHCND_TXTN_acts_1950-2014_padded.zarr'

padded_days = 31

ds = xr.open_zarr(zarr_path)

print('Original')
print(ds)

# Extract the original data and convert to NumPy arrays
original_data = {var: ds[var].values for var in ds.data_vars}

# Get dimensions
time_size, lat_size, lon_size = len(ds['time']), len(ds['latitude']), len(ds['longitude'])

# Create arrays for the new padded time
time_before = [ds['time'].values[0] - np.timedelta64(time_delta, 'D') for time_delta in range(padded_days,0,-1)]
time_after = [ds['time'].values[-1] + np.timedelta64(time_delta, 'D') for time_delta in range(1, padded_days +1)]
padded_times = np.concatenate([time_before, ds['time'].values, time_after])

print('padded_times_len:', padded_times.shape)

# Create padded data arrays with zeros for the new time points
padded_data = {
    var: np.zeros((time_size + 2 * padded_days, lat_size, lon_size), dtype=np.float32)
    for var in ds.data_vars
}

# Fill the padded array with original data in the correct positions
for var in ds.data_vars:
    padded_data[var][padded_days:-padded_days, :, :] = original_data[var]

# Create new dataset
new_ds = xr.Dataset(
    {
        var: (('time', 'latitude', 'longitude'), padded_data[var])
        for var in padded_data
    },
    coords={
        'time': padded_times,
        'latitude': ds['latitude'].values,
        'longitude': ds['longitude'].values,
    }
)

#23739 = len(original_data)
chunking = {'latitude':73, 'longitude':96, 'time':23739 + 2 * padded_days}
# # Re-chunk the new dataset
new_ds = new_ds.chunk(chunking)
new_ds.to_zarr(out_path, mode='w', consolidated=True)
