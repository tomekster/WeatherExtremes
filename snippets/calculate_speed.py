import xarray as xr

# Load the two datasets
u = xr.open_zarr('u_10m_wind_speed.zarr')
v = xr.open_zarr('v_10m_wind_speed.zarr')

# Ensure both datasets are aligned before averaging
ds_u, ds_v = xr.align(u, v)

# Calculate the mean of the u and v components
speed = (ds_u['10m_u_component_of_wind']**2 + ds_v['10m_v_component_of_wind']**2) ** 0.5

# Create a new dataset with the mean wind
ds_speed = xr.Dataset({'10m_wind_speed': speed})

# if wind_speed.chunks is None:
#     wind_speed = wind_speed.chunk({"time": 1, "latitude":721, "longitude":1440})  # Set chunking as desired

# Save to a new Zarr file
ds_speed.to_zarr('/weather/10m_wind_speed_calculated.zarr', mode='w', compute=True)