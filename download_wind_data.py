from dataloader import GCSDataLoader

# # DOWNLOAD
# start_date = "1959-01-01"
# end_date = "2021-12-31"

# ds = GCSDataLoader().load_weatherbench()
# ds = ds.sel(time=slice(start_date, end_date))
# ds = ds['10m_wind_speed']
# output_filename = f"data/10m_wind_speed_1959_2021.zarr"
# ds.to_zarr(output_filename)


# CALCULATE DAILY MAX
import xarray as xr

def calculate_daily_max_wind_speed(input_zarr_path, output_zarr_path):
    # Load the dataset from the Zarr file
    ds = xr.open_zarr(input_zarr_path)
    
    # Ensure the data has a time dimension that can be resampled
    if 'time' not in ds.coords:
        raise ValueError("Dataset does not have a 'time' coordinate for resampling.")
    
    # Calculate daily maximum wind speed by resampling along the time dimension
    daily_max_wind_speed = ds.resample(time="1D").max()
    
    # Save the daily maximum wind speed data to a new Zarr file
    daily_max_wind_speed.to_zarr(output_zarr_path, mode="w")
    
    print(f"Daily maximum wind speed data saved to {output_zarr_path}")

# Specify input and output Zarr file paths
input_zarr = "data/10m_wind_speed_1959_2021.zarr"
output_zarr = "data/daily_max_wind_speed_1959_2021.zarr"

# Call the function
calculate_daily_max_wind_speed(input_zarr, output_zarr)