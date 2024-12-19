from dataloader import GCSDataLoader
from tqdm.auto import tqdm
import xarray as xr

# from dask.diagnostics import Callback
from tqdm.auto import tqdm
from dask.diagnostics import ProgressBar

# class TqdmCallback(Callback):
#     def __init__(self):
#         super().__init__()
#         self.tqdm_bar = None
#         self.total_tasks = 0

#     def _start_state(self, dsk, state):
#         self.total_tasks = len(state['all'])
#         self.tqdm_bar = tqdm(total=self.total_tasks, unit='task', desc='Resampling')

#     def _pretask(self, key, dsk, state):
#         self.tqdm_bar.update(1)

#     def _finish(self, dsk, state, errored):
        # self.tqdm_bar.close()

# Initialize tqdm progress bar
def tqdm_hook(num_tasks):
    # Set up the progress bar
    pbar = tqdm(total=num_tasks, desc="Writing to Zarr", unit="chunk")

    # Define a hook function to update progress
    def update_hook(n):
        pbar.update(n)

    return update_hook

# DOWNLOAD
start_date = "1959-01-01"
end_date = "2021-12-31"

print("Connecting to GCS")
ds = GCSDataLoader().load_weatherbench()
print("Selecting the timespan")
ds = ds.sel(time=slice(start_date, end_date))
print("Selecting the variable")
ds = ds['10m_wind_speed']
print("Chunking")
ds = ds.chunk({'latituted': 721, 'longitude': 1440, 'time': 24})

print("Resampling")
# with TqdmCallback():
with ProgressBar():
    ds = ds.resample(time="1D").max()

num_chunks = sum([v.chunks for v in ds.data_vars.values() if v.chunks])

output_filename = f"/outputs/10m_wind_speed_1959_2021.zarr"
ds.to_zarr(output_filename, mode="w", compute=True, progress=tqdm_hook(num_chunks) )



# CALCULATE DAILY MAX
# import xarray as xr

# def calculate_daily_max_wind_speed(input_zarr_path, output_zarr_path):
#     # Load the dataset from the Zarr file
#     ds = xr.open_zarr(input_zarr_path)
    
#     # Ensure the data has a time dimension that can be resampled
#     if 'time' not in ds.coords:
#         raise ValueError("Dataset does not have a 'time' coordinate for resampling.")
    
#     # Calculate daily maximum wind speed by resampling along the time dimension
#     daily_max_wind_speed = ds.resample(time="1D").max()
    
#     # Save the daily maximum wind speed data to a new Zarr file
#     daily_max_wind_speed.to_zarr(output_zarr_path, mode="w")
    
#     print(f"Daily maximum wind speed data saved to {output_zarr_path}")

# # Specify input and output Zarr file paths
# input_zarr = "data/10m_wind_speed_1959_2021.zarr"
# output_zarr = "data/daily_max_wind_speed_1959_2021.zarr"

# # Call the function
# calculate_daily_max_wind_speed(input_zarr, output_zarr)