import gcsfs
import xarray as xr
import os
import re
import netCDF4
import cftime
from enums import DAYS_PER_MONTH

import os 
print(os.getcwd())

class GCSDataLoader:

    def __init__(self):
        self.fs = gcsfs.GCSFileSystem()

    def load_weatherbench(self):
        return xr.open_zarr(self.fs.get_mapper(f'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'))        
        
class ZarrLoader:
    def __init__(self, input_zarr_path):
        self.input_zarr_path = input_zarr_path
    
    def load(self):
        return xr.open_zarr(self.input_zarr_path)
    
    def download_and_store_as_zarr(self, variable, start_date, end_date, output_dir):
        """
        Download and store the data for a specific variable and time range as Zarr.

        Args:
        - variable (str): The name of the variable to download (e.g., 'temperature', 'precipitation').
        - start_date (str): The start date in the format 'YYYY-MM-DD'.
        - end_date (str): The end date in the format 'YYYY-MM-DD'.
        - output_dir (str): The directory where the Zarr file will be stored.
        """
        # Load the entire dataset
        ds = self.load()

        # Select the subset of data based on the variable and time range
        ds_subset = ds[variable].sel(time=slice(start_date, end_date))

        # Define Zarr store path
        zarr_store = f'{output_dir}/{variable}_{start_date}_to_{end_date}.zarr'

        # Save the subset of data to the specified directory as Zarr
        ds_subset.to_zarr(zarr_store, mode='w', consolidated=True)

        print(f"Data for {variable} from {start_date} to {end_date} saved to {zarr_store}.")
        
    def calculate_daily_mean_2m_temperature(self, input_zarr_path, output_zarr_path):
        """
        Calculate the daily mean 2m temperature from the downloaded Zarr file
        and save the result as a new Zarr file in the specified directory.

        Args:
        - input_zarr_path (str): The path to the input Zarr file.
        - output_zarr_path (str): The path to save the output Zarr file.
        """
        # Open the input Zarr dataset
        ds = xr.open_zarr(input_zarr_path)

        # Check if the 2m temperature variable is available
        if '2m_temperature' not in ds:
            raise ValueError("The dataset does not contain the '2m_temperature' variable.")

        # Resample the data to daily mean (assuming time dimension is named 'time')
        daily_mean_t2m = ds['2m_temperature'].resample(time='1D').mean()

        # Create a new dataset to hold the daily mean temperature
        ds_daily_mean = xr.Dataset({'daily_mean_2m_temperature': daily_mean_t2m})

        # Save the new dataset to Zarr
        ds_daily_mean.to_zarr(output_zarr_path, mode='w', consolidated=True)

        print(f"Daily mean 2m_temperature saved to {output_zarr_path}.")
    
class MichaelDataloader:
    def save_michaels_files_as_zarr(self, directory, start_date, end_date, variable_name, target_variable_name, output_zarr):
        """
        Combine netCDF files from a given directory into a single Zarr dataset along the time dimension,
        appending new data from each file to the existing Zarr dataset.
        
        Parameters:
        - directory: Path to the directory containing netCDF files.
        - start_date: The start date in 'YYYY-MM-DD' format.
        - end_date: The end date in 'YYYY-MM-DD' format.
        - variable_name: The name of the variable to extract (e.g., 't2m').
        - target_variable_name: The target variable name in the output Zarr store.
        - output_zarr: The output path for the Zarr dataset.
        """
        
        assert os.path.exists(directory), directory
        
        # Convert the start and end dates into pandas timestamps for comparison
        start_timestamp = cftime.DatetimeGregorian.strptime(start_date, "%Y-%m-%d")
        end_timestamp = cftime.DatetimeGregorian.strptime(end_date, "%Y-%m-%d")

        # Define a regular expression pattern to extract year and month from the filenames
        file_pattern = re.compile(r"T2MEAN-(\d{4})-(\d{2})")

        # Check if the Zarr store already exists
        zarr_exists = os.path.exists(output_zarr)

        files = list(os.listdir(directory))
        assert len(files) > 0

        # Loop through all files in the directory
        for file_name in sorted(files):
            # Match the pattern for filenames like T2MEAN-1969-10
            match = file_pattern.match(file_name)
            if match:
                year = int(match.group(1))
                month = int(match.group(2))

                # Construct the full file path
                file_path = os.path.join(directory, file_name)

                # Convert the year and month into a timestamp to check if it's within the date range
                file_date1 = cftime.DatetimeGregorian.strptime(f"{year}-{month:02d}-01", "%Y-%m-%d") 
                file_date2 = cftime.DatetimeGregorian.strptime(f"{year}-{month:02d}-{DAYS_PER_MONTH[month-1]:02d}", "%Y-%m-%d") 
                if (start_timestamp <= file_date1 <= end_timestamp) or (start_timestamp <= file_date2 <= end_timestamp):
                    print(file_name)
                    # Read the netCDF file
                    dataset = netCDF4.Dataset(file_path)
                    # Extract the variable
                    data_var = dataset.variables[variable_name][:]

                    # Extract time dimension
                    time_units = dataset.variables['time'].units
                    time_values = dataset.variables['time'][:]

                    # Convert the time to a pandas datetime index
                    time_index = netCDF4.num2date(time_values, time_units)

                    # Create an xarray dataset for this file
                    ds = xr.Dataset(
                        {target_variable_name: (["time", "lat", "lon"], data_var)},
                        coords={
                            "time": time_index,
                            "lat": dataset.variables['lat'][:],
                            "lon": dataset.variables['lon'][:]
                        }
                    )
                    ds['time'] = ds.indexes['time'].to_datetimeindex().values
                    
                    # Check if the Zarr store already exists
                    zarr_exists = os.path.exists(output_zarr)
                    if not zarr_exists:
                        # If the Zarr store does not exist, save the first dataset and create the Zarr store
                        ds.chunk({'time': 1, 'lat': 721, 'lon': 1440}).to_zarr(output_zarr, mode='w')
                    else:
                        # If the Zarr store exists, append the new data to it
                        ds.chunk({'time': 1, 'lat': 721, 'lon': 1440}).to_zarr(output_zarr, mode='a', append_dim='time')

        # Open the final Zarr dataset for verification
        dataset = xr.open_zarr(output_zarr)
        print(dataset)
        return dataset
    
        # Example of usage:
        # combine_nc_to_zarr("/path/to/nc/files", "1969-01-01", "1970-12-31", "t2m", "/output/path/dataset.zarr")