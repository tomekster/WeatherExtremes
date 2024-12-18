import gcsfs
import xarray as xr

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