import numpy as np
import pandas as pd
import xarray as xr

def convert_multiple_hadghcnd_to_zarr(input_nc_files, output_zarr_path):
    """
    Read multiple daily min/max temperature data files from HadGHCND netCDF files 
    and combine them into a single zarr format.
    
    Args:
        input_nc_files (list): List of paths to input HadGHCND netCDF files
        output_zarr_path (str): Path where to save the output zarr
    """
    print(f"Converting multiple HadGHCND files to zarr format")
    
    # Read and combine all datasets
    datasets = []
    for file_path in sorted(input_nc_files):  # Sort to ensure chronological order
        print(f"Reading {file_path}")
        ds = xr.open_dataset(file_path, decode_times = False)
        # Convert time values to datetime using days since year 0
        ds['time'] = pd.to_datetime('1950-01-01') + pd.to_timedelta(ds.time.values - 712224 , unit='D')
        
        # Standardize dimension names
        if 'latitude' not in ds.dims:
            ds = ds.rename({'lat': 'latitude'})
        if 'longitude' not in ds.dims:
            ds = ds.rename({'lon': 'longitude'})
            
        # Convert to standard variable names if needed
        var_mapping = {
            'tmax': 'temperature_2m_max',
            'tmin': 'temperature_2m_min'
        }
        ds = ds.rename({old: new for old, new in var_mapping.items() if old in ds})
        
        datasets.append(ds)
    
    # Combine all datasets along the time dimension
    combined_ds = xr.concat(datasets, dim='time')
    
    # Sort by time to ensure chronological order
    combined_ds = combined_ds.sortby('time')
    
    # Save to zarr with appropriate chunking
    chunks = {
        'time': -1,
        'latitude': 1,
        'longitude': combined_ds.dims['longitude']
    }
    
    combined_ds = combined_ds.chunk(chunks)
    print(f"Saving combined dataset to {output_zarr_path}")
    combined_ds.to_zarr(output_zarr_path, mode='w')
    
    return combined_ds

# Example usage with your files
input_files = [
    'data/HadGHCND/HadGHCND_TXTN_acts_1950-1960_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_1961-1970_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_1971-1980_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_1981-1990_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_1991-2000_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_2001-2010_15102015.nc',
    'data/HadGHCND/HadGHCND_TXTN_acts_2011-2014_15102015.nc'
]

convert_multiple_hadghcnd_to_zarr(input_files, 'data/HadGHCND/HadGHCND_TXTN_acts_1950-2014_combined.zarr')