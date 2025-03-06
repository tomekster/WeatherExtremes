import os
from utils.utils import zarr_has_nans, ensure_dimensions
import xarray as xr

WEATHERBENECH2_PATH = "/ERA5/weatherbench2_original"
PREPRCESSED_DIR = "data/preprocessed/"

def preprocess_weatherbench2(variable, aggregation_method):
    """
    Calculate daily aggregations (min/max/avg) for ERA5 data and save to zarr.
    
    Args:
        input_zarr_path (str): Path to input ERA5 zarr dataset
        start_date (datetime): Start date for the analysis
        end_date (datetime): End date for the analysis
        variable (str): Name of the variable to process
        aggregation_method (str): One of 'min', 'max', 'mean'
        output_zarr_path (str): Path where to save the output zarr
    """

    # Read the data
    data = xr.open_zarr(WEATHERBENECH2_PATH)[variable]
    
    # Verify we have data
    if data.sizes['time'] == 0:
        raise ValueError(f"No data found in zarr file at {WEATHERBENECH2_PATH}")
    
    print(f"Loaded data with time range: {data.time[0].values} to {data.time[-1].values}")
    
    # Convert to no-leap calendar first
    data = data.convert_calendar('noleap')
    
    # Already verified for weatherbench2
    # # check the time-range for NaNs
    # if zarr_has_nans(data):
    #     raise Exception(f"Raw data contains NaNs")
    # else:
    #     print("No NaNs in the data")
    
    if data.sizes['time'] == 0:
        raise ValueError(f"No data found")
    
    # Calculate daily aggregation using groupby instead of resample for more robust handling
    # TODO(tsternal): nmake sure this calculation is correct
    # daily_data = data.groupby(group=data.time.dt.floor('D'))

    # if aggregation_method == 'min':
    #     result = daily_data.min()
    # elif aggregation_method == 'max':
    #     result = daily_data.max()
    # elif aggregation_method in 'mean':
    #     result = daily_data.mean()
    # else:
    #     raise ValueError(f"Invalid aggregation method: {aggregation_method}. Must be one of: min, max, mean")
    
    # result = result.rename({'floor': 'time'})
    
    # Non-overlapping windows and convert to daily dates
    # 'D' means daily frequency, starting at midnight
    if aggregation_method == 'min':
        result = data.resample(time='D').min()
    elif aggregation_method == 'max':
        result = data.resample(time='D').max()
    elif aggregation_method in 'mean':
        result = data.resample(time='D').mean()
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}. Must be one of: min, max, mean")
    
    ensure_dimensions(result)
    
    processed_zarr_path = os.path.join(PREPRCESSED_DIR, f'2_weatherbench2_{variable}_daily_{aggregation_method}.zarr')
    
    print(f"Preprocessed raw data from {WEATHERBENECH2_PATH}, saving to {processed_zarr_path}")
    
    # Save to zarr
    result.to_zarr(processed_zarr_path, mode='w')
    
    if zarr_has_nans(result):
        raise Exception(f"The raw data did not contain any NaNs in the selected time period, but the saved zarr {processed_zarr_path} contains NaNs!")

    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess WeatherBench2 data with daily aggregations.')
    parser.add_argument('var', type=str, help='Variable name to process')
    parser.add_argument('agg', type=str, choices=['min', 'max', 'mean'], 
                        help='Aggregation method (min, max, or mean)')
    
    args = parser.parse_args()
    
    result = preprocess_weatherbench2(args.var, args.agg)
    print(f"Successfully processed {args.var} with {args.agg} aggregation")
