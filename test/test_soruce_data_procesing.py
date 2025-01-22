import xarray as xr
from numcodecs.registry import codec_registry

print(codec_registry.keys())  # Check available codecs

# def test_michaels_data_processing():
#     pass

# def test_weatherbench_data_processing():
#     pass

def test_era5_data_processing():
    # ds = xr.open_zarr('/ERA5/experiment')
    ds = xr.open_zarr('data/era5_wb2_1950-2020_combined.zarr')
    print(ds.info)
    print(ds['2m_temperature'].isel(time=0).values)
    # print(z['2m_temperature'])
    # print(z['time'])

test_era5_data_processing()