from dataloader import GCSDataLoader

# Run this script to download the ERA5 data and calculate daily means for 2m_temperature and save them as .zarr 

dl = GCSDataLoader()

VAR = '2m_temperature'
START = '1960-01-01'
END = '1970-01-01'
DATA_FOLDER_PATH = 'data'

dl.download_and_store_as_zarr(VAR, START , END, DATA_FOLDER_PATH)
dl.calculate_daily_mean_2m_temperature(f"{DATA_FOLDER_PATH}/2m_temperature_1960-01-01_to_1970-01-01.zarr", f"{DATA_FOLDER_PATH}/daily_mean_2m_temperature.zarr")
