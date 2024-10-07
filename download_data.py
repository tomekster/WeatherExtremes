from dataloader import GCSDataLoader

dl = GCSDataLoader()

dl.download_and_store_as_zarr('2m_temperature', '1960-01-01', '1970-01-01', 'data')

dl.calculate_daily_mean_2m_temperature("data/2m_temperature_1960-01-01_to_1970-01-01.zarr", "data/daily_mean_2m_temperature.zarr")
