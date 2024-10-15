from dataloader import GCSDataLoader, MichaelDataloader
from ftp import FTP
# Run this script to download the ERA5 data and calculate daily means for 2m_temperature and save them as .zarr 

def download_from_gcs():
    dl = GCSDataLoader()

    VAR = '2m_temperature'
    START = '1959-01-01'
    END = '1980-01-01'
    DATA_FOLDER_PATH = 'data'

    dl.download_and_store_as_zarr(VAR, START , END, DATA_FOLDER_PATH)
    dl.calculate_daily_mean_2m_temperature(f"{DATA_FOLDER_PATH}/2m_temperature_1959-01-01_to_1980-01-01.zarr", f"{DATA_FOLDER_PATH}/daily_mean_2m_temperature_1959_1980.zarr")

def download_michaels_data(years=['1942-02']):
    ### Download Michaels data from the FTP
    ftp = FTP()
    ftp.cd('michaesp/tomasz.sternal/era5.daily')
    ftp.cd('t2mean')
    for year in years:
        files = [file for file in ftp.ls() if year in file]
        for file in files:
            print(f"Downloading year {year}")
            ftp.download(file, f'data/michael_t2_mean/{file}')
            
# download_michaels_data()

def convert_michaels_data_to_zarr(nc_files_dir='data/michael_t2_mean', start="1964-12-01", end="1971-02-01", varname='t2m', target_var_name='daily_mean_2m_temperature', out_zarr_path='data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr'):
    dl = MichaelDataloader()
    dl.save_michaels_files_as_zarr(nc_files_dir, start, end, varname, target_var_name, out_zarr_path)
    
convert_michaels_data_to_zarr(nc_files_dir='data/michael_t2_mean', start="1950-01-01", end="1980-12-31", varname='t2m', target_var_name='daily_mean_2m_temperature', out_zarr_path='data/michaels_t2_mean_as_zarr_1950-01-01_1980-12-31.zarr')