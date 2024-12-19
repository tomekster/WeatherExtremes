from data_loading.dataloader import GCSDataLoader, MichaelDataloader
from data_loading.ftp import FTP

def download_from_gcs():
    dl = GCSDataLoader()

    VAR = '2m_temperature'
    START = '1959-01-01'
    END = '1980-01-01'
    DATA_FOLDER_PATH = 'data'

    dl.download_and_store_as_zarr(VAR, START , END, DATA_FOLDER_PATH)
    dl.calculate_daily_mean_2m_temperature(f"{DATA_FOLDER_PATH}/2m_temperature_1959-01-01_to_1980-01-01.zarr", f"{DATA_FOLDER_PATH}/daily_mean_2m_temperature_1959_1980.zarr")

def download_michaels_data(years=['1942']):
    ### Download Michaels data from the FTP
    ftp = FTP()
    ftp.cd('michaesp/tomasz.sternal/era5.daily')
    ftp.cd('t2mean')
    for year in years:
        fnames = ftp.ls()
        files = [file for file in fnames if year in file]
        print(f"Downloading files {files}")
        for file in files:
            print(f"Downloading year {year}")
            ftp.download(file, f'data/michael_t2_mean/{file}')

nc_files_dir='data/michael_t2_mean'
start_year = 1990
end_year = 2006
start=f"{start_year-1}-11-01"
end=f"{end_year+1}-02-01"
varname='t2m'
target_var_name='daily_mean_2m_temperature'
out_zarr_path=f'data/michaels_t2_single_arr_mean_zarr_{start_year}_{end_year}.zarr'

download_michaels_data(years=[str(x) for x in range(start_year-1, end_year + 1)])
dl = MichaelDataloader()
dl.save_michaels_files_as_zarr(nc_files_dir, start, end, varname, target_var_name, out_zarr_path)