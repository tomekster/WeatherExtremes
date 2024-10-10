import netCDF4
import numpy as np
import cftime
from enums import AGG
from impl_xarray import main
from ftp import FTP
from dataloader import ZarrLoader, MichaelDataloader
from vis import plot_and_save_arrays
 
### Download Michaels data from the FTP
# ftp = FTP()
# ftp.cd('michaesp/tomasz.sternal/era5.daily')
# ftp.cd('t2mean')
# files = [file for file in ftp.ls() if '1959' in file]
# for file in files:
#     ftp.download(file, f'data/michael_t2_mean/{file}')
# exit()

def convert_michaels_data_to_zarr():
    # Convert Michael's data to zarr format
    dl = MichaelDataloader()
    dl.save_michaels_files_as_zarr('data/michael_t2_mean', "1964-12-01", "1971-02-01", 't2m', 'daily_mean_2m_temperature', 'data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr')

def compare_agg():
    # dl = ZarrLoader('data/daily_mean_2m_temperature_1959_1980.zarr') # Raw Weatherbench2 data
    dl = ZarrLoader('data/michaels_t2_mean_as_zarr_1964-12-01_1971-02-01.zarr') # Michales data in Zarr format
    data = dl.load()
    data = data.convert_calendar('noleap')
    data = data['daily_mean_2m_temperature'].sel(time=slice(cftime.DatetimeNoLeap(1962,8,30), cftime.DatetimeNoLeap(1962,9,3))).to_numpy()
    # Flip and roll if weatherbench to match Michael's nc files (unsure about the exact roll value...)
    # data = np.flip(data, axis=1) 
    # data = np.roll(data, 720, axis=2)
    p1 = data[0]
    data = data[:,0,:3]
    print(data)

    print('Michaels RAW nc files')
    data = []
    path1 = 'data/michael_t2_mean/T2MEAN-1962-08'
    aug = netCDF4.Dataset(path1)
    aug = aug.variables['t2m']
    print(np.array(aug[29:,0,:3]))
    p2 = np.array(aug[29])

    # file_path='T2MEAN-1962-09'
    # dataset = netCDF4.Dataset(file_path)
    # # print(dataset.variables.keys())
    # variable = dataset.variables['t2m']
    # data_raw = np.array(variable[:3,0,:3])
    # print(data_raw)

    print("Sven's aggregation in T2MEAN-AGG-1962-09-01")
    file_path = 'data/comparison_with_sven/T2MEAN-AGG-1962-09-01'
    dataset = netCDF4.Dataset(file_path)
    variable = dataset.variables['T2MEAN']
    data_sven = np.array(variable[:])
    p3 = data_sven[0]
    print("SVEN AGG")
    print(data_sven)

    print("Tomasz's aggregation for 1962-09-01")
    # Comparison
    PARAMS = {
        'var': 'daily_mean_2m_temperature',
        'reference_period': (cftime.DatetimeNoLeap(1962, 1, 1), cftime.DatetimeNoLeap(1962, 12, 31)), # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 5,
        'perc_boosting_window': 5,
        'percentile': 0.99,
    }
    reference_period_agg, perc243, perc244 = main(PARAMS)
    # print(reference_period_agg)
    # print(reference_period_agg['time'].values)
    specific_date = cftime.DatetimeNoLeap(1962,9,1)
    data_tomasz = reference_period_agg.sel(time=specific_date).to_numpy()
    print("AGG TOMASZ")
    print(data_tomasz)

def compare_percentile():
    PARAMS = {
        'var': 'daily_mean_2m_temperature',
        'reference_period': (cftime.DatetimeNoLeap(1965, 1, 1), cftime.DatetimeNoLeap(1970, 12, 31)), # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 5,
        'perc_boosting_window': 5,
        'percentile': 0.99,
    }
    reference_period_agg, perc243, perc244 = main(PARAMS)
    
    print("Sven's percentile in T2MEAN-PERC-09-01")
    file_path = 'data/comparison_with_sven/T2MEAN-PERC-09-01'
    dataset = netCDF4.Dataset(file_path)
    print(dataset.variables.keys())
    variable = dataset.variables['t2m']
    data_sven = np.array(variable[:])
    # print(data_sven.shape)
    print("SVEN PERC")
    print(data_sven)
    print(data_sven.shape)

    print("TOMASZ PERC")
    print(perc243)
    print(perc243.shape)
    print(perc244)
    print(perc244.shape)
    
    np.save('perc243.np', perc243)
    np.save('perc244.np', perc244)
    
# convert_michaels_data_to_zarr()
compare_percentile()