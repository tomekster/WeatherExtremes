import netCDF4
import numpy as np
import cftime
from enums import AGG, DAYS_PER_MONTH
from impl_xarray import optimised
from ftp import FTP
from dataloader import ZarrLoader, MichaelDataloader
from diagnostics import run_direct
from tqdm import tqdm

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
    reference_period_agg, pre_perc = optimised(PARAMS)
    # print(reference_period_agg)
    # print(reference_period_agg['time'].values)
    specific_date = cftime.DatetimeNoLeap(1962,9,1)
    data_tomasz = reference_period_agg.sel(time=specific_date).to_numpy()
    print("AGG TOMASZ")
    print(data_tomasz)




def compare_data_fields(params, comp_month=9):
    """
    comp_month (int (1-12)): month for which data_fields should be compared
    """
    
    print("Running a comparison for the following params:")
    print(params)
    print('comp_month:', comp_month)
        
    direct_data_fields_dir='data/direct_data_fields'
    
    ref_y0 = params['reference_period'][0].year
    ref_y1 = params['reference_period'][1].year
    nyears = ref_y1 - ref_y0 + 1
    
    run_direct(
        ndays = params['aggregation_window'] // 2,
        timewin = params['perc_boosting_window'] // 2,
        y_0 = params['analysis_period'][0].year,
        y_1 = params['analysis_period'][1].year,
        m_0 = params['analysis_period'][0].month,
        m_1 = params['analysis_period'][1].month,
        ref_y0 = ref_y0,
        ref_y1 = ref_y1,
        perc = int(params['percentile'] * 100),
        aggmode = str(params['aggregation']).lower().split('.')[-1],
        direct_data_fields_dir=direct_data_fields_dir,
        save_datafields=True,
        analysis_months = [comp_month]
        )
    
    EPS = 0.00001
    
    _, optimized_df = optimised(params)
    np.save("optimized_full_df", optimized_df)
    perc_boost = params['perc_boosting_window']

    for d in tqdm(list(range(DAYS_PER_MONTH[comp_month-1]))):
        direct_df_slice = np.load(f"{direct_data_fields_dir}/day_{d}.npy")
        day = sum(DAYS_PER_MONTH[:comp_month-1]) + d
        
        direct_df_slice = np.sort(direct_df_slice, axis=0)
        
        optimized_df_slice = optimized_df[day * nyears:(day+perc_boost) * nyears,:,:]
        optimized_df_slice = np.sort(optimized_df_slice, axis=0)
        
        assert not np.isnan(direct_df_slice).any()
        assert not np.isnan(optimized_df_slice).any()
        
        try:
            assert np.allclose(direct_df_slice, optimized_df_slice, atol=EPS)
        except:
            dir_slice_path = 'direct_data_field_slice'
            opt_slice_path = 'optimized_data_field'
            np.save(dir_slice_path, direct_df_slice)
            np.save(opt_slice_path, optimized_df_slice)
            print(f"Warning! Data Fields for calculating percentiels disagree on day {d} ({day+1}/365)")
            print(f"Saved disagreeing slices to {dir_slice_path} and {opt_slice_path}")
            assert np.allclose(direct_df_slice, optimized_df_slice, atol=EPS)
            
    print("Success! Pre-Percentile data fields agree on all days of the month!")
    
year1_agg1_boost3_params = {
        'var': 'daily_mean_2m_temperature',
        # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'reference_period': (cftime.DatetimeNoLeap(1965, 1, 1), cftime.DatetimeNoLeap(1965, 12, 31)),
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 1,
        'perc_boosting_window': 3,
        'percentile': 0.99,
    }

compare_data_fields(year1_agg1_boost3_params, comp_month=1)
compare_data_fields(year1_agg1_boost3_params, comp_month=9)
compare_data_fields(year1_agg1_boost3_params, comp_month=12)

year2_agg1_boost3_params = {
        'var': 'daily_mean_2m_temperature',
        # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'reference_period': (cftime.DatetimeNoLeap(1965, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 1,
        'perc_boosting_window': 3,
        'percentile': 0.99,
    }

compare_data_fields(year2_agg1_boost3_params, comp_month=1)
compare_data_fields(year2_agg1_boost3_params, comp_month=9)
compare_data_fields(year2_agg1_boost3_params, comp_month=12)

year2_agg3_boost3_params = {
        'var': 'daily_mean_2m_temperature',
        # NOTE: For correctness this ref-period has to start on Jan 1st and end on Dec 31st
        'reference_period': (cftime.DatetimeNoLeap(1965, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'analysis_period': (cftime.DatetimeNoLeap(1966, 1, 1), cftime.DatetimeNoLeap(1966, 12, 31)),
        'aggregation': AGG.MAX,
        'aggregation_window': 3,
        'perc_boosting_window': 3,
        'percentile': 0.99,
    }
compare_data_fields(year2_agg3_boost3_params, comp_month=1)
compare_data_fields(year2_agg3_boost3_params, comp_month=9)
compare_data_fields(year2_agg3_boost3_params, comp_month=12)