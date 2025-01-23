import os
from datetime import timedelta
import xarray as xr
from enums import AGG
import zarr
import time
from tqdm import tqdm
import dask.array as da
import numpy as np
import pandas as pd

class Experiment:
    
    def __init__(self, params, raw_data_path, input_zarr_path, percentile):
        self.input_zarr_path = input_zarr_path
        
        # PARAMETERS INITIALIZATION
        self.var = params['var']
        self.aggregation = params['aggregation']
        self.ref_start, self.ref_end = params['reference_period']
        self.an_start, self.an_end = params['analysis_period']
        self.perc_boost = params['perc_boosting_window']
        self.agg_window = params['aggregation_window']
        self.half_perc_boost = self.perc_boost // 2
        self.n_years = self.ref_end.year - self.ref_start.year + 1
        self.percentile = percentile

        self.half_agg_days = timedelta(days=self.agg_window//2)
        self.half_perc_boost_days = timedelta(days=self.half_perc_boost)
        
        self.agg_start = min(self.ref_start, self.an_start) - (self.half_agg_days + self.half_perc_boost_days)
        self.agg_end = max(self.ref_end, self.an_end) + (self.half_agg_days + self.half_perc_boost_days)
        
        self.lat_size=params['lat_size']
        self.lon_size=params['lon_size']
        
        # EXPERIMENT PATHS AND DIRECTORY INITIALIZATION
        self.experiment_dir = f"experiments/{self.var}_{self.ref_start.year}_{self.ref_end.year}_{str(self.aggregation)}_aggrwindow_{params['aggregation_window']}_percboost_{self.perc_boost}"
        self.pre_percentile_zarr_path = os.path.join(self.experiment_dir, 'pre_precentile.zarr')
        
        perc_string = str(self.percentile).replace('.','_')
        self.percentiles_path = os.path.join(f"{self.experiment_dir}", f"percentiles_{perc_string}.npy")
        self.monthly_exceedances_path = os.path.join(self.experiment_dir, f"monthly_exceedances_{perc_string}.zarr")
    
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        if raw_data_path:
            if 'ERA5' in raw_data_path:
                if not os.path.exists(input_zarr_path):
                    Experiment.calculate_daily_aggregation_for_era5(
                        raw_data_path,
                        self.agg_start,
                        self.agg_end,
                        self.var,
                        str(self.aggregation).lower(),
                        input_zarr_path
                    )
            elif 'HadGHCND' in raw_data_path:
                pass
            else:
                raise "Unsupported raw data path"

    def aggregate(self, data):
        
        print("Aggregating, checking if there are no NaNs in the data")
        try:
            assert not np.isnan(data.values).any()
        except:
            raise Exception(f"The data passed to aggregation function contains NaNs")
        
        print("Converting to no-leap calendar")
        print(data)
        data = data.convert_calendar('noleap')
        
        data = data.sel(time=slice(self.agg_start, self.agg_end))
        
        assert data['time'][0] == self.agg_start, f"Missing data at the beginning of the reference period. Data: {data['time'][0]}, Start: {self.agg_start}"
        assert data['time'][-1] == self.agg_end, f"Missing data at the end of the reference period. Data: {data['time'][-1]}, End: {self.agg_end}"
        
        print("Calculating Aggregations...")
        rolling_data = data.rolling(time=self.agg_window, center=True)

        if self.aggregation == AGG.MEAN:
            aggregated_data = rolling_data.mean()
        elif self.aggregation == AGG.SUM:
            aggregated_data = rolling_data.sum()
        elif self.aggregation == AGG.MIN:
            aggregated_data = rolling_data.min()
        elif self.aggregation == AGG.MAX:
            aggregated_data = rolling_data.max()
        else:
            raise Exception("Wrong type of aggregation provided: params['aggregation'] = ", self.aggregation)
        
        # Make sure we use the correct dimension names
        if 'lat' in aggregated_data.dims:
            aggregated_data = aggregated_data.rename({"lat": "latitude"})
        if 'lon' in aggregated_data.dims:
            aggregated_data = aggregated_data.rename({"lon": "longitude"})
        assert 'latitude' in aggregated_data.dims
        assert 'longitude' in aggregated_data.dims
        
        return aggregated_data
    
    def load_pre_perc_zarr(self):
        print(f"PrePercentile Zarr exists, reading from {self.pre_percentile_zarr_path}")
        return zarr.open(self.pre_percentile_zarr_path)
    
    def compute_pre_perc_zarr(self, aggregated_data):
        print("PrePercentile Zarr not found. Calcluating and saving pre-percentiles")
        pre_perc_array = self.parallel_pre_percentile_arrange(aggregated_data)
        pre_perc_zarr = self.save_pre_percentile_to_zarr(pre_perc_array)
        return pre_perc_zarr

    def parallel_pre_percentile_arrange(self, agg_data, start_doy=1, end_doy=365):
        """
        Rearrange the grouped by DOY data and save to zarr
        """
        perc_start = self.ref_start - self.half_perc_boost_days
        perc_end = self.ref_end + self.half_perc_boost_days
        
        jan1doy = 1
        dec31doy = 365
        prefix_to_append = dec31doy + self.half_perc_boost - 365 # How many doy_groups from the beginning should be appended
        suffix_to_preppend = -(jan1doy - self.half_perc_boost - 1) # How many doy_groups from the end should be prepended
        
        print("Rearranging data into pre-percentile format")
        
        assert prefix_to_append >= 0
        assert suffix_to_preppend >= 0
        
        # Read the data and group by Day Of Year
        agg_data = agg_data.sel(time=slice(perc_start, perc_end))
        doy_grouped = agg_data.groupby('time.dayofyear')
        
        prefix_to_append_index=prefix_to_append
        suffix_to_preppend_index=len(doy_grouped)-suffix_to_preppend
        
        prefix_arrays = []
        main_arrays = []
        suffix_arrays = []
        
        for day_of_year in range(start_doy,end_doy+1):
            # Select the group corresponding to this day of year
            doy_groups_np = doy_grouped[day_of_year].data  # Get the underlying dask array

            if day_of_year-1 < prefix_to_append_index:
                suffix_arrays.append(doy_groups_np[1:])
                doy_groups_np = doy_groups_np[:-1]
            elif day_of_year-1 >= suffix_to_preppend_index:
                prefix_arrays.append(doy_groups_np[:-1])
                doy_groups_np = doy_groups_np[1:]
                
            assert doy_groups_np.shape[0] == self.n_years, f"Wrong array shape! Shape: {doy_groups_np.shape}"
            main_arrays.append(doy_groups_np)
            
        arrays = prefix_arrays + main_arrays + suffix_arrays
        return arrays

    def save_pre_percentile_to_zarr(self, arrays, batch_size=1):
        start = time.time()
        
        pre_percentile_zarr_store = zarr.open(self.pre_percentile_zarr_path, mode='w', shape=(len(arrays) * self.n_years, self.lat_size, self.lon_size), chunks=(batch_size * self.n_years, 1, self.lon_size), dtype=arrays[0].dtype)
        
        for i in tqdm(list(range(0, len(arrays), batch_size))):
            batch = da.concatenate(arrays[i:i + batch_size])
            start_index = i * self.n_years
            end_index = start_index + batch.shape[0]
            da.to_zarr(batch, pre_percentile_zarr_store, region=(slice(start_index, end_index), slice(None), slice(None)))

        end = time.time()
        print(f"Rearranging: {end - start} seconds")
        
        return pre_percentile_zarr_store
    
    def _calculate_band_percentiles(self, band):
        assert np.any(band > 0)
        percentiles = []
        for doy in tqdm(range(365), leave=False):
            expected_band_len = self.n_years * (365 + self.perc_boost-1)
            assert len(band) == expected_band_len, f'Wrong band shape: {band.shape}. The expected length is {expected_band_len}'
            
            # Select the days in the current day-of-year percentile boosting window
            pre_percentile_window = band[self.n_years * doy: self.n_years * (self.perc_boost + doy)]
            
            try:
                assert not np.isnan(pre_percentile_window).any()
            except:
                nan_indices = np.argwhere(np.isnan(pre_percentile_window))
                raise Exception(f"The array contains NaN values in band {band} for day-of-year {doy} (indexed from 0)! {nan_indices.shape}")
            
            percentile =np.quantile(pre_percentile_window, self.percentile, axis=0) 
            percentiles.append(percentile)
    
        band_percentiles = np.stack(percentiles)
        print('Percentiles Band Shape', band_percentiles.shape)
        assert np.any(band_percentiles > 0)
        return band_percentiles
            
    def _calculate_percentiles(self, pre_perc_zarr):
        print("Percentiles file not found. Calculating percentiles")
        # To reduce the memory footprint and enable easy parallelization we will calculate the percentiles band, by band
        # Since the original weather state array is 721 x 1440, we will split the first dimension into bands 
        # Every band is a set of consecutive latitudes
        max_mem = 1000000000 # ~1 GB
        max_floats = max_mem // 4 # 4 bytes per float
        band_size = max_floats // (self.n_years * (365 + self.perc_boost - 1) * self.lon_size)
        band_size = min(self.lat_size, band_size)
        
        # Calculate Band Indices
        print("BAND_SIZE", band_size)
        bands_indices = list(range(0, self.lat_size, band_size))
        if bands_indices[-1] < self.lat_size:
            bands_indices.append(self.lat_size)
        bands = list(zip(bands_indices, bands_indices[1:]))
        
        # Calculate percentiles for every Band
        all_percentiles = [self._calculate_band_percentiles(pre_perc_zarr[:,start:end,:]) for (start, end) in bands]
        percentiles = np.concatenate(all_percentiles, axis=1)
        
        print(f"Saving percentiles to {self.percentiles_path}")
        np.save(self.percentiles_path.removesuffix('.npy'), percentiles)
        return percentiles
            
    def load_percentiles(self):
        path = self.percentiles_path
        print(f"Percentiles file exists. Loading from {path}")
        return np.load(path)
        
    def calculate_percentiles(self):
        
        data = xr.open_zarr(self.input_zarr_path)[self.var]
        
        aggregated_data = self.aggregate(data)
        
        pre_perc_zarr = self.load_pre_perc_zarr() if os.path.exists(self.pre_percentile_zarr_path) else self.compute_pre_perc_zarr(aggregated_data)
        
        percentiles = self.load_percentiles() if os.path.exists(self.percentiles_path) else self._calculate_percentiles(pre_perc_zarr)
        
        return percentiles
        
    def calcluate_exceedances(self):
        data = xr.open_zarr(self.input_zarr_path)[self.var]
        
        aggregated_data = self.aggregate(data)
        aggregated_data = aggregated_data.sel(time=slice(self.an_start, self.an_end))
        
        percentiles = self.calculate_percentiles()        
        
        print('Aggregated Data', aggregated_data)
        
        assert aggregated_data.shape[0] % percentiles.shape[0] == 0, f"{aggregated_data.shape}, {percentiles.shape}"
        aggregated_data_doy = aggregated_data.groupby('time.dayofyear')
        print('Aggregated Data DOY', aggregated_data_doy)
        
        threshhold_da = xr.DataArray(percentiles, dims=["dayofyear", 'latitude', 'longitude'])
        exceedances_doy = (aggregated_data_doy > threshhold_da)
        exceedances_doy = exceedances_doy.chunk({"time": -1})
        exceedances_doy.resample(time="1D").sum(dim="time")
        return exceedances_doy
        
    @staticmethod
    def calculate_daily_aggregation_for_era5(input_zarr_path, start_date, end_date, variable, aggregation_method, output_zarr_path):
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
        print(f"Converting hourly ERA5 values to daily values for range: {start_date} {end_date}")

        # Read the data
        data = xr.open_zarr(input_zarr_path)[variable]
        
        # Verify we have data
        if data.sizes['time'] == 0:
            raise ValueError(f"No data found in zarr file at {input_zarr_path}")
        
        print(f"Loaded data with time range: {data.time[0].values} to {data.time[-1].values}")
        
        # Convert to no-leap calendar first
        data = data.convert_calendar('noleap')
        
        # Now select the time slice with converted dates
        data = data.sel(time=slice(start_date, end_date))
        
        if data.sizes['time'] == 0:
            raise ValueError(f"No data found for time range {start_date} to {end_date}")
        
        # Calculate daily aggregation using groupby instead of resample for more robust handling
        daily_data = data.groupby(group=data.time.dt.floor('D'))

        
        agg_methd = aggregation_method.split('.')[-1].lower()
        if agg_methd == 'min':
            result = daily_data.min()
        elif agg_methd == 'max':
            result = daily_data.max()
        elif agg_methd in ['mean', 'avg']:
            result = daily_data.mean()
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}. Must be one of: min, max, mean")
        
        result = result.rename({'floor': 'time'})
        
        # Ensure consistent dimension names
        if 'lat' in result.dims:
            result = result.rename({"lat": "latitude"})
        if 'lon' in result.dims:
            result = result.rename({"lon": "longitude"})
        
        print(f"Converted, saving to {output_zarr_path}")
        # Save to zarr
        result.to_zarr(output_zarr_path, mode='w')
        
        return result
        