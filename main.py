#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------
# Frequency of extremes in ERA5
# -----------------------------------------------

import datetime as dt
import os.path

#%%
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
from cdo import Cdo
from netCDF4 import Dataset

from ftp import FTP

cdo = Cdo()
import warnings

from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')


#%%
# -----------------------------------------
# set parameters and variables
# -----------------------------------------

ftp = FTP()
ftp.cd('michaesp/tomasz.sternal/era5.daily')

# Calculation mode (aggregate | seasonality | percentile | event | climatology | trend | test); compute = aggegate + percentile + event
# calcmode = 'percentile'
calcmode = 'percentile'

# choose meteorological field
fieldname = 't2m'
unit = 'K'

# set paths
# input_dir = '/net/thermo/atmosdyn/michaesp/era5.frequency.of.extremes/era5.daily/t2mean/'
input_dir = 't2mean/'
output_dir = 'output/'

# prefix of data files
prefix = 'T2MEAN'

# Number of aggregation/consecutive days (1 corresponds to single day)
ndays = 2

# Aggregation mode (mean|median|min|max|test)
aggmode = 'mean'

# Start and end of analysis period (year, month)
y_0 = 1950; m_0 = 1
y_1 = 1952; m_1 = 1 

# Start and end of reference period
ref_y0 = 1951
ref_y1 = 1951

# Percentile for extreme-event identification
perc = 99.9

# Timewindow (in days) for percentile boosting
timewin = 2

# Set months for analysis
month = [1]

# number of gridpoints in model
nlon = 1440
nlat = 721


#%%
# functions

# read netCDF file
def ncread(file, fieldname):
    if not os.path.isfile(file):
        print(file)
        ftp.download(file, file)
    with Dataset(file, 'r') as ncfile:
        field = np.array(ncfile.variables[fieldname][:])
    return field

# write netCDF file
def ncwrite(path, field, date, prefix, description, unit):
    times = pd.date_range(date.strftime('%Y%m%d'), periods=field.shape[0], freq='d')
    latitudes = np.linspace(-90,90,field.shape[1])
    longitudes = np.linspace(-180,180,field.shape[2])

    # create dataarray with coordinates
    data_nc = xr.DataArray(
        data = field,
        name = prefix,
        dims = ['time', 'lat', 'lon'],
        coords={'time': times, 'lat': latitudes, 'lon': longitudes},
        attrs={
        'description': description,
        'units': unit}
    )

    dataset = xr.Dataset({
        prefix: data_nc
    })

    dataset.to_netcdf(path, encoding={'time': {'dtype': 'i4'}})

#%%
# -----------------------------------------------
# Aggregation
# -----------------------------------------------

if calcmode == 'aggregate':

    curr_year = y_0; curr_month = m_0

    # read data from current month
    input_file = input_dir + f'{prefix}-{curr_year:04d}-{curr_month:02d}'
    curr_field = ncread(input_file, fieldname)

    while (curr_year < y_1) or (curr_month < m_1):
        print('Aggregate values for '+str(curr_year)+'-'+str(curr_month))

        curr_nt, curr_ny, curr_nx = curr_field.shape

        # read data from next month
        next_month = curr_month + 1
        next_year = curr_year
        if next_month == 13:
            next_month = 1
            next_year += 1
        input_file = input_dir + f'{prefix}-{next_year:04d}-{next_month:02d}'
        next_field = ncread(input_file, fieldname)
        next_nt = next_field.shape[0]

        # reshape and combine both datasets
        data = np.zeros((curr_nt + next_nt, curr_ny, curr_nx))
        data[:curr_nt,:,:] = curr_field
        data[curr_nt:,:,:] = next_field

        # loop over all days of the month
        agg_field = np.zeros_like(curr_field)
        for i in range(curr_nt):
            # collect data for aggregation
            if ndays == 1:
                agg = data[i,:,:]
            elif aggmode == 'mean':
                agg = np.mean(data[i:i + ndays,:,:], axis=0)
            elif aggmode == 'median':
                agg = np.median(data[i:i + ndays,:,:], axis=0)
            elif aggmode == 'min':
                agg = np.min(data[i:i + ndays,:,:], axis=0)
            elif aggmode == 'max':
                agg = np.max(data[i:i + ndays,:,:], axis=0)
            elif aggmode == 'sum':
                agg = np.sum(data[i:i + ndays,:,:], axis=0)

            # write agg to aggregated field
            agg_field[i,:,:] = agg

        # save agg_field to netcdf file
        outpath = output_dir + f'{prefix}-AGG-{curr_year:04d}-{curr_month:02d}'
        description = prefix + ', Number of aggregation: ' + str(ndays)
        ncwrite(outpath, agg_field, dt.datetime(curr_year,curr_month,1), prefix, description, unit)

        # update for next month
        curr_year = next_year
        curr_month = next_month
        curr_field = next_field




#%%

if calcmode == 'seasonality':
    # seasonality calculation on a day-of-the-year basis (without running window) for each grid point separately
    nday_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # -------------
    # what should we do with 29. february?
    # -------------

    for month in range(1,13):
        seasonality_field = np.zeros((nday_per_month[month-1],nlat,nlon))
        for year in range(ref_y0,ref_y1+1):
            inpfile = input_dir + f'{prefix}-{year:04d}-{month:02d}'
            monthly_field = ncread(inpfile,fieldname)

            seasonality_field += monthly_field

        # divide per number of years to get the mean
        seasonality_field /= (ref_y1 - ref_y0 + 1)
        
        
        # save to netcdf file
        outpath = output_dir + f'{prefix}-SEASONALITY-{month:02d}'
        description = 'Mean seasonality of ' + str(prefix) + ' from ' + str(ref_y0) + '-' + str(ref_y1)
        ncwrite(outpath, seasonality_field, dt.datetime(ref_y0,1,1),fieldname,description, unit)




#%%

if calcmode == 'percentile':
    # Number of days per month
    nday_per_month = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # loop over all months
    for mm in range(1,13):
        perc_field = np.zeros((nday_per_month[mm-1],nlat,nlon))

        # loop over all days of the month
        for dd in range(1, nday_per_month[mm - 1] + 1):
            print(str(mm)+'-'+str(dd))
            index = 0
            data_field = None
            data_keep = None

            for yyyy in range(ref_y0, ref_y1 + 1):
                # collect data from current month
                read_mm = mm
                read_yyyy = yyyy
                yyyystr = f'{read_yyyy:04d}'
                mmstr = f'{read_mm:02d}'
                inpfile = output_dir + f'{prefix}-AGG-{yyyystr}-{mmstr}'
                inp_field = ncread(inpfile,prefix)
                nt, ny, nx = inp_field.shape

                if data_field is None:
                    # time dimension is number of years in reference period times boosting window (2*timewin+), the plus one comes from the day we're looking at
                    data_field = np.zeros(((ref_y1-ref_y0+1)*(2*timewin+1),ny,nx))

                # if data from previous month is needed
                if dd <= timewin:
                    read_mm_prev = mm - 1 if mm > 1 else 12
                    read_yyyy_prev = yyyy if mm > 1 else yyyy - 1
                    yyyystr = f'{read_yyyy_prev:04d}'
                    mmstr = f'{read_mm_prev:02d}'
                    inpfile = output_dir + f'{prefix}-AGG-{yyyystr}-{mmstr}'
                    inp_field_prev = ncread(inpfile,prefix)
                    nt_prev, ny_prev, nx_prev = inp_field_prev.shape

                    # calculate the number of days needed in the previous month
                    days_previous = timewin + 1 - dd
                    data_field[index:index+days_previous,:,:] = inp_field_prev[-days_previous:,:,:]
                else:
                    days_previous = 0


                # if data from next month is needed
                if dd + timewin > nday_per_month[mm - 1]:
                    read_mm_next = mm + 1 if mm < 12 else 1
                    read_yyyy_next = yyyy if mm < 12 else yyyy + 1
                    yyyystr = f'{read_yyyy_next:04d}'
                    mmstr = f'{read_mm_next:02d}'
                    inpfile = output_dir + f'{prefix}-AGG-{yyyystr}-{mmstr}'
                    inp_field_next = ncread(inpfile,prefix)
                    nt_next, ny_next, nx_next = inp_field_next.shape

                    # calculate the number of days needed in the next month
                    days_next = timewin + dd - nt
                    data_field[index+days_previous:index+days_previous+days_next,:,:] = inp_field_next[days_next,:,:]
                else:
                    days_next = 0

                data_field[index+days_previous+days_next:index+2*timewin+1,:,:] = inp_field[dd-1:dd-1+2*timewin+1-days_previous-days_next,:,:]

                index += 2*timewin+1

            # ---------------------------------------------
            # subtract seasonality from data_field here
            # ---------------------------------------------

            # calculate percentiles for this day
            perc_field[dd-1,:,:] = np.percentile(data_field, perc, axis=0)

        # write netcdf file with percentiles
        outpath = output_dir + f'{prefix}-PERC-{mm:02d}'
        description = str(perc) + 'th percentile of ' + fieldname +' for reference period ' + str(ref_y0) + '-' + str(ref_y1)
        ncwrite(outpath,perc_field,dt.datetime(ref_y0,1,1),fieldname,description,unit)


#%%

if calcmode == 'event':

    # loop over all years and months
    for year in range(1950,1951):#y_0,y_1+1):
        for month in range(1,2):#13):

            # read aggregated field for respective month
            year_str = f'{year:04d}'
            month_str = f'{month:02d}'
            inpfile = output_dir + f'{prefix}-AGG-{year_str}-{month_str}'
            inp_field = ncread(inpfile,prefix)
            nt, ny, nx = inp_field.shape

            # load corresponding percentiles
            inpfile = output_dir + f'{prefix}-PERC-{month_str}'
            perc_field = ncread(inpfile,fieldname)
            nt_perc, ny_perc, nx_perc = perc_field.shape

            # handle february 29
            if nt < nt_perc:
                nt_perc -= 1
                perc_field = perc_field[:nt_perc,:,:]

            # initialize event array
            events_field = np.zeros((nt,ny,nx))

            event_mask = (inp_field > perc_field)
            events_field[event_mask] = 1

            # write extremes to netcdf file
            outpath = output_dir + f'{prefix}-EVENT-{year_str}-{month_str}'
            description = fieldname + '-extreme events'
            ncwrite(outpath,events_field,dt.datetime(year,month,1),fieldname,description,unit)