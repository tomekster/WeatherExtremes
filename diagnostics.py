#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------
# Frequency of extremes in ERA5
# -----------------------------------------------

#%%
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import cartopy as cart
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import matplotlib.ticker as mticker
from matplotlib import colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import xarray as xr
import time
import pandas as pd
import cftime
import math
import netCDF4 as nc
from netCDF4 import Dataset
from cdo import *
cdo = Cdo()
from scipy.spatial import cKDTree
import calendar

import os,sys,re,fnmatch,datetime

import warnings
warnings.filterwarnings('ignore')

#%%

# take time at the start
time_start = time.time()


#%%
# -----------------------------------------
# set parameters and variables
# -----------------------------------------

# Calculation mode (aggregate | seasonality | percentile | event | climatology | trend | test); compute = aggregate + seasonality + percentile + event + climatology
calcmode = 'compute'
with_seasonality = False

# choose meteorological field
fieldname = 't2m'
unit = 'K'

# set paths
input_dir = '/net/thermo/atmosdyn/michaesp/era5.frequency.of.extremes/era5.daily/t2mean/'
output_dir = '/net/litho/atmosdyn2/svoigt/project_extremes/temp/measure_time/'

# prefix of data files
prefix = 'T2MEAN'

# Number of aggregation/consecutive days (0 corresponds to single day), 1 to 3 days(day before, current day, and day after)
ndays = 2

# Aggregation mode (mean|median|min|max|test)
aggmode = 'max'

# Start and end of analysis period (year, month)
y_0 = 1960; m_0 = 1
y_1 = 1965; m_1 = 1 

# Start and end of reference period
ref_y0 = 1965
ref_y1 = 1970

# Percentile for extreme-event identification
perc = 99

# Timewindow (in days) for percentile boosting
timewin = 2

# Set months for analysis
analysis_months = [9]

# set period for seasonality calculation
seasonality_y0 = 1960
seasonality_y1 = 1990

# number of gridpoints in model
nlon = 1440
nlat = 721


#%%
# functions

# read netCDF file
def ncread(file, fieldname):
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

if calcmode == 'aggregate' or calcmode == 'compute':

    curr_year = np.min([y_0,ref_y0])-1; curr_month = m_0

    # read data from current month
    input_file = input_dir + f'{prefix}-{curr_year:04d}-{curr_month:02d}'
    curr_field = ncread(input_file, fieldname)
    if (calendar.isleap(curr_year)) and (curr_month == 2):
        curr_field = curr_field[:-1,:,:]
    

    while (curr_year <= np.max([y_1,ref_y1])+1) or (curr_month < m_1):
        print('Aggregate values for '+str(curr_year)+'-'+str(curr_month))

        curr_nt, curr_ny, curr_nx = curr_field.shape

        # read data from previous month
        prev_month = curr_month - 1
        prev_year = curr_year
        if prev_month == 0:
            prev_month = 12
            prev_year -= 1
        input_file = input_dir + f'{prefix}-{prev_year:04d}-{prev_month:02d}'
        prev_field = ncread(input_file, fieldname)
        if (calendar.isleap(prev_year)) and (prev_month == 2):
            prev_field = prev_field[:-1,:,:]
        prev_nt = prev_field.shape[0]

        # read data from next month
        next_month = curr_month + 1
        next_year = curr_year
        if next_month == 13:
            next_month = 1
            next_year += 1
        input_file = input_dir + f'{prefix}-{next_year:04d}-{next_month:02d}'
        next_field = ncread(input_file, fieldname)
        if (calendar.isleap(next_year)) and (next_month == 2):
            next_field = next_field[:-1,:,:]
        next_nt = next_field.shape[0]

        # reshape and combine all three datasets
        data = np.zeros((prev_nt + curr_nt + next_nt, curr_ny, curr_nx))
        data[:prev_nt,:,:] = prev_field
        data[prev_nt:prev_nt+curr_nt,:,:] = curr_field
        data[prev_nt+curr_nt:,:,:] = next_field


        # loop over all days of the current month
        agg_field = np.zeros_like(curr_field)
        for i in range(prev_nt,prev_nt+curr_nt):
            # collect data for aggregation
            if ndays == 0:
                agg = data[i,:,:]
            elif aggmode == 'mean':
                agg = np.mean(data[i - ndays:i + ndays,:,:], axis=0)
            elif aggmode == 'median':
                agg = np.median(data[i - ndays:i + ndays,:,:], axis=0)
            elif aggmode == 'min':
                agg = np.min(data[i - ndays:i + ndays,:,:], axis=0)
            elif aggmode == 'max':
                agg = np.max(data[i - ndays:i + ndays,:,:], axis=0)
            elif aggmode == 'sum':
                agg = np.sum(data[i - ndays:i + ndays,:,:], axis=0)

            # write agg to aggregated field
            agg_field[i-prev_nt,:,:] = agg

        # save agg_field to netcdf file
        outpath = output_dir + f'{prefix}-AGG-{curr_year:04d}-{curr_month:02d}'
        description = prefix + ', Number of aggregation: current day plus/minus ' + str(ndays)
        ncwrite(outpath, agg_field, dt.datetime(curr_year,curr_month,1), prefix, description, unit)

        # update for next month
        curr_year = next_year
        curr_month = next_month
        curr_field = next_field




#%%

if calcmode == 'seasonality' or calcmode == 'compute':
    if with_seasonality:
        # seasonality calculation on a day-of-the-year basis (without running window) for each grid point separately
        nday_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        

        for month in range(1,13):
            print('Working on month '+str(month))
            seasonality_field = np.zeros((nday_per_month[month-1],nlat,nlon))
            for year in range(seasonality_y0,seasonality_y1+1):
                inpfile = input_dir + f'{prefix}-{year:04d}-{month:02d}'
                monthly_field = ncread(inpfile,fieldname)
                if month == 2 and monthly_field.shape[0] == 29:
                    monthly_field = monthly_field[:-1,:,:]

                seasonality_field += monthly_field

            # divide per number of years to get the mean
            seasonality_field /= (seasonality_y1 - seasonality_y0 + 1)
            
            
            # save to netcdf file
            outpath = output_dir + f'{prefix}-SEASONALITY-{month:02d}'
            description = 'Mean seasonality of ' + str(prefix) + ' from ' + str(ref_y0) + '-' + str(ref_y1)
            ncwrite(outpath, seasonality_field, dt.datetime(ref_y0,1,1),fieldname,description, unit)




#%%

if calcmode == 'percentile' or calcmode == 'compute':
    # Number of days per month
    nday_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # loop over all months
    for mm in analysis_months:
        perc_field = np.zeros((nday_per_month[mm-1],nlat,nlon))

        if with_seasonality:
            # read in seasonality
            read_mm = mm
            mmstr = f'{read_mm:02d}'
            inpfile = output_dir + f'{prefix}-SEASONALITY-{mmstr}'
            seasonality_field = ncread(inpfile,fieldname)

            # read in seasonality from previous month
            read_mm = mm - 1
            if read_mm == 0:
                read_mm = 12
            mmstr = f'{read_mm:02d}'
            inpfile = output_dir + f'{prefix}-SEASONALITY-{mmstr}'
            seasonality_field_prev = ncread(inpfile,fieldname)

            # read in seasonality from next month
            read_mm = mm + 1
            if read_mm == 13:
                read_mm = 1
            mmstr = f'{read_mm:02d}'
            inpfile = output_dir + f'{prefix}-SEASONALITY-{mmstr}'
            seasonality_field_next = ncread(inpfile,fieldname)

        # loop over all days of the month
        for dd in range(1, nday_per_month[mm - 1] + 1):
            print('Month: '+str(mm)+'; Day: '+str(dd))
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
                if with_seasonality:
                    # subtract seasonality
                    inp_field -= seasonality_field

                if data_field is None:
                    # time dimension is number of years in reference period times boosting window (2*timewin+1), the plus one comes from the day we're looking at
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
                    if with_seasonality:
                        # subtract seasonality
                        inp_field_prev -= seasonality_field_prev

                    # calculate the number of days needed in the previous month
                    days_previous = timewin + 1 - dd
                    # add values from previous month to data_field
                    # place to add values is between index (symbolizing the place of the year within the reference period) and the number of days from the previous month
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
                    if with_seasonality:
                        # subtract seasonality
                        inp_field_next -= seasonality_field_next

                    # calculate the number of days needed in the next month
                    days_next = timewin + dd - nt
                    # add values from next month to data_field
                    # place: between index and number of days from the next month
                    data_field[index+days_previous:index+days_previous+days_next,:,:] = inp_field_next[:days_next,:,:]
                    
                else:
                    days_next = 0


                # add values from current month. if no values from previous/next month is needed: values are between index (symbol for year)
                # and index + the length of the boosting window. if there is data from the previous or next month, one has to change the place
                # where the data from inp_field is added: dd -3 + days_previous comes from: dd-1 is place of the day we are looking at. we need
                # to subtract timewin because of the size of the boosting window. then we add days_previous in case there are days from the previous month
                # (to avoid overwriting those values assigned in the if-statement). dd+2-days_next comes from: dd-1 is the place in inp_field of
                # the day we are looking at. we need to add timewin because of the size of the boosting window. then we add 1 because of the python structure
                # where a range x:y only runs until y-1. Finally we subtract days_next in case values were already assigned in the if-statement before
                data_field[index+days_previous+days_next:index+2*timewin+1,:,:] = inp_field[dd-1-timewin+days_previous:dd-1+timewin+1-days_next,:,:]


                # ---------------------------------------------
                # update index
                index += 2*timewin+1

            # ---------------------------------------------
            # subtract seasonality from data_field here
            # ---------------------------------------------
            '''
            # read in seasonality
            read_mm = mm
            mmstr = f'{read_mm:02d}'
            inpfile = output_dir + f'{prefix}-SEASONALITY-{mmstr}'
            seasonality_field = ncread(inpfile,fieldname)
            seasonality_nt, seasonality_ny, seasonality_nx = seasonality_field.shape

            # subtract seasonaliy field from data_field
            data_field -= seasonality_field[dd-1,:,:]
            '''
            if with_seasonality:
                # calculate percentiles for this day and add seasonality field
                perc_field[dd-1,:,:] = np.percentile(data_field, perc, axis=0) + seasonality_field[dd-1,:,:]
            else:
                perc_field[dd-1,:,:] = np.percentile(data_field, perc, axis=0)

        # write netcdf file with percentiles
        outpath = output_dir + f'{prefix}-PERC-{mm:02d}'
        description = str(perc) + 'th percentile of ' + fieldname +' for reference period ' + str(ref_y0) + '-' + str(ref_y1)
        ncwrite(outpath,perc_field,dt.datetime(ref_y0,1,1),fieldname,description,unit)



                



#%%

if calcmode == 'event' or calcmode == 'compute':

    # loop over all years and months
    for month in analysis_months:
        for year in range(y_0,y_1+1):
        
            print('Working on '+str(year)+'-'+str(month))

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


#%%

# take time at the end
time_end = time.time()

print(str(np.round(time_end-time_start))+'s')

#%%

if calcmode == 'climatology' or calcmode == 'compute':

    # initialize dataframe counting number of extremes
    event_counts = np.zeros((1,nlat,nlon))

    # loop over all years and months
    for month in analysis_months:
        for year in range(y_0,y_1+1):

            # read events-field for respective month
            year_str = f'{year:04d}'
            month_str = f'{month:02d}'
            inpfile = output_dir + f'{prefix}-EVENT-{year_str}-{month_str}'
            inp_field = ncread(inpfile,fieldname)
            nt, ny, nx = inp_field.shape

            # count number of extremes at every gridpoint and add them to event_counts
            event_counts[0,:,:] += np.sum(inp_field,axis=0)


        # write event climatology to netcdf file
        outpath = output_dir + f'{prefix}-CLIMATOLOGY-{month_str}_{y_0}-{y_1}'
        description = 'Number of ' + fieldname + '-extreme events from '+str(y_0)+' to '+str(y_1)
        ncwrite(outpath,event_counts,dt.datetime(y_0,month,1),fieldname,description,'')



#%%

# take time after climatology
time_end = time.time()

print(str(np.round(time_end-time_start))+'s')