import numpy as np
from netCDF4 import Dataset

"""
Discrepancy example 

Consider the following parameters:

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


Crucially note that both the aggregation_window and the perc_boosting_window are >1

We are interested in 99th percentile for 1st Jan 1965.
Let's check what aggregated mean temeparture values are used to calculate that percentile in both direct and optimized methods. Let's restrict ourselves to the location (3,3)

Let's start with the raw data in T2MEAN nc files:

    Values for 30 Dec 1964 - 3 Jan 1965: 
    [243.8285369873047, 243.59710693359375, 242.8618621826172, 242.03143310546875, 241.74830627441406]
    Values for 30 Dec 1965 - 3 Jan 1965: 
    [245.33184814453125, 245.8216552734375, 246.34466552734375, 244.8524169921875, 244.9216766357422]

Aggregations for those periods:
[243.8285369873047, 243.59710693359375, 242.8618621826172]
[246.34466552734375, 246.34466552734375, 246.34466552734375]

Sorted aggregations: 
[242.8618621826172, 243.59710693359375, 243.8285369873047, 246.34466552734375, 246.34466552734375, 246.34466552734375]

Now let's take a look at what values are considered by the direct and optimized methods when calculating the percentile for the location (3,3) for 1st Jan

direct:
[242.8618621826172, 243.59710693359375, 243.8285369873047, 245.8216552734375, 246.34466552734375, 246.34466552734375]
optimized:
[242.8618621826172, 243.59710693359375, 243.8285369873047, 246.34466552734375, 246.34466552734375, 246.34466552734375]


Notice that the direct method considers the value 245.8216552734375, which is the RAW value for 31 Dec 1965, but which should not appear in the Aggregated data since it
Notice that the direct method considers the value 245.8216552734375, which is the RAW value for 31 Dec 1965, but which should not appear in the Aggregated data since it is lower than the mean temp value for 1st Jan 1965
"""

a = np.load('direct_data_field_slice.npy')
b = np.load('optimized_data_field.npy')

print('direct')
a = a[:,3,3]
a = [float(x) for x in sorted(a)]
print(a)
print('opt')
b = b[:,3,3]
b = [float(x) for x in sorted(b)]
print(b)

with Dataset('/home/tsternal/WeatherExtremes/data/michael_t2_mean/T2MEAN-1964-12', 'r') as ncfile:
    dec = np.array(ncfile.variables['t2m'][:])
with Dataset('/home/tsternal/WeatherExtremes/data/michael_t2_mean/T2MEAN-1965-01', 'r') as ncfile:
    jan = np.array(ncfile.variables['t2m'][:])
with Dataset('/home/tsternal/WeatherExtremes/data/michael_t2_mean/T2MEAN-1965-12', 'r') as ncfile:
    dec2 = np.array(ncfile.variables['t2m'][:])
with Dataset('/home/tsternal/WeatherExtremes/data/michael_t2_mean/T2MEAN-1966-01', 'r') as ncfile:
    jan2 = np.array(ncfile.variables['t2m'][:])
    

l1 = [dec[29,3,3], dec[30,3,3], jan[0,3,3], jan[1,3,3], jan[2,3,3]]
l2 = [dec2[29,3,3], dec2[30,3,3], jan2[0,3,3], jan2[1,3,3], jan2[2,3,3]]
l1 = [float(x) for x in l1]
l2 = [float(x) for x in l2]

agg1 = [float(max(l1[i:i+3])) for i in range(len(l1)-2)]
agg2 = [float(max(l2[i:i+3])) for i in range(len(l2)-2)]

print('TRUE RAW')
print(l1)
print(l2)
print('TRUE AGG')
print(sorted(agg1+agg2))