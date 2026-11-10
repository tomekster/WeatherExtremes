# """LOAD DATA"""

# import xarray as xr
# import matplotlib.pyplot as plt

# from utils.utils import lat_lon_to_grid_index

# import os
# os.chdir("/home/tsternal/phd/WeatherExtremes2")  # adjust

# data = xr.open_zarr('data/preprocessed/2_weatherbench2_2m_temperature_daily_mean_vienna.zarr')

# print(data)

# # vienna_lat, vienna_lon = 48.210033, 16.363449
# array_2d = data['2m_temperature'].squeeze().values  # remove single-dimensional entries, if any
# # vienna_x, vienna_y = lat_lon_to_grid_index(vienna_lat, vienna_lon, data)

# print(array_2d)

# # for i in range(5):
# #     for j in range(5):
# #         array_2d[vienna_x+i, vienna_y+j] = 200

# # plt.imshow(array_2d, cmap='viridis', aspect='auto')
# # plt.colorbar(label='Temperature')
# # plt.title('2m Temperature, 2020-01-01')
# # plt.xlabel('Longitude Index')
# # plt.ylabel('Latitude Index')
# # plt.imsave("map.png", array_2d)

# # data_vienna = xr.open_zarr('data/preprocessed/2_weatherbench2_2m_temperature_daily_mean.zarr')['2m_temperature'].isel(latitude=vienna_x, longitude=vienna_y)

# # print(len(data_vienna.values))
# # print(data_vienna.values)


import zarr

path = "/home/tsternal/phd/WeatherExtremes2/experiments/daily_max_2m_temperature_1960_1989_AGG.MEAN_aggrwindow_3_percboost_1_seasonality_0/pre_precentile.zarr"

arr = zarr.open(path, mode="r")
print(type(arr))
print(arr.shape, arr.dtype)
print(arr[:5])   # or another small slice



# Zurich (549, 754), (47.25, 8.5)
# San Francisco (511, 230), (37.75, -122.5)
# Cape Town (224, 794), (-34.0, 18.5)


# #LAT
# San Fran 511 -> 37.75
# Zurich 549  -> 47.25


# 38 -> 9.5

# 1 step lat = 0 .25 lat

# # LON
# Zurich 754 -> 8.5 
# Cape 794 -> 18.5

# 40 -> 10
# 1 step lon -> 0.25 lon



# LAT: 224 = -34 -> 360 = 0
# LON: 754 = 8.5  -> 720 = 0
