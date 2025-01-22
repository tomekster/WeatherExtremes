import xarray as xr
import numpy as np

# ds = xr.open_zarr('monthly_exceedances.zarr')

# a = []
# for i in range(360):
#     data = ds['__xarray_dataarray_variable__'][i].compute().to_numpy()
#     a.append(data.sum())
# print(a)
# print(sum(a))
#     # max_index = np.argmax(data)

#     # max_index_2d = np.unravel_index(max_index, data.shape)

#     # print(max_index_2d)
#     # print(data[max_index_2d])

ds = xr.open_zarr('trends.zarr').isel(month=1).compute().values
print(ds)
