import gcsfs
import xarray as xr


class GCSDataLoader:

    def __init__(self):
        self.fs = gcsfs.GCSFileSystem()

    def load(self):
        return xr.open_zarr(self.fs.get_mapper(f'gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'))
        
