def zarr_has_nans(data):
    return bool(data.isnull().any().compute())

    # Throws AttributeError: 'DataArray' object has no attribute 'to_array'. Did you mean: 'to_zarr'? when preprocessing weatherbench2
    # return bool(data.isnull().any().to_array().any().compute())

def ensure_dimensions(data):
    # Make sure we use the correct dimension names
    if 'lat' in data.dims:
        data = data.rename({"lat": "latitude"})
    if 'lon' in data.dims:
        data = data.rename({"lon": "longitude"})
    
    for dim in ['latitude', 'longitude', 'time']:
        if dim not in data.dims:
            raise ValueError(f"Dimension {dim} is missing from the resulting zarr: \n{data}")