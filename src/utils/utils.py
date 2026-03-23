# Define season boundaries (days within year)
SEASON_BOUNDS = [
    (-31, 59),   # DJF: Dec(-31 to -1), Jan(0-30), Feb(31-58)
    (59, 151),   # MAM: Mar-May
    (151, 243),  # JJA: Jun-Aug
    (243, 334)   # SON: Sep-Nov
]

def zarr_has_nans(data):
    return bool(data.isnull().any().compute())

    # Throws AttributeError: 'DataArray' object has no attribute 'to_array'. Did you mean: 'to_zarr'? when preprocessing weatherbench2
    # return bool(data.isnull().any().to_array().any().compute())

def ensure_dimensions(data):
    """Normalise dimension names to latitude/longitude and validate all three exist."""
    if 'lat' in data.dims:
        data = data.rename({"lat": "latitude"})
    if 'lon' in data.dims:
        data = data.rename({"lon": "longitude"})

    for dim in ['latitude', 'longitude', 'time']:
        if dim not in data.dims:
            raise ValueError(f"Dimension '{dim}' is missing: \n{data}")

    return data
        
def lat_lon_to_grid_index(lat, lon, data):
    # Use xarray's "nearest" method to find the closest latitude and longitude indices
    nearest = data.sel(latitude=lat, longitude=lon, method="nearest")
    # Find the integer indices by searching for the position in the coordinate arrays
    lat_idx = int((data.latitude.values == nearest.latitude.values).nonzero()[0][0])
    lon_idx = int((data.longitude.values == nearest.longitude.values).nonzero()[0][0])
    return lat_idx, lon_idx
    
# print(grid_index_to_lat_long(334,576))
# print(grid_index_to_lat_long(390,1030))


# Zurich (549, 754), (47.25, 8.5)
# San Francisco (511, 230), (37.75, -122.5)
# Cape Town (224, 794), (-34.0, 18.5)
    