# Basic Regridding Example

This example demonstrates how to regrid a simple dataset from a source grid to a destination grid using bilinear interpolation.

## Prerequisites

Ensure you have the following libraries installed:
- `pyregrid`
- `xarray`
- `numpy`

## Example Code

```python
import xarray as xr
import numpy as np
import pyregrid

# 1. Create a source dataset with a simple grid
lats_source = np.linspace(0, 90, 10)
lons_source = np.linspace(0, 180, 20)
source_data_vars = {
    'temperature': (('lat', 'lon'), np.random.rand(10, 20))
}
source_coords = {
    'lat': lats_source,
    'lon': lons_source
}
source_ds = xr.Dataset(source_data_vars, coords=source_coords)

# 2. Create a destination grid (e.g., a coarser grid)
lats_dest = np.linspace(10, 80, 5)
lons_dest = np.linspace(20, 160, 10)
destination_ds = xr.Dataset(coords={'lat': lats_dest, 'lon': lons_dest})

# 3. Instantiate the regridder
regridder = pyregrid.GridRegridder(
    source_grid=source_ds,
    destination_grid=destination_ds,
    method='bilinear'
)

# 4. Perform the regridding
regridded_data = regridder.regrid(source_ds['temperature'])

# 5. Display the result
print("Source Dataset:")
print(source_ds)
print("\nDestination Grid:")
print(destination_ds)
print("\nRegridded Data:")
print(regridded_data)

# You can also save the regridded data
# regridded_data.to_netcdf("regridded_temperature.nc")
```

This example creates two simple grids, instantiates the `GridRegridder` with bilinear interpolation, performs the regridding, and prints the resulting `xarray.DataArray`.