# Getting Started

This guide will help you get started with PyRegrid by showing basic usage examples and common patterns.

## Basic Example

Here's a simple example of how to use PyRegrid to regrid data from one grid to another:

```python
import xarray as xr
import pyregrid

# Load source and destination grids
source_data = xr.open_dataset('source.nc')
destination_grid = xr.open_dataset('destination.nc')

# Create a regridder
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=destination_grid,
    method='bilinear'
)

# Perform the regridding
result = regridder.regrid(source_data['temperature'])
```

## Creating Grids from Points

PyRegrid also supports creating grids from scattered point data:

```python
import numpy as np
import xarray as xr
import pyregrid

# Example scattered data points
lats = np.array([30, 35, 40, 45])
lons = np.array([-120, -115, -110, -105])
values = np.array([20, 22, 25, 23])

# Create a regular grid from scattered points
grid_data = pyregrid.grid_from_points(
    lats=lats,
    lons=lons,
    values=values,
    method='idw'  # Inverse distance weighting
)
```

## Using with Dask

For large datasets, PyRegrid integrates with Dask for parallel processing:

```python
import dask.array as da
import xarray as xr
import pyregrid

# Load data with Dask
source_data = xr.open_dataset('large_source.nc', chunks={'time': 10})
destination_grid = xr.open_dataset('destination.nc')

# Create regridder (works the same way)
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=destination_grid,
    method='conservative'
)

# The result will be a Dask array
result = regridder.regrid(source_data['temperature'])
# Computation happens when you call .compute()
final_result = result.compute()
```

## Available Interpolation Methods

PyRegrid supports several interpolation methods:

- `'bilinear'`: Bilinear interpolation for smooth fields
- `'nearest'`: Nearest neighbor interpolation
- `'conservative'`: Conservative remapping for conservative quantities
- `'patch'`: Patch recovery for higher-order interpolation
- `'weights'`: Use precomputed weights

## Accessor Interface

PyRegrid provides an xarray accessor for convenient access to regridding methods:

```python
import xarray as xr
import pyregrid

# Load your data
ds = xr.open_dataset('data.nc')

# Use the regrid accessor
result = ds.regrid(target_grid=destination_grid, method='bilinear')
```

## Next Steps

- Explore the [User Guide](user-guide/core-concepts.md) for detailed explanations of PyRegrid's features
- Check out the [Tutorials](tutorials/index.md) for more comprehensive examples
- Review the [API Reference](api-reference/index.md) for detailed function documentation