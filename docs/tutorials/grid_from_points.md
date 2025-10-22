# Tutorial: Creating Grids from Scattered Points

This tutorial demonstrates how to create a regular grid from scattered data points using PyRegrid's `grid_from_points` function. This is useful when your initial data is not on a structured grid but rather a collection of points with associated values.

## Prerequisites

Ensure you have the following libraries installed:
- `pyregrid`
- `xarray`
- `numpy`

## Understanding `grid_from_points`

The `pyregrid.grid_from_points` function takes scattered latitude, longitude, and value data and interpolates it onto a regular grid.

**Key Parameters:**

*   `lats`: A NumPy array of latitude values for the scattered points.
*   `lons`: A NumPy array of longitude values for the scattered points.
*   `values`: A NumPy array of data values corresponding to each scattered point.
*   `method`: The interpolation method to use. Common options include:
    *   `'idw'` (Inverse Distance Weighting): A simple and common method.
    *   `'nearest'`: Nearest neighbor interpolation.
    *   `'linear'`: Linear interpolation (if applicable, though often requires more complex setup for scattered data).
*   `grid_shape`: A tuple `(n_lat, n_lon)` specifying the desired shape of the output regular grid. If not provided, PyRegrid will attempt to infer a suitable grid shape.
*   `grid_bounds`: A tuple `((min_lat, max_lat), (min_lon, max_lon))` defining the spatial extent of the output grid.

## Example Code

This example creates sample scattered data and then interpolates it onto a regular grid.

```python
import xarray as xr
import numpy as np
import pyregrid

# 1. Generate sample scattered data points
num_points = 100
lats_scatter = np.random.uniform(low=0, high=90, size=num_points)
lons_scatter = np.random.uniform(low=-180, high=180, size=num_points)
# Simulate some data values, e.g., temperature, based on location
values_scatter = (
    20 +
    5 * np.sin(np.deg2rad(lats_scatter)) +
    3 * np.cos(np.deg2rad(lons_scatter)) +
    np.random.normal(0, 1, num_points)
)

print(f"Generated {num_points} scattered data points.")

# 2. Define the desired output grid shape and bounds
# Let's create a grid with 30 latitude points and 60 longitude points
output_grid_shape = (30, 60)
# Define the spatial extent for the output grid
output_grid_bounds = ((0, 90), (-180, 180)) # Min/max lat, Min/max lon

# 3. Create the regular grid using grid_from_points
# We'll use Inverse Distance Weighting (IDW) for interpolation
try:
    regular_grid_da = pyregrid.grid_from_points(
        lats=lats_scatter,
        lons=lons_scatter,
        values=values_scatter,
        method='idw',
        grid_shape=output_grid_shape,
        grid_bounds=output_grid_bounds
    )

    # 4. Display the resulting DataArray
    print("\nSuccessfully created regular grid DataArray:")
    print(regular_grid_da)

    # The result is an xarray.DataArray with 'lat' and 'lon' coordinates
    # You can now use this DataArray for further analysis or regridding
    # For example, to save it:
    # regular_grid_da.to_netcdf("scattered_data_on_regular_grid.nc")

except Exception as e:
    print(f"\nAn error occurred during grid creation: {e}")
    print("Please ensure your input data and parameters are valid.")

```

## Further Exploration

*   Experiment with different `method` parameters (e.g., `'nearest'`).
*   Adjust `grid_shape` and `grid_bounds` to control the resolution and extent of your output grid.
*   Consider how to handle data with missing values or more complex spatial relationships.

This tutorial provides a foundation for working with scattered data in PyRegrid. For more complex scenarios, refer to the [User Guide](user-guide/core-concepts.md) and other tutorials.