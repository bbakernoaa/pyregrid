# Coordinate Systems

This guide explains how PyRegrid handles different coordinate reference systems (CRS) and coordinate transformations.

## Overview

PyRegrid provides robust support for various coordinate systems commonly used in geospatial analysis. Understanding coordinate systems is crucial for accurate regridding operations, especially when working with data from different sources or projections.

## Supported Coordinate Systems

PyRegrid supports several types of coordinate systems:

### Geographic Coordinates (WGS84)
- Standard longitude/latitude coordinates
- EPSG:4326 is the most common representation
- Used for global datasets
- Requires special handling for interpolation across datelines

### Projected Coordinates
- Cartesian coordinate systems derived from geographic coordinates
- Examples: UTM, Albers Equal Area, Lambert Conformal Conic
- Units typically in meters
- Preserves certain properties (area, shape, distance)

### Native Grid Coordinates
- Model-specific coordinate systems
- Often used in climate and weather models
- May require specific transformation parameters

## Coordinate Handling in PyRegrid

PyRegrid automatically detects and handles coordinate systems when possible:

```python
import xarray as xr
import pyregrid

# Load data with coordinate information
source_data = xr.open_dataset('source.nc')
target_grid = xr.open_dataset('target.nc')

# PyRegrid will use coordinate information for proper transformations
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=target_grid,
    method='bilinear'
)
```

## Specifying Coordinate Reference Systems

You can explicitly specify coordinate reference systems:

```python
# Using EPSG codes
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=target_grid,
    source_crs='EPSG:4326',
    destination_crs='EPSG:3857',
    method='bilinear'
)

# Using proj4 strings
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=target_grid,
    source_crs='+proj=longlat +datum=WGS84',
    destination_crs='+proj=merc +datum=WGS84',
    method='bilinear'
)
```

## Handling Datelines and Poles

Special care is needed when regridding across datelines or near poles:

- Data spanning the dateline (±180° longitude) requires special handling
- Polar regions may have coordinate singularities
- PyRegrid attempts to automatically detect and handle these cases
- For complex cases, consider preprocessing data to avoid discontinuities

## Best Practices

1. **Ensure coordinate consistency**: Make sure all input data has properly defined coordinates
2. **Check coordinate bounds**: Verify that coordinate ranges are appropriate
3. **Consider projection effects**: Different projections have different properties and limitations
4. **Validate results**: Always verify that regridded data appears reasonable geographically
5. **Use appropriate methods**: Some interpolation methods work better with certain coordinate systems

## Common Issues and Solutions

### Dateline Wrapping
If your data crosses the dateline, consider using a coordinate system that doesn't have a discontinuity there, or split the data before regridding.

### Coordinate Order
Ensure longitude/latitude order is consistent (typically longitude first for xarray coordinates).

### Units
Verify that coordinate units are consistent between source and destination grids.