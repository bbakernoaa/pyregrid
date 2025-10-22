# Core Concepts

This guide introduces the fundamental concepts behind PyRegrid and how to effectively use the library for regridding geospatial data.

## What is Regridding?

Regridding is the process of mapping data from one grid structure to another. This is commonly needed in geospatial analysis when:

- Combining datasets with different spatial resolutions
- Converting between different coordinate systems
- Standardizing data to a common grid for analysis
- Downscaling or upscaling spatial data

## Grid Structures

PyRegrid supports various grid structures:

- **Rectilinear grids**: Regular grids with coordinates that vary along each axis independently
- **Curvilinear grids**: Grids where coordinates can vary in both dimensions
- **Unstructured grids**: Collections of points without regular connectivity
- **Scattered points**: Irregularly distributed data points

## Interpolation Methods

PyRegrid provides multiple interpolation methods, each suited for different types of data:

- **Bilinear**: Smooth interpolation for continuous fields like temperature
- **Nearest neighbor**: Preserves original values, good for categorical data
- **Conservative**: Preserves integrals, essential for flux calculations
- **Higher-order methods**: More accurate for smooth fields

## Xarray Integration

PyRegrid has deep integration with xarray, allowing you to:

- Work directly with DataArray and Dataset objects
- Preserve metadata and coordinates during regridding
- Handle multiple variables simultaneously
- Chain operations efficiently

## Dask Integration

For large datasets, PyRegrid integrates with Dask to:

- Process data in chunks
- Distribute computations across multiple cores
- Handle datasets larger than memory
- Maintain lazy evaluation until computation is needed

## Coordinate Reference Systems

PyRegrid handles coordinate reference systems (CRS) through integration with Cartopy and other geospatial libraries, allowing you to:

- Reproject between different coordinate systems
- Handle complex grid transformations
- Maintain geospatial accuracy
- Work with various projection types