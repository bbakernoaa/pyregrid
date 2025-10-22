"""
Grid from points utility function.

This module provides the grid_from_points function for creating regular grids from scattered point data.
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any
import warnings
from pyregrid.point_interpolator import PointInterpolator


def grid_from_points(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_grid: Union[xr.Dataset, xr.DataArray, Dict[str, np.ndarray]],
    method: str = "idw",
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[str] = None,
    target_crs: Optional[str] = None,
    use_dask: Optional[bool] = None,
    chunk_size: Optional[Union[int, tuple]] = None,
    **kwargs
) -> xr.Dataset:
    """
    Create a regular grid from scattered point data.
    
    This function interpolates values from scattered points to a regular grid,
    similar to GDAL's gdal_grid tool.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        The source scattered point data to interpolate from.
        For DataFrame, should contain coordinate columns (e.g., 'longitude', 'latitude').
        For Dataset, should contain coordinate variables.
        For dict, should have coordinate keys like {'longitude': [...], 'latitude': [...]}.
    target_grid : xr.Dataset, xr.DataArray, or dict
        The target grid definition to interpolate to
        For xarray objects: regular grid with coordinate variables
        For dict: grid specification with coordinate arrays like {'lon': [...], 'lat': [...]}
    method : str, optional
        The interpolation method to use (default: 'idw')
        Options: 'idw', 'linear', 'nearest', 'moving_average', 'gaussian', 'exponential'
    x_coord : str, optional
        Name of the x coordinate column/variable (e.g., 'longitude', 'x', 'lon')
        If None, will be inferred from common coordinate names
    y_coord : str, optional
        Name of the y coordinate column/variable (e.g., 'latitude', 'y', 'lat')
        If None, will be inferred from common coordinate names
    source_crs : str, optional
        The coordinate reference system of the source points
    target_crs : str, optional
        The coordinate reference system of the target grid (if different from source)
    use_dask : bool, optional
        Whether to use Dask for computation. If None, automatically detected
        based on data type (default: None)
    chunk_size : int or tuple, optional
        Chunk size for Dask arrays. If None, automatic chunking is used
    **kwargs
        Additional keyword arguments for the interpolation method:
        - For IDW: power (default 2), search_radius (default None)
        - For KNN methods: n_neighbors (default 8), weights (default 'distance')
        
    Returns
    -------
    xr.Dataset
        The interpolated grid data as an xarray Dataset with proper coordinate variables and metadata
    """
    # Validate method
    valid_methods = ['idw', 'linear', 'nearest', 'moving_average', 'gaussian', 'exponential']
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
    
    # Validate input types
    if not isinstance(source_points, (pd.DataFrame, xr.Dataset, dict)):
        raise TypeError(
            f"source_points must be pandas.DataFrame, xarray.Dataset, or dict, "
            f"got {type(source_points)}"
        )
    
    if not isinstance(target_grid, (xr.Dataset, xr.DataArray, dict)):
        raise TypeError(
            f"target_grid must be xr.Dataset, xr.DataArray, or dict, "
            f"got {type(target_grid)}"
        )
    
    # Handle target grid specification
    if isinstance(target_grid, dict):
        # Convert dict to xarray Dataset
        if 'lon' in target_grid and 'lat' in target_grid:
            lon_coords = target_grid['lon']
            lat_coords = target_grid['lat']
        elif 'x' in target_grid and 'y' in target_grid:
            lon_coords = target_grid['x']
            lat_coords = target_grid['y']
        else:
            # Try to infer coordinate names
            coord_keys = [k for k in target_grid.keys() if 'lon' in k.lower() or 'x' in k.lower()]
            lat_keys = [k for k in target_grid.keys() if 'lat' in k.lower() or 'y' in k.lower()]
            if coord_keys and lat_keys:
                lon_coords = target_grid[coord_keys[0]]
                lat_coords = target_grid[lat_keys[0]]
            else:
                raise ValueError("Could not find longitude/latitude coordinates in target_grid dict")
        
        # Create coordinate arrays
        lon_coords = np.asarray(lon_coords)
        lat_coords = np.asarray(lat_coords)
        
        # Create target grid Dataset
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], lon_coords),
                'lat': (['lat'], lat_coords)
            }
        )
    elif isinstance(target_grid, xr.DataArray):
        # Convert DataArray to Dataset while preserving coordinates
        target_grid = target_grid.to_dataset()
    
    # Extract coordinate names from the target grid
    if isinstance(target_grid, xr.Dataset):
        lon_name = [str(coord) for coord in target_grid.coords
                   if 'lon' in str(coord).lower() or 'x' in str(coord).lower()]
        lat_name = [str(coord) for coord in target_grid.coords
                   if 'lat' in str(coord).lower() or 'y' in str(coord).lower()]
    else:  # This shouldn't happen due to type check, but just in case
        raise TypeError(f"target_grid must be xr.Dataset or converted to xr.Dataset, got {type(target_grid)}")
    
    # Default to common names if not found
    if not lon_name:
        lon_name = ['lon'] if 'lon' in target_grid.coords else ['x']
    if not lat_name:
        lat_name = ['lat'] if 'lat' in target_grid.coords else ['y']
        
    lon_name = lon_name[0]
    lat_name = lat_name[0]
    
    # Check if Dask is available and should be used
    try:
        import dask.array as da
        dask_available = True
    except ImportError:
        dask_available = False
        da = None
    
    # Determine whether to use Dask
    if use_dask is None:
        # Check if source_points or target_grid contains Dask arrays
        use_dask = False
        if isinstance(source_points, (xr.Dataset, xr.DataArray)):
            use_dask = hasattr(source_points.data, 'chunks') if hasattr(source_points, 'data') else False
        elif isinstance(target_grid, (xr.Dataset, xr.DataArray)):
            use_dask = hasattr(target_grid.data, 'chunks') if hasattr(target_grid, 'data') else False
    
    # Create PointInterpolator instance
    # The grid_from_points function is meant to interpolate scattered points to a grid
    # So we should use the PointInterpolator from point_interpolator.py which handles scattered data
    try:
        from pyregrid.point_interpolator import PointInterpolator
        interpolator = PointInterpolator(
            source_points=source_points,
            method=method,
            x_coord=x_coord,
            y_coord=y_coord,
            source_crs=source_crs,
            use_dask=use_dask,
            chunk_size=chunk_size,
            **kwargs
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create PointInterpolator: {str(e)}")
    
    # Interpolate to the target grid
    try:
        result = interpolator.interpolate_to_grid(target_grid, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to interpolate to grid: {str(e)}")
    
    # Ensure the result is an xarray Dataset with proper metadata
    if not isinstance(result, xr.Dataset):
        raise RuntimeError(f"Interpolation result is not an xarray Dataset: {type(result)}")
    
    # Add metadata to the result
    result.attrs["interpolation_method"] = method
    result.attrs["source_type"] = type(source_points).__name__
    result.attrs["description"] = f"Grid created from scattered points using {method} method"
    
    # Add any additional attributes from kwargs
    for key, value in kwargs.items():
        if key not in result.attrs:
            result.attrs[f"param_{key}"] = value
    
    return result