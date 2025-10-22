"""
PyRegrid Accessor implementation.

This module implements the xarray accessor that provides the .pyregrid interface.
"""

import xarray as xr
import numpy as np
import pandas as pd
from typing import Union, Optional, Dict, Any, Tuple
import warnings
from pyproj import CRS


@xr.register_dataset_accessor("pyregrid")
@xr.register_dataarray_accessor("pyregrid")
class PyRegridAccessor:
    """
    xarray accessor for PyRegrid functionality.
    
    This accessor provides methods for:
    - Grid-to-grid regridding
    - Grid-to-point interpolation
    """
    
    def __init__(self, xarray_obj: Union[xr.Dataset, xr.DataArray]):
        self._obj = xarray_obj
        self._name = "pyregrid"

    def regrid_to(
        self,
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear",
        use_dask: Optional[bool] = None,
        chunk_size: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Regrid the current dataset/dataarray to the target grid.
        
        Parameters
        ----------
        target_grid : xr.Dataset or xr.DataArray
            The target grid to regrid to
        method : str, optional
            The regridding method to use (default: 'bilinear')
            Options: 'bilinear', 'cubic', 'nearest', 'conservative'
        use_dask : bool, optional
            Whether to use Dask for computation. If None, automatically detected
            based on data type (default: None)
        chunk_size : int or tuple, optional
            Chunk size for Dask arrays. If None, automatic chunking is used
        **kwargs
            Additional keyword arguments for the regridding method
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The regridded data
        """
        from ..core import GridRegridder
        from ..dask import DaskRegridder, ChunkingStrategy
        
        # Validate inputs
        if not isinstance(target_grid, (xr.Dataset, xr.DataArray)):
            raise TypeError(f"target_grid must be xr.Dataset or xr.DataArray, got {type(target_grid)}")
        
        if not isinstance(method, str):
            raise TypeError(f"method must be str, got {type(method)}")
            
        # Check if the source object has appropriate dimensions
        self._validate_source_data()
        
        # Determine whether to use Dask
        if use_dask is None:
            use_dask = self.has_dask() or self._has_dask_arrays(target_grid)
        
        # Prepare chunking information
        chunking_info = {}
        if chunk_size is not None:
            chunking_info['chunk_size'] = chunk_size
        
        if use_dask:
            try:
                # Use DaskRegridder if Dask arrays are present or requested
                regridder = DaskRegridder(
                    source_grid=self._obj,
                    target_grid=target_grid,
                    method=method,
                    **chunking_info,
                    **kwargs
                )
                
                # Apply chunking strategy if needed
                if chunk_size is None and self.has_dask():
                    chunking_strategy = ChunkingStrategy()
                    optimal_chunk_size = chunking_strategy.determine_chunk_size(
                        self._obj, target_grid
                    )
                    if optimal_chunk_size is not None:
                        regridder.chunk_size = optimal_chunk_size
                
                return regridder.regrid(self._obj)
            except ImportError:
                # If Dask is not available, fall back to regular GridRegridder
                warnings.warn(
                    "Dask not available, falling back to regular regridding. "
                    "For better performance with large datasets, install Dask."
                )
                regridder = GridRegridder(
                    source_grid=self._obj,
                    target_grid=target_grid,
                    method=method,
                    **kwargs
                )
                return regridder.regrid(self._obj)
        else:
            # Use regular GridRegridder for numpy arrays
            regridder = GridRegridder(
                source_grid=self._obj,
                target_grid=target_grid,
                method=method,
                **kwargs
            )
            return regridder.regrid(self._obj)

    def interpolate_to(
        self,
        target_points,
        method: str = "bilinear",
        use_dask: Optional[bool] = None,
        chunk_size: Optional[Union[int, Tuple[int, ...]]] = None,
        **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Interpolate the current dataset/dataarray to the target points.
        
        Parameters
        ----------
        target_points : pandas.DataFrame or xarray.Dataset or dict
            The target points to interpolate to. For DataFrame, should contain
            coordinate columns (e.g., 'longitude', 'latitude' or 'x', 'y').
        method : str, optional
            The interpolation method to use (default: 'bilinear')
            Options: 'bilinear', 'cubic', 'nearest', 'idw', 'linear'
        use_dask : bool, optional
            Whether to use Dask for computation. If None, automatically detected
            based on data type (default: None)
        chunk_size : int or tuple, optional
            Chunk size for Dask arrays. If None, automatic chunking is used
        **kwargs
            Additional keyword arguments for the interpolation method
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The interpolated data
        """
        from ..core import PointInterpolator
        from ..dask import ChunkingStrategy
        
        # Validate inputs
        if not isinstance(method, str):
            raise TypeError(f"method must be str, got {type(method)}")
        
        # Check if the source object has appropriate dimensions
        self._validate_source_data()
        
        # Validate target_points format
        if not isinstance(target_points, (pd.DataFrame, xr.Dataset, dict)):
            raise TypeError(
                f"target_points must be pandas.DataFrame, xarray.Dataset, or dict, "
                f"got {type(target_points)}"
            )
        
        # Determine whether to use Dask
        if use_dask is None:
            use_dask = self.has_dask()
        
        # Prepare chunking information
        chunking_info = {}
        if chunk_size is not None:
            chunking_info['chunk_size'] = chunk_size
        
        if use_dask:
            try:
                # Use Dask-enabled PointInterpolator if Dask arrays are present or requested
                interpolator = PointInterpolator(
                    source_data=self._obj,
                    target_points=target_points,
                    method=method,
                    **chunking_info,
                    **kwargs
                )
                
                # Apply chunking strategy if needed
                if chunk_size is None and self.has_dask():
                    chunking_strategy = ChunkingStrategy()
                    # For point interpolation, use a default chunk size strategy
                    optimal_chunk_size = chunking_strategy.determine_chunk_size(
                        self._obj, self._obj  # Use source grid as reference
                    )
                    if optimal_chunk_size is not None:
                        # Pass chunk size through kwargs to PointInterpolator
                        kwargs['chunk_size'] = optimal_chunk_size
                
                return interpolator.interpolate()
            except ImportError:
                # If Dask is not available, fall back to regular PointInterpolator
                warnings.warn(
                    "Dask not available, falling back to regular interpolation. "
                    "For better performance with large datasets, install Dask."
                )
                interpolator = PointInterpolator(
                    source_data=self._obj,
                    target_points=target_points,
                    method=method,
                    **kwargs
                )
                return interpolator.interpolate()
        else:
            # Use regular PointInterpolator for numpy arrays
            interpolator = PointInterpolator(
                source_data=self._obj,
                target_points=target_points,
                method=method,
                **kwargs
            )
            return interpolator.interpolate()

    def _validate_source_data(self):
        """
        Validate that the source xarray object has appropriate dimensions and coordinates
        for regridding or interpolation operations.
        """
        if not isinstance(self._obj, (xr.Dataset, xr.DataArray)):
            raise TypeError(
                f"Source object must be xr.Dataset or xr.DataArray, got {type(self._obj)}"
            )
        
        # Check for coordinate variables
        if isinstance(self._obj, xr.DataArray):
            coords = self._obj.coords
        else:  # xr.Dataset
            coords = self._obj.coords
        
        # Look for latitude and longitude coordinates
        lat_coords = [str(name) for name in coords if
                      any(lat_name in str(name).lower() for lat_name in ['lat', 'latitude', 'y'])]
        lon_coords = [str(name) for name in coords if
                      any(lon_name in str(name).lower() for lon_name in ['lon', 'longitude', 'x'])]
        
        if not lat_coords or not lon_coords:
            warnings.warn(
                "Could not automatically detect latitude/longitude coordinates. "
                "Make sure your data has appropriate coordinate variables."
            )
    
    def get_coordinates(self) -> Dict[str, Any]:
        """
        Extract coordinate information from the xarray object.
        
        Returns
        -------
        dict
            A dictionary containing coordinate information including names and values
        """
        if isinstance(self._obj, xr.DataArray):
            coords = self._obj.coords
        else:  # xr.Dataset
            coords = self._obj.coords
        
        coordinate_info = {}
        
        # Identify latitude and longitude coordinates
        lat_coords = [str(name) for name in coords if
                      any(lat_name in str(name).lower() for lat_name in ['lat', 'latitude', 'y'])]
        lon_coords = [str(name) for name in coords if
                      any(lon_name in str(name).lower() for lon_name in ['lon', 'longitude', 'x'])]
        
        if lat_coords:
            coordinate_info['latitude_coord'] = lat_coords[0]
            coordinate_info['latitude_values'] = coords[lat_coords[0]].values
        if lon_coords:
            coordinate_info['longitude_coord'] = lon_coords[0]
            coordinate_info['longitude_values'] = coords[lon_coords[0]].values
        
        # Add coordinate reference system if available
        if hasattr(self._obj, 'attrs') and 'crs' in self._obj.attrs:
            coordinate_info['crs'] = self._obj.attrs['crs']
        elif hasattr(self._obj, 'rio') and hasattr(self._obj.rio, 'crs'):
            # If using rioxarray, try to get CRS from there
            coordinate_info['crs'] = self._obj.rio.crs
        
        return coordinate_info
    
    def has_dask(self) -> bool:
        """
        Check if the xarray object contains Dask arrays.
        
        Returns
        -------
        bool
            True if any data variables use Dask arrays, False otherwise
        """
        return self._has_dask_arrays(self._obj)
    
    def _has_dask_arrays(self, obj) -> bool:
        """
        Check if the xarray object contains Dask arrays.
        
        Parameters
        ----------
        obj : xr.Dataset or xr.DataArray
            The xarray object to check
        
        Returns
        -------
        bool
            True if any data variables use Dask arrays, False otherwise
        """
        if isinstance(obj, xr.DataArray):
            return hasattr(obj.data, 'chunks')
        elif isinstance(obj, xr.Dataset):
            for var_name, var_data in obj.data_vars.items():
                if hasattr(var_data.data, 'chunks'):
                    return True
        return False