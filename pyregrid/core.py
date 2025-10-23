"""
Core PyRegrid classes.

This module contains the main regridding and interpolation classes:
- GridRegridder: For grid-to-grid operations
- PointInterpolator: For scattered data interpolation
"""

import xarray as xr
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from typing import Union, Optional, Dict, Any, Tuple
import warnings
import pyproj
from pyproj import CRS, Transformer

from pyregrid.crs.crs_manager import CRSManager


class GridRegridder:
    """
    Grid-to-grid regridding engine.
    
    This class implements the prepare-execute pattern for efficient regridding,
    where interpolation weights are computed once and can be reused.
    """
    
    def __init__(
        self,
        source_grid: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear",
        source_crs: Optional[Union[str, CRS]] = None,
        target_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ):
        """
        Initialize the GridRegridder.
        
        Parameters
        ----------
        source_grid : xr.Dataset or xr.DataArray
            The source grid to regrid from
        target_grid : xr.Dataset or xr.DataArray
            The target grid to regrid to
        method : str, optional
            The regridding method to use (default: 'bilinear')
            Options: 'bilinear', 'cubic', 'nearest'
        source_crs : str, CRS, optional
            The coordinate reference system of the source grid
        target_crs : str, CRS, optional
            The coordinate reference system of the target grid
        **kwargs
            Additional keyword arguments for the regridding method
        """
        self.source_grid = source_grid
        self.target_grid = target_grid
        self.method = method
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.kwargs = kwargs
        self.weights = None
        self.transformer = None
        self._source_coords = None
        self._target_coords = None
        
        # Initialize CRS manager for coordinate system handling
        self.crs_manager = CRSManager()
        
        # Validate method
        valid_methods = ['bilinear', 'cubic', 'nearest', 'conservative']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Extract coordinate information
        self._extract_coordinates()
        
        # Determine CRS if not provided explicitly using the "strict but helpful" policy
        # Track whether CRS was explicitly provided vs auto-detected
        source_crs_explicitly_provided = self.source_crs is not None
        target_crs_explicitly_provided = self.target_crs is not None
        
        if self.source_crs is None:
            self.source_crs = self.crs_manager.get_crs_from_source(
                self.source_grid,
                self._source_lon,
                self._source_lat,
                self._source_lon_name,
                self._source_lat_name
            )
        
        if self.target_crs is None:
            self.target_crs = self.crs_manager.get_crs_from_source(
                self.target_grid,
                self._target_lon,
                self._target_lat,
                self._target_lon_name,
                self._target_lat_name
            )
        
        # Initialize CRS transformation if needed
        # Only create transformer if both source and target CRS are explicitly provided (not auto-detected)
        if (source_crs_explicitly_provided and target_crs_explicitly_provided):
            # Convert string CRS to CRS objects if needed
            if isinstance(self.source_crs, str):
                self.source_crs = CRS.from_string(self.source_crs)
            if isinstance(self.target_crs, str):
                self.target_crs = CRS.from_string(self.target_crs)
            
            if isinstance(self.source_crs, CRS) and isinstance(self.target_crs, CRS):
                if self.source_crs != self.target_crs:
                    self._setup_crs_transformation()
                else:
                    # Create a no-op transformer for same CRS
                    self.transformer = Transformer.from_crs(self.source_crs, self.target_crs, always_xy=True)
            else:
                self.transformer = None  # No transformation needed for invalid CRS objects
        else:
            self.transformer = None
        
        # Prepare the regridding weights (following the two-phase model)
        # Weights will be computed and stored for reuse
        self.prepare()
    
    def _setup_crs_transformation(self):
        """Setup coordinate reference system transformation."""
        if self.source_crs is None or self.target_crs is None:
            raise ValueError("Both source_crs and target_crs must be provided for CRS transformation")
        
        # Create transformer for coordinate transformation
        self.transformer = Transformer.from_crs(
            self.source_crs, self.target_crs, always_xy=True
        )
    
    def _extract_coordinates(self):
        """Extract coordinate information from source and target grids."""
        # Determine coordinate names for source grid
        if isinstance(self.source_grid, xr.DataArray):
            source_coords = self.source_grid.coords
            # Find longitude/latitude coordinates
            lon_names = [str(name) for name in source_coords if 'lon' in str(name).lower() or 'x' in str(name).lower()]
            lat_names = [str(name) for name in source_coords if 'lat' in str(name).lower() or 'y' in str(name).lower()]
        else:  # Dataset
            try:
                source_coords = self.source_grid.coords
                lon_names = [str(name) for name in source_coords if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                lat_names = [str(name) for name in source_coords if 'lat' in str(name).lower() or 'y' in str(name).lower()]
            except (AttributeError, TypeError):
                # If the source grid doesn't have proper coordinates, raise an error
                raise ValueError(
                    f"Source grid does not have valid coordinate information. "
                    f"Please ensure your grid is a proper xarray DataArray or Dataset."
                )
        
        # Use common coordinate names if not found
        if not lon_names:
            lon_names = ['lon'] if 'lon' in [str(name) for name in source_coords] else ['x']
        if not lat_names:
            lat_names = ['lat'] if 'lat' in [str(name) for name in source_coords] else ['y']
        
        # If still no coordinates found, use the first coordinate names
        if not lon_names:
            lon_names = [list(source_coords.keys())[0]]
        if not lat_names:
            lat_names = [list(source_coords.keys())[1]] if len(source_coords) > 1 else [list(source_coords.keys())[0]]
        
        # Validate that coordinate names exist in the grid
        valid_lon_names = []
        valid_lat_names = []
        
        for name in lon_names:
            if str(name) in [str(coord) for coord in source_coords]:
                valid_lon_names.append(str(name))
                
        for name in lat_names:
            if str(name) in [str(coord) for coord in source_coords]:
                valid_lat_names.append(str(name))
        
        # If no valid coordinate names found, raise an error
        if not valid_lon_names or not valid_lat_names:
            available_coords = list(source_coords.keys())
            raise ValueError(
                f"Could not identify valid longitude and latitude coordinates in the source grid. "
                f"Available coordinates: {available_coords}. "
                f"Please ensure your grid has properly named coordinate variables (e.g., 'lon', 'lat', 'x', 'y')."
            )
        
        self._source_lon_name = valid_lon_names[0]
        self._source_lat_name = valid_lat_names[0]
        
        self._source_lon_name = lon_names[0]
        self._source_lat_name = lat_names[0]
        
        # Similarly for target grid
        if isinstance(self.target_grid, xr.DataArray):
            target_coords = self.target_grid.coords
            lon_names = [str(name) for name in target_coords if 'lon' in str(name).lower() or 'x' in str(name).lower()]
            lat_names = [str(name) for name in target_coords if 'lat' in str(name).lower() or 'y' in str(name).lower()]
        else:  # Dataset
            target_coords = self.target_grid.coords
            lon_names = [str(name) for name in target_coords if 'lon' in str(name).lower() or 'x' in str(name).lower()]
            lat_names = [str(name) for name in target_coords if 'lat' in str(name).lower() or 'y' in str(name).lower()]
        
        if not lon_names:
            lon_names = ['lon'] if 'lon' in [str(name) for name in target_coords] else ['x']
        if not lat_names:
            lat_names = ['lat'] if 'lat' in [str(name) for name in target_coords] else ['y']
        
        # If still no coordinates found, use the first coordinate names
        if not lon_names:
            lon_names = [list(target_coords.keys())[0]]
        if not lat_names:
            lat_names = [list(target_coords.keys())[1]] if len(target_coords) > 1 else [list(target_coords.keys())[0]]
        
        # Validate that coordinate names exist in the grid
        valid_lon_names = []
        valid_lat_names = []
        
        for name in lon_names:
            if str(name) in [str(coord) for coord in target_coords]:
                valid_lon_names.append(str(name))
                
        for name in lat_names:
            if str(name) in [str(coord) for coord in target_coords]:
                valid_lat_names.append(str(name))
        
        # If no valid coordinate names found, raise an error
        if not valid_lon_names or not valid_lat_names:
            available_coords = list(target_coords.keys())
            raise ValueError(
                f"Could not identify valid longitude and latitude coordinates in the target grid. "
                f"Available coordinates: {available_coords}. "
                f"Please ensure your grid has properly named coordinate variables (e.g., 'lon', 'lat', 'x', 'y')."
            )
        
        self._target_lon_name = valid_lon_names[0]
        self._target_lat_name = valid_lat_names[0]
        
        # Store coordinate arrays
        self._source_lon = self.source_grid[self._source_lon_name].values
        self._source_lat = self.source_grid[self._source_lat_name].values
        self._target_lon = self.target_grid[self._target_lon_name].values
        self._target_lat = self.target_grid[self._target_lat_name].values
    
    def prepare(self):
        """
        Prepare the regridding by calculating interpolation weights.
        
        This method computes the interpolation weights based on the source and target grids
        and the specified method. The weights can be reused for multiple regridding operations.
        """
        # Determine interpolation order based on method
        if self.method == 'bilinear':
            order = 1
        elif self.method == 'cubic':
            order = 3
        elif self.method == 'nearest':
            order = 0
        elif self.method == 'conservative':
            # For conservative method, we'll use a different approach
            # Store the source and target coordinates for conservative interpolation
            self.weights = {
                'source_lon': self._source_lon,
                'source_lat': self._source_lat,
                'target_lon': self._target_lon,
                'target_lat': self._target_lat,
                'method': self.method
            }
            return # Return early as conservative interpolation handles weights differently
        else:
            raise ValueError(f"Unsupported method: {self.method}")
        
        # Prepare coordinate transformation if needed
        if self.transformer:
            # Transform target coordinates to source CRS
            target_lon_flat = self._target_lon.flatten()
            target_lat_flat = self._target_lat.flatten()
            try:
                source_target_lon, source_target_lat = self.transformer.transform(
                    target_lon_flat, target_lat_flat, direction='INVERSE'
                )
                # Reshape back to original grid shape
                source_target_lon = source_target_lon.reshape(self._target_lon.shape)
                source_target_lat = source_target_lat.reshape(self._target_lat.shape)
            except Exception as e:
                # If transformation fails, use original coordinates
                warnings.warn(f"Coordinate transformation failed: {e}. Using original coordinates.")
                source_target_lon = self._target_lon
                source_target_lat = self._target_lat
        else:
            source_target_lon = self._target_lon
            source_target_lat = self._target_lat
        
        # Calculate normalized coordinates for map_coordinates
        # Find the index coordinates in the source grid
        # For longitude (x-axis) and latitude (y-axis), we need to create 2D index grids
        # that match the target grid shape, not just 1D coordinate arrays
        
        # For identity regridding (source and target grids are the same),
        # we need to handle the coordinate mapping differently
        if np.array_equal(self._source_lon, self._target_lon) and np.array_equal(self._source_lat, self._target_lat):
            # For identity regridding, we should map each point to itself
            # Create identity mapping indices
            if self._source_lon.ndim == 2 and self._source_lat.ndim == 2:
                # For curvilinear grids, create identity mapping
                # The indices should map each point in the target grid to the same position in the source grid
                target_shape = self._target_lon.shape
                lat_indices, lon_indices = np.meshgrid(
                    np.arange(target_shape[0]),
                    np.arange(target_shape[1]),
                    indexing='ij'
                )
            else:
                # For rectilinear grids, create 2D coordinate grids that match the target grid shape
                # Create meshgrids with the correct shape for identity mapping
                target_lat_idx = np.arange(len(self._target_lat))
                target_lon_idx = np.arange(len(self._target_lon))
                lat_indices, lon_indices = np.meshgrid(target_lat_idx, target_lon_idx, indexing='ij')
        else:
            # Create 2D meshgrids for target coordinates
            target_lon_2d, target_lat_2d = np.meshgrid(
                self._target_lon, self._target_lat, indexing='xy'
            )
            
            # Prepare coordinate transformation if needed
            if self.transformer:
                # Transform target coordinates to source CRS
                try:
                    source_target_lon_2d, source_target_lat_2d = self.transformer.transform(
                        target_lon_2d, target_lat_2d, direction='INVERSE'
                    )
                except Exception as e:
                    # If transformation fails, use original coordinates
                    warnings.warn(f"Coordinate transformation failed: {e}. Using original coordinates.")
                    source_target_lon_2d = target_lon_2d
                    source_target_lat_2d = target_lat_2d
            else:
                source_target_lon_2d = target_lon_2d
                source_target_lat_2d = target_lat_2d
            
            # For longitude (x-axis)
            # Check if coordinates are in ascending or descending order
            # Handle both 1D (rectilinear) and 2D (curvilinear) coordinate arrays
            if self._source_lon.ndim == 1:
                # 1D coordinates (rectilinear grid)
                if len(self._source_lon) > 1 and self._source_lon[0] > self._source_lon[-1]:
                    # Coordinates are in descending order, need to reverse the index mapping
                    lon_indices = len(self._source_lon) - 1 - np.interp(
                        source_target_lon_2d,
                        self._source_lon[::-1],  # Reverse the coordinate array
                        np.arange(len(self._source_lon))  # Normal index array
                    )
                else:
                    # Coordinates are in ascending order (normal case)
                    lon_indices = np.interp(
                        source_target_lon_2d,
                        self._source_lon,
                        np.arange(len(self._source_lon))
                    )
            else:
                # 2D coordinates (curvilinear grid) - need special handling
                # For curvilinear grids, we need to map each target point to the nearest source point
                # This is more complex than simple interpolation
                
                # Create coordinate grids for the source
                source_lon_grid, source_lat_grid = np.meshgrid(
                    np.arange(self._source_lon.shape[1]),  # longitude indices
                    np.arange(self._source_lon.shape[0]),  # latitude indices
                    indexing='xy'
                )
                
                # Flatten the source coordinates and create points
                source_points = np.column_stack([
                    source_lat_grid.flatten(),
                    source_lon_grid.flatten()
                ])
                
                # Flatten the target coordinates
                target_points = np.column_stack([
                    source_target_lat_2d.flatten(),
                    source_target_lon_2d.flatten()
                ])
                
                # Use KDTree for nearest neighbor search
                from scipy.spatial import cKDTree
                tree = cKDTree(source_points)
                
                # Find nearest neighbors
                distances, indices = tree.query(target_points)
                
                # Reshape indices back to target grid shape
                lon_indices = indices.reshape(source_target_lon_2d.shape)
            
            # For latitude (y-axis) - for curvilinear grids, we use the same indices as longitude
            # since we're doing nearest neighbor mapping
            if self._source_lat.ndim == 1:
                # 1D coordinates (rectilinear grid)
                if len(self._source_lat) > 1 and self._source_lat[0] > self._source_lat[-1]:
                    # Coordinates are in descending order, need to reverse the index mapping
                    lat_indices = len(self._source_lat) - 1 - np.interp(
                        source_target_lat_2d,
                        self._source_lat[::-1], # Reverse the coordinate array
                        np.arange(len(self._source_lat))  # Normal index array
                    )
                else:
                    # Coordinates are in ascending order (normal case)
                    lat_indices = np.interp(
                        source_target_lat_2d,
                        self._source_lat,
                        np.arange(len(self._source_lat))
                    )
            else:
                # For curvilinear grids, lat_indices should be the same as lon_indices
                # because we're mapping each target point to a specific source point
                lat_indices = lon_indices
        
        # Store the coordinate mapping
        self.weights = {
            'lon_indices': lon_indices,
            'lat_indices': lat_indices,
            'order': order,
            'method': self.method
        }
    
    def regrid(self, data: Union[xr.Dataset, xr.DataArray]) -> Union[xr.Dataset, xr.DataArray]:
        """
        Apply the regridding to the input data using precomputed weights.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to regrid, must be compatible with the source grid
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The regridded data on the target grid
        """
        if self.weights is None:
            raise RuntimeError("Weights not prepared. Call prepare() first.")
        
        # Check if data is compatible with source grid
        if isinstance(data, xr.DataArray):
            return self._regrid_dataarray(data)
        elif isinstance(data, xr.Dataset):
            return self._regrid_dataset(data)
        else:
            raise TypeError(f"Input data must be xr.DataArray or xr.Dataset, got {type(data)}")
    
    def _regrid_dataarray(self, data: xr.DataArray) -> xr.DataArray:
        """Regrid a DataArray using precomputed weights."""
        # Check if the data has the expected dimensions
        if self._source_lon_name not in data.dims or self._source_lat_name not in data.dims:
            raise ValueError(f"Data must have dimensions '{self._source_lon_name}' and '{self._source_lat_name}'")
        
        # Check that weights have been prepared
        if self.weights is None:
            raise RuntimeError("Weights not prepared. Call prepare() first.")
        
        # Handle conservative method separately
        if self.method == 'conservative':
            # Import ConservativeInterpolator
            from pyregrid.algorithms.interpolators import ConservativeInterpolator
            
            # Create the conservative interpolator
            interpolator = ConservativeInterpolator(
                source_lon=self._source_lon,
                source_lat=self._source_lat,
                target_lon=self._target_lon,
                target_lat=self._target_lat
            )
            
            # Perform conservative interpolation
            result_data = interpolator.interpolate(
                data.values,
                source_lon=self._source_lon,
                source_lat=self._source_lat,
                target_lon=self._target_lon,
                target_lat=self._target_lat
            )
            
            # Create output coordinates
            output_coords = {}
            for coord_name in data.coords:
                if coord_name == self._source_lon_name:
                    output_coords[self._target_lon_name] = self._target_lon
                elif coord_name == self._source_lat_name:
                    output_coords[self._target_lat_name] = self._target_lat
                elif coord_name in [self._source_lon_name, self._source_lat_name]:
                    # Skip the original coordinate axes, they'll be replaced
                    continue
                else:
                    # Keep other coordinates as they are
                    output_coords[coord_name] = data.coords[coord_name]
            
            # Create the output DataArray
            output_dims = list(data.dims)
            output_dims[output_dims.index(self._source_lon_name)] = self._target_lon_name
            output_dims[output_dims.index(self._source_lat_name)] = self._target_lat_name
            
            result = xr.DataArray(
                result_data,
                dims=output_dims,
                coords=output_coords,
                attrs=data.attrs
            )
            
            return result
        else:
            # Prepare coordinate indices for map_coordinates (for non-conservative methods)
            lon_indices = self.weights['lon_indices']
            lat_indices = self.weights['lat_indices']
            order = self.weights['order']
            
            # Determine which axes correspond to longitude and latitude in the data
            lon_axis = data.dims.index(self._source_lon_name)
            lat_axis = data.dims.index(self._source_lat_name)
            
            # For map_coordinates, we need to handle the coordinate transformation properly
            # We'll use a more direct approach by creating a function that handles the regridding
            
            # Determine output shape first
            output_shape = list(data.shape)
            # For curvilinear grids, target coordinates are 2D arrays
            # The output should match the shape of the target coordinate arrays
            if self._target_lon.ndim == 2 and self._target_lat.ndim == 2:
                # For curvilinear grids, both coordinate arrays should have the same shape
                output_shape[lon_axis] = self._target_lon.shape[1]  # longitude dimension size
                output_shape[lat_axis] = self._target_lon.shape[0]  # latitude dimension size
            else:
                # For rectilinear grids, coordinates are 1D
                output_shape[lon_axis] = len(self._target_lon)
                output_shape[lat_axis] = len(self._target_lat)
            
            # Create output coordinates
            output_coords = {}
            for coord_name in data.coords:
                if coord_name == self._source_lon_name:
                    # For curvilinear grids, preserve the 2D coordinate structure
                    if self._target_lon.ndim == 2:
                        # For 2D coordinates, create a Variable with proper dimensions and attributes
                        # Use the target coordinate dimensions instead of data.dims to avoid conflicts
                        from xarray.core.variable import Variable
                        # For curvilinear grids, the coordinate should have the same dimensions as the target coordinate
                        # For curvilinear grids, the coordinate should have the same dimensions as the data array
                        # We need to use the actual dimensions of the data array to avoid conflicts
                        coord_var = Variable(data.dims, self._target_lon)
                        # Preserve original attributes if they exist in the source grid
                        if hasattr(self.source_grid, 'coords') and self._source_lon_name in self.source_grid.coords:
                            coord_var.attrs.update(self.source_grid.coords[self._source_lon_name].attrs)
                        output_coords[self._target_lon_name] = coord_var
                    else:
                        output_coords[self._target_lon_name] = self._target_lon
                elif coord_name == self._source_lat_name:
                    # For curvilinear grids, preserve the 2D coordinate structure
                    if self._target_lat.ndim == 2:
                        # For 2D coordinates, create a Variable with proper dimensions and attributes
                        # Use the target coordinate dimensions instead of data.dims to avoid conflicts
                        from xarray.core.variable import Variable
                        # For curvilinear grids, the coordinate should have the same dimensions as the data array
                        coord_var = Variable(data.dims, self._target_lat)
                        # Preserve original attributes if they exist in the source grid
                        if hasattr(self.source_grid, 'coords') and self._source_lat_name in self.source_grid.coords:
                            coord_var.attrs.update(self.source_grid.coords[self._source_lat_name].attrs)
                        output_coords[self._target_lat_name] = coord_var
                    else:
                        output_coords[self._target_lat_name] = self._target_lat
                elif coord_name in [self._source_lon_name, self._source_lat_name]:
                    # Skip the original coordinate axes, they'll be replaced
                    continue
                else:
                    # Keep other coordinates as they are
                    output_coords[coord_name] = data.coords[coord_name]
            
            # Check if data contains Dask arrays
            is_dask = hasattr(data.data, 'chunks') and data.data.__class__.__module__.startswith('dask')
            
            if is_dask:
                # For Dask arrays, we need to use dask-compatible operations
                try:
                    import dask.array as da
                    
                    # Use the _interpolate_along_axes method which now handles Dask arrays
                    result_data = self._interpolate_along_axes(
                        data.values,
                        (lon_axis, lat_axis),
                        (lon_indices, lat_indices),
                        order
                    )
                except ImportError:
                    # If Dask is not available, fall back to numpy computation
                    # Since is_dask is True, data.values should be a dask array
                    # Compute the dask array and use numpy approach
                    computed_data = data.values.compute()
                    if lon_axis == len(data.dims) - 1 and lat_axis == len(data.dims) - 2:
                        result_data = map_coordinates(
                            computed_data,
                            [lat_indices, lon_indices],  # [lat_idx, lon_idx] for each output point
                            order=order,
                            mode='nearest',  # Use 'nearest' for out-of-bounds values
                            cval=np.nan
                        )
                    else:
                        # More complex case - need to handle arbitrary axis positions
                        result_data = self._interpolate_along_axes(
                            computed_data,
                            (lon_axis, lat_axis),
                            (lon_indices, lat_indices),
                            order
                        )
            else:
                # For numpy arrays, use the original approach
                # But check if we have curvilinear grids (2D coordinate arrays)
                if (self._source_lon.ndim == 2 or self._source_lat.ndim == 2):
                    # For curvilinear grids, we need special handling
                    # Use direct indexing with the precomputed indices
                    result_data = self._interpolate_curvilinear(
                        data.values,
                        (lon_axis, lat_axis),
                        (lon_indices, lat_indices),
                        order
                    )
                else:
                    # Use the _interpolate_along_axes method which handles multi-dimensional data properly
                    result_data = self._interpolate_along_axes(
                        data.values,
                        (lon_axis, lat_axis),
                        (lon_indices, lat_indices),
                        order
                    )
            
            # Add error handling for potential issues with the result
            if result_data is None:
                raise RuntimeError(f"Interpolation failed for method {self.method} with order {order}")
            
            # Ensure the result has the expected shape
            expected_shape = list(data.shape)
            # For curvilinear grids, target coordinates are 2D arrays
            # The output should match the shape of the target coordinate arrays
            if self._target_lon.ndim == 2 and self._target_lat.ndim == 2:
                # For curvilinear grids, both coordinate arrays should have the same shape
                expected_shape[lon_axis] = self._target_lon.shape[1]  # longitude dimension size
                expected_shape[lat_axis] = self._target_lon.shape[0]  # latitude dimension size
            else:
                # For rectilinear grids, coordinates are 1D
                expected_shape[lon_axis] = len(self._target_lon)
                expected_shape[lat_axis] = len(self._target_lat)
    
            if result_data.shape != tuple(expected_shape):
                raise ValueError(
                    f"Result shape {result_data.shape} does not match expected shape {tuple(expected_shape)}"
                )
            
            # Create the output DataArray
            output_dims = list(data.dims)
            output_dims[lon_axis] = self._target_lon_name
            output_dims[lat_axis] = self._target_lat_name
            
            result = xr.DataArray(
                result_data,
                dims=output_dims,
                coords=output_coords,
                attrs=data.attrs,
                name=data.name  # Preserve the original data variable name
            )
            
            return result
    
    def _interpolate_along_axes(self, data: np.ndarray, axes: Tuple[int, int], coordinate_grids: Tuple[np.ndarray, np.ndarray], order: int) -> np.ndarray:
       """
       Interpolate data along specific axes using coordinate grids.
       
       Parameters
       ----------
       data : np.ndarray
           Input data array to interpolate
       axes : Tuple[int, int]
           Tuple of axis indices (lon_axis, lat_axis) to interpolate along
       coordinate_grids : Tuple[np.ndarray, np.ndarray]
           Tuple of coordinate grids (lon_indices, lat_indices) for interpolation
       order : int
           Interpolation order (0 for nearest, 1 for bilinear, etc.)
           
       Returns
       -------
       np.ndarray
           Interpolated data array with updated spatial dimensions
       """
       # Get the source coordinate values
       lon_indices, lat_indices = coordinate_grids
       lon_axis, lat_axis = axes
       
       # Since lon_indices and lat_indices are now 2D arrays with the target grid shape,
       # we can use them directly as coordinate mappings for map_coordinates
       # The coordinates for map_coordinates should be [lat_idx, lon_idx] for each output point
       coordinates = [lat_indices, lon_indices]
       
       # Prepare the output shape
       output_shape = list(data.shape)
       # Handle both 1D and 2D coordinate index arrays for identity regridding
       if lon_indices.ndim == 1:
           # For 1D coordinate indices (rectilinear grids in identity regridding)
           output_shape[lon_axis] = len(lon_indices)
           output_shape[lat_axis] = len(lat_indices)
       else:
           # For 2D coordinate indices (normal regridding or curvilinear grids)
           output_shape[lon_axis] = lon_indices.shape[1] # Target longitude size
           output_shape[lat_axis] = lon_indices.shape[0] # Target latitude size
       
       # For regular numpy arrays, use the original approach
       # Transpose the data so that the spatial axes are at the end
       non_spatial_axes = [i for i in range(len(data.shape)) if i not in axes]
       transposed_axes = non_spatial_axes + [lat_axis, lon_axis]
       transposed_data = np.transpose(data, transposed_axes)
       
       # Get the shape of non-spatial dimensions
       non_spatial_shape = transposed_data.shape[:len(non_spatial_axes)]
       # Handle both 1D and 2D coordinate index arrays for identity regridding
       if lon_indices.ndim == 1:
           # For 1D coordinate indices (rectilinear grids in identity regridding)
           result_shape = non_spatial_shape + (len(lat_indices), len(lon_indices))
       else:
           # For 2D coordinate indices (normal regridding or curvilinear grids)
           result_shape = non_spatial_shape + (lon_indices.shape[0], lon_indices.shape[1])
       result = np.full(result_shape, np.nan, dtype=data.dtype)
       
       # Process each slice along non-spatial dimensions
       for idx in np.ndindex(non_spatial_shape):
           slice_2d = transposed_data[idx]
           # Use map_coordinates with the precomputed coordinate arrays
           # For each point in the output grid, we specify which input indices to use
           # Handle both 1D and 2D coordinate index arrays for identity regridding
           if lon_indices.ndim == 1:
               # For 1D coordinate indices (rectilinear grids in identity regridding)
               # Need to create 2D coordinate grids for each slice
               lat_idx_grid, lon_idx_grid = np.meshgrid(
                   lat_indices.astype(float),
                   lon_indices.astype(float),
                   indexing='ij'
               )
               interpolated_slice = map_coordinates(
                   slice_2d,
                   [lat_idx_grid, lon_idx_grid],
                   order=order,
                   mode='nearest',
                   cval=np.nan
               )
           else:
               # For 2D coordinate indices (normal regridding or curvilinear grids)
               interpolated_slice = map_coordinates(
                   slice_2d,
                   coordinates,
                   order=order,
                   mode='nearest',
                   cval=np.nan
               )
           result[idx] = interpolated_slice
       
       # Transpose back to original axis order but with new spatial dimensions
       final_axes = []
       ax_idx = 0
       for i in range(len(output_shape)):
           if i == lat_axis:
               final_axes.append(len(non_spatial_shape))
           elif i == lon_axis:
               final_axes.append(len(non_spatial_shape) + 1)
           else:
               final_axes.append(ax_idx)
               ax_idx += 1
       
       output = np.transpose(result, final_axes)
       
       # Check if data is a Dask array for out-of-core processing
       is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
       
       if is_dask:
           # For Dask arrays, we need to use dask-compatible operations
           try:
               import dask.array as da
               
               # Create a function to apply the interpolation
               def apply_interp(block, block_info=None):
                   # Apply the same interpolation logic to each block
                   # This is a simplified version - a full implementation would be more complex
                   # For now, we'll just use the numpy approach on each block
                   # Transpose the block so that the spatial axes are at the end
                   block_transposed_axes = non_spatial_axes + [lat_axis, lon_axis]
                   block_transposed = np.transpose(block, block_transposed_axes)
                   
                   # Get the shape of non-spatial dimensions for this block
                   block_non_spatial_shape = block_transposed.shape[:len(non_spatial_axes)]
                   # Handle both 1D and 2D coordinate index arrays
                   if lon_indices.ndim == 1:
                       # For 1D coordinate indices (rectilinear grids in identity regridding)
                       block_result_shape = block_non_spatial_shape + (len(lat_indices), len(lon_indices))
                   else:
                       # For 2D coordinate indices (normal regridding or curvilinear grids)
                       block_result_shape = block_non_spatial_shape + (lon_indices.shape[0], lon_indices.shape[1])
                   block_result = np.full(block_result_shape, np.nan, dtype=block.dtype)
                   
                   # Process each slice along non-spatial dimensions
                   for idx in np.ndindex(block_non_spatial_shape):
                       slice_2d = block_transposed[idx]
                       # Use map_coordinates with the precomputed coordinate arrays
                       interpolated_slice = map_coordinates(
                           slice_2d,
                           coordinates,  # Use the same coordinates for all blocks
                           order=order,
                           mode='nearest',
                           cval=np.nan
                       )
                       block_result[idx] = interpolated_slice
                   
                   # Transpose back to original axis order but with new spatial dimensions
                   block_final_axes = []
                   ax_idx = 0
                   for j in range(len(output_shape)):
                       if j == lat_axis:
                           block_final_axes.append(len(block_non_spatial_shape))
                       elif j == lon_axis:
                           block_final_axes.append(len(block_non_spatial_shape) + 1)
                       else:
                           block_final_axes.append(ax_idx)
                           ax_idx += 1
                   
                   return np.transpose(block_result, block_final_axes)
               
               # Use map_blocks for Dask arrays
               output = da.map_blocks(
                   apply_interp,
                   data,
                   dtype=data.dtype,
                   drop_axis=[lat_axis, lon_axis],  # Remove the old spatial axes
                   new_axis=list(range(len(non_spatial_shape), len(non_spatial_shape) + 2)),  # Add new spatial axes
                   chunks=output_shape
               )
               return output
           except ImportError:
               # If Dask is not available, use the numpy implementation
               pass
       
       # Return the result after proper transposition
       return output
   
    def _interpolate_curvilinear(self, data: np.ndarray, axes: Tuple[int, int], coordinate_grids: Tuple[np.ndarray, np.ndarray], order: int) -> np.ndarray:
        """
        Interpolate data for curvilinear grids using direct indexing.
        
        Parameters
        ----------
        data : np.ndarray
            Input data array to interpolate
        axes : Tuple[int, int]
            Tuple of axis indices (lon_axis, lat_axis) to interpolate along
        coordinate_grids : Tuple[np.ndarray, np.ndarray]
            Tuple of coordinate grids (lon_indices, lat_indices) for interpolation
        order : int
            Interpolation order (0 for nearest, 1 for bilinear, etc.)
            
        Returns
        -------
        np.ndarray
            Interpolated data array with updated spatial dimensions
        """
        # Get the source coordinate values
        lon_indices, lat_indices = coordinate_grids
        lon_axis, lat_axis = axes
        
        # For curvilinear grids, we use direct indexing with the precomputed indices
        # The indices should already be in the correct format for direct indexing
        
        # For regular numpy arrays, use direct indexing
        # Transpose the data so that the spatial axes are at the end
        non_spatial_axes = [i for i in range(len(data.shape)) if i not in [lon_axis, lat_axis]]
        transposed_axes = non_spatial_axes + [lat_axis, lon_axis]
        transposed_data = np.transpose(data, transposed_axes)
        
        # Get the shape of non-spatial dimensions
        non_spatial_shape = transposed_data.shape[:len(non_spatial_axes)]
        
        # Determine output shape based on the coordinate indices
        # For identity regridding, the target grid shape should match the source grid shape
        # For regular regridding, it should match the target grid shape
        # Handle both 1D and 2D coordinate index arrays
        if lon_indices.ndim == 1:
            # For 1D coordinate indices (rectilinear grids in identity regridding)
            result_shape = non_spatial_shape + (len(lat_indices), len(lon_indices))
        else:
            # For 2D coordinate indices (normal regridding or curvilinear grids)
            result_shape = non_spatial_shape + lon_indices.shape
        result = np.full(result_shape, np.nan, dtype=data.dtype)
        
        # Process each slice along non-spatial dimensions
        for idx in np.ndindex(non_spatial_shape):
            slice_2d = transposed_data[idx]
            
            # For curvilinear grids, we need to use advanced indexing
            # The indices are already computed to map target to source points
            if order == 0:  # Nearest neighbor
                # Use direct indexing with the precomputed indices
                # Make sure indices are within bounds
                lat_idx = np.clip(lat_indices.astype(int), 0, slice_2d.shape[0] - 1)
                lon_idx = np.clip(lon_indices.astype(int), 0, slice_2d.shape[1] - 1)
                # Use advanced indexing to select the values
                interpolated_slice = slice_2d[lat_idx, lon_idx]
            else:
                # For higher order interpolation, we need to use a different approach
                # Since we have curvilinear grids, we'll use nearest neighbor for now
                # This could be extended to use bilinear or cubic interpolation
                # by interpolating between the nearest neighbors
                lat_idx = np.clip(lat_indices.astype(int), 0, slice_2d.shape[0] - 1)
                lon_idx = np.clip(lon_indices.astype(int), 0, slice_2d.shape[1] - 1)
                # Use advanced indexing to select the values
                interpolated_slice = slice_2d[lat_idx, lon_idx]
            
            result[idx] = interpolated_slice
        
        # Transpose back to original axis order but with new spatial dimensions
        final_axes = []
        ax_idx = 0
        for i in range(len(data.shape)):
            if i == lat_axis:
                final_axes.append(len(non_spatial_shape))
            elif i == lon_axis:
                final_axes.append(len(non_spatial_shape) + 1)
            else:
                final_axes.append(ax_idx)
                ax_idx += 1
        
        output = np.transpose(result, final_axes)
        
        # Check if data is a Dask array for out-of-core processing
        is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
        
        if is_dask:
            # For Dask arrays, we need to use dask-compatible operations
            try:
                import dask.array as da
                
                # Create a function to apply the interpolation
                def apply_interp(block, block_info=None):
                    # Apply the same interpolation logic to each block
                    block_transposed_axes = non_spatial_axes + [lat_axis, lon_axis]
                    block_transposed = np.transpose(block, block_transposed_axes)
                    
                    # Get the shape of non-spatial dimensions for this block
                    block_non_spatial_shape = block_transposed.shape[:len(non_spatial_axes)]
                    # Determine output shape based on the coordinate indices
                    # Handle both 1D and 2D coordinate index arrays
                    if lon_indices.ndim == 1:
                        # For 1D coordinate indices (rectilinear grids in identity regridding)
                        block_result_shape = block_non_spatial_shape + (len(lat_indices), len(lon_indices))
                    else:
                        # For 2D coordinate indices (normal regridding or curvilinear grids)
                        block_result_shape = block_non_spatial_shape + lon_indices.shape
                    block_result = np.full(block_result_shape, np.nan, dtype=block.dtype)
                    
                    # Process each slice along non-spatial dimensions
                    for idx in np.ndindex(block_non_spatial_shape):
                        slice_2d = block_transposed[idx]
                        
                        # For curvilinear grids, use direct indexing
                        if order == 0:  # Nearest neighbor
                            lat_idx = np.clip(lat_indices.astype(int), 0, slice_2d.shape[0] - 1)
                            lon_idx = np.clip(lon_indices.astype(int), 0, slice_2d.shape[1] - 1)
                            # Use advanced indexing to select the values
                            interpolated_slice = slice_2d[lat_idx, lon_idx]
                        else:
                            # For higher order interpolation, use nearest neighbor for now
                            lat_idx = np.clip(lat_indices.astype(int), 0, slice_2d.shape[0] - 1)
                            lon_idx = np.clip(lon_indices.astype(int), 0, slice_2d.shape[1] - 1)
                            # Use advanced indexing to select the values
                            interpolated_slice = slice_2d[lat_idx, lon_idx]
                        
                        block_result[idx] = interpolated_slice
                    
                    # Transpose back to original axis order but with new spatial dimensions
                    block_final_axes = []
                    ax_idx = 0
                    for j in range(len(block.shape)):
                        if j == lat_axis:
                            block_final_axes.append(len(block_non_spatial_shape))
                        elif j == lon_axis:
                            block_final_axes.append(len(block_non_spatial_shape) + 1)
                        else:
                            block_final_axes.append(ax_idx)
                            ax_idx += 1
                    
                    return np.transpose(block_result, block_final_axes)
                
                # Use map_blocks for Dask arrays
                output = da.map_blocks(
                    apply_interp,
                    data,
                    dtype=data.dtype,
                    drop_axis=[lat_axis, lon_axis],  # Remove the old spatial axes
                    new_axis=list(range(len(non_spatial_shape), len(non_spatial_shape) + 2)),  # Add new spatial axes
                    chunks=output.shape
                )
                return output
            except ImportError:
                # If Dask is not available, use the numpy implementation
                pass
        
        # Return the result after proper transposition
        return output
    
    def _interpolate_2d_slice(self, data_slice, lon_axis, lat_axis, lon_indices, lat_indices, order):
        """Interpolate a 2D slice along longitude and latitude axes."""
        # Ensure data_slice is at least 2D
        if data_slice.ndim < 2:
            return data_slice
        
        # For the simple case where we have a 2D grid with lon and lat dimensions
        if data_slice.ndim == 2:
            # Determine which axis is which - map_coordinates expects [axis0_idx, axis1_idx, ...]
            # where axis0_idx corresponds to the first dimension of the array, etc.
            if lat_axis == 0 and lon_axis == 1:  # Standard case: lat first, lon second
                indices = [lat_indices, lon_indices]
                result = map_coordinates(
                    data_slice,
                    indices,
                    order=order,
                    mode='nearest',
                    cval=np.nan
                )
            elif lat_axis == 1 and lon_axis == 0:  # Transposed case: lon first, lat second
                # Need to transpose the data and indices to match expected format
                indices = [lon_indices, lat_indices]
                result = map_coordinates(
                    data_slice,
                    indices,
                    order=order,
                    mode='nearest',
                    cval=np.nan
                )
            else:
                # For non-standard axis orders, transpose to standard format
                data_2d = np.moveaxis(data_slice, [lat_axis, lon_axis], [0, 1])
                indices = [lat_indices, lon_indices]
                result = map_coordinates(
                    data_2d,
                    indices,
                    order=order,
                    mode='nearest',
                    cval=np.nan
                )
        else:
            # For higher-dimensional data, we need to work slice by slice
            # This is a more complex case that requires careful handling of axis positions
            # First, transpose the data so that spatial dimensions are at the end
            non_spatial_axes = [i for i in range(data_slice.ndim) if i not in [lat_axis, lon_axis]]
            transposed_axes = non_spatial_axes + [lat_axis, lon_axis]
            transposed_data = np.transpose(data_slice, transposed_axes)
            
            # Get the shape of non-spatial dimensions
            non_spatial_shape = transposed_data.shape[:len(non_spatial_axes)]
            # Handle both 1D and 2D coordinate index arrays
            if lon_indices.ndim == 1:
                # For 1D coordinate indices (rectilinear grids in identity regridding)
                result_shape = non_spatial_shape + (len(lat_indices), len(lon_indices))
            else:
                # For 2D coordinate indices (normal regridding or curvilinear grids)
                result_shape = non_spatial_shape + (lat_indices.shape[0], lon_indices.shape[1])
            result = np.full(result_shape, np.nan, dtype=data_slice.dtype)
            
            # Iterate over all combinations of non-spatial dimensions
            for idx in np.ndindex(non_spatial_shape):
                # Extract the 2D slice
                slice_2d = transposed_data[idx]
                
                # Apply interpolation to the 2D slice
                interpolated_slice = map_coordinates(
                    slice_2d,
                    [lat_indices, lon_indices],
                    order=order,
                    mode='nearest',
                    cval=np.nan
                )
                
                # Store the result
                result[idx] = interpolated_slice
        
        return result
    
    def _regrid_dataset(self, data: xr.Dataset) -> xr.Dataset:
        """Regrid a Dataset using precomputed weights."""
        # Apply regridding to each data variable in the dataset
        regridded_vars = {}
        for var_name, var_data in data.data_vars.items():
            regridded_vars[var_name] = self._regrid_dataarray(var_data)
        
        # Create output coordinates
        output_coords = {}
        for coord_name in data.coords:
            if coord_name == self._source_lon_name:
                output_coords[self._target_lon_name] = self._target_lon
            elif coord_name == self._source_lat_name:
                output_coords[self._target_lat_name] = self._target_lat
            elif coord_name in [self._source_lon_name, self._source_lat_name]:
                # Skip the original coordinate axes, they'll be replaced
                continue
            else:
                # Keep other coordinates as they are
                output_coords[coord_name] = data.coords[coord_name]
        
        result = xr.Dataset(
            regridded_vars,
            coords=output_coords,
            attrs=data.attrs
        )
        
        return result


class PointInterpolator:
    """
    Scattered data interpolation engine.
    
    This class manages interpolation from scattered point data to grids or other points,
    with intelligent selection of spatial indexing backends.
    """
    
    def __init__(
        self,
        source_data: Union[xr.Dataset, xr.DataArray],
        target_points,
        method: str = "idw",
        source_crs: Optional[Union[str, CRS]] = None,
        target_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ):
        """
        Initialize the PointInterpolator.
        
        Parameters
        ----------
        source_data : xr.Dataset or xr.DataArray
            The source gridded data to interpolate from
        target_points : pandas.DataFrame or xarray.Dataset
            The target points to interpolate to
        method : str, optional
            The interpolation method to use (default: 'idw')
            Options: 'idw', 'linear', 'nearest', 'moving_average', 'gaussian', 'exponential'
        source_crs : str, CRS, optional
            The coordinate reference system of the source data
        target_crs : str, CRS, optional
            The coordinate reference system of the target points
        **kwargs
            Additional keyword arguments for the interpolation method
        """
        self.source_data = source_data
        self.target_points = target_points
        self.method = method
        self.source_crs = source_crs
        self.target_crs = target_crs
        self.kwargs = kwargs
        
        # Initialize CRS manager for coordinate system handling
        self.crs_manager = CRSManager()
        
        # Validate method
        valid_methods = ['idw', 'linear', 'nearest', 'moving_average', 'gaussian', 'exponential', 'bilinear']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Prepare the interpolation
        self._prepare_interpolation()
        
        # Initialize CRS transformation if needed
        if self.source_crs is not None and self.target_crs is not None:
            self._setup_crs_transformation()
    
    def _setup_crs_transformation(self):
        """Setup coordinate reference system transformation."""
        if self.source_crs is None or self.target_crs is None:
            raise ValueError("Both source_crs and target_crs must be provided for CRS transformation")
        
        # Create transformer for coordinate transformation
        self.transformer = Transformer.from_crs(
            self.source_crs, self.target_crs, always_xy=True
        )
    
    def _prepare_interpolation(self):
        """Prepare the interpolation setup."""
        # Determine CRS for source data if not provided
        if self.source_crs is None:
            # Extract coordinates from source data to determine CRS
            if isinstance(self.source_data, xr.DataArray):
                source_coords = self.source_data.coords
            else:  # xr.Dataset
                source_coords = self.source_data.coords
            
            # Find latitude and longitude coordinates in source data
            source_lat_names = [str(name) for name in source_coords
                               if 'lat' in str(name).lower() or 'y' in str(name).lower()]
            source_lon_names = [str(name) for name in source_coords
                               if 'lon' in str(name).lower() or 'x' in str(name).lower()]
            
            if source_lat_names and source_lon_names:
                source_lons = source_coords[source_lon_names[0]].values
                source_lats = source_coords[source_lat_names[0]].values
                self.source_crs = self.crs_manager.get_crs_from_source(
                    self.source_data,
                    source_lons,
                    source_lats,
                    source_lon_names[0],
                    source_lat_names[0]
                )
            else:
                raise ValueError("Could not find latitude/longitude coordinates in source data")
        
        # Determine CRS for target points if not provided
        if self.target_crs is None:
            # Extract coordinates from target points to determine CRS
            if isinstance(self.target_points, pd.DataFrame):
                # Look for common coordinate names in the DataFrame
                lon_col = None
                lat_col = None
                for col in self.target_points.columns:
                    if 'lon' in col.lower() or 'x' in col.lower():
                        lon_col = col
                    elif 'lat' in col.lower() or 'y' in col.lower():
                        lat_col = col
                
                if lon_col is not None and lat_col is not None:
                    target_lons = np.asarray(self.target_points[lon_col].values)
                    target_lats = np.asarray(self.target_points[lat_col].values)
                    self.target_crs = self.crs_manager.get_crs_from_source(
                        self.target_points,
                        target_lons,
                        target_lats,
                        lon_col,
                        lat_col
                    )
                else:
                    raise ValueError(
                        "Could not find longitude/latitude columns in target_points DataFrame. "
                        "Expected column names containing 'lon', 'lat', 'x', or 'y'."
                    )
            elif isinstance(self.target_points, xr.Dataset):
                # Extract coordinates from xarray Dataset
                lat_names = [str(name) for name in self.target_points.coords
                            if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                lon_names = [str(name) for name in self.target_points.coords
                            if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                
                # Also check data variables for coordinates
                if not lat_names or not lon_names:
                    lat_names = [str(name) for name in self.target_points.data_vars
                                if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                    lon_names = [str(name) for name in self.target_points.data_vars
                                if 'lon' in str(name).lower() or 'x' in str(name).lower()]

                if lat_names and lon_names:
                    # Use the first coordinate found
                    target_lons = np.asarray(self.target_points[lon_names[0]].values)
                    target_lats = np.asarray(self.target_points[lat_names[0]].values)
                    self.target_crs = self.crs_manager.get_crs_from_source(
                        self.target_points,
                        target_lons,
                        target_lats,
                        lon_names[0],
                        lat_names[0]
                    )
                else:
                    raise ValueError("Could not find latitude/longitude coordinates in target_points Dataset")
            elif isinstance(self.target_points, dict):
                # Extract coordinates from dictionary
                if 'longitude' in self.target_points:
                    target_lons = np.asarray(self.target_points['longitude'])
                elif 'lon' in self.target_points:
                    target_lons = np.asarray(self.target_points['lon'])
                elif 'x' in self.target_points:
                    target_lons = np.asarray(self.target_points['x'])
                else:
                    raise ValueError("Dictionary must contain 'longitude', 'lon', or 'x' key")
                
                if 'latitude' in self.target_points:
                    target_lats = np.asarray(self.target_points['latitude'])
                elif 'lat' in self.target_points:
                    target_lats = np.asarray(self.target_points['lat'])
                elif 'y' in self.target_points:
                    target_lats = np.asarray(self.target_points['y'])
                else:
                    raise ValueError("Dictionary must contain 'latitude', 'lat', or 'y' key")
                
                # For dictionary, we'll use the first key names found as the coordinate names
                lon_name = ('longitude' if 'longitude' in self.target_points else
                           'lon' if 'lon' in self.target_points else 'x')
                lat_name = ('latitude' if 'latitude' in self.target_points else
                           'lat' if 'lat' in self.target_points else 'y')
                
                self.target_crs = self.crs_manager.get_crs_from_source(
                    self.target_points,
                    target_lons,
                    target_lats,
                    lon_name,
                    lat_name
                )
            else:
                raise TypeError(
                    f"target_points must be pandas.DataFrame, xarray.Dataset, or dict, "
                    f"got {type(self.target_points)}"
                )
    
    def interpolate(self) -> Union[xr.Dataset, xr.DataArray]:
        """
        Perform the interpolation.
        
        Returns
        -------
        xr.Dataset or xr.DataArray
            The interpolated data at target points
        """
        # Validate target_points format and extract coordinates
        if isinstance(self.target_points, pd.DataFrame):
            # Look for common coordinate names in the DataFrame
            lon_col = None
            lat_col = None
            for col in self.target_points.columns:
                if 'lon' in col.lower() or 'x' in col.lower():
                    lon_col = col
                elif 'lat' in col.lower() or 'y' in col.lower():
                    lat_col = col
            
            if lon_col is None or lat_col is None:
                raise ValueError(
                    "Could not find longitude/latitude columns in target_points DataFrame. "
                    "Expected column names containing 'lon', 'lat', 'x', or 'y'."
                )
            
            target_lons = np.asarray(self.target_points[lon_col].values)
            target_lats = np.asarray(self.target_points[lat_col].values)
            
        elif isinstance(self.target_points, xr.Dataset):
            # Extract coordinates from xarray Dataset
            lat_names = [str(name) for name in self.target_points.coords
                        if 'lat' in str(name).lower() or 'y' in str(name).lower()]
            lon_names = [str(name) for name in self.target_points.coords
                        if 'lon' in str(name).lower() or 'x' in str(name).lower()]
            
            # Also check data variables for coordinates
            if not lat_names or not lon_names:
                lat_names = [str(name) for name in self.target_points.data_vars
                            if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                lon_names = [str(name) for name in self.target_points.data_vars
                            if 'lon' in str(name).lower() or 'x' in str(name).lower()]

            if not lat_names or not lon_names:
                raise ValueError("Could not find latitude/longitude coordinates in target_points Dataset")
            
            target_lons = np.asarray(self.target_points[lon_names[0]].values)
            target_lats = np.asarray(self.target_points[lat_names[0]].values)
            
        elif isinstance(self.target_points, dict):
            # Extract coordinates from dictionary
            if 'longitude' in self.target_points:
                target_lons = np.asarray(self.target_points['longitude'])
            elif 'lon' in self.target_points:
                target_lons = np.asarray(self.target_points['lon'])
            elif 'x' in self.target_points:
                target_lons = np.asarray(self.target_points['x'])
            else:
                raise ValueError("Dictionary must contain 'longitude', 'lon', or 'x' key")
            
            if 'latitude' in self.target_points:
                target_lats = np.asarray(self.target_points['latitude'])
            elif 'lat' in self.target_points:
                target_lats = np.asarray(self.target_points['lat'])
            elif 'y' in self.target_points:
                target_lats = np.asarray(self.target_points['y'])
            else:
                raise ValueError("Dictionary must contain 'latitude', 'lat', or 'y' key")
        else:
            raise TypeError(
                f"target_points must be pandas.DataFrame, xarray.Dataset, or dict, "
                f"got {type(self.target_points)}"
            )
        
        # Extract coordinate information from source data
        if isinstance(self.source_data, xr.DataArray):
            source_coords = self.source_data.coords
        else:  # xr.Dataset
            source_coords = self.source_data.coords
        
        # Find latitude and longitude coordinates in source data
        source_lat_names = [str(name) for name in source_coords
                           if 'lat' in str(name).lower() or 'y' in str(name).lower()]
        source_lon_names = [str(name) for name in source_coords
                           if 'lon' in str(name).lower() or 'x' in str(name).lower()]
        
        if not source_lat_names or not source_lon_names:
            raise ValueError("Could not find latitude/longitude coordinates in source data")
        
        source_lons = np.asarray(source_coords[source_lon_names[0]].values)
        source_lats = np.asarray(source_coords[source_lat_names[0]].values)
        
        # If CRS transformation is needed, transform coordinates
        if self.source_crs is not None and self.target_crs is not None and self.source_crs != self.target_crs:
            # Transform target coordinates to source CRS for interpolation
            transformer = Transformer.from_crs(self.target_crs, self.source_crs, always_xy=True)
            target_lons_transformed, target_lats_transformed = transformer.transform(target_lons, target_lats)
            # Use the transformed coordinates for interpolation
            interp_target_lons, interp_target_lats = target_lons_transformed, target_lats_transformed
        else:
            # No transformation needed
            interp_target_lons, interp_target_lats = target_lons, target_lats
        
        # Perform interpolation based on method using transformed coordinates
        if self.method == 'bilinear':
            return self._interpolate_bilinear(interp_target_lons, interp_target_lats, source_lons, source_lats)
        elif self.method == 'nearest':
            return self._interpolate_nearest(interp_target_lons, interp_target_lats, source_lons, source_lats)
        elif self.method in ['idw', 'linear', 'moving_average', 'gaussian', 'exponential']:
            # For more complex methods, we need a different approach
            # This is a simplified implementation that can be expanded
            warnings.warn(
                f"Method '{self.method}' is not fully implemented for grid-to-point interpolation. "
                f"Falling back to bilinear interpolation.",
                UserWarning
            )
            return self._interpolate_bilinear(interp_target_lons, interp_target_lats, source_lons, source_lats)
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")
    
    def _interpolate_bilinear(self, target_lons, target_lats, source_lons, source_lats):
        """Perform bilinear interpolation from source grid to target points."""
        from scipy.interpolate import RegularGridInterpolator
        
        # Check if the source data contains Dask arrays
        is_dask = False
        if isinstance(self.source_data, xr.DataArray):
            is_dask = hasattr(self.source_data.data, 'chunks') and self.source_data.data.__class__.__module__.startswith('dask')
        else:  # xr.Dataset
            for var_name, var_data in self.source_data.data_vars.items():
                if hasattr(var_data.data, 'chunks') and var_data.data.__class__.__module__.startswith('dask'):
                    is_dask = True
                    break
        
        if is_dask:
            # For Dask arrays, we need to use dask-compatible operations
            try:
                import dask.array as da
                
                # For now, we'll compute the dask arrays to perform the interpolation
                # A more sophisticated implementation would handle chunked interpolation
                if isinstance(self.source_data, xr.DataArray):
                    # For DataArray, interpolate the values directly
                    computed_values = self.source_data.values.compute() if hasattr(self.source_data.values, 'compute') else self.source_data.values
                    interpolator = RegularGridInterpolator(
                        (source_lats, source_lons),
                        computed_values,
                        method='linear',
                        bounds_error=False,
                        fill_value=np.nan
                    )
                    
                    # Create coordinate pairs for interpolation
                    points = np.column_stack([target_lats, target_lons])
                    interpolated_values = interpolator(points)
                    
                    # Create result DataArray with target coordinates
                    result_coords = {self.source_data.dims[-2]: target_lats,
                                   self.source_data.dims[-1]: target_lons}
                    result = xr.DataArray(
                        interpolated_values,
                        dims=[self.source_data.dims[-2], self.source_data.dims[-1]],
                        coords=result_coords,
                        attrs=self.source_data.attrs
                    )
                    
                    return result
                else:  # xr.Dataset
                    # For Dataset, interpolate each data variable
                    interpolated_vars = {}
                    for var_name, var_data in self.source_data.data_vars.items():
                        # Find spatial dimensions in the variable
                        spatial_dims = []
                        for dim in var_data.dims:
                            if any(name in str(dim).lower() for name in ['lat', 'lon', 'y', 'x']):
                                spatial_dims.append(dim)
                        
                        if len(spatial_dims) >= 2:
                            # Extract spatial coordinates for this variable
                            var_coords = var_data.coords
                            var_lat_names = [str(name) for name in var_coords
                                         if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                            var_lon_names = [str(name) for name in var_coords
                                         if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                            
                            if var_lat_names and var_lon_names:
                                var_lats = np.asarray(var_coords[var_lat_names[0]].values)
                                var_lons = np.asarray(var_coords[var_lon_names[0]].values)
                                
                                # Compute the dask array values
                                computed_values = var_data.values.compute() if hasattr(var_data.values, 'compute') else var_data.values
                                
                                # Create interpolator for this variable
                                interpolator = RegularGridInterpolator(
                                    (var_lats, var_lons),
                                    computed_values,
                                    method='linear',
                                    bounds_error=False,
                                    fill_value=np.nan
                                )
                                
                                # Interpolate
                                points = np.column_stack([target_lats, target_lons])
                                # For variables with additional dimensions, we need to handle them appropriately
                                interpolated_values = interpolator(points)
                                
                                # Create result DataArray
                                result_coords = {var_lat_names[0]: target_lats, var_lon_names[0]: target_lons}
                                interpolated_vars[var_name] = xr.DataArray(
                                    interpolated_values,
                                    dims=[var_lat_names[0], var_lon_names[0]],
                                    coords=result_coords,
                                    attrs=var_data.attrs
                                )
                    
                    # Create result Dataset
                    # Use the last available coordinate names if any were found
                    result_coords = {}
                    if interpolated_vars and len(target_lats) > 0 and len(target_lons) > 0:
                        # Get the coordinate names from the last processed variable
                        # Since all variables should have the same coordinate system in a dataset
                        last_var = list(interpolated_vars.values())[-1]
                        # Extract the coordinate names from the last variable
                        for coord_name, coord_vals in last_var.coords.items():
                            if any(name in coord_name.lower() for name in ['lat', 'y']):
                                result_coords[coord_name] = target_lats
                            elif any(name in coord_name.lower() for name in ['lon', 'x']):
                                result_coords[coord_name] = target_lons
                    
                    result = xr.Dataset(interpolated_vars, coords=result_coords)
                    return result
            except ImportError:
                # If Dask is not available, fall back to numpy computation
                pass
        
        # For numpy arrays or if Dask is not available, use the original approach
        try:
            if isinstance(self.source_data, xr.DataArray):
                # For DataArray, interpolate the values directly
                interpolator = RegularGridInterpolator(
                    (source_lats, source_lons),
                    self.source_data.values,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                
                # Create coordinate pairs for interpolation
                points = np.column_stack([target_lats, target_lons])
                interpolated_values = interpolator(points)
                
                # Create result DataArray with target coordinates
                # For point interpolation, we want a 1D result with coordinates as non-dimension coordinates
                result = xr.DataArray(
                    interpolated_values,
                    dims=['points'],
                    coords={'points': np.arange(len(interpolated_values))},
                    attrs=self.source_data.attrs
                )
                # Add longitude and latitude as non-dimension coordinates
                result = result.assign_coords(longitude=('points', target_lons))
                result = result.assign_coords(latitude=('points', target_lats))
                
                return result
            else:  # xr.Dataset
                # For Dataset, interpolate each data variable
                interpolated_vars = {}
                for var_name, var_data in self.source_data.data_vars.items():
                    # Find spatial dimensions in the variable
                    spatial_dims = []
                    for dim in var_data.dims:
                        if any(name in str(dim).lower() for name in ['lat', 'lon', 'y', 'x']):
                            spatial_dims.append(dim)
                    
                    if len(spatial_dims) >= 2:
                        # Extract spatial coordinates for this variable
                        var_coords = var_data.coords
                        var_lat_names = [str(name) for name in var_coords
                                       if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                        var_lon_names = [str(name) for name in var_coords
                                       if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                        
                        if var_lat_names and var_lon_names:
                            var_lats = np.asarray(var_coords[var_lat_names[0]].values)
                            var_lons = np.asarray(var_coords[var_lon_names[0]].values)
                            
                            # Create interpolator for this variable
                            interpolator = RegularGridInterpolator(
                                (var_lats, var_lons),
                                var_data.values,
                                method='linear',
                                bounds_error=False,
                                fill_value=np.nan
                            )
                            
                            # Interpolate
                            points = np.column_stack([target_lats, target_lons])
                            # For variables with additional dimensions, we need to handle them appropriately
                            interpolated_values = interpolator(points)
                            
                            # Create result DataArray
                            # For point interpolation, we want a 1D result with coordinates as non-dimension coordinates
                            interpolated_vars[var_name] = xr.DataArray(
                                interpolated_values,
                                dims=['points'],
                                coords={'points': np.arange(len(interpolated_values))},
                                attrs=var_data.attrs
                            )
                            # Add longitude and latitude as non-dimension coordinates
                            interpolated_vars[var_name] = interpolated_vars[var_name].assign_coords(
                                longitude=('points', target_lons)
                            )
                            interpolated_vars[var_name] = interpolated_vars[var_name].assign_coords(
                                latitude=('points', target_lats)
                            )
                
                # Create result Dataset
                # Use the last available coordinate names if any were found
                result_coords = {}
                if interpolated_vars and len(target_lats) > 0 and len(target_lons) > 0:
                    # Get the coordinate names from the last processed variable
                    # Since all variables should have the same coordinate system in a dataset
                    last_var = list(interpolated_vars.values())[-1]
                    # Extract the coordinate names from the last variable
                    for coord_name, coord_vals in last_var.coords.items():
                        if any(name in coord_name.lower() for name in ['lat', 'y']):
                            result_coords[coord_name] = target_lats
                        elif any(name in coord_name.lower() for name in ['lon', 'x']):
                            result_coords[coord_name] = target_lons
                
                result = xr.Dataset(interpolated_vars, coords=result_coords)
                return result
        except Exception as e:
            raise RuntimeError(f"Interpolation failed: {str(e)}")
    
    def _interpolate_nearest(self, target_lons, target_lats, source_lons, source_lats):
        """Perform nearest neighbor interpolation from source grid to target points."""
        
        # Check if the source data contains Dask arrays
        is_dask = False
        if isinstance(self.source_data, xr.DataArray):
            is_dask = hasattr(self.source_data.data, 'chunks') and self.source_data.data.__class__.__module__.startswith('dask')
        else:  # xr.Dataset
            for var_name, var_data in self.source_data.data_vars.items():
                if hasattr(var_data.data, 'chunks') and var_data.data.__class__.__module__.startswith('dask'):
                    is_dask = True
                    break
        
        if is_dask:
            # For Dask arrays, we need to compute them to perform the interpolation
            try:
                # Create a grid of source coordinates
                source_lon_grid, source_lat_grid = np.meshgrid(source_lons, source_lats)
                source_points = np.column_stack([source_lat_grid.ravel(), source_lon_grid.ravel()])
                
                # Create KDTree for nearest neighbor search
                tree = cKDTree(source_points)
                
                # Query points for target coordinates
                target_points = np.column_stack([target_lats, target_lons])
                distances, indices = tree.query(target_points)
                
                # Interpolate values from source data
                if isinstance(self.source_data, xr.DataArray):
                    # Compute the dask array values and flatten to match the grid points
                    computed_values = self.source_data.values.compute() if hasattr(self.source_data.values, 'compute') else self.source_data.values
                    flat_source_data = computed_values.ravel()
                    interpolated_values = flat_source_data[indices]
                    
                    # Create result DataArray
                    result_coords = {self.source_data.dims[-2]: target_lats,
                                   self.source_data.dims[-1]: target_lons}
                    result = xr.DataArray(
                        interpolated_values,
                        dims=[self.source_data.dims[-2], self.source_data.dims[-1]],
                        coords=result_coords,
                        attrs=self.source_data.attrs
                    )
                    
                    return result
                else:  # xr.Dataset
                    # For Dataset, interpolate each data variable
                    interpolated_vars = {}
                    for var_name, var_data in self.source_data.data_vars.items():
                        # Compute the dask array values and flatten to match the grid points
                        computed_values = var_data.values.compute() if hasattr(var_data.values, 'compute') else var_data.values
                        flat_var_data = computed_values.ravel()
                        interpolated_values = flat_var_data[indices]
                        
                        # Create result DataArray
                        result_coords = {}
                        # Find the lat/lon dimension names for this variable
                        var_lat_names = [str(name) for name in var_data.coords
                                       if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                        var_lon_names = [str(name) for name in var_data.coords
                                       if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                        
                        if var_lat_names and var_lon_names:
                            result_coords = {}
                            result_coords[var_lat_names[0]] = target_lats
                            result_coords[var_lon_names[0]] = target_lons
                            
                            interpolated_vars[var_name] = xr.DataArray(
                                interpolated_values,
                                dims=[var_lat_names[0], var_lon_names[0]],
                                coords=result_coords,
                                attrs=var_data.attrs
                            )
                    
                    # Create result Dataset
                    result = xr.Dataset(interpolated_vars)
                    return result
            except ImportError:
                # If Dask is not available, fall back to numpy computation
                pass
        
        # For numpy arrays or if Dask is not available, use the original approach
        try:
            # Create a grid of source coordinates
            source_lon_grid, source_lat_grid = np.meshgrid(source_lons, source_lats)
            source_points = np.column_stack([source_lat_grid.ravel(), source_lon_grid.ravel()])
            
            # Create KDTree for nearest neighbor search
            tree = cKDTree(source_points)
            
            # Query points for target coordinates
            target_points = np.column_stack([target_lats, target_lons])
            distances, indices = tree.query(target_points)
            
            # Interpolate values from source data
            if isinstance(self.source_data, xr.DataArray):
                # Flatten the source data to match the grid points
                flat_source_data = self.source_data.values.ravel()
                interpolated_values = flat_source_data[indices]
                
                # Create result DataArray
                # For point interpolation, we want a 1D result with coordinates as non-dimension coordinates
                result = xr.DataArray(
                    interpolated_values,
                    dims=['points'],
                    coords={'points': np.arange(len(interpolated_values))},
                    attrs=self.source_data.attrs
                )
                # Add longitude and latitude as non-dimension coordinates
                result = result.assign_coords(longitude=('points', target_lons))
                result = result.assign_coords(latitude=('points', target_lats))
                
                return result
            else:  # xr.Dataset
                # For Dataset, interpolate each data variable
                interpolated_vars = {}
                for var_name, var_data in self.source_data.data_vars.items():
                    # Flatten the variable data to match the grid points
                    flat_var_data = var_data.values.ravel()
                    interpolated_values = flat_var_data[indices]
                    
                    # Create result DataArray
                    result_coords = {}
                    # Find the lat/lon dimension names for this variable
                    var_lat_names = [str(name) for name in var_data.coords
                                   if 'lat' in str(name).lower() or 'y' in str(name).lower()]
                    var_lon_names = [str(name) for name in var_data.coords
                                   if 'lon' in str(name).lower() or 'x' in str(name).lower()]
                    
                    if var_lat_names and var_lon_names:
                        result_coords = {}
                        result_coords[var_lat_names[0]] = target_lats
                        result_coords[var_lon_names[0]] = target_lons
                        
                        interpolated_vars[var_name] = xr.DataArray(
                            interpolated_values,
                            dims=[var_lat_names[0], var_lon_names[0]],
                            coords=result_coords,
                            attrs=var_data.attrs
                        )
                
                # Create result Dataset
                result = xr.Dataset(interpolated_vars)
                return result
        except Exception as e:
            raise RuntimeError(f"Nearest neighbor interpolation failed: {str(e)}")