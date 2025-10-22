"""
Scattered data interpolation module.

This module provides the PointInterpolator class for interpolating from scattered point data
to grids or other points using various interpolation methods like IDW, linear, nearest neighbor, etc.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Optional, Dict, Any, Tuple, List
import warnings
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
# Import sklearn neighbors conditionally for optional functionality
# try:
#     from sklearn.neighbors import KNeighborsRegressor
#     HAS_SKLEARN = True
# except ImportError:
#     KNeighborsRegressor = None
#     HAS_SKLEARN = False
HAS_SKLEARN = False # sklearn functionality removed for now
from pyproj import CRS, Transformer

from pyregrid.crs.crs_manager import CRSManager
from pyregrid.algorithms.interpolators import BaseInterpolator, BilinearInterpolator, CubicInterpolator, NearestInterpolator


class PointInterpolator:
    """
    Scattered data interpolation engine.
    
    This class handles interpolation from scattered point data to grids or other points,
    with intelligent selection of spatial indexing backends based on coordinate system type.
    """
    
    def __init__(
        self,
        source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
        method: str = "idw",
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        source_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ):
        """
        Initialize the PointInterpolator.
        
        Parameters
        ----------
        source_points : pandas.DataFrame, xarray.Dataset, or dict
            The source scattered point data to interpolate from.
            For DataFrame, should contain coordinate columns (e.g., 'longitude', 'latitude').
            For Dataset, should contain coordinate variables.
            For dict, should have coordinate keys like {'longitude': [...], 'latitude': [...]}.
        method : str, optional
            The interpolation method to use (default: 'idw')
            Options: 'idw', 'linear', 'nearest', 'bilinear', 'cubic', 'moving_average', 
                     'gaussian', 'exponential'
        x_coord : str, optional
            Name of the x coordinate column/variable (e.g., 'longitude', 'x', 'lon')
            If None, will be inferred from common coordinate names
        y_coord : str, optional
            Name of the y coordinate column/variable (e.g., 'latitude', 'y', 'lat')
            If None, will be inferred from common coordinate names
        source_crs : str, CRS, optional
            The coordinate reference system of the source points
        **kwargs
            Additional keyword arguments for the interpolation method:
            - For IDW: power (default 2), search_radius (default None)
            - For KNN methods: n_neighbors (default 8), weights (default 'distance')
        """
        self.source_points = source_points
        self.method = method
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.source_crs = source_crs
        self.kwargs = kwargs
        
        # Initialize CRS manager for coordinate system handling
        self.crs_manager = CRSManager()
        
        # Validate method
        valid_methods = ['idw', 'linear', 'nearest', 'bilinear', 'cubic', 
                        'moving_average', 'gaussian', 'exponential']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Extract and validate coordinates
        self._extract_coordinates()
        
        # Validate coordinate arrays
        if not self.crs_manager.validate_coordinate_arrays(self.x_coords, self.y_coords,
                                                          self.source_crs if isinstance(self.source_crs, CRS) else None):
            raise ValueError("Invalid coordinate arrays detected")
        
        # Determine CRS if not provided explicitly
        if self.source_crs is None:
            # Use the "strict but helpful" policy to determine CRS
            if isinstance(self.source_points, pd.DataFrame):
                self.source_crs = self.crs_manager.get_crs_from_source(
                    self.source_points,
                    self.x_coords,
                    self.y_coords,
                    self.x_coord if self.x_coord is not None else 'x',
                    self.y_coord if self.y_coord is not None else 'y'
                )
            elif isinstance(self.source_points, xr.Dataset):
                self.source_crs = self.crs_manager.get_crs_from_source(
                    self.source_points,
                    self.x_coords,
                    self.y_coords,
                    self.x_coord if self.x_coord is not None else 'x',
                    self.y_coord if self.y_coord is not None else 'y'
                )
            elif isinstance(self.source_points, dict):
                # For dict, we need to create a minimal object that can be handled
                # For now, just detect from coordinates
                detected_crs = self.crs_manager.detect_crs_from_coordinates(
                    self.x_coords, self.y_coords,
                    self.x_coord if self.x_coord is not None else 'x',
                    self.y_coord if self.y_coord is not None else 'y'
                )
                if detected_crs is not None:
                    self.source_crs = detected_crs
                else:
                    raise ValueError(
                        f"No coordinate reference system (CRS) information found for coordinates "
                        f"'{self.x_coord if self.x_coord is not None else 'x'}' and '{self.y_coord if self.y_coord is not None else 'y'}'. Coordinate names do not clearly indicate "
                        f"geographic coordinates (latitude/longitude). Please provide explicit "
                        f"CRS information to avoid incorrect assumptions about the coordinate system."
                    )
        
        # Determine coordinate system type to select appropriate spatial backend
        self.coord_system_type = self.crs_manager.detect_coordinate_system_type(
            self.source_crs if isinstance(self.source_crs, CRS) else None
        )
        
        # Build spatial index for efficient neighbor search
        self._build_spatial_index()
        
        # Store the original point data for interpolation
        self._extract_point_data()
    
    def _extract_coordinates(self):
        """Extract coordinate information from source points."""
        if isinstance(self.source_points, pd.DataFrame):
            # Look for common coordinate names in the DataFrame if not specified
            if self.x_coord is None:
                for col in self.source_points.columns:
                    if any(name in col.lower() for name in ['lon', 'x', 'longitude']):
                        self.x_coord = col
                        break
                if self.x_coord is None:
                    raise ValueError("Could not find x coordinate column in DataFrame")
            
            if self.y_coord is None:
                for col in self.source_points.columns:
                    if any(name in col.lower() for name in ['lat', 'y', 'latitude']):
                        self.y_coord = col
                        break
                if self.y_coord is None:
                    raise ValueError("Could not find y coordinate column in DataFrame")
            
            self.x_coords = np.asarray(self.source_points[self.x_coord].values)
            self.y_coords = np.asarray(self.source_points[self.y_coord].values)
            
        elif isinstance(self.source_points, xr.Dataset):
            # Extract coordinates from xarray Dataset
            if self.x_coord is None:
                for coord_name in self.source_points.coords:
                    if any(name in str(coord_name).lower() for name in ['lon', 'x', 'longitude']):
                        self.x_coord = str(coord_name)
                        break
                if self.x_coord is None:
                    raise ValueError("Could not find x coordinate in Dataset")
            
            if self.y_coord is None:
                for coord_name in self.source_points.coords:
                    if any(name in str(coord_name).lower() for name in ['lat', 'y', 'latitude']):
                        self.y_coord = str(coord_name)
                        break
                if self.y_coord is None:
                    raise ValueError("Could not find y coordinate in Dataset")
            
            self.x_coords = np.asarray(self.source_points[self.x_coord].values)
            self.y_coords = np.asarray(self.source_points[self.y_coord].values)
            
        elif isinstance(self.source_points, dict):
            # Extract coordinates from dictionary
            if self.x_coord is None:
                for key in self.source_points.keys():
                    if any(name in key.lower() for name in ['lon', 'x', 'longitude']):
                        self.x_coord = key
                        break
                if self.x_coord is None:
                    raise ValueError("Could not find x coordinate key in dictionary")
            
            if self.y_coord is None:
                for key in self.source_points.keys():
                    if any(name in key.lower() for name in ['lat', 'y', 'latitude']):
                        self.y_coord = key
                        break
                if self.y_coord is None:
                    raise ValueError("Could not find y coordinate key in dictionary")
            
            self.x_coords = np.asarray(self.source_points[self.x_coord])
            self.y_coords = np.asarray(self.source_points[self.y_coord])
        else:
            raise TypeError(
                f"source_points must be pandas.DataFrame, xarray.Dataset, or dict, "
                f"got {type(self.source_points)}"
            )
        
        # Validate that coordinates have the same length
        if len(self.x_coords) != len(self.y_coords):
            raise ValueError("x and y coordinate arrays must have the same length")
        
        # Check for duplicate points
        unique_points, unique_indices = np.unique(
            np.column_stack([self.x_coords, self.y_coords]), 
            axis=0, 
            return_index=True
        )
        if len(unique_points) != len(self.x_coords):
            warnings.warn(
                f"Found {len(self.x_coords) - len(unique_points)} duplicate points in source data. "
                f"Only unique points will be used for interpolation.",
                UserWarning
            )
            # Keep only unique points
            self.x_coords = self.x_coords[unique_indices]
            self.y_coords = self.y_coords[unique_indices]
            # Update source_points to only contain unique points
            if isinstance(self.source_points, pd.DataFrame):
                self.source_points = self.source_points.iloc[unique_indices]
            elif isinstance(self.source_points, xr.Dataset):
                # For xarray, this is more complex - we'll just issue a warning
                warnings.warn(
                    "Duplicate point removal for xarray Dataset is not fully implemented. "
                    "Consider preprocessing your data to remove duplicates.",
                    UserWarning
                )
    
    def _build_spatial_index(self):
        """Build spatial index for efficient neighbor search."""
        # Create point array for spatial indexing
        self.points = np.column_stack([self.y_coords, self.x_coords])  # lat, lon format for consistency
        
        # Select appropriate spatial index based on coordinate system type
        if self.coord_system_type == 'geographic':
            # For geographic coordinates, use BallTree which handles great-circle distances
            # For now, we'll use cKDTree with a warning that for geographic data, 
            # more sophisticated methods may be needed
            warnings.warn(
                "Using cKDTree for geographic coordinates. For more accurate results with "
                "geographic data, consider using a specialized geographic interpolation method.",
                UserWarning
            )
            self.spatial_index = cKDTree(self.points)
        else:
            # For projected coordinates, cKDTree is appropriate
            self.spatial_index = cKDTree(self.points)
    
    def _extract_point_data(self):
        """Extract data values from source points."""
        if isinstance(self.source_points, pd.DataFrame):
            # Get all columns except coordinate columns as data variables
            data_cols = [col for col in self.source_points.columns 
                        if col not in [self.x_coord, self.y_coord]]
            self.data_vars = {}
            for col in data_cols:
                self.data_vars[col] = np.asarray(self.source_points[col].values)
        elif isinstance(self.source_points, xr.Dataset):
            # Extract all data variables
            self.data_vars = {}
            for var_name, var_data in self.source_points.data_vars.items():
                self.data_vars[var_name] = var_data.values
        elif isinstance(self.source_points, dict):
            # All keys that are not coordinates are considered data
            data_keys = [key for key in self.source_points.keys() 
                        if key not in [self.x_coord, self.y_coord]]
            self.data_vars = {}
            for key in data_keys:
                self.data_vars[key] = np.asarray(self.source_points[key])
    
    def interpolate_to(
        self,
        target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        target_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Interpolate from source points to target points.
        
        Parameters
        ----------
        target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
            Target points to interpolate to.
            If DataFrame/Dataset/dict: same format as source_points with coordinate columns.
            If np.ndarray: shape (n, 2) with [y, x] coordinates for each point.
        x_coord : str, optional
            Name of x coordinate in target points (if not using np.ndarray)
        y_coord : str, optional
            Name of y coordinate in target points (if not using np.ndarray)
        target_crs : str, CRS, optional
            Coordinate reference system of target points (if different from source)
        **kwargs
            Additional interpolation parameters
        
        Returns
        -------
        xr.Dataset, xr.DataArray, or dict
            Interpolated data at target points
        """
        # Extract target coordinates
        if isinstance(target_points, np.ndarray):
            # Direct coordinate array format: (n, 2) with [y, x] for each point
            if target_points.ndim != 2 or target_points.shape[1] != 2:
                raise ValueError("Target coordinates array must have shape (n, 2) with [y, x] format")
            target_ys = target_points[:, 0]
            target_xs = target_points[:, 1]
        else:
            # DataFrame, Dataset, or dict format
            if isinstance(target_points, pd.DataFrame):
                if x_coord is None:
                    for col in target_points.columns:
                        if any(name in col.lower() for name in ['lon', 'x', 'longitude']):
                            x_coord = col
                            break
                    if x_coord is None:
                        raise ValueError("Could not find x coordinate column in target DataFrame")
                
                if y_coord is None:
                    for col in target_points.columns:
                        if any(name in col.lower() for name in ['lat', 'y', 'latitude']):
                            y_coord = col
                            break
                if y_coord is None:
                    raise ValueError("Could not find y coordinate column in target DataFrame")
                
                target_xs = np.asarray(target_points[x_coord].values)
                target_ys = np.asarray(target_points[y_coord].values)
                
            elif isinstance(target_points, xr.Dataset):
                if x_coord is None:
                    for coord_name in target_points.coords:
                        if any(name in str(coord_name).lower() for name in ['lon', 'x', 'longitude']):
                            x_coord = str(coord_name)
                            break
                    if x_coord is None:
                        raise ValueError("Could not find x coordinate in target Dataset")
                
                if y_coord is None:
                    for coord_name in target_points.coords:
                        if any(name in str(coord_name).lower() for name in ['lat', 'y', 'latitude']):
                            y_coord = str(coord_name)
                            break
                    if y_coord is None:
                        raise ValueError("Could not find y coordinate in target Dataset")
                
                target_xs = np.asarray(target_points[x_coord].values)
                target_ys = np.asarray(target_points[y_coord].values)
                
            elif isinstance(target_points, dict):
                if x_coord is None:
                    for key in target_points.keys():
                        if any(name in key.lower() for name in ['lon', 'x', 'longitude']):
                            x_coord = key
                            break
                    if x_coord is None:
                        raise ValueError("Could not find x coordinate key in target dictionary")
                
                if y_coord is None:
                    for key in target_points.keys():
                        if any(name in key.lower() for name in ['lat', 'y', 'latitude']):
                            y_coord = key
                            break
                    if y_coord is None:
                        raise ValueError("Could not find y coordinate key in target dictionary")
                
                target_xs = np.asarray(target_points[x_coord])
                target_ys = np.asarray(target_points[y_coord])
            else:
                raise TypeError(
                    f"target_points must be pandas.DataFrame, xarray.Dataset, dict, or np.ndarray, "
                    f"got {type(target_points)}"
                )
        
        # Handle CRS transformation if needed
        if target_crs is not None and self.source_crs != target_crs:
            # Transform target coordinates to source CRS for interpolation
            transformer = Transformer.from_crs(target_crs, self.source_crs, always_xy=True)
            target_xs_transformed, target_ys_transformed = transformer.transform(target_xs, target_ys)
            interp_target_xs, interp_target_ys = target_xs_transformed, target_ys_transformed
        else:
            # No transformation needed
            interp_target_xs, interp_target_ys = target_xs, target_ys
        
        # Perform interpolation based on method
        interpolated_results = {}
        
        for var_name, var_data in self.data_vars.items():
            if self.method == 'idw':
                interpolated_values = self._interpolate_idw(
                    interp_target_xs, interp_target_ys, var_data, **kwargs
                )
            elif self.method == 'nearest':
                interpolated_values = self._interpolate_nearest(
                    interp_target_xs, interp_target_ys, var_data
                )
            elif self.method == 'linear':
                interpolated_values = self._interpolate_linear(
                    interp_target_xs, interp_target_ys, var_data
                )
            elif self.method == 'bilinear':
                # For scattered data, bilinear is not directly applicable
                # Use IDW with linear weights instead
                interpolated_values = self._interpolate_knn(
                    interp_target_xs, interp_target_ys, var_data, 
                    method='linear', **kwargs
                )
            elif self.method == 'cubic':
                # For scattered data, cubic is not directly applicable
                # Use IDW with higher-order weights instead
                interpolated_values = self._interpolate_knn(
                    interp_target_xs, interp_target_ys, var_data, 
                    method='cubic', **kwargs
                )
            elif self.method in ['moving_average', 'gaussian', 'exponential']:
                interpolated_values = self._interpolate_knn(
                    interp_target_xs, interp_target_ys, var_data, 
                    method=self.method, **kwargs
                )
            else:
                raise ValueError(f"Unsupported interpolation method: {self.method}")
            
            interpolated_results[var_name] = interpolated_values
        
        # Return appropriate format based on input type
        if isinstance(target_points, xr.Dataset):
            # Create result as xarray Dataset
            result_coords = {y_coord: target_ys, x_coord: target_xs}
            result_vars = {}
            for var_name, var_values in interpolated_results.items():
                result_vars[var_name] = xr.DataArray(
                    var_values,
                    dims=[y_coord, x_coord] if var_values.ndim == 2 else [y_coord] if var_values.ndim == 1 else [],
                    coords=result_coords if var_values.ndim > 0 else {},
                    name=var_name
                )
            result_dataset = xr.Dataset(result_vars, coords=result_coords)
            return result_dataset
        elif isinstance(target_points, pd.DataFrame):
            # Create result as DataFrame
            result_df = pd.DataFrame({x_coord: target_xs, y_coord: target_ys})
            for var_name, var_values in interpolated_results.items():
                result_df[var_name] = var_values
            return result_df
        else:
            # Return as dictionary
            result_dict = {}
            if x_coord is not None:
                result_dict[x_coord] = target_xs
            else:
                result_dict['x'] = target_xs
            if y_coord is not None:
                result_dict[y_coord] = target_ys
            else:
                result_dict['y'] = target_ys
            result_dict.update(interpolated_results)
            return result_dict
    
    def _interpolate_idw(self, target_xs, target_ys, data, **kwargs):
        """Perform Inverse Distance Weighting interpolation."""
        # Check if data is a Dask array for out-of-core processing
        is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
        
        if is_dask:
            return self._interpolate_idw_dask(target_xs, target_ys, data, **kwargs)
        else:
            return self._interpolate_idw_numpy(target_xs, target_ys, data, **kwargs)
    
    def _interpolate_idw_numpy(self, target_xs, target_ys, data, **kwargs):
        """Perform Inverse Distance Weighting interpolation for numpy arrays."""
        # Get parameters
        power = kwargs.get('power', 2)
        search_radius = kwargs.get('search_radius', None)
        n_neighbors = kwargs.get('n_neighbors', min(10, len(self.x_coords)))
        
        # Prepare target points for querying
        target_points = np.column_stack([target_ys, target_xs])
        
        # Find nearest neighbors for each target point
        if search_radius is not None:
            # Use radius-based search
            distances, indices = self.spatial_index.query_ball_point(target_points, search_radius, return_distance=True)
            # For each target point, get the corresponding distances and indices
            interpolated_values = []
            for i, (dist_list, idx_list) in enumerate(zip(distances, target_points)):
                if len(idx_list) == 0:
                    # No neighbors found, return NaN
                    interpolated_values.append(np.nan)
                else:
                    # Calculate inverse distance weights
                    dists = np.array([np.sqrt((target_ys[i] - self.y_coords[j])**2 + (target_xs[i] - self.x_coords[j])**2)
                                     for j in idx_list])
                    # Avoid division by zero
                    dists = np.maximum(dists, 1e-10)
                    weights = 1.0 / (dists ** power)
                    # Calculate weighted average
                    weighted_sum = np.sum(weights * data[idx_list])
                    weight_sum = np.sum(weights)
                    interpolated_values.append(weighted_sum / weight_sum if weight_sum != 0 else np.nan)
            return np.array(interpolated_values)
        else:
            # Use k-nearest neighbors
            distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
            
            # Calculate inverse distance weights
            distances = np.maximum(distances, 1e-10)  # Avoid division by zero
            weights = 1.0 / (distances ** power)
            
            # Calculate weighted average for each target point
            interpolated_values = []
            for i in range(len(target_points)):
                if distances[i, 0] < 1e-8:  # Exact match
                    interpolated_values.append(data[indices[i, 0]])
                else:
                    weight_sum = np.sum(weights[i, :])
                    if weight_sum == 0:
                        interpolated_values.append(np.nan)
                    else:
                        weighted_sum = np.sum(weights[i, :] * data[indices[i, :]])
                        interpolated_values.append(weighted_sum / weight_sum)
            
            return np.array(interpolated_values)
    
    def _interpolate_idw_dask(self, target_xs, target_ys, data, **kwargs):
        """Perform Inverse Distance Weighting interpolation for Dask arrays."""
        try:
            import dask.array as da
            import numpy as np
            
            # Get parameters
            power = kwargs.get('power', 2)
            search_radius = kwargs.get('search_radius', None)
            n_neighbors = kwargs.get('n_neighbors', min(10, len(self.x_coords)))
            
            # Prepare target points for querying
            target_points = np.column_stack([target_ys, target_xs])
            
            # For Dask processing, we need to chunk the target points and process each chunk
            # This is a simplified approach - a more sophisticated implementation would handle chunking better
            if search_radius is not None:
                # Use radius-based search
                distances, indices = self.spatial_index.query_ball_point(target_points, search_radius, return_distance=True)
                # For each target point, get the corresponding distances and indices
                interpolated_values = []
                for i, (dist_list, idx_list) in enumerate(zip(distances, target_points)):
                    if len(idx_list) == 0:
                        # No neighbors found, return NaN
                        interpolated_values.append(np.nan)
                    else:
                        # Calculate inverse distance weights
                        dists = np.array([np.sqrt((target_ys[i] - self.y_coords[j])**2 + (target_xs[i] - self.x_coords[j])**2)
                                         for j in idx_list])
                        # Avoid division by zero
                        dists = np.maximum(dists, 1e-10)
                        weights = 1.0 / (dists ** power)
                        # Calculate weighted average
                        # For Dask arrays, we need to handle the indexing differently
                        selected_data = data[idx_list]
                        if hasattr(selected_data, 'compute'):
                            selected_data = selected_data.compute()
                        weighted_sum = np.sum(weights * selected_data)
                        weight_sum = np.sum(weights)
                        interpolated_values.append(weighted_sum / weight_sum if weight_sum != 0 else np.nan)
                return np.array(interpolated_values)
            else:
                # Use k-nearest neighbors
                distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
                
                # Calculate inverse distance weights
                distances = np.maximum(distances, 1e-10)  # Avoid division by zero
                weights = 1.0 / (distances ** power)
                
                # Calculate weighted average for each target point
                interpolated_values = []
                for i in range(len(target_points)):
                    if distances[i, 0] < 1e-8:  # Exact match
                        # For Dask arrays, handle indexing appropriately
                        selected_data = data[indices[i, 0]]
                        if hasattr(selected_data, 'compute'):
                            selected_data = selected_data.compute()
                        interpolated_values.append(selected_data)
                    else:
                        # For Dask arrays, handle indexing appropriately
                        selected_data = data[indices[i, :]]
                        if hasattr(selected_data, 'compute'):
                            selected_data = selected_data.compute()
                        weight_sum = np.sum(weights[i, :])
                        if weight_sum == 0:
                            interpolated_values.append(np.nan)
                        else:
                            weighted_sum = np.sum(weights[i, :] * selected_data)
                            interpolated_values.append(weighted_sum / weight_sum)
                
                return np.array(interpolated_values)
        except ImportError:
            # If Dask is not available, fall back to numpy computation
            if hasattr(data, 'compute'):
                data = data.compute()
            return self._interpolate_idw_numpy(target_xs, target_ys, data, **kwargs)
    
    def _interpolate_nearest(self, target_xs, target_ys, data):
        """Perform nearest neighbor interpolation."""
        # Check if data is a Dask array for out-of-core processing
        is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
        
        if is_dask:
            return self._interpolate_nearest_dask(target_xs, target_ys, data)
        else:
            return self._interpolate_nearest_numpy(target_xs, target_ys, data)
    
    def _interpolate_nearest_numpy(self, target_xs, target_ys, data):
        """Perform nearest neighbor interpolation for numpy arrays."""
        target_points = np.column_stack([target_ys, target_xs])
        distances, indices = self.spatial_index.query(target_points, k=1)
        return data[indices]
    
    def _interpolate_nearest_dask(self, target_xs, target_ys, data):
        """Perform nearest neighbor interpolation for Dask arrays."""
        try:
            import dask.array as da
            import numpy as np
            
            target_points = np.column_stack([target_ys, target_xs])
            distances, indices = self.spatial_index.query(target_points, k=1)
            
            # For Dask arrays, we need to handle indexing differently
            # Since Dask doesn't support fancy indexing the same way as numpy,
            # we need to compute the result in chunks
            if hasattr(data, 'compute'):
                # If it's a Dask array, we compute the indices selection
                selected_data = data[indices]
                return selected_data
            else:
                return data[indices]
        except ImportError:
            # If Dask is not available, fall back to numpy computation
            if hasattr(data, 'compute'):
                data = data.compute()
            return self._interpolate_nearest_numpy(target_xs, target_ys, data)
    
    def _interpolate_linear(self, target_xs, target_ys, data):
        """Perform linear interpolation using Delaunay triangulation."""
        # Check if data is a Dask array for out-of-core processing
        is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
        
        if is_dask:
            return self._interpolate_linear_dask(target_xs, target_ys, data)
        else:
            return self._interpolate_linear_numpy(target_xs, target_ys, data)
    
    def _interpolate_linear_numpy(self, target_xs, target_ys, data):
        """Perform linear interpolation using Delaunay triangulation for numpy arrays."""
        try:
            from scipy.interpolate import griddata
            source_points = np.column_stack([self.x_coords, self.y_coords])
            target_points = np.column_stack([target_xs, target_ys])
            return griddata(
                source_points,
                data,
                target_points,
                method='linear',
                fill_value=np.nan
            )
        except Exception as e:
            warnings.warn(f"Linear interpolation failed: {str(e)}. Falling back to nearest neighbor.", UserWarning)
            return self._interpolate_nearest(target_xs, target_ys, data)
    
    def _interpolate_linear_dask(self, target_xs, target_ys, data):
        """Perform linear interpolation using Delaunay triangulation for Dask arrays."""
        try:
            import dask.array as da
            from scipy.interpolate import griddata
            import numpy as np
            
            source_points = np.column_stack([self.x_coords, self.y_coords])
            target_points = np.column_stack([target_xs, target_ys])
            
            # For Dask arrays, we need to handle this differently
            # Since griddata doesn't work directly with Dask arrays, we need to process in chunks
            # or compute the result differently
            if hasattr(data, 'compute'):
                # If it's a Dask array, compute it for the interpolation
                computed_data = data.compute()
                result = griddata(
                    source_points,
                    computed_data,
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )
                # Convert back to Dask array if needed
                return da.from_array(result, chunks='auto')
            else:
                return griddata(
                    source_points,
                    data,
                    target_points,
                    method='linear',
                    fill_value=np.nan
                )
        except Exception as e:
            warnings.warn(f"Linear interpolation failed: {str(e)}. Falling back to nearest neighbor.", UserWarning)
            return self._interpolate_nearest(target_xs, target_ys, data)
    
    def _interpolate_knn(self, target_xs, target_ys, data, method='moving_average', **kwargs):
        """Perform interpolation using K-nearest neighbors with various weighting schemes."""
        # Check if data is a Dask array for out-of-core processing
        is_dask = hasattr(data, 'chunks') and data.__class__.__module__.startswith('dask')
        
        if is_dask:
            return self._interpolate_knn_dask(target_xs, target_ys, data, method, **kwargs)
        else:
            return self._interpolate_knn_numpy(target_xs, target_ys, data, method, **kwargs)
    
    def _interpolate_knn_numpy(self, target_xs, target_ys, data, method='moving_average', **kwargs):
        """Perform interpolation using K-nearest neighbors with various weighting schemes for numpy arrays."""
        # Get parameters
        n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
        
        # Prepare target points
        target_points = np.column_stack([target_ys, target_xs])
        
        # Get the spatial neighbors
        distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
        
        # Prepare source points for sklearn
        source_points = np.column_stack([self.y_coords, self.x_coords])
        
        # Select appropriate weighting function based on method
        if method == 'moving_average':
            weights_func = 'uniform' # Equal weights for all neighbors
        elif method == 'gaussian':
            # Use a custom distance-based Gaussian weighting
            def gaussian_weights(distances):
                sigma = kwargs.get('sigma', np.std(distances) if len(distances) > 1 else 1.0)
                return np.exp(-0.5 * (distances / sigma) ** 2)
            weights_func = gaussian_weights
        elif method == 'exponential':
            # Use a custom distance-based exponential weighting
            def exp_weights(distances):
                scale = kwargs.get('scale', 1.0)
                return np.exp(-distances / scale)
            weights_func = exp_weights
        else:  # 'linear' or 'idw' style
            def idw_weights(distances):
                power = kwargs.get('power', 2)
                return 1.0 / np.maximum(distances ** power, 1e-10)
            weights_func = idw_weights
        
        # For each target point, calculate the weighted average
        interpolated_values = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            # Get the source data values for the neighbors
            neighbor_data = data[idx_row]
            
            # Calculate weights
            if callable(weights_func):
                weights = weights_func(dist_row)
            else:
                weights = weights_func  # For 'uniform' case
            
            # Calculate weighted average
            if np.sum(weights) > 0:
                weighted_avg = np.average(neighbor_data, weights=weights)
                interpolated_values.append(weighted_avg)
            else:
                interpolated_values.append(np.nan)
        
        return np.array(interpolated_values)
    
    def _interpolate_knn_dask(self, target_xs, target_ys, data, method='moving_average', **kwargs):
        """Perform interpolation using K-nearest neighbors with various weighting schemes for Dask arrays."""
        try:
            import dask.array as da
            import numpy as np
            
            # Get parameters
            n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
            
            # Prepare target points
            target_points = np.column_stack([target_ys, target_xs])
            
            # Get the spatial neighbors
            distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
            
            # Select appropriate weighting function based on method
            if method == 'moving_average':
                weights_func = 'uniform' # Equal weights for all neighbors
            elif method == 'gaussian':
                # Use a custom distance-based Gaussian weighting
                def gaussian_weights(distances):
                    sigma = kwargs.get('sigma', np.std(distances) if len(distances) > 1 else 1.0)
                    return np.exp(-0.5 * (distances / sigma) ** 2)
                weights_func = gaussian_weights
            elif method == 'exponential':
                # Use a custom distance-based exponential weighting
                def exp_weights(distances):
                    scale = kwargs.get('scale', 1.0)
                    return np.exp(-distances / scale)
                weights_func = exp_weights
            else:  # 'linear' or 'idw' style
                def idw_weights(distances):
                    power = kwargs.get('power', 2)
                    return 1.0 / np.maximum(distances ** power, 1e-10)
                weights_func = idw_weights
            
            # For each target point, calculate the weighted average
            interpolated_values = []
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                # Get the source data values for the neighbors
                # For Dask arrays, we need to handle indexing differently
                neighbor_data = data[idx_row]
                if hasattr(neighbor_data, 'compute'):
                    neighbor_data = neighbor_data.compute()
                
                # Calculate weights
                if callable(weights_func):
                    weights = weights_func(dist_row)
                else:
                    weights = weights_func  # For 'uniform' case
                
                # Calculate weighted average
                if np.sum(weights) > 0:
                    weighted_avg = np.average(neighbor_data, weights=weights)
                    interpolated_values.append(weighted_avg)
                else:
                    interpolated_values.append(np.nan)
            
            return np.array(interpolated_values)
        except ImportError:
            # If Dask is not available, fall back to numpy computation
            if hasattr(data, 'compute'):
                data = data.compute()
            return self._interpolate_knn_numpy(target_xs, target_ys, data, method, **kwargs)
    
    def interpolate_to_grid(self, target_grid, **kwargs):
        """
        Interpolate from scattered points to a regular grid.
        
        Parameters
        ----------
        target_grid : xr.Dataset or xr.DataArray
            Target grid to interpolate to
        **kwargs
            Additional interpolation parameters
            
        Returns
        -------
        xr.Dataset
            Interpolated data on the target grid
        """
        # Extract grid coordinates
        if isinstance(target_grid, xr.DataArray):
            target_coords = target_grid.coords
        else:  # xr.Dataset
            target_coords = target_grid.coords
        
        # Find latitude and longitude coordinates in target grid
        target_lat_names = [str(name) for name in target_coords
                           if 'lat' in str(name).lower() or 'y' in str(name).lower()]
        target_lon_names = [str(name) for name in target_coords
                           if 'lon' in str(name).lower() or 'x' in str(name).lower()]
        
        if not target_lat_names or not target_lon_names:
            raise ValueError("Could not find latitude/longitude coordinates in target grid")
        
        target_lons = np.asarray(target_grid[target_lon_names[0]].values)
        target_lats = np.asarray(target_grid[target_lat_names[0]].values)
        
        # Create meshgrid for all target points
        if target_lons.ndim == 1 and target_lats.ndim == 1:
            # 1D coordinate arrays - create 2D meshgrid
            lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
            target_points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
        else:
            # Already 2D coordinate arrays
            target_points = np.column_stack([target_lats.ravel(), target_lons.ravel()])
        
        # Interpolate to all target points
        result_dict = self.interpolate_to(target_points, **kwargs)
        
        # Reshape results back to grid shape
        if isinstance(result_dict, dict):
            reshaped_results = {}
            for key, values in result_dict.items():
                if key not in [target_lon_names[0], target_lat_names[0]]:
                    reshaped_results[key] = values.reshape(target_lats.shape + target_lons.shape)
            
            # Create output dataset
            result_vars = {}
            for var_name, reshaped_data in reshaped_results.items():
                result_vars[var_name] = xr.DataArray(
                    reshaped_data,
                    dims=[target_lat_names[0], target_lon_names[0]],
                    coords={target_lat_names[0]: target_lats, target_lon_names[0]: target_lons},
                    name=var_name
                )
            
            return xr.Dataset(result_vars)
        else:
            return result_dict