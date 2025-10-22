"""
Scattered data interpolation module.

This module provides comprehensive scattered data interpolation functionality
with neighbor-based weighting methods, triangulation-based interpolation,
and spatial indexing with hybrid backend approach.
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Union, Optional, Dict, Any, Tuple, Callable, List
import warnings
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsRegressor, BallTree
from pyproj import CRS, Transformer

from .crs.crs_manager import CRSManager

# Import spatial modules with error handling
try:
    from scipy.spatial import cKDTree, Delaunay
    HAS_SCIPY_SPATIAL = True
except ImportError:
    cKDTree = None
    Delaunay = None
    HAS_SCIPY_SPATIAL = False


class BaseScatteredInterpolator:
    """
    Base class for scattered data interpolation methods.
    
    Provides common functionality for all scattered interpolation methods.
    """
    
    def __init__(
        self,
        source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        source_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ):
        """
        Initialize the scattered interpolator.
        
        Parameters
        ----------
        source_points : pandas.DataFrame, xarray.Dataset, or dict
            The source scattered point data to interpolate from.
        x_coord : str, optional
            Name of the x coordinate column/variable
        y_coord : str, optional
            Name of the y coordinate column/variable
        source_crs : str, CRS, optional
            The coordinate reference system of the source points
        **kwargs
            Additional keyword arguments
        """
        self.source_points = source_points
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.source_crs = source_crs
        self.kwargs = kwargs
        
        # Initialize CRS manager for coordinate system handling
        self.crs_manager = CRSManager()
        
        # Extract and validate coordinates
        self._extract_coordinates()
        
        # Validate coordinate arrays
        if not self.crs_manager.validate_coordinate_arrays(
            self.x_coords, self.y_coords,
            self.source_crs if isinstance(self.source_crs, CRS) else None
        ):
            raise ValueError("Invalid coordinate arrays detected")
        
        # Determine CRS if not provided explicitly
        if self.source_crs is None:
            self.source_crs = self._determine_crs()
        
        # Determine coordinate system type to select appropriate spatial backend
        self.coord_system_type = self.crs_manager.detect_coordinate_system_type(
            self.source_crs if isinstance(self.source_crs, CRS) else None
        )
        
        # Extract the point data values
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
    
    def _determine_crs(self) -> Optional[CRS]:
        """Determine CRS from source points based on coordinate system policy."""
        if isinstance(self.source_points, pd.DataFrame):
            return self.crs_manager.get_crs_from_source(
                self.source_points,
                self.x_coords,
                self.y_coords,
                self.x_coord if self.x_coord is not None else 'x',
                self.y_coord if self.y_coord is not None else 'y'
            )
        elif isinstance(self.source_points, xr.Dataset):
            return self.crs_manager.get_crs_from_source(
                self.source_points,
                self.x_coords,
                self.y_coords,
                self.x_coord if self.x_coord is not None else 'x',
                self.y_coord if self.y_coord is not None else 'y'
            )
        elif isinstance(self.source_points, dict):
            # For dict, detect from coordinates
            detected_crs = self.crs_manager.detect_crs_from_coordinates(
                self.x_coords, self.y_coords,
                self.x_coord if self.x_coord is not None else 'x',
                self.y_coord if self.y_coord is not None else 'y'
            )
            if detected_crs is not None:
                return detected_crs
            else:
                raise ValueError(
                    f"No coordinate reference system (CRS) information found for coordinates "
                    f"'{self.x_coord if self.x_coord is not None else 'x'}' and '{self.y_coord if self.y_coord is not None else 'y'}'. Coordinate names do not clearly indicate "
                    f"geographic coordinates (latitude/longitude). Please provide explicit "
                    f"CRS information to avoid incorrect assumptions about the coordinate system."
                )
        else:
            raise TypeError(
                f"source_points must be pandas.DataFrame, xarray.Dataset, or dict, "
                f"got {type(self.source_points)}"
            )
    
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


class NeighborBasedInterpolator(BaseScatteredInterpolator):
    """
    Neighbor-based interpolation methods using sklearn.neighbors.KNeighborsRegressor.
    
    Supports Inverse Distance Weighting (IDW), Moving Average, Gaussian, and Exponential weighting.
    """
    
    def __init__(
        self,
        source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
        method: str = "idw",
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        source_crs: Optional[Union[str, CRS]] = None,
        chunk_size: Optional[int] = 10000,  # For performance optimization with large datasets
        **kwargs
    ):
        """
        Initialize the neighbor-based interpolator.
        
        Parameters
        ----------
        source_points : pandas.DataFrame, xarray.Dataset, or dict
            The source scattered point data to interpolate from.
        method : str
            The interpolation method ('idw', 'moving_average', 'gaussian', 'exponential')
        x_coord : str, optional
            Name of the x coordinate column/variable
        y_coord : str, optional
            Name of the y coordinate column/variable
        source_crs : str, CRS, optional
            The coordinate reference system of the source points
        chunk_size : int, optional
            Size of chunks for processing large datasets (default: 10000)
        **kwargs
            Additional keyword arguments:
            - n_neighbors: number of neighbors to use (default: min(8, len(points)))
            - power: power parameter for IDW (default: 2)
            - sigma: sigma parameter for Gaussian (default: std of distances)
            - scale: scale parameter for Exponential (default: 1.0)
        """
        self.method = method
        self.chunk_size = chunk_size if chunk_size is not None else 10000
        super().__init__(source_points, x_coord, y_coord, source_crs, **kwargs)
        
        # Validate method
        valid_methods = ['idw', 'moving_average', 'gaussian', 'exponential']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
        
        # Build spatial index based on coordinate system type
        self._build_spatial_index()
    
    def _build_spatial_index(self):
        """Build spatial index for neighbor search with appropriate metric."""
        # Create point array for spatial indexing
        self.points = np.column_stack([self.x_coords, self.y_coords])
        
        # Select appropriate spatial index based on coordinate system type
        if self.coord_system_type == 'geographic':
            # For geographic coordinates, use BallTree with haversine metric
            # Note: BallTree with haversine expects [lat, lon] format in radians
            points_rad = np.column_stack([np.radians(self.y_coords), np.radians(self.x_coords)])
            self.spatial_index = BallTree(points_rad, metric='haversine')
            self.is_geographic = True
        else:
            # For projected coordinates, use scipy's cKDTree for efficiency
            if HAS_SCIPY_SPATIAL and cKDTree is not None:
                self.spatial_index = cKDTree(self.points)
                self.is_geographic = False
            else:
                # Fallback to BallTree if cKDTree is not available
                points_rad = np.column_stack([np.radians(self.y_coords), np.radians(self.x_coords)])
                self.spatial_index = BallTree(points_rad, metric='haversine')
                self.is_geographic = True
                warnings.warn(
                    "scipy not available, using BallTree as fallback for projected coordinates. "
                    "This may affect performance.",
                    UserWarning
                )
    
    def interpolate_to(
        self,
        target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        target_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Interpolate from source points to target points using neighbor-based methods.
        
        Parameters
        ----------
        target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
            Target points to interpolate to.
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
        target_xs, target_ys = self._extract_target_coordinates(
            target_points, x_coord, y_coord
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
        
        # Prepare target points for interpolation
        target_points_array = np.column_stack([interp_target_xs, interp_target_ys])
        
        # Perform interpolation based on method with chunking for large datasets
        interpolated_results = {}
        
        for var_name, var_data in self.data_vars.items():
            # Process in chunks for memory efficiency
            if len(target_points_array) > self.chunk_size:
                interpolated_values = self._interpolate_in_chunks(
                    target_points_array, var_data, **kwargs
                )
            else:
                if self.method == 'idw':
                    interpolated_values = self._interpolate_idw(
                        target_points_array, var_data, **kwargs
                    )
                elif self.method == 'moving_average':
                    interpolated_values = self._interpolate_moving_average(
                        target_points_array, var_data, **kwargs
                    )
                elif self.method == 'gaussian':
                    interpolated_values = self._interpolate_gaussian(
                        target_points_array, var_data, **kwargs
                    )
                elif self.method == 'exponential':
                    interpolated_values = self._interpolate_exponential(
                        target_points_array, var_data, **kwargs
                    )
                else:
                    raise ValueError(f"Unsupported interpolation method: {self.method}")
            
            interpolated_results[var_name] = interpolated_values
        
        # Return appropriate format based on input type
        return self._format_output(target_points, target_xs, target_ys, interpolated_results)
    
    def _extract_target_coordinates(self, target_points, x_coord, y_coord):
        """Extract coordinates from target points."""
        if isinstance(target_points, np.ndarray):
            # Direct coordinate array format: (n, 2) with [x, y] for each point
            if target_points.ndim != 2 or target_points.shape[1] != 2:
                raise ValueError("Target coordinates array must have shape (n, 2) with [x, y] format")
            target_xs = target_points[:, 0]
            target_ys = target_points[:, 1]
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
        
        return target_xs, target_ys
    
    def _format_output(self, target_points, target_xs, target_ys, interpolated_results):
        """Format output based on input type."""
        if isinstance(target_points, xr.Dataset):
            # Create result as xarray Dataset
            result_coords = {self.y_coord: ('y', target_ys),
                           self.x_coord: ('x', target_xs)}
            result_vars = {}
            for var_name, var_values in interpolated_results.items():
                result_vars[var_name] = (['y'], var_values)  # Using 'y' dimension for 1D case
            return xr.Dataset(result_vars, coords=result_coords)
        elif isinstance(target_points, pd.DataFrame):
            # Create result as DataFrame
            result_df = pd.DataFrame({self.x_coord: target_xs, self.y_coord: target_ys})
            for var_name, var_values in interpolated_results.items():
                result_df[var_name] = var_values
            return result_df
        else:
            # Return as dictionary
            result_dict = {self.x_coord: target_xs, self.y_coord: target_ys}
            result_dict.update(interpolated_results)
            return result_dict
    
    def _interpolate_idw(self, target_points, data, **kwargs):
        """Perform Inverse Distance Weighting interpolation."""
        n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
        power = kwargs.get('power', 2)
        search_radius = kwargs.get('search_radius', None)
        
        # Find nearest neighbors for each target point
        if search_radius is not None:
            # Use radius-based search
            if hasattr(self.spatial_index, 'query_ball_point') and not self.is_geographic:
                indices = self.spatial_index.query_ball_point(
                    target_points, search_radius
                )
            else:
                # For BallTree, use query_radius
                target_points_rad = np.column_stack([
                    np.radians(target_points[:, 1]),  # lat in radians
                    np.radians(target_points[:, 0])   # lon in radians
                ])
                from pyproj import Geod
                geod = Geod(ellps='WGS84')
                radius_rad = search_radius / geod.a  # Convert meters to radians
                indices = self.spatial_index.query_radius(target_points_rad, radius_rad)
            
            # For each target point, calculate IDW
            interpolated_values = []
            for i, idx_list in enumerate(indices):
                if len(idx_list) == 0:
                    # No neighbors found, return NaN
                    interpolated_values.append(np.nan)
                else:
                    # Get actual distances to neighbors
                    actual_dists = []
                    for j in idx_list:
                        if not self.is_geographic:
                            dist = np.sqrt(
                                (target_points[i, 0] - self.points[j, 0])**2 +
                                (target_points[i, 1] - self.points[j, 1])**2
                            )
                        else:
                            # For geographic coordinates, compute great circle distance
                            lat1, lon1 = np.radians(target_points[i, 1]), np.radians(target_points[i, 0])
                            lat2, lon2 = np.radians(self.y_coords[j]), np.radians(self.x_coords[j])
                            dlat = lat2 - lat1
                            dlon = lon2 - lon1
                            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                            c = 2 * np.arcsin(np.sqrt(a))
                            dist = c * 637100  # Earth's radius in meters
                        actual_dists.append(dist)
                    actual_dists = np.array(actual_dists)
                    
                    # Avoid division by zero
                    actual_dists = np.maximum(actual_dists, 1e-10)
                    weights = 1.0 / (actual_dists ** power)
                    
                    # Calculate weighted average
                    neighbor_data = data[np.array(idx_list)]
                    weighted_sum = np.sum(weights * neighbor_data)
                    weight_sum = np.sum(weights)
                    interpolated_values.append(weighted_sum / weight_sum if weight_sum != 0 else np.nan)
            return np.array(interpolated_values)
        else:
            # Use k-nearest neighbors
            if hasattr(self.spatial_index, 'query'):
                distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
            else:
                # For geographic coordinates, transform target points
                target_points_rad = np.column_stack([
                    np.radians(target_points[:, 1]),  # lat in radians
                    np.radians(target_points[:, 0])   # lon in radians
                ])
                distances, indices = self.spatial_index.query(target_points_rad, k=n_neighbors)
                # Convert distances from radians to meters
                from pyproj import Geod
                geod = Geod(ellps='WGS84')
                distances = distances * geod.a  # Convert radians to meters
            
            # Calculate inverse distance weights
            distances = np.maximum(distances, 1e-10) # Avoid division by zero
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
    
    def _interpolate_in_chunks(self, target_points, data, **kwargs):
        """
        Process interpolation in chunks to handle large datasets efficiently.
        """
        n_targets = len(target_points)
        chunk_size = self.chunk_size if self.chunk_size is not None else 10000
        
        if self.method == 'idw':
            interpolate_func = self._interpolate_idw
        elif self.method == 'moving_average':
            interpolate_func = self._interpolate_moving_average
        elif self.method == 'gaussian':
            interpolate_func = self._interpolate_gaussian
        elif self.method == 'exponential':
            interpolate_func = self._interpolate_exponential
        else:
            raise ValueError(f"Unsupported interpolation method: {self.method}")
        
        results = []
        for start_idx in range(0, n_targets, chunk_size):
            end_idx = min(start_idx + chunk_size, n_targets)
            chunk_targets = target_points[start_idx:end_idx]
            chunk_result = interpolate_func(chunk_targets, data, **kwargs)
            results.append(chunk_result)
        
        return np.concatenate(results)
    
    def _interpolate_moving_average(self, target_points, data, **kwargs):
        """Perform Moving Average interpolation."""
        n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
        
        # Find k-nearest neighbors for each target point
        if hasattr(self.spatial_index, 'query'):
            distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
        else:
            # For geographic coordinates, transform target points
            target_points_rad = np.column_stack([
                np.radians(target_points[:, 1]),  # lat in radians
                np.radians(target_points[:, 0])   # lon in radians
            ])
            distances, indices = self.spatial_index.query(target_points_rad, k=n_neighbors)
        
        # Calculate simple average for each target point
        interpolated_values = []
        for i in range(len(target_points)):
            neighbor_data = data[indices[i, :]]
            interpolated_values.append(np.mean(neighbor_data))
        
        return np.array(interpolated_values)
    
    def _interpolate_gaussian(self, target_points, data, **kwargs):
        """Perform Gaussian interpolation."""
        n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
        sigma = kwargs.get('sigma', None)
        
        # Find k-nearest neighbors for each target point
        if HAS_SCIPY_SPATIAL and hasattr(self.spatial_index, 'query'):
            distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
        else:
            # For geographic coordinates, transform target points
            target_points_rad = np.column_stack([
                np.radians(target_points[:, 1]),  # lat in radians
                np.radians(target_points[:, 0])   # lon in radians
            ])
            distances, indices = self.spatial_index.query(target_points_rad, k=n_neighbors)
            # Convert distances from radians to meters
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            distances = distances * geod.a  # Convert radians to meters
        
        # If sigma not provided, use standard deviation of distances
        if sigma is None:
            # Use a heuristic: average distance to neighbors
            sigma = np.std(distances) if len(distances) > 1 and np.std(distances) > 0 else 1.0
            if sigma == 0:
                sigma = 1.0
        
        # Calculate Gaussian weights and weighted average for each target point
        interpolated_values = []
        for i in range(len(target_points)):
            dists = distances[i, :]
            weights = np.exp(-0.5 * (dists / sigma) ** 2)
            
            # Avoid division by zero
            weight_sum = np.sum(weights)
            if weight_sum == 0:
                interpolated_values.append(np.nan)
            else:
                weighted_sum = np.sum(weights * data[indices[i, :]])
                interpolated_values.append(weighted_sum / weight_sum)
        
        return np.array(interpolated_values)
    
    def _interpolate_exponential(self, target_points, data, **kwargs):
        """Perform Exponential interpolation."""
        n_neighbors = kwargs.get('n_neighbors', min(8, len(self.x_coords)))
        scale = kwargs.get('scale', 1.0)
        
        # Find k-nearest neighbors for each target point
        if HAS_SCIPY_SPATIAL and hasattr(self.spatial_index, 'query') and not self.is_geographic:
            distances, indices = self.spatial_index.query(target_points, k=n_neighbors)
        else:
            # For geographic coordinates, transform target points
            target_points_rad = np.column_stack([
                np.radians(target_points[:, 1]),  # lat in radians
                np.radians(target_points[:, 0])   # lon in radians
            ])
            distances, indices = self.spatial_index.query(target_points_rad, k=n_neighbors)
            # Convert distances from radians to meters
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            distances = distances * geod.a  # Convert radians to meters
        
        # Calculate exponential weights and weighted average for each target point
        interpolated_values = []
        for i in range(len(target_points)):
            dists = distances[i, :]
            weights = np.exp(-dists / scale)
            
            # Avoid division by zero
            weight_sum = np.sum(weights)
            if weight_sum == 0:
                interpolated_values.append(np.nan)
            else:
                weighted_sum = np.sum(weights * data[indices[i, :]])
                interpolated_values.append(weighted_sum / weight_sum)
        
        return np.array(interpolated_values)


class TriangulationBasedInterpolator(BaseScatteredInterpolator):
    """
    Triangulation-based linear interpolation using scipy.spatial.Delaunay.
    
    Performs linear barycentric interpolation within Delaunay triangles.
    """
    
    def __init__(
        self,
        source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        source_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ):
        """
        Initialize the triangulation-based interpolator.
        
        Parameters
        ----------
        source_points : pandas.DataFrame, xarray.Dataset, or dict
            The source scattered point data to interpolate from.
        x_coord : str, optional
            Name of the x coordinate column/variable
        y_coord : str, optional
            Name of the y coordinate column/variable
        source_crs : str, CRS, optional
            The coordinate reference system of the source points
        **kwargs
            Additional keyword arguments
        """
        super().__init__(source_points, x_coord, y_coord, source_crs, **kwargs)
        
        # Build Delaunay triangulation
        self._build_triangulation()
    
    def _build_triangulation(self):
        """Build Delaunay triangulation from source points."""
        # Create point array for triangulation
        self.points = np.column_stack([self.x_coords, self.y_coords])
        
        # Perform Delaunay triangulation
        if not HAS_SCIPY_SPATIAL or Delaunay is None:
            raise ImportError("Delaunay triangulation not available. scipy is required.")
        
        try:
            self.triangulation = Delaunay(self.points)
        except Exception as e:
            raise ValueError(f"Delaunay triangulation failed: {str(e)}")
    
    def interpolate_to(
        self,
        target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
        x_coord: Optional[str] = None,
        y_coord: Optional[str] = None,
        target_crs: Optional[Union[str, CRS]] = None,
        **kwargs
    ) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
        """
        Interpolate from source points to target points using triangulation-based methods.
        
        Parameters
        ----------
        target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
            Target points to interpolate to.
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
        target_xs, target_ys = self._extract_target_coordinates(
            target_points, x_coord, y_coord
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
        
        # Prepare target points for interpolation
        target_points_array = np.column_stack([interp_target_xs, interp_target_ys])
        
        # Perform triangulation-based interpolation
        interpolated_results = {}
        
        for var_name, var_data in self.data_vars.items():
            interpolated_values = self._interpolate_linear(
                target_points_array, var_data
            )
            interpolated_results[var_name] = interpolated_values
        
        # Return appropriate format based on input type
        return self._format_output(target_points, target_xs, target_ys, interpolated_results)
    
    def _extract_target_coordinates(self, target_points, x_coord, y_coord):
        """Extract coordinates from target points."""
        if isinstance(target_points, np.ndarray):
            # Direct coordinate array format: (n, 2) with [x, y] for each point
            if target_points.ndim != 2 or target_points.shape[1] != 2:
                raise ValueError("Target coordinates array must have shape (n, 2) with [x, y] format")
            target_xs = target_points[:, 0]
            target_ys = target_points[:, 1]
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
        
        return target_xs, target_ys
    
    def _format_output(self, target_points, target_xs, target_ys, interpolated_results):
        """Format output based on input type."""
        if isinstance(target_points, xr.Dataset):
            # Create result as xarray Dataset
            result_coords = {self.y_coord: ('y', target_ys),
                           self.x_coord: ('x', target_xs)}
            result_vars = {}
            for var_name, var_values in interpolated_results.items():
                result_vars[var_name] = (['y'], var_values)  # Using 'y' dimension for 1D case
            return xr.Dataset(result_vars, coords=result_coords)
        elif isinstance(target_points, pd.DataFrame):
            # Create result as DataFrame
            result_df = pd.DataFrame({self.x_coord: target_xs, self.y_coord: target_ys})
            for var_name, var_values in interpolated_results.items():
                result_df[var_name] = var_values
            return result_df
        else:
            # Return as dictionary
            result_dict = {self.x_coord: target_xs, self.y_coord: target_ys}
            result_dict.update(interpolated_results)
            return result_dict
    
    def _interpolate_linear(self, target_points, data):
        """Perform linear interpolation using Delaunay triangulation."""
        if not HAS_SCIPY_SPATIAL:
            raise ImportError("Linear interpolation not available. scipy is required.")
        
        try:
            from scipy.interpolate import LinearNDInterpolator
        except ImportError:
            raise ImportError("Linear interpolation not available. scipy is required.")
        
        # Create interpolator using the triangulation
        interpolator = LinearNDInterpolator(self.points, data)
        
        # Interpolate to target points
        interpolated_values = interpolator(target_points)
        
        return interpolated_values


class HybridSpatialIndex:
    """
    Hybrid spatial indexing system that automatically selects the appropriate
    backend based on coordinate system type:
    - scipy.spatial.cKDTree for projected data (Euclidean distance)
    - sklearn.neighbors.BallTree with metric='haversine' for geographic data
    """
    
    def __init__(self, x_coords, y_coords, crs: Optional[CRS] = None):
        """
        Initialize the hybrid spatial index.
        
        Parameters
        ----------
        x_coords : array-like
            X coordinate array (longitude or easting)
        y_coords : array-like
            Y coordinate array (latitude or northing)
        crs : CRS, optional
            Coordinate reference system
        """
        self.x_coords = np.asarray(x_coords)
        self.y_coords = np.asarray(y_coords)
        self.crs = crs
        
        # Determine coordinate system type
        self.crs_manager = CRSManager()
        if crs is not None:
            self.coord_system_type = self.crs_manager.detect_coordinate_system_type(crs)
        else:
            self.coord_system_type = "unknown"  # Will need to determine from data
        
        # Build the appropriate spatial index
        self._build_index()
    
    def _build_index(self):
        """Build the spatial index based on coordinate system type."""
        if self.coord_system_type == 'geographic':
            # For geographic coordinates, use BallTree with haversine metric
            # Haversine metric expects [lat, lon] in radians
            points_rad = np.column_stack([np.radians(self.y_coords), np.radians(self.x_coords)])
            self.spatial_index = BallTree(points_rad, metric='haversine')
            self.is_geographic = True
        else:
            # For projected coordinates, use cKDTree with Euclidean distance
            if HAS_SCIPY_SPATIAL and cKDTree is not None:
                self.spatial_index = cKDTree(np.column_stack([self.x_coords, self.y_coords]))
                self.is_geographic = False
            else:
                # Fallback to BallTree if cKDTree is not available
                points_rad = np.column_stack([np.radians(self.y_coords), np.radians(self.x_coords)])
                self.spatial_index = BallTree(points_rad, metric='haversine')
                self.is_geographic = True
                warnings.warn(
                    "scipy not available, using BallTree as fallback for projected coordinates. "
                    "This may affect performance.",
                    UserWarning
                )
    
    def query(self, target_points, k=1):
        """
        Query the spatial index for k nearest neighbors.
        
        Parameters
        ----------
        target_points : array-like
            Target points to query, shape (n, 2) with [x, y] or [lon, lat]
        k : int
            Number of nearest neighbors to find
        
        Returns
        -------
        distances : array
            Distances to k nearest neighbors
        indices : array
            Indices of k nearest neighbors
        """
        target_points = np.asarray(target_points)
        
        if self.is_geographic:
            # For geographic data, convert target points to radians
            target_points_rad = np.column_stack([
                np.radians(target_points[:, 1]),  # lat in radians
                np.radians(target_points[:, 0])   # lon in radians
            ])
            distances, indices = self.spatial_index.query(target_points_rad, k=k)
            # Convert distances from radians to actual distance (in the same units as Earth's radius)
            # By default, BallTree with haversine returns distances in radians
            # Multiply by Earth's radius to get distance in the same units as the radius (typically km)
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            distances = distances * geod.a  # Convert radians to meters
        else:
            # For projected data, use Euclidean distance directly
            if hasattr(self.spatial_index, 'query'):
                target_points_xy = np.column_stack([target_points[:, 0], target_points[:, 1]])
                distances, indices = self.spatial_index.query(target_points_xy, k=k)
            else:
                # Fallback for BallTree if needed
                target_points_rad = np.column_stack([
                    np.radians(target_points[:, 1]),  # lat in radians
                    np.radians(target_points[:, 0])   # lon in radians
                ])
                distances, indices = self.spatial_index.query(target_points_rad, k=k)
                from pyproj import Geod
                geod = Geod(ellps='WGS84')
                distances = distances * geod.a  # Convert radians to meters
        
        return distances, indices
    
    def query_radius(self, target_points, radius):
        """
        Query the spatial index for neighbors within a radius.
        
        Parameters
        ----------
        target_points : array-like
            Target points to query, shape (n, 2) with [x, y] or [lon, lat]
        radius : float
            Search radius
        
        Returns
        -------
        indices : list of arrays
            Indices of neighbors within radius for each target point
        """
        target_points = np.asarray(target_points)
        
        if self.is_geographic:
            # For geographic data, convert radius to radians
            from pyproj import Geod
            geod = Geod(ellps='WGS84')
            radius_rad = radius / geod.a  # Convert meters to radians
            target_points_rad = np.column_stack([
                np.radians(target_points[:, 1]),  # lat in radians
                np.radians(target_points[:, 0])   # lon in radians
            ])
            indices = self.spatial_index.query_radius(target_points_rad, radius_rad)
        else:
            # For projected data, use radius directly
            if hasattr(self.spatial_index, 'query_ball_point') and not self.is_geographic:
                target_points_xy = np.column_stack([target_points[:, 0], target_points[:, 1]])
                indices = self.spatial_index.query_ball_point(target_points_xy, radius)
            else:
                # For geographic coordinates or when query_ball_point is not available, use query_radius
                target_points_rad = np.column_stack([
                    np.radians(target_points[:, 1]),  # lat in radians
                    np.radians(target_points[:, 0])   # lon in radians
                ])
                from pyproj import Geod
                geod = Geod(ellps='WGS84')
                radius_rad = radius / geod.a  # Convert meters to radians
                indices = self.spatial_index.query_radius(target_points_rad, radius_rad)
        
        # Convert numpy array to list to match expected interface
        return indices.tolist()


# Convenience functions for common interpolation methods
def idw_interpolation(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[Union[str, CRS]] = None,
    target_crs: Optional[Union[str, CRS]] = None,
    **kwargs
) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function for Inverse Distance Weighting interpolation.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        Source scattered point data
    target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
        Target points to interpolate to
    x_coord, y_coord : str, optional
        Coordinate names
    source_crs, target_crs : str, CRS, optional
        Coordinate reference systems
    **kwargs
        Additional interpolation parameters (n_neighbors, power, etc.)
    
    Returns
    -------
    Interpolated data at target points
    """
    interpolator = NeighborBasedInterpolator(
        source_points, method='idw', x_coord=x_coord, y_coord=y_coord, 
        source_crs=source_crs, **kwargs
    )
    return interpolator.interpolate_to(
        target_points, x_coord=x_coord, y_coord=y_coord, 
        target_crs=target_crs, **kwargs
    )


def moving_average_interpolation(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[Union[str, CRS]] = None,
    target_crs: Optional[Union[str, CRS]] = None,
    **kwargs
) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function for Moving Average interpolation.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        Source scattered point data
    target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
        Target points to interpolate to
    x_coord, y_coord : str, optional
        Coordinate names
    source_crs, target_crs : str, CRS, optional
        Coordinate reference systems
    **kwargs
        Additional interpolation parameters (n_neighbors, etc.)
    
    Returns
    -------
    Interpolated data at target points
    """
    interpolator = NeighborBasedInterpolator(
        source_points, method='moving_average', x_coord=x_coord, y_coord=y_coord, 
        source_crs=source_crs, **kwargs
    )
    return interpolator.interpolate_to(
        target_points, x_coord=x_coord, y_coord=y_coord, 
        target_crs=target_crs, **kwargs
    )


def gaussian_interpolation(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[Union[str, CRS]] = None,
    target_crs: Optional[Union[str, CRS]] = None,
    **kwargs
) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function for Gaussian interpolation.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        Source scattered point data
    target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
        Target points to interpolate to
    x_coord, y_coord : str, optional
        Coordinate names
    source_crs, target_crs : str, CRS, optional
        Coordinate reference systems
    **kwargs
        Additional interpolation parameters (n_neighbors, sigma, etc.)
    
    Returns
    -------
    Interpolated data at target points
    """
    interpolator = NeighborBasedInterpolator(
        source_points, method='gaussian', x_coord=x_coord, y_coord=y_coord, 
        source_crs=source_crs, **kwargs
    )
    return interpolator.interpolate_to(
        target_points, x_coord=x_coord, y_coord=y_coord, 
        target_crs=target_crs, **kwargs
    )


def exponential_interpolation(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[Union[str, CRS]] = None,
    target_crs: Optional[Union[str, CRS]] = None,
    **kwargs
) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function for Exponential interpolation.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        Source scattered point data
    target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
        Target points to interpolate to
    x_coord, y_coord : str, optional
        Coordinate names
    source_crs, target_crs : str, CRS, optional
        Coordinate reference systems
    **kwargs
        Additional interpolation parameters (n_neighbors, scale, etc.)
    
    Returns
    -------
    Interpolated data at target points
    """
    interpolator = NeighborBasedInterpolator(
        source_points, method='exponential', x_coord=x_coord, y_coord=y_coord, 
        source_crs=source_crs, **kwargs
    )
    return interpolator.interpolate_to(
        target_points, x_coord=x_coord, y_coord=y_coord, 
        target_crs=target_crs, **kwargs
    )


def linear_interpolation(
    source_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray]],
    target_points: Union[pd.DataFrame, xr.Dataset, Dict[str, np.ndarray], np.ndarray],
    x_coord: Optional[str] = None,
    y_coord: Optional[str] = None,
    source_crs: Optional[Union[str, CRS]] = None,
    target_crs: Optional[Union[str, CRS]] = None,
    **kwargs
) -> Union[xr.Dataset, pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Convenience function for triangulation-based linear interpolation.
    
    Parameters
    ----------
    source_points : pandas.DataFrame, xarray.Dataset, or dict
        Source scattered point data
    target_points : pandas.DataFrame, xarray.Dataset, dict, or np.ndarray
        Target points to interpolate to
    x_coord, y_coord : str, optional
        Coordinate names
    source_crs, target_crs : str, CRS, optional
        Coordinate reference systems
    **kwargs
        Additional interpolation parameters
    
    Returns
    -------
    Interpolated data at target points
    """
    interpolator = TriangulationBasedInterpolator(
        source_points, x_coord=x_coord, y_coord=y_coord, 
        source_crs=source_crs, **kwargs
    )
    return interpolator.interpolate_to(
        target_points, x_coord=x_coord, y_coord=y_coord, 
        target_crs=target_crs, **kwargs
    )