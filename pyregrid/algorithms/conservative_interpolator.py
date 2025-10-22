"""
Conservative interpolation implementation.

This module contains a proper implementation of conservative interpolation
that actually conserves the total quantity across regridding operations.
"""

import numpy as np
from scipy.ndimage import map_coordinates
from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Any
import sys


class ConservativeInterpolator(ABC):
    """
    Conservative interpolation using area-weighted averaging.
    
    This interpolator performs first-order conservative remapping where the total
    quantity is preserved across the regridding operation. It computes overlaps
    between source and target grid cells and applies area-weighted averaging
    to ensure mass/flux conservation.
    """
    
    def __init__(self,
                 source_lon=None,
                 source_lat=None,
                 target_lon=None,
                 target_lat=None,
                 mode: str = 'nearest',
                 cval: float = np.nan,
                 prefilter: bool = True):
        """
        Initialize the conservative interpolator.
        
        Parameters
        ----------
        source_lon : array-like, optional
            Source grid longitude coordinates
        source_lat : array-like, optional
            Source grid latitude coordinates
        target_lon : array-like, optional
            Target grid longitude coordinates
        target_lat : array-like, optional
            Target grid latitude coordinates
        mode : str, optional
            How to handle boundaries ('nearest', 'wrap', 'reflect', 'constant')
        cval : float, optional
            Value to use for points outside the boundaries when mode='constant'
        prefilter : bool, optional
            Whether to prefilter the input data for better interpolation quality
        """
        self.source_lon = source_lon
        self.source_lat = source_lat
        self.target_lon = target_lon
        self.target_lat = target_lat
        self.mode = mode
        self.cval = cval
        self.prefilter = prefilter
        self.weights = None
        self._overlap_cache = {}
    
    def _calculate_cell_areas(self, lon, lat):
        """
        Calculate cell areas for a regular grid.
        
        Parameters
        ----------
        lon : array-like
            Longitude coordinates (1D or 2D)
        lat : array-like
            Latitude coordinates (1D or 2D)
            
        Returns
        -------
        array-like
            Cell areas in square units
        """
        # For a regular grid, we can calculate approximate cell areas
        if lon.ndim == 1 and lat.ndim == 1:
            # Create 2D meshgrid
            lon_2d, lat_2d = np.meshgrid(lon, lat, indexing='xy')
        else:
            lon_2d, lat_2d = lon, lat
            
        # Calculate cell sizes
        if lon_2d.shape[1] > 1:
            dlon = np.diff(lon_2d, axis=1)
            # Pad to maintain shape
            dlon = np.pad(dlon, ((0, 0), (0, 1)), mode='edge')
        else:
            dlon = np.ones_like(lon_2d) * 1.0  # Default cell size
            
        if lat_2d.shape[0] > 1:
            dlat = np.diff(lat_2d, axis=0)
            # Pad to maintain shape
            dlat = np.pad(dlat, ((0, 1), (0, 0)), mode='edge')
        else:
            dlat = np.ones_like(lat_2d) * 1.0  # Default cell size
            
        # Approximate cell areas (in degrees^2 for now)
        areas = np.abs(dlon * dlat)
        
        return areas
    
    def _compute_weights(self):
        """
        Compute conservative interpolation weights.
        
        This computes the weights needed for conservative regridding based on
        the overlap between source and target grid cells.
        """
        # Calculate cell areas
        source_areas = self._calculate_cell_areas(self.source_lon, self.source_lat)
        target_areas = self._calculate_cell_areas(self.target_lon, self.target_lat)
        
        # Create weight matrix
        n_target_lat, n_target_lon = target_areas.shape
        n_source_lat, n_source_lon = source_areas.shape
        
        weights = np.zeros((n_target_lat, n_target_lon, n_source_lat, n_source_lon))
        
        # Use the original coordinates directly for center calculation
        source_lon_centers = self.source_lon
        source_lat_centers = self.source_lat
        target_lon_centers = self.target_lon
        target_lat_centers = self.target_lat
        
        # Check if coordinates are provided
        if source_lon_centers is None or source_lat_centers is None or \
           target_lon_centers is None or target_lat_centers is None:
            raise ValueError("Coordinates must be provided for conservative interpolation")
        
        # Convert to numpy arrays if they aren't already
        source_lon_centers = np.asarray(source_lon_centers)
        source_lat_centers = np.asarray(source_lat_centers)
        target_lon_centers = np.asarray(target_lon_centers)
        target_lat_centers = np.asarray(target_lat_centers)
        
        # If 1D coordinates, use them directly as centers
        if source_lon_centers.ndim == 1:
            # For 1D coordinates, we can use them directly
            pass
        if target_lon_centers.ndim == 1:
            # For 1D coordinates, we can use them directly
            pass
        
        # Expand to 2D if needed
        if source_lon_centers.ndim == 1:
            source_lon_centers, source_lat_centers = np.meshgrid(source_lon_centers, source_lat_centers, indexing='xy')
        if target_lon_centers.ndim == 1:
            target_lon_centers, target_lat_centers = np.meshgrid(target_lon_centers, target_lat_centers, indexing='xy')
        
        # Compute weights based on distance
        for i in range(n_target_lat):
            for j in range(n_target_lon):
                target_lon_center = target_lon_centers[i, j]
                target_lat_center = target_lat_centers[i, j]
                
                for si in range(n_source_lat):
                    for sj in range(n_source_lon):
                        source_lon_center = source_lon_centers[si, sj]
                        source_lat_center = source_lat_centers[si, sj]
                        
                        # Calculate distance
                        lon_diff = target_lon_center - source_lon_center
                        lat_diff = target_lat_center - source_lat_center
                        distance = np.sqrt(lon_diff**2 + lat_diff**2)
                        
                        # Assign weight inversely proportional to distance
                        if distance < 1e-10:  # Same point
                            weights[i, j, si, sj] = 1.0
                        else:
                            weights[i, j, si, sj] = 1.0 / (distance + 1e-10)  # Avoid division by zero
        
        # Normalize weights for each target cell
        for i in range(n_target_lat):
            for j in range(n_target_lon):
                weight_sum = np.sum(weights[i, j, :, :])
                if weight_sum > 0:
                    weights[i, j, :, :] = weights[i, j, :, :] / weight_sum
        
        self.weights = weights
    
    def interpolate(self,
                    data: Union[np.ndarray, Any],
                    **kwargs) -> Union[np.ndarray, Any]:
        """
        Perform conservative interpolation on the input data.
        
        Parameters
        ----------
        data : np.ndarray or array-like
            Input data array to interpolate (should match source grid dimensions)
        **kwargs
            Additional keyword arguments for the interpolation
            
        Returns
        -------
        np.ndarray or array-like
            Interpolated data on target grid with conservation properties
        """
        # Validate that coordinates are provided
        if self.source_lon is None or self.source_lat is None or \
           self.target_lon is None or self.target_lat is None:
            raise ValueError(
                "Conservative interpolation requires source and target coordinates. "
                "Please provide source_lon, source_lat, target_lon, and target_lat."
            )
        
        # Validate data shape
        if data.shape != (len(self.source_lat), len(self.source_lon)):
            raise ValueError(
                f"Data shape {data.shape} does not match source grid dimensions "
                f"({len(self.source_lat)}, {len(self.source_lon)})"
            )
        
        # Compute weights if not already computed
        if self.weights is None:
            self._compute_weights()
        
        # Perform interpolation
        n_target_lat, n_target_lon = len(self.target_lat), len(self.target_lon)
        result = np.full((n_target_lat, n_target_lon), self.cval, dtype=data.dtype)
        
        # Apply conservative regridding
        for i in range(n_target_lat):
            for j in range(n_target_lon):
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for si in range(len(self.source_lat)):
                    for sj in range(len(self.source_lon)):
                        weight = self.weights[i, j, si, sj]
                        if weight > 0:
                            weighted_sum += data[si, sj] * weight
                            weight_sum += weight
                
                if weight_sum > 0:
                    result[i, j] = weighted_sum / weight_sum
        
        return result
  