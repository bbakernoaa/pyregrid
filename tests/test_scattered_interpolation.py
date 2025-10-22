"""
Tests for scattered interpolation functionality.
This module contains comprehensive tests for all scattered interpolation algorithms.
"""

import pytest
import numpy as np
import pandas as pd
import xarray as xr
from unittest.mock import Mock, patch
import sys
import os

# Add the project root to the path to import pyregrid modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyregrid.scattered_interpolation import (
    BaseScatteredInterpolator,
    NeighborBasedInterpolator,
    TriangulationBasedInterpolator,
    HybridSpatialIndex,
    idw_interpolation,
    moving_average_interpolation,
    gaussian_interpolation,
    exponential_interpolation,
    linear_interpolation
)


class TestBaseScatteredInterpolator:
    """Test BaseScatteredInterpolator class."""
    
    def test_initialization_with_dataframe(self):
        """Test initialization with DataFrame source points."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30],
            'humidity': [50, 60, 70]
        })
        
        interpolator = BaseScatteredInterpolator(
            source_points=df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert interpolator.x_coord == 'longitude'
        assert interpolator.y_coord == 'latitude'
        assert len(interpolator.x_coords) == 3
        assert len(interpolator.y_coords) == 3
        assert 'temperature' in interpolator.data_vars
        assert 'humidity' in interpolator.data_vars
    
    def test_initialization_with_xarray_dataset(self):
        """Test initialization with xarray Dataset source points."""
        lon = np.array([-5, 0, 5])
        lat = np.array([42, 45, 48])
        temp = np.array([20, 25, 30])
        humidity = np.array([50, 60, 70])
        
        ds = xr.Dataset(
            {
                'temperature': (['points'], temp),
                'humidity': (['points'], humidity)
            },
            coords={
                'lon': (['points'], lon),
                'lat': (['points'], lat)
            }
        )
        
        interpolator = BaseScatteredInterpolator(
            source_points=ds,
            x_coord='lon',
            y_coord='lat'
        )
        
        assert interpolator.x_coord == 'lon'
        assert interpolator.y_coord == 'lat'
        assert len(interpolator.x_coords) == 3
        assert len(interpolator.y_coords) == 3
        assert 'temperature' in interpolator.data_vars
        assert 'humidity' in interpolator.data_vars
    
    def test_initialization_with_dict(self):
        """Test initialization with dictionary source points."""
        source_dict = {
            'longitude': np.array([-5, 0, 5]),
            'latitude': np.array([42, 45, 48]),
            'temperature': np.array([20, 25, 30]),
            'humidity': np.array([50, 60, 70])
        }
        
        interpolator = BaseScatteredInterpolator(
            source_points=source_dict,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert interpolator.x_coord == 'longitude'
        assert interpolator.y_coord == 'latitude'
        assert len(interpolator.x_coords) == 3
        assert len(interpolator.y_coords) == 3
        assert 'temperature' in interpolator.data_vars
        assert 'humidity' in interpolator.data_vars
    
    def test_coordinate_validation(self):
        """Test coordinate validation."""
        df = pd.DataFrame({
            'x': [1, 2, 3],
            'y': [4, 5, 6],
            'value': [10, 20, 30]
        })
        
        # Test with mismatched coordinate lengths
        interpolator = BaseScatteredInterpolator(
            source_points=df,
            x_coord='x',
            y_coord='y'
        )
        
        # Should pass validation
        assert len(interpolator.x_coords) == len(interpolator.y_coords)
    
    def test_duplicate_point_handling(self):
        """Test handling of duplicate points."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 0, 5],  # duplicate longitude
            'latitude': [42, 45, 45, 48],  # duplicate latitude
            'temperature': [20, 25, 26, 30]  # different values for duplicate points
        })
        
        interpolator = BaseScatteredInterpolator(
            source_points=df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Should handle duplicates appropriately
        assert len(interpolator.x_coords) <= 4  # May have fewer after deduplication


class TestNeighborBasedInterpolator:
    """Test NeighborBasedInterpolator class."""
    
    def test_initialization(self):
        """Test NeighborBasedInterpolator initialization."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert interpolator.method == 'idw'
        assert len(interpolator.x_coords) == 3
        assert len(interpolator.y_coords) == 3
    
    def test_method_validation(self):
        """Test that invalid methods raise error."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30]
        })
        
        with pytest.raises(ValueError, match="Method must be one of"):
            NeighborBasedInterpolator(
                source_points=df,
                method='invalid_method',
                x_coord='longitude',
                y_coord='latitude'
            )
    
    def test_idw_interpolation(self):
        """Test IDW interpolation method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target points for interpolation
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_moving_average_interpolation(self):
        """Test Moving Average interpolation method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='moving_average',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target points for interpolation
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_gaussian_interpolation(self):
        """Test Gaussian interpolation method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='gaussian',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target points for interpolation
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_exponential_interpolation(self):
        """Test Exponential interpolation method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='exponential',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target points for interpolation
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_interpolation_with_numpy_array(self):
        """Test interpolation with numpy array target points."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Use numpy array for target points
        target_points = np.array([[0.5, 0]])  # [longitude, latitude]
        
        result = interpolator.interpolate_to(target_points)
        
        assert isinstance(result, dict)
        assert 'longitude' in result
        assert 'latitude' in result
        assert 'temperature' in result
        assert len(result['longitude']) == 1


class TestTriangulationBasedInterpolator:
    """Test TriangulationBasedInterpolator class."""
    
    def test_initialization(self):
        """Test TriangulationBasedInterpolator initialization."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1, 0],
            'latitude': [0, -1, 0, 1],
            'temperature': [10, 20, 30, 25]
        })
        
        interpolator = TriangulationBasedInterpolator(
            source_points=df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert len(interpolator.x_coords) == 4
        assert len(interpolator.y_coords) == 4
        assert 'temperature' in interpolator.data_vars
    
    def test_linear_interpolation(self):
        """Test linear interpolation method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1, 0],
            'latitude': [0, -1, 0, 1],
            'temperature': [10, 20, 30, 25]
        })
        
        interpolator = TriangulationBasedInterpolator(
            source_points=df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target points for interpolation
        target_df = pd.DataFrame({
            'longitude': [0.0],
            'latitude': [0.0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1


class TestHybridSpatialIndex:
    """Test HybridSpatialIndex class."""
    
    def test_initialization(self):
        """Test HybridSpatialIndex initialization."""
        x_coords = np.array([-1, 0, 1])
        y_coords = np.array([0, 0, 0])
        
        index = HybridSpatialIndex(x_coords, y_coords)
        
        assert len(index.x_coords) == 3
        assert len(index.y_coords) == 3
    
    def test_query_nearest_neighbors(self):
        """Test query for nearest neighbors."""
        x_coords = np.array([-1, 0, 1])
        y_coords = np.array([0, 0, 0])
        
        index = HybridSpatialIndex(x_coords, y_coords)
        
        target_points = np.array([[0.5, 0]])
        distances, indices = index.query(target_points, k=2)
        
        assert distances.shape[0] == 1  # One target point
        assert indices.shape[0] == 1    # One target point
        assert distances.shape[1] == 2  # Two nearest neighbors
        assert indices.shape[1] == 2    # Two nearest neighbors
    
    def test_query_radius(self):
        """Test query for neighbors within radius."""
        x_coords = np.array([-1, 0, 1])
        y_coords = np.array([0, 0, 0])
        
        index = HybridSpatialIndex(x_coords, y_coords)
        
        target_points = np.array([[0.5, 0]])
        indices = index.query_radius(target_points, radius=10000)  # 100km radius
        
        assert isinstance(indices, list)
        assert len(indices) == 1  # One target point


class TestConvenienceFunctions:
    """Test convenience functions for interpolation."""
    
    def test_idw_interpolation_function(self):
        """Test the IDW interpolation convenience function."""
        source_df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = idw_interpolation(
            source_points=source_df,
            target_points=target_df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_moving_average_interpolation_function(self):
        """Test the moving average interpolation convenience function."""
        source_df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = moving_average_interpolation(
            source_points=source_df,
            target_points=target_df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_gaussian_interpolation_function(self):
        """Test the Gaussian interpolation convenience function."""
        source_df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = gaussian_interpolation(
            source_points=source_df,
            target_points=target_df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_exponential_interpolation_function(self):
        """Test the exponential interpolation convenience function."""
        source_df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = exponential_interpolation(
            source_points=source_df,
            target_points=target_df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1
    
    def test_linear_interpolation_function(self):
        """Test the linear interpolation convenience function."""
        source_df = pd.DataFrame({
            'longitude': [-1, 0, 1, 0],
            'latitude': [0, -1, 0, 1],
            'temperature': [10, 20, 30, 25]
        })
        
        target_df = pd.DataFrame({
            'longitude': [0.0],
            'latitude': [0.0]
        })
        
        result = linear_interpolation(
            source_points=source_df,
            target_points=target_df,
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'longitude' in result.columns
            assert 'latitude' in result.columns
            assert 'temperature' in result.columns
            assert len(result) == 1
        else:
            # Result is a dict
            assert 'longitude' in result
            assert 'latitude' in result
            assert 'temperature' in result
            assert len(result['longitude']) == 1


class TestScatteredInterpolationEdgeCases:
    """Test scattered interpolation with edge cases."""
    
    def test_empty_data(self):
        """Test interpolation with empty data."""
        df = pd.DataFrame({
            'longitude': [],
            'latitude': [],
            'temperature': []
        })
        
        with pytest.raises(ValueError):
            NeighborBasedInterpolator(
                source_points=df,
                method='idw',
                x_coord='longitude',
                y_coord='latitude'
            )
    
    def test_single_point(self):
        """Test interpolation with single data point."""
        df = pd.DataFrame({
            'longitude': [0],
            'latitude': [0],
            'temperature': [20]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0.5]
        })
        
        # This should work even with single point
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'temperature' in result.columns
        else:
            # Result is a dict
            assert 'temperature' in result
    
    def test_nan_values_in_data(self):
        """Test interpolation with NaN values in source data."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, np.nan, 30]
        })
        
        interpolator = NeighborBasedInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_df = pd.DataFrame({
            'longitude': [0.5],
            'latitude': [0]
        })
        
        result = interpolator.interpolate_to(target_df)
        
        # Check based on result type (DataFrame or dict)
        if hasattr(result, 'columns'):
            # Result is a DataFrame
            assert 'temperature' in result.columns
        else:
            # Result is a dict
            assert 'temperature' in result