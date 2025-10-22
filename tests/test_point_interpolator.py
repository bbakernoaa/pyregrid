"""
Tests for point interpolation functionality.
This module contains comprehensive tests for the PointInterpolator class.
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

from pyregrid.point_interpolator import PointInterpolator


class TestPointInterpolator:
    """Test PointInterpolator class."""
    
    def test_initialization(self):
        """Test PointInterpolator initialization."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30],
            'humidity': [50, 60, 70]
        })
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert interpolator.method == 'idw'
        assert interpolator.x_coord == 'longitude'
        assert interpolator.y_coord == 'latitude'
        assert len(interpolator.source_points) == 3
    
    def test_initialization_with_xarray(self):
        """Test PointInterpolator initialization with xarray Dataset."""
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
        
        interpolator = PointInterpolator(
            source_points=ds,
            method='idw',
            x_coord='lon',
            y_coord='lat'
        )
        
        assert interpolator.method == 'idw'
        assert interpolator.x_coord == 'lon'
        assert interpolator.y_coord == 'lat'
        assert len(interpolator.source_points['lon']) == 3
    
    def test_interpolate_to_grid(self):
        """Test interpolation to grid functionality."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target grid
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], np.linspace(-1, 1, 3)),
                'lat': (['lat'], np.linspace(-0.1, 0.1, 2))
            }
        )
        
        result = interpolator.interpolate_to_grid(target_grid)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert result['temperature'].shape == (2, 3)  # lat, lon dimensions
    
    def test_method_validation(self):
        """Test that invalid methods raise error."""
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30]
        })
        
        with pytest.raises(ValueError, match="Method must be one of"):
            PointInterpolator(
                source_points=df,
                method='invalid_method',
                x_coord='longitude',
                y_coord='latitude'
            )
    
    def test_interpolation_with_different_methods(self):
        """Test interpolation with different methods."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        # Test with IDW method
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], np.linspace(-0.5, 0.5, 2)),
                'lat': (['lat'], np.linspace(-0.1, 0.1, 2))
            }
        )
        
        result = interpolator.interpolate_to_grid(target_grid)
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
    
    def test_interpolation_with_nan_values(self):
        """Test interpolation with NaN values in source data."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, np.nan, 30]
        })
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], np.linspace(-0.5, 0.5, 2)),
                'lat': (['lat'], np.linspace(-0.1, 0.1, 2))
            }
        )
        
        result = interpolator.interpolate_to_grid(target_grid)
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars




class TestPointInterpolatorEdgeCases:
    """Test PointInterpolator with edge cases."""
    
    def test_empty_data(self):
        """Test interpolation with empty data."""
        df = pd.DataFrame({
            'longitude': [],
            'latitude': [],
            'temperature': []
        })
        
        with pytest.raises(ValueError):
            PointInterpolator(
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
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], np.array([0.5])),
                'lat': (['lat'], np.array([0.5]))
            }
        )
        
        result = interpolator.interpolate_to_grid(target_grid)
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
    
    def test_duplicate_points(self):
        """Test interpolation with duplicate points."""
        df = pd.DataFrame({
            'longitude': [0, 0, 1],  # duplicate longitude
            'latitude': [0, 0, 1],  # duplicate latitude
            'temperature': [20, 25, 30]  # different values for duplicate points
        })
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        target_grid = xr.Dataset(
            coords={
                'lon': (['lon'], np.array([0.5])),
                'lat': (['lat'], np.array([0.5]))
            }
        )
        
        # This should handle duplicates appropriately
        result = interpolator.interpolate_to_grid(target_grid)
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
    
    def test_interpolate_to_grid(self):
        """Test the interpolate_to_grid method."""
        df = pd.DataFrame({
            'longitude': [-1, 0, 1],
            'latitude': [0, 0, 0],
            'temperature': [10, 20, 30]
        })
        
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Create target grid
        target_grid = xr.Dataset(
            coords={
                'longitude': (['longitude'], np.linspace(-1, 1, 3)),
                'latitude': (['latitude'], np.linspace(-0.1, 0.1, 2))
            }
        )
        
        result = interpolator.interpolate_to_grid(target_grid)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert result['temperature'].shape == (2, 3)  # lat, lon dimensions