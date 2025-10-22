"""
Tests for core module functionality.
This module contains comprehensive tests for the core GridRegridder class.
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

from pyregrid.core import GridRegridder


class TestGridRegridder:
    """Test GridRegridder class."""
    
    def test_initialization(self):
        """Test GridRegridder initialization."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        assert regridder is not None
        assert regridder.method == 'bilinear'
    
    def test_initialization_with_invalid_method(self):
        """Test GridRegridder initialization with invalid method."""
        # Create source and target grids
        source_lon = np.linspace(-10, 5, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        with pytest.raises(ValueError, match="Method must be one of"):
            GridRegridder(
                source_grid=source_ds,
                target_grid=target_ds,
                method='invalid_method'
            )
    
    def test_regrid_method(self):
        """Test the regrid method."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        result = regridder.regrid(source_ds)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert result['temperature'].shape == (8, 10)  # Should match target grid
    
    def test_regrid_method_with_multiple_variables(self):
        """Test the regrid method with multiple variables."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_temp = np.random.random((4, 5))
        source_humidity = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {
                'temperature': (['lat', 'lon'], source_temp),
                'humidity': (['lat', 'lon'], source_humidity)
            },
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {
                'temperature': (['lat', 'lon'], target_data),
                'humidity': (['lat', 'lon'], target_data)
            },
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        result = regridder.regrid(source_ds)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'humidity' in result.data_vars
        assert result['temperature'].shape == (8, 10)
        assert result['humidity'].shape == (8, 10)
    
    def test_regrid_with_different_methods(self):
        """Test regrid with different interpolation methods."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Test bilinear method
        regridder_bilinear = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        result_bilinear = regridder_bilinear.regrid(source_ds)
        assert isinstance(result_bilinear, xr.Dataset)
        assert 'temperature' in result_bilinear.data_vars
        
        # Test nearest method
        regridder_nearest = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='nearest'
        )
        result_nearest = regridder_nearest.regrid(source_ds)
        assert isinstance(result_nearest, xr.Dataset)
        assert 'temperature' in result_nearest.data_vars
    
    def test_regrid_with_weights(self):
        """Test regrid with precomputed weights."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Initialize regridder to compute weights
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        # Verify weights are computed
        assert regridder.weights is not None
    
    def test_regrid_with_crs_transformation(self):
        """Test regrid with CRS transformation."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear',
            source_crs='EPSG:4326',
            target_crs='EPSG:4326'
        )
        
        result = regridder.regrid(source_ds)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert result['temperature'].shape == (8, 10)


class TestGridRegridderEdgeCases:
    """Test GridRegridder with edge cases."""
    
    def test_empty_grids(self):
        """Test regrid with empty grids."""
        # Create empty grids
        source_lon = np.array([])
        source_lat = np.array([])
        target_lon = np.array([])
        target_lat = np.array([])
        
        source_data = np.array([]).reshape(0, 0)
        target_data = np.array([]).reshape(0, 0)
        
        with pytest.raises(ValueError):
            source_ds = xr.Dataset(
                {'temperature': (['lat', 'lon'], source_data)},
                coords={'lon': source_lon, 'lat': source_lat}
            )
            target_ds = xr.Dataset(
                {'temperature': (['lat', 'lon'], target_data)},
                coords={'lon': target_lon, 'lat': target_lat}
            )
            
            GridRegridder(
                source_grid=source_ds,
                target_grid=target_ds,
                method='bilinear'
            )
    
    def test_single_point_grids(self):
        """Test regrid with single point grids."""
        # Create single point grids
        source_lon = np.array([0.0])
        source_lat = np.array([0.0])
        target_lon = np.array([0.5])
        target_lat = np.array([0.5])
        
        source_data = np.array([[20.0]])  # 20 degrees at single point
        target_data = np.array([[0.0]])
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='nearest'
        )
        
        result = regridder.regrid(source_ds)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert result['temperature'].shape == (1, 1)
    
    def test_nan_values_in_source(self):
        """Test regrid with NaN values in source data."""
        # Create source and target grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        # Create source data with NaN values
        source_data = np.random.random((4, 5))
        source_data[0, 0] = np.nan  # Add NaN value
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        regridder = GridRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        result = regridder.regrid(source_ds)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        # Result may contain NaN values where appropriate