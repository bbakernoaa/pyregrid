
"""
Comprehensive integration tests for pyregrid data processing workflows.

This test suite verifies end-to-end functionality including:
- xarray accessor integration
- CRS transformation workflows  
- Dask integration with core functionality
- Memory management with large datasets
- Cross-module interactions
- Real-world usage scenarios
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
import dask.array as da
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path

# Import pyregrid modules
import pyregrid
from pyregrid import GridRegridder, PointInterpolator
from pyregrid.crs.crs_manager import CRSManager
from pyregrid.dask.dask_regridder import DaskRegridder
from pyregrid.dask.memory_management import MemoryManager
from pyregrid.dask.chunking import ChunkingStrategy
from pyregrid.algorithms.interpolators import BilinearInterpolator, CubicInterpolator, NearestInterpolator
from pyregrid.point_interpolator import PointInterpolator as PointInterpolatorImpl
from pyregrid.scattered_interpolation import (
    NeighborBasedInterpolator, 
    TriangulationBasedInterpolator,
    idw_interpolation,
    linear_interpolation
)


class TestXarrayAccessorIntegration:
    """Test xarray accessor integration and workflows."""
    
    def test_accessor_basic_functionality(self, simple_2d_grid_dataset):
        """Test basic .pyregrid accessor functionality."""
        ds = simple_2d_grid_dataset
        
        # Test accessor exists and has expected methods
        assert hasattr(ds, 'pyregrid')
        assert hasattr(ds.pyregrid, 'regrid_to')
        assert hasattr(ds.pyregrid, 'interpolate_to')
        assert hasattr(ds.pyregrid, 'get_coordinates')
        assert hasattr(ds.pyregrid, 'has_dask')
        
        # Test coordinate extraction
        coords = ds.pyregrid.get_coordinates()
        assert 'latitude_coord' in coords
        assert 'longitude_coord' in coords
        assert 'latitude_values' in coords
        assert 'longitude_values' in coords
        
        # Test Dask detection
        has_dask = ds.pyregrid.has_dask()
        assert isinstance(has_dask, bool)
    
    def test_accessor_regrid_workflow(self, simple_2d_grid_dataset, simple_target_grid):
        """Test complete regridding workflow through accessor."""
        ds = simple_2d_grid_dataset
        target_grid = simple_target_grid
        
        # Prepare target grid as xarray Dataset
        target_ds = xr.Dataset({
            'lon': (['lon'], target_grid['lon'].data),
            'lat': (['lat'], target_grid['lat'].data)
        })
        
        # Test regridding through accessor
        result = ds.pyregrid.regrid_to(target_ds, method='bilinear')
        
        # Verify result structure
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'lon' in result.coords
        assert 'lat' in result.coords
        
        # Verify coordinate alignment
        np.testing.assert_array_equal(result['lon'].values, target_grid['lon'])
        np.testing.assert_array_equal(result['lat'].values, target_grid['lat'])
    
    def test_accessor_interpolate_workflow(self, simple_point_data, simple_target_grid):
        """Test interpolation workflow through accessor."""
        # Create source gridded data
        source_ds = xr.Dataset({
            'temperature': (['lat', 'lon'], np.random.rand(5, 10)),
        }, coords={
            'lon': (['lon'], np.linspace(-180, 180, 10)),
            'lat': (['lat'], np.linspace(-90, 90, 5))
        })
        
        # Prepare target points as DataFrame
        target_points = pd.DataFrame({
            'longitude': simple_point_data['longitude'],
            'latitude': simple_point_data['latitude']
        })
        
        # Test interpolation through accessor
        result = source_ds.pyregrid.interpolate_to(target_points, method='idw')
        
        # Verify result structure
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords
    
    def test_accessor_crs_detection(self, simple_2d_grid_dataset):
        """Test CRS detection through accessor."""
        ds = simple_2d_grid_dataset
        
        # Test CRS detection without explicit CRS
        result = ds.pyregrid.get_coordinates()
        # The warning is emitted by the CRS manager, not by the accessor directly
        
        # Test with explicit CRS
        ds_with_crs = ds.copy()
        ds_with_crs.attrs['crs'] = 'EPSG:4326'
        
        result = ds_with_crs.pyregrid.get_coordinates()
        assert result['crs'] is not None
    
    def test_accessor_dask_integration(self, simple_2d_grid_dataset):
        """Test Dask integration through accessor."""
        ds = simple_2d_grid_dataset
        
        # Convert to Dask array
        ds_dask = ds.chunk({'lat': 2, 'lon': 5})
        
        # Test Dask detection
        assert ds_dask.pyregrid.has_dask() == True
        
        # Test regridding with Dask - use same coordinate dimensions as source
        target_ds = xr.Dataset({
            'lon': (['lon'], np.linspace(-180, 180, 10)),  # Match source dimension size
            'lat': (['lat'], np.linspace(-90, 90, 5))     # Match source dimension size
        })
        
        result = ds_dask.pyregrid.regrid_to(target_ds, method='bilinear')