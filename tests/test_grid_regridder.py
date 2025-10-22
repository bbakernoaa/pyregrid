"""
Tests for GridRegridder class.

This module contains comprehensive tests for the GridRegridder class,
including initialization, regridding functionality, parameter validation,
and edge cases.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import Mock, patch

from pyregrid.core import GridRegridder


class TestGridRegridderInitialization:
    """Test GridRegridder initialization and basic setup."""
    
    def test_init_with_dataarray(self, simple_2d_grid, simple_target_grid):
        """Test initialization with DataArray inputs."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        assert regridder.source_grid is simple_2d_grid
        assert regridder.target_grid is simple_target_grid
        assert regridder.method == 'bilinear'
        assert regridder.weights is not None  # Should be prepared after init
        assert hasattr(regridder, '_source_lon_name')
        assert hasattr(regridder, '_source_lat_name')
        assert hasattr(regridder, '_target_lon_name')
        assert hasattr(regridder, '_target_lat_name')
    
    def test_init_with_dataset(self, simple_2d_grid_dataset, simple_target_grid):
        """Test initialization with Dataset inputs."""
        regridder = GridRegridder(simple_2d_grid_dataset, simple_target_grid)
        
        assert regridder.source_grid is simple_2d_grid_dataset
        assert regridder.target_grid is simple_target_grid
        assert regridder.method == 'bilinear'
        assert regridder.weights is not None
    
    def test_init_with_custom_method(self, simple_2d_grid, simple_target_grid):
        """Test initialization with custom method."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid, method='nearest')
        
        assert regridder.method == 'nearest'
        # weights should be prepared after initialization
        assert regridder.weights is not None
        assert regridder.weights['order'] == 0  # nearest neighbor order
    
    def test_init_with_invalid_method(self, simple_2d_grid, simple_target_grid):
        """Test initialization with invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be one of"):
            GridRegridder(simple_2d_grid, simple_target_grid, method='invalid_method')
    
    def test_init_with_crs(self, simple_2d_grid, simple_target_grid):
        """Test initialization with CRS parameters."""
        regridder = GridRegridder(
            simple_2d_grid, 
            simple_target_grid, 
            source_crs='EPSG:4326',
            target_crs='EPSG:4326'
        )
        
        assert regridder.source_crs is not None
        assert regridder.target_crs is not None
        assert regridder.transformer is not None
    
    def test_init_with_different_crs(self, simple_2d_grid, simple_target_grid):
        """Test initialization with different CRS values."""
        regridder = GridRegridder(
            simple_2d_grid, 
            simple_target_grid, 
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        assert regridder.source_crs is not None
        assert regridder.target_crs is not None
        assert regridder.transformer is not None


class TestGridRegridderFunctionality:
    """Test GridRegridder core functionality."""
    
    def test_regrid_dataarray(self, simple_2d_grid, simple_target_grid):
        """Test regridding a DataArray."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        result = regridder.regrid(simple_2d_grid)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (4, 8)  # Target grid shape
        assert result.dims == ('lat', 'lon')
        assert 'lat' in result.coords
        assert 'lon' in result.coords
    
    def test_regrid_dataset(self, simple_2d_grid_dataset, simple_target_grid):
        """Test regridding a Dataset."""
        regridder = GridRegridder(simple_2d_grid_dataset, simple_target_grid)
        result = regridder.regrid(simple_2d_grid_dataset)
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        assert result.temperature.shape == (4, 8)
        assert result.pressure.shape == (4, 8)
    
    def test_regrid_different_methods(self, simple_2d_grid, simple_target_grid):
        """Test regridding with different methods."""
        methods = ['bilinear', 'cubic', 'nearest']
        
        for method in methods:
            regridder = GridRegridder(simple_2d_grid, simple_target_grid, method=method)
            result = regridder.regrid(simple_2d_grid)
            
            assert isinstance(result, xr.DataArray)
            assert result.shape == (4, 8)
    
    def test_regrid_without_prepare(self, simple_2d_grid, simple_target_grid):
        """Test regridding without calling prepare first raises error."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        regridder.weights = None  # Simulate not being prepared
        
        with pytest.raises(RuntimeError, match="Weights not prepared"):
            regridder.regrid(simple_2d_grid)
    
    def test_regrid_invalid_input_type(self, simple_2d_grid, simple_target_grid):
        """Test regridding with invalid input type raises error."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        with pytest.raises(TypeError, match="Input data must be xr.DataArray or xr.Dataset"):
            # Use a mock object instead of string to avoid type checking issues
            invalid_input = Mock()
            type(invalid_input).__name__ = "str"
            regridder.regrid(invalid_input)


class TestGridRegridderCoordinateHandling:
    """Test GridRegridder coordinate extraction and handling."""
    
    def test_coordinate_extraction(self, simple_2d_grid, simple_target_grid):
        """Test coordinate extraction from grids."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        # Check that coordinates were extracted correctly
        assert hasattr(regridder, '_source_lon')
        assert hasattr(regridder, '_source_lat')
        assert hasattr(regridder, '_target_lon')
        assert hasattr(regridder, '_target_lat')
        assert len(regridder._source_lon) == 10
        assert len(regridder._source_lat) == 5
        assert len(regridder._target_lon) == 8
        assert len(regridder._target_lat) == 4
    
    def test_coordinate_names_detection(self, simple_2d_grid, simple_target_grid):
        """Test detection of coordinate names."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        assert regridder._source_lon_name == 'lon'
        assert regridder._source_lat_name == 'lat'
        assert regridder._target_lon_name == 'lon'
        assert regridder._target_lat_name == 'lat'
    
    def test_coordinate_names_with_different_names(self):
        """Test coordinate detection with different coordinate names."""
        # Create grid with different coordinate names
        lons = np.linspace(-180, 180, 10)
        lats = np.linspace(-90, 90, 5)
        data = np.random.rand(5, 10)
        
        source = xr.DataArray(
            data,
            dims=['y', 'x'],
            coords={'x': lons, 'y': lats}
        )
        
        target = xr.DataArray(
            np.zeros((4, 8)),
            dims=['y', 'x'],
            coords={'x': np.linspace(-170, 170, 8), 'y': np.linspace(-80, 80, 4)}
        )
        
        regridder = GridRegridder(source, target)
        
        assert regridder._source_lon_name == 'x'
        assert regridder._source_lat_name == 'y'
        assert regridder._target_lon_name == 'x'
        assert regridder._target_lat_name == 'y'


class TestGridRegridderEdgeCases:
    """Test GridRegridder edge cases and error conditions."""
    
    def test_regrid_with_nan_values(self, simple_2d_grid_with_nan, simple_target_grid):
        """Test regridding data with NaN values."""
        regridder = GridRegridder(simple_2d_grid_with_nan, simple_target_grid)
        result = regridder.regrid(simple_2d_grid_with_nan)
        
        # Should handle NaN values gracefully
        assert isinstance(result, xr.DataArray)
        assert result.shape == (4, 8)
    
    def test_regrid_with_mismatched_dimensions(self, simple_2d_grid, simple_target_grid):
        """Test regridding with dimension mismatch."""
        # Create a 3D grid
        times = pd.date_range('2020-01-01', periods=3)
        lons = np.linspace(-180, 180, 10)
        lats = np.linspace(-90, 90, 5)
        data_3d = np.random.rand(3, 5, 10)
        
        source_3d = xr.DataArray(
            data_3d,
            dims=['time', 'lat', 'lon'],
            coords={'time': times, 'lon': lons, 'lat': lats}
        )
        
        regridder = GridRegridder(source_3d, simple_target_grid)
        result = regridder.regrid(source_3d)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3, 4, 8)  # Should preserve time dimension
    
    def test_regrid_with_single_point(self):
        """Test regridding with single point grids."""
        # Create single point source grid
        source = xr.DataArray(
            np.array([[42.0]]),
            dims=['lat', 'lon'],
            coords={'lon': [0], 'lat': [0]}
        )
        
        # Create single point target grid
        target = xr.DataArray(
            np.array([[0.0]]),
            dims=['lat', 'lon'],
            coords={'lon': [0], 'lat': [0]}
        )
        
        regridder = GridRegridder(source, target)
        result = regridder.regrid(source)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (1, 1)
    
    def test_regrid_with_large_grid(self):
        """Test regridding with larger grids."""
        # Create larger grids
        lons = np.linspace(-180, 180, 50)
        lats = np.linspace(-90, 90, 25)
        data = np.random.rand(25, 50)
        
        source = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={'lon': lons, 'lat': lats}
        )
        
        target_lons = np.linspace(-170, 170, 40)
        target_lats = np.linspace(-80, 80, 20)
        target = xr.DataArray(
            np.zeros((20, 40)),
            dims=['lat', 'lon'],
            coords={'lon': target_lons, 'lat': target_lats}
        )
        
        regridder = GridRegridder(source, target)
        result = regridder.regrid(source)
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (20, 40)


class TestGridRegridderPrepareMethod:
    """Test GridRegridder prepare method functionality."""
    
    def test_prepare_method(self, simple_2d_grid, simple_target_grid):
        """Test the prepare method explicitly."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        # Reset weights to test prepare method
        regridder.weights = None
        regridder.prepare()
        
        assert regridder.weights is not None
        assert 'lon_indices' in regridder.weights
        assert 'lat_indices' in regridder.weights
        assert 'order' in regridder.weights
        assert 'method' in regridder.weights
    
    def test_prepare_with_different_methods(self, simple_2d_grid, simple_target_grid):
        """Test prepare method with different interpolation methods."""
        methods = ['bilinear', 'cubic', 'nearest']
        
        for method in methods:
            regridder = GridRegridder(simple_2d_grid, simple_target_grid, method=method)
            regridder.weights = None  # Reset to test prepare
            regridder.prepare()
            
            assert regridder.weights is not None
            assert regridder.weights['method'] == method
            assert regridder.weights['order'] == {'bilinear': 1, 'cubic': 3, 'nearest': 0}[method]


class TestGridRegridderCRSTransformation:
    """Test GridRegridder CRS transformation functionality."""
    
    @patch('pyregrid.core.Transformer')
    def test_crs_transformation_setup(self, mock_transformer, simple_2d_grid, simple_target_grid):
        """Test CRS transformation setup."""
        regridder = GridRegridder(
            simple_2d_grid,
            simple_target_grid,
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        # Verify transformer was created
        mock_transformer.from_crs.assert_called_once()
        assert regridder.transformer is not None
    
    def test_crs_transformation_without_crs(self, simple_2d_grid, simple_target_grid):
        """Test behavior without CRS specification."""
        regridder = GridRegridder(simple_2d_grid, simple_target_grid)
        
        # Should not have transformer if no CRS specified
        assert regridder.transformer is None
    
    def test_crs_transformation_error(self, simple_2d_grid, simple_target_grid):
        """Test error when CRS transformation fails."""
        regridder = GridRegridder(
            simple_2d_grid,
            simple_target_grid,
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        # Test that _setup_crs_transformation works when transformer exists
        assert regridder.transformer is not None
        
        # Test error when source_crs is None
        regridder.source_crs = None
        with pytest.raises(ValueError, match="Both source_crs and target_crs must be provided"):
            regridder._setup_crs_transformation()


class TestGridRegridderValidation:
    """Test GridRegridder validation and error handling."""
    
    def test_invalid_grid_type(self):
       """Test initialization with invalid grid types."""
       # Test with Mock objects that don't have proper xarray structure
       # Mock objects are not iterable, so this should raise ValueError
       with pytest.raises(ValueError, match="Source grid does not have valid coordinate information"):
           GridRegridder(Mock(), xr.DataArray())
    
       with pytest.raises(ValueError, match="Could not identify valid longitude and latitude coordinates"):
           GridRegridder(xr.DataArray(), Mock())
    
    def test_missing_coordinates(self):
        """Test initialization with grids missing coordinates."""
        # Create grid without coordinates
        data = np.random.rand(5, 10)
        source = xr.DataArray(data, dims=['dim1', 'dim2'])
        target = xr.DataArray(np.zeros((4, 8)), dims=['dim1', 'dim2'])
        
        with pytest.raises(ValueError):
            GridRegridder(source, target)
    
    def test_dimension_mismatch(self):
        """Test initialization with dimension mismatch."""
        # Create source with different dimensions
        source = xr.DataArray(
            np.random.rand(5, 10),
            dims=['lat', 'lon'],
            coords={'lon': np.linspace(-180, 180, 10), 'lat': np.linspace(-90, 90, 5)}
        )
        
        # Create target with different coordinate structure
        target = xr.DataArray(
            np.zeros((4, 8)),
            dims=['y', 'x'],
            coords={'x': np.linspace(-170, 170, 8), 'y': np.linspace(-80, 80, 4)}
        )
        
        # This should work as coordinate names are detected
        regridder = GridRegridder(source, target)
        assert regridder is not None