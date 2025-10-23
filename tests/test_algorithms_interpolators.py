"""
Tests for interpolation algorithms.

This module contains comprehensive tests for all interpolation algorithms,
including BaseInterpolator, BilinearInterpolator, ConservativeInterpolator,
CubicInterpolator, and NearestInterpolator.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project root to the path to import pyregrid modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pyregrid.algorithms.interpolators import (
    BaseInterpolator,
    BilinearInterpolator,
    ConservativeInterpolator,
    CubicInterpolator,
    NearestInterpolator
)


class TestBaseInterpolator:
    """Test BaseInterpolator abstract base class."""
    
    def test_base_interpolator_is_abstract(self):
        """Test that BaseInterpolator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseInterpolator()
    
    def test_base_interpolator_requires_interpolate_method(self):
        """Test that concrete interpolators must implement interpolate method."""
        class ConcreteInterpolator(BaseInterpolator):
            def __init__(self):
                super().__init__(order=1)
            
            def interpolate(self, data, coordinates, **kwargs):
                return np.zeros((10, 10))
        
        # Should be able to instantiate concrete class
        interpolator = ConcreteInterpolator()
        assert interpolator.order == 1
        assert interpolator.mode == 'nearest'
        assert np.isnan(interpolator.cval)
        assert interpolator.prefilter is True
    
    def test_base_interpolator_custom_parameters(self):
        """Test BaseInterpolator with custom parameters."""
        class CustomInterpolator(BaseInterpolator):
            def __init__(self):
                super().__init__(order=2, mode='reflect', cval=0.0, prefilter=False)
            
            def interpolate(self, data, coordinates, **kwargs):
                return np.zeros((10, 10))
        
        interpolator = CustomInterpolator()
        assert interpolator.order == 2
        assert interpolator.mode == 'reflect'
        assert interpolator.cval == 0.0
        assert interpolator.prefilter is False


class TestBilinearInterpolator:
    """Test BilinearInterpolator class."""
    
    def test_initialization(self):
        """Test BilinearInterpolator initialization."""
        interpolator = BilinearInterpolator()
        assert interpolator.order == 1
        assert interpolator.mode == 'nearest'
        assert np.isnan(interpolator.cval)
        assert interpolator.prefilter is True
    
    def test_initialization_with_custom_params(self):
        """Test BilinearInterpolator initialization with custom parameters."""
        interpolator = BilinearInterpolator(mode='reflect', cval=0.0, prefilter=False)
        assert interpolator.order == 1
        assert interpolator.mode == 'reflect'
        assert interpolator.cval == 0.0
        assert interpolator.prefilter is False
    
    def test_interpolate_numpy_2d(self):
        """Test bilinear interpolation with 2D numpy array."""
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Create coordinates for interpolation (center of each cell)
        # For a 3x3 grid, interpolate at center points
        y_coords = np.array([0.5, 0.5, 0.5])  # y coordinates
        x_coords = np.array([0.5, 0.5, 0.5])  # x coordinates
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        # Should be reasonable values (not exact due to interpolation)
        assert not np.isnan(result).any()
        assert np.all(result >= 1) and np.all(result <= 9)
    
    def test_interpolate_numpy_1d(self):
        """Test bilinear interpolation with 1D numpy array."""
        # Create test data
        data = np.array([1, 2, 3, 4, 5], dtype=float)
        
        # Create coordinates for interpolation
        coordinates = np.array([np.array([0.5, 1.5, 2.5, 3.5, 4.5])])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (5,)
        assert not np.isnan(result).any()
    
    def test_interpolate_numpy_with_boundary_modes(self):
        """Test bilinear interpolation with different boundary modes."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Coordinates outside the original bounds
        y_coords = np.array([-0.5, 1.0, 2.5])  # Outside bounds
        x_coords = np.array([1.0, 1.0, 1.0])
        coordinates = np.array([y_coords, x_coords])
        
        modes = ['nearest', 'reflect', 'wrap', 'constant']
        
        for mode in modes:
            interpolator = BilinearInterpolator(mode=mode)
            if mode == 'constant':
                interpolator.cval = -999.0
            result = interpolator._interpolate_numpy(data, coordinates)
            assert result.shape == (3,)
            assert not np.isnan(result).any()
    
    def test_interpolate_dask_lazy_evaluation(self):
        """Test bilinear interpolation with dask array lazy evaluation."""
        # Create mock dask array
        mock_dask_array = Mock()
        mock_dask_array.__class__.__module__ = 'dask.array'
        mock_dask_array.chunks = ((3,), (3,))
        mock_dask_array.compute.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        data = mock_dask_array
        coordinates = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        # Should NOT call compute (lazy evaluation)
        mock_dask_array.compute.assert_not_called()
        # Result should be a delayed object for lazy evaluation
        # The exact type depends on dask.delayed implementation
        assert hasattr(result, 'compute') or hasattr(result, 'dask')
    
    @patch('pyregrid.algorithms.interpolators.map_coordinates')
    def test_interpolate_with_kwargs(self, mock_map_coordinates):
        """Test bilinear interpolation with additional keyword arguments."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        coordinates = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        
        interpolator = BilinearInterpolator()
        interpolator._interpolate_numpy(data, coordinates, test_param='test_value')
        
        # Verify that map_coordinates was called with the kwargs
        mock_map_coordinates.assert_called_once()
        call_kwargs = mock_map_coordinates.call_args[1]
        assert call_kwargs['test_param'] == 'test_value'


class TestCubicInterpolator:
    """Test CubicInterpolator class."""
    
    def test_initialization(self):
        """Test CubicInterpolator initialization."""
        interpolator = CubicInterpolator()
        assert interpolator.order == 3
        assert interpolator.mode == 'nearest'
        assert np.isnan(interpolator.cval)
        assert interpolator.prefilter is True
    
    def test_interpolate_numpy_2d(self):
        """Test cubic interpolation with 2D numpy array."""
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Create coordinates for interpolation
        y_coords = np.array([0.5, 0.5, 0.5])
        x_coords = np.array([0.5, 0.5, 0.5])
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = CubicInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        assert not np.isnan(result).any()
    
    def test_interpolate_numpy_with_prefilter(self):
        """Test cubic interpolation with prefilter enabled/disabled."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        coordinates = np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        
        # Test with prefilter enabled
        interpolator_enabled = CubicInterpolator(prefilter=True)
        result_enabled = interpolator_enabled._interpolate_numpy(data, coordinates)
        
        # Test with prefilter disabled
        interpolator_disabled = CubicInterpolator(prefilter=False)
        result_disabled = interpolator_disabled._interpolate_numpy(data, coordinates)
        
        assert result_enabled.shape == (3,)
        assert result_disabled.shape == (3,)
        # Results may differ due to prefiltering
        assert not np.array_equal(result_enabled, result_disabled)


class TestNearestInterpolator:
    """Test NearestInterpolator class."""
    
    def test_initialization(self):
        """Test NearestInterpolator initialization."""
        interpolator = NearestInterpolator()
        assert interpolator.order == 0
        assert interpolator.mode == 'nearest'
        assert np.isnan(interpolator.cval)
        assert interpolator.prefilter is True
    
    def test_interpolate_numpy_2d(self):
        """Test nearest neighbor interpolation with 2D numpy array."""
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Create coordinates for interpolation
        y_coords = np.array([0.1, 0.9, 1.9])  # Should map to indices 0, 1, 2
        x_coords = np.array([0.1, 0.9, 1.9])  # Should map to indices 0, 1, 2
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = NearestInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        # Should be close to original values (nearest neighbor)
        expected_values = [data[0, 0], data[1, 1], data[2, 2]]  # [1, 5, 9]
        assert np.allclose(result, expected_values, rtol=0.1)
    
    def test_interpolate_numpy_with_cval(self):
        """Test nearest neighbor interpolation with constant fill value."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Coordinates far outside bounds
        y_coords = np.array([-10.0, 1.0, 10.0])
        x_coords = np.array([1.0, 1.0, 1.0])
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = NearestInterpolator(mode='constant', cval=-999.0)
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        # First and last should be fill value, middle should be interpolated
        assert result[0] == -999.0
        assert result[2] == -999.0
        assert result[1] == 5.0  # Middle value


class TestConservativeInterpolator:
    """Test ConservativeInterpolator class."""
    
    def test_initialization(self):
        """Test ConservativeInterpolator initialization."""
        interpolator = ConservativeInterpolator()
        assert interpolator.order == 0
        assert interpolator.mode == 'nearest'
        assert np.isnan(interpolator.cval)
        assert interpolator.prefilter is True
        assert interpolator.weights is None
        assert interpolator._overlap_cache == {}
    
    def test_initialization_with_coordinates(self):
        """Test ConservativeInterpolator initialization with coordinates."""
        source_lon = np.linspace(-180, 180, 10)
        source_lat = np.linspace(-90, 90, 5)
        target_lon = np.linspace(-170, 170, 8)
        target_lat = np.linspace(-80, 80, 4)
        
        interpolator = ConservativeInterpolator(
            source_lon=source_lon,
            source_lat=source_lat,
            target_lon=target_lon,
            target_lat=target_lat
        )
        
        assert interpolator.source_lon is source_lon
        assert interpolator.source_lat is source_lat
        assert interpolator.target_lon is target_lon
        assert interpolator.target_lat is target_lat
        assert interpolator.weights is None
    
    def test_validate_coordinates_success(self):
        """Test coordinate validation with valid coordinates."""
        source_lon = np.linspace(-180, 180, 10)
        source_lat = np.linspace(-90, 90, 5)
        target_lon = np.linspace(-170, 170, 8)
        target_lat = np.linspace(-80, 80, 4)
        
        interpolator = ConservativeInterpolator(
            source_lon=source_lon,
            source_lat=source_lat,
            target_lon=target_lon,
            target_lat=target_lat
        )
        
        # Should not raise exception
        interpolator._validate_coordinates()
    
    def test_validate_coordinates_failure(self):
        """Test coordinate validation with missing coordinates."""
        interpolator = ConservativeInterpolator()
        
        with pytest.raises(ValueError, match="Conservative interpolation requires source and target coordinates"):
            interpolator._validate_coordinates()
    
    def test_calculate_grid_cell_area_1d_coordinates(self):
        """Test grid cell area calculation with 1D coordinates."""
        interpolator = ConservativeInterpolator()
        
        # Create 1D coordinates
        lon = np.linspace(-180, 180, 5)
        lat = np.linspace(-90, 90, 3)
        
        areas = interpolator._calculate_grid_cell_area(lon, lat)
        
        assert areas.shape == (3, 5)  # Should be 2D grid
        assert np.all(areas > 0)  # All areas should be positive
        assert not np.isnan(areas).any()
    
    def test_calculate_grid_cell_area_2d_coordinates(self):
        """Test grid cell area calculation with 2D coordinates."""
        interpolator = ConservativeInterpolator()
        
        # Create 2D coordinates
        lon_2d = np.array([[-180, -90, 0, 90, 180],
                          [-180, -90, 0, 90, 180],
                          [-180, -90, 0, 90, 180]])
        lat_2d = np.array([[-90, -90, -90, -90, -90],
                          [0, 0, 0, 0, 0],
                          [90, 90, 90, 90, 90]])
        
        areas = interpolator._calculate_grid_cell_area(lon_2d, lat_2d)
        
        assert areas.shape == (3, 5)
        assert np.all(areas > 0)
        assert not np.isnan(areas).any()
    
    def test_calculate_grid_cell_area_single_point(self):
        """Test grid cell area calculation with single point."""
        interpolator = ConservativeInterpolator()
        
        # Single point coordinates
        lon = np.array([0])
        lat = np.array([0])
        
        areas = interpolator._calculate_grid_cell_area(lon, lat)
        
        assert areas.shape == (1, 1)
        assert areas[0, 0] > 0  # Should have positive area
    
    @patch('pyregrid.algorithms.interpolators.map_coordinates')
    def test_interpolate_numpy_simple_case(self, mock_map_coordinates):
        """Test conservative interpolation simple case."""
        # Mock the map_coordinates to avoid complex dependencies
        mock_map_coordinates.return_value = np.array([[1, 2], [3, 4]])
        
        interpolator = ConservativeInterpolator()
        
        # Create simple test data
        data = np.array([[1, 2], [3, 4]], dtype=float)
        source_lon = np.array([0, 1])
        source_lat = np.array([0, 1])
        target_lon = np.array([0.5, 1.5])
        target_lat = np.array([0.5, 1.5])
        
        result = interpolator._interpolate_numpy(
            data, source_lon=source_lon, source_lat=source_lat,
            target_lon=target_lon, target_lat=target_lat
        )
        
        assert result.shape == (2, 2)
        assert not np.isnan(result).any()
    
    def test_interpolate_numpy_without_coordinates(self):
        """Test conservative interpolation without coordinates raises error."""
        interpolator = ConservativeInterpolator()
        data = np.array([[1, 2], [3, 4]], dtype=float)
        
        with pytest.raises(ValueError, match="Conservative interpolation requires source and target coordinates"):
            interpolator._interpolate_numpy(data)
    
    def test_interpolate_with_coordinate_override(self):
        """Test conservative interpolation with coordinate override."""
        interpolator = ConservativeInterpolator()
        
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        source_lon = np.linspace(-180, 180, 3)
        source_lat = np.linspace(-90, 90, 3)
        target_lon = np.linspace(-170, 170, 2)
        target_lat = np.linspace(-80, 80, 2)
        
        # Override coordinates in interpolate call
        result = interpolator.interpolate(
            data,
            source_lon=source_lon,
            source_lat=source_lat,
            target_lon=target_lon,
            target_lat=target_lat
        )
        
        assert result.shape == (2, 2)
        assert not np.isnan(result).any()
    
    def test_interpolate_dask_lazy_evaluation(self):
        """Test conservative interpolation with dask array lazy evaluation."""
        # Create mock dask array
        mock_dask_array = Mock()
        mock_dask_array.__class__.__module__ = 'dask.array'
        mock_dask_array.chunks = ((3,), (3,))
        mock_dask_array.compute.return_value = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        interpolator = ConservativeInterpolator()
        source_lon = np.linspace(-180, 180, 3)
        source_lat = np.linspace(-90, 90, 3)
        target_lon = np.linspace(-170, 170, 2)
        target_lat = np.linspace(-80, 80, 2)
        
        result = interpolator.interpolate(
            mock_dask_array,
            source_lon=source_lon,
            source_lat=source_lat,
            target_lon=target_lon,
            target_lat=target_lat
        )
        
        # Should NOT call compute (lazy evaluation)
        mock_dask_array.compute.assert_not_called()
        # Result should be a delayed object for lazy evaluation
        # The exact type depends on dask.delayed implementation
        assert hasattr(result, 'compute') or hasattr(result, 'dask')


class TestInterpolationConsistency:
    """Test consistency across different interpolation methods."""
    
    def test_same_data_different_methods(self):
        """Test that different interpolation methods work on same data."""
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Create coordinates for interpolation
        y_coords = np.array([0.5, 0.5, 0.5])
        x_coords = np.array([0.5, 0.5, 0.5])
        coordinates = np.array([y_coords, x_coords])
        
        # Test different interpolation methods
        bilinear = BilinearInterpolator()
        cubic = CubicInterpolator()
        nearest = NearestInterpolator()
        
        result_bilinear = bilinear._interpolate_numpy(data, coordinates)
        result_cubic = cubic._interpolate_numpy(data, coordinates)
        result_nearest = nearest._interpolate_numpy(data, coordinates)
        
        # All should have same shape
        assert result_bilinear.shape == result_cubic.shape == result_nearest.shape
        assert result_bilinear.shape == (3,)
        
        # Results should be reasonable (not NaN)
        assert not np.isnan(result_bilinear).any()
        assert not np.isnan(result_cubic).any()
        assert not np.isnan(result_nearest).any()
    
    def test_boundary_conditions_consistency(self):
        """Test that all interpolators handle boundary conditions consistently."""
        # Create test data
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Create coordinates outside bounds
        y_coords = np.array([-0.5, 1.0, 2.5])  # Outside bounds
        x_coords = np.array([1.0, 1.0, 1.0])
        coordinates = np.array([y_coords, x_coords])
        
        # Test different interpolation methods with same boundary mode
        bilinear = BilinearInterpolator(mode='nearest')
        cubic = CubicInterpolator(mode='nearest')
        nearest = NearestInterpolator(mode='nearest')
        
        result_bilinear = bilinear._interpolate_numpy(data, coordinates)
        result_cubic = cubic._interpolate_numpy(data, coordinates)
        result_nearest = nearest._interpolate_numpy(data, coordinates)
        
        # All should have same shape
        assert result_bilinear.shape == result_cubic.shape == result_nearest.shape
        assert result_bilinear.shape == (3,)
        
        # All should handle boundaries without crashing
        assert not np.isnan(result_bilinear).any()
        assert not np.isnan(result_cubic).any()
        assert not np.isnan(result_nearest).any()


class TestInterpolationEdgeCases:
    """Test interpolation algorithms with edge cases."""
    
    def test_empty_data_array(self):
        """Test interpolation with empty data array."""
        data = np.array([], dtype=float).reshape(0, 0)
        coordinates = np.array([[], []])
        
        interpolator = BilinearInterpolator()
        
        # Should handle empty arrays gracefully
        with pytest.raises((ValueError, IndexError)):
            interpolator._interpolate_numpy(data, coordinates)
    
    def test_single_value_data(self):
        """Test interpolation with single value data."""
        data = np.array([[5.0]])
        coordinates = np.array([[0.0], [0.0]])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (1,)
        assert result[0] == 5.0  # Should return the single value
    
    def test_nan_values_in_data(self):
        """Test interpolation with NaN values in data."""
        data = np.array([[1, 2, 3], [np.nan, 5, 6], [7, 8, 9]], dtype=float)
        coordinates = np.array([[0.5, 1.0, 1.5], [0.5, 1.0, 1.5]])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        # Should handle NaN values gracefully
        assert not np.isnan(result).all()  # Not all should be NaN
    
    def test_large_coordinates(self):
        """Test interpolation with large coordinate values."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        coordinates = np.array([[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
        
        interpolator = BilinearInterpolator(mode='constant', cval=-999.0)
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        assert np.all(result == -999.0)  # Should be fill value
    
    def test_inf_values_in_coordinates(self):
        """Test interpolation with infinite coordinate values."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        coordinates = np.array([[np.inf, 1.0, -np.inf], [1.0, 1.0, 1.0]])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (3,)
        # Should handle infinite coordinates gracefully
        assert not np.isnan(result).all()


class TestInterpolationPerformance:
    """Test interpolation algorithm performance characteristics."""
    
    def test_large_array_performance(self):
        """Test interpolation performance with large arrays."""
        import time
        
        # Create large test data
        data = np.random.rand(100, 100)
        
        # Create coordinates for interpolation
        y_coords = np.random.rand(1000) * 99
        x_coords = np.random.rand(1000) * 99
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = BilinearInterpolator()
        
        # Time the interpolation
        start_time = time.time()
        result = interpolator._interpolate_numpy(data, coordinates)
        end_time = time.time()
        
        assert result.shape == (1000,)
        assert end_time - start_time < 5.0  # Should complete in reasonable time
    
    def test_memory_usage_with_large_arrays(self):
        """Test memory usage with large arrays."""
        # Create moderately large test data
        data = np.random.rand(50, 50)
        
        # Create many coordinates
        y_coords = np.random.rand(10000) * 49
        x_coords = np.random.rand(10000) * 49
        coordinates = np.array([y_coords, x_coords])
        
        interpolator = BilinearInterpolator()
        result = interpolator._interpolate_numpy(data, coordinates)
        
        assert result.shape == (10000,)
        # Should handle large coordinate arrays without memory issues
        assert not np.isnan(result).all()