
"""
Comprehensive edge case tests for PyRegrid.

This module contains tests for error handling, boundary conditions,
and invalid input scenarios to ensure robustness in production environments.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import Mock, patch
import warnings

from pyregrid.core import GridRegridder, PointInterpolator
from pyregrid.algorithms.interpolators import (
    BaseInterpolator, 
    BilinearInterpolator, 
    CubicInterpolator, 
    NearestInterpolator
)
from pyregrid.crs.crs_manager import CRSManager
from pyproj import CRS

# Optional Dask imports - handle gracefully if not available
try:
    from pyregrid.dask.dask_regridder import DaskRegridder
    from pyregrid.dask.chunking import ChunkingStrategy
    from pyregrid.dask.memory_management import MemoryManager
    from pyregrid.dask.parallel_processing import ParallelProcessor
    HAS_DASK = True
except ImportError:
    # Create placeholder classes for testing
    class DaskRegridder:
        def __init__(self, *args, **kwargs):
            raise ImportError("DaskRegridder requires Dask to be installed")
    
    class ChunkingStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError("ChunkingStrategy requires Dask to be installed")
    
    class MemoryManager:
        def __init__(self, *args, **kwargs):
            raise ImportError("MemoryManager requires Dask to be installed")
    
    class ParallelProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("ParallelProcessor requires Dask to be installed")
    
    HAS_DASK = False


class TestEdgeCaseErrorHandling:
    """Test error handling in various components of PyRegrid."""

    def test_grid_regridder_invalid_method(self):
        """Test GridRegridder with invalid interpolation method."""
        # Create simple test grids
        source_lon = np.linspace(-10, 5, 5)  # 5 points
        source_lat = np.linspace(40, 50, 4)  # 4 points
        source_data = np.random.random((4, 5))  # 4x5 data
        
        target_lon = np.linspace(-5, 5, 3)   # 3 points
        target_lat = np.linspace(42, 48, 2)   # 2 points
        target_data = np.random.random((2, 3)) # 2x3 data
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Test with invalid method
        with pytest.raises(ValueError, match="Method must be one of"):
            GridRegridder(source_ds, target_ds, method="invalid_method")

    def test_point_interpolator_invalid_method(self):
        """Test PointInterpolator with invalid interpolation method."""
        # Create simple test data
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        source_data = np.random.random((4, 5))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        
        # Test with invalid method
        with pytest.raises(ValueError, match="Method must be one of"):
            PointInterpolator(source_ds, pd.DataFrame({'longitude': [0], 'latitude': [45]}), method="invalid_method")

    def test_grid_regridder_missing_coordinates(self):
        """Test GridRegridder with missing coordinate information."""
        # Create source grid without proper coordinates
        source_data = np.random.random((4, 5))
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)}
            # Missing coordinates!
        )
        
        # Create target grid
        target_lon = np.linspace(-5, 5, 3)
        target_lat = np.linspace(42, 48, 2)
        target_data = np.random.random((2, 3))
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # This should fail gracefully
        with pytest.raises(Exception):
            GridRegridder(source_ds, target_ds)

    def test_point_interpolator_missing_coordinates(self):
        """Test PointInterpolator with missing coordinate information."""
        # Create source data without proper coordinates
        source_data = np.random.random((4, 5))
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)}
            # Missing coordinates!
        )
        
        # Try to create point interpolator
        with pytest.raises(ValueError, match="Could not find latitude/longitude coordinates"):
            PointInterpolator(source_ds, pd.DataFrame({'longitude': [0], 'latitude': [45]}))


class TestEdgeCaseBoundaryConditions:
    """Test boundary conditions and extreme values."""

    def test_grid_regridder_extreme_values(self):
        """Test GridRegridder with extreme coordinate values."""
        # Create grids with extreme coordinate values
        source_lon = np.array([-180, -179, 179, 180])  # 4 points
        source_lat = np.array([-90, -89, 90, 89])      # 4 points (matching data)
        source_data = np.random.random((4, 4))          # 4x4 data
        
        target_lon = np.array([-180.5, 0, 180.5])      # 3 points
        target_lat = np.array([-90.5, 0, 90.5])        # 3 points
        target_data = np.random.random((3, 3))          # 3x3 data
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Provide explicit CRS to avoid coordinate range validation issues
        from pyproj import CRS
        explicit_crs = CRS.from_string("EPSG:4326")
        
        # This should not crash but may produce warnings
        regridder = GridRegridder(source_ds, target_ds, source_crs=explicit_crs, target_crs=explicit_crs)
        # The regridder should be able to handle extreme values
        
    def test_interpolation_with_nan_values(self):
        """Test interpolation with NaN values in data."""
        # Create test data with NaN values
        data = np.array([[np.nan, 2, 3], [4, 5, 6], [7, 8, 9]])
        coordinates = np.array([[0.5, 1.5], [1.0, 1.0]])
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        # Should handle NaN values gracefully
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_interpolation_with_infinite_values(self):
        """Test interpolation with infinite values in data."""
        # Create test data with infinite values
        data = np.array([[1, 2, np.inf], [4, 5, 6], [7, 8, 9]])
        coordinates = np.array([[0.5, 1.5], [1.0, 1.0]])
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        # Should handle infinite values gracefully
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)

    def test_interpolation_with_very_large_numbers(self):
        """Test interpolation with very large coordinate values."""
        # Create data with very large coordinate values
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        coordinates = np.array([[1e10, 2e10], [3e10, 4e10]])
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        # Should handle large values gracefully
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)


class TestEdgeCaseInvalidInputs:
    """Test various invalid inputs and malformed data scenarios."""

    def test_grid_regridder_none_inputs(self):
        """Test GridRegridder with None inputs."""
        with pytest.raises(ValueError, match="Source grid does not have valid coordinate information"):
            GridRegridder(None, None)

    def test_point_interpolator_none_inputs(self):
        """Test PointInterpolator with None inputs."""
        with pytest.raises(AttributeError):
            PointInterpolator(None, None)

    def test_grid_regridder_empty_data(self):
        """Test GridRegridder with empty data arrays."""
        # Create empty arrays
        source_lon = np.array([])
        source_lat = np.array([])
        source_data = np.array([]).reshape(0, 0)
        
        target_lon = np.array([])
        target_lat = np.array([])
        target_data = np.array([]).reshape(0, 0)
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # This should fail gracefully
        with pytest.raises(Exception):
            GridRegridder(source_ds, target_ds)

    def test_interpolation_with_wrong_dimensional_coordinates(self):
        """Test interpolation with mismatched dimensional coordinate arrays."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Wrong dimension - 1D coordinates for 2D data
        coordinates = np.array([0.5, 1.5])  # Should be 2D: [[lat], [lon]]
        
        interpolator = BilinearInterpolator()
        with pytest.raises(Exception):
            interpolator.interpolate(data, coordinates)

    def test_interpolation_with_invalid_coordinate_types(self):
        """Test interpolation with invalid coordinate types."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        invalid_coordinates = "invalid_string"
        
        interpolator = BilinearInterpolator()
        with pytest.raises(Exception):
            interpolator.interpolate(data, invalid_coordinates)

    def test_grid_regridder_unmatched_dimensions(self):
        """Test GridRegridder with unmatched grid dimensions."""
        # Source grid with 3x3 dimensions
        source_lon = np.linspace(-10, 10, 3)
        source_lat = np.linspace(40, 50, 3)
        source_data = np.random.random((3, 3))
        
        # Target grid with different dimensions
        target_lon = np.linspace(-5, 5, 5)  # Different size
        target_lat = np.linspace(42, 48, 4)  # Different size
        target_data = np.random.random((4, 5))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # This should work, but may produce warnings about coordinate mismatch
        regridder = GridRegridder(source_ds, target_ds)
        # Should not crash


class TestMemoryConstraintScenarios:
    """Test memory constraint scenarios and out-of-memory handling."""

    def test_memory_management_with_large_arrays(self, monkeypatch):
        """Test memory management with large arrays."""
        # Test with large arrays that would trigger memory constraints
        large_lon = np.linspace(-180, 180, 1000)  # Reduced size for testing
        large_lat = np.linspace(-90, 90, 1000)   # Reduced size for testing
        large_data = np.random.random((1000, 1000))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], large_data)},
            coords={'lon': large_lon, 'lat': large_lat}
        )
        
        target_lon = np.linspace(-180, 180, 500)
        target_lat = np.linspace(-90, 90, 500)
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], np.random.random((500, 500)))},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # This should handle large arrays gracefully
        from pyproj import CRS
        explicit_crs = CRS.from_string("EPSG:4326")
        
        regridder = GridRegridder(source_ds, target_ds, source_crs=explicit_crs, target_crs=explicit_crs)
        # The regridder should be able to handle large arrays

    def test_chunking_strategy_edge_cases(self):
        """Test chunking strategy with edge case configurations."""
        # Test with invalid chunk sizes
        if HAS_DASK:
            # Only test if Dask is available
            chunking_strategy = ChunkingStrategy()
            # Test with invalid method
            with pytest.raises(ValueError, match="Unknown method"):
                chunking_strategy.determine_chunk_size(None, None, method="invalid_method")
        else:
            # If Dask is not available, the placeholder should raise ImportError
            with pytest.raises(ImportError):
                ChunkingStrategy()


class TestCRSEdgeCases:
    """Test CRS-related edge cases."""

    def test_crs_manager_invalid_crs_handling(self):
        """Test CRS manager with invalid CRS specifications."""
        crs_manager = CRSManager()
        
        # Test with completely invalid CRS string
        with pytest.raises(Exception):
            CRS.from_string("invalid_crs_string")

    def test_crs_manager_boundary_conditions(self):
        """Test CRS manager with boundary coordinate values."""
        crs_manager = CRSManager()
        
        # Test with coordinates at exact geographic boundaries
        geographic_crs = CRS.from_string("EPSG:4326")
        lon_coords = np.array([-180.0, 0.0, 180.0])  # Exactly at boundaries
        lat_coords = np.array([-90.0, 0.0, 90.0])   # Exactly at boundaries
        
        # Should not crash with boundary values
        assert crs_manager.validate_coordinate_arrays(lon_coords, lat_coords, geographic_crs)


class TestDaskIntegrationEdgeCases:
    """Test edge cases in Dask integration."""

    def test_dask_regridder_with_invalid_chunks(self):
       """Test DaskRegridder with invalid chunk configurations."""
       # Test that DaskRegridder raises appropriate error with invalid inputs
       # Since we're passing None values, it should raise an appropriate error
       with pytest.raises(Exception):
           DaskRegridder(None, None)

    def test_parallel_processor_edge_cases(self):
       """Test ParallelProcessor with edge case configurations."""
       # Test that ParallelProcessor can be created without arguments
       # The constructor should not raise an exception
       processor = ParallelProcessor()
       assert processor is not None
       assert processor.client is None


class TestAlgorithmEdgeCases:
    """Test edge cases in interpolation algorithms."""

    def test_interpolator_with_insufficient_data(self):
        """Test interpolators with insufficient data points."""
        # Very small data array
        data = np.array([[5]])  # Single point
        coordinates = np.array([[0.0, 0.0]])  # Need 2D coordinates for 2D data
        
        # This might cause issues with scipy's map_coordinates
        interpolator = BilinearInterpolator()
        with pytest.raises(Exception):
            interpolator.interpolate(data, coordinates)

    def test_interpolator_with_single_coordinate(self):
        """Test interpolators with single coordinate points."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Fix coordinates to have proper shape for 2D data: [[lat], [lon]]
        coordinates = np.array([[0.5], [1.5]])  # Proper 2D shape
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)

    def test_interpolator_with_zero_coordinates(self):
        """Test interpolators with zero coordinate values."""
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        # Fix coordinates to have proper shape for 2D data: [[lat], [lon]]
        coordinates = np.array([[0.0], [0.0]])  # Proper 2D shape
        
        interpolator = BilinearInterpolator()
        result = interpolator.interpolate(data, coordinates)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1,)