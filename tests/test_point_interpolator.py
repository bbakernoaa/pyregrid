"""
Tests for PointInterpolator class.

This module contains comprehensive tests for the PointInterpolator class,
including initialization, interpolation functionality, parameter validation,
and edge cases.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pyregrid.core import PointInterpolator


class TestPointInterpolatorInitialization:
    """Test PointInterpolator initialization and basic setup."""
    
    def test_init_with_dataarray_and_dataframe(self, simple_2d_grid, target_points_df):
        """Test initialization with DataArray source and DataFrame target points."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df)
        
        assert interpolator.source_data is simple_2d_grid
        assert interpolator.target_points is target_points_df
        assert interpolator.method == 'idw'
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
    
    def test_init_with_dataset_and_dict(self, simple_2d_grid_dataset, target_points_dict):
        """Test initialization with Dataset source and dict target points."""
        interpolator = PointInterpolator(simple_2d_grid_dataset, target_points_dict)
        
        assert interpolator.source_data is simple_2d_grid_dataset
        assert interpolator.target_points is target_points_dict
        assert interpolator.method == 'idw'
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
    
    def test_init_with_custom_method(self, simple_2d_grid, target_points_df):
        """Test initialization with custom method."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='linear')
        
        assert interpolator.method == 'linear'
    
    def test_init_with_invalid_method(self, simple_2d_grid, target_points_df):
        """Test initialization with invalid method raises error."""
        with pytest.raises(ValueError, match="Method must be one of"):
            PointInterpolator(simple_2d_grid, target_points_df, method='invalid_method')
    
    def test_init_with_crs(self, simple_2d_grid, target_points_df):
        """Test initialization with CRS parameters."""
        interpolator = PointInterpolator(
            simple_2d_grid, 
            target_points_df, 
            source_crs='EPSG:4326',
            target_crs='EPSG:4326'
        )
        
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
        assert interpolator.transformer is not None
    
    def test_init_with_different_crs(self, simple_2d_grid, target_points_df):
        """Test initialization with different CRS values."""
        interpolator = PointInterpolator(
            simple_2d_grid, 
            target_points_df, 
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
        assert interpolator.transformer is not None


class TestPointInterpolatorCoordinateExtraction:
    """Test PointInterpolator coordinate extraction and handling."""
    
    def test_coordinate_extraction_from_dataarray(self, simple_2d_grid, target_points_df):
        """Test coordinate extraction from DataArray source."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df)
        
        # Check that coordinates were extracted correctly
        assert hasattr(interpolator, 'source_crs')
        assert hasattr(interpolator, 'target_crs')
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
    
    def test_coordinate_extraction_from_dataset(self, simple_2d_grid_dataset, target_points_dict):
        """Test coordinate extraction from Dataset source."""
        interpolator = PointInterpolator(simple_2d_grid_dataset, target_points_dict)
        
        # Check that coordinates were extracted correctly
        assert hasattr(interpolator, 'source_crs')
        assert hasattr(interpolator, 'target_crs')
        assert interpolator.source_crs is not None
        assert interpolator.target_crs is not None
    
    def test_coordinate_extraction_from_dataframe(self, simple_2d_grid, target_points_df):
        """Test coordinate extraction from DataFrame target points."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df)
        
        # Check that coordinates were extracted correctly
        assert hasattr(interpolator, 'target_crs')
        assert interpolator.target_crs is not None
    
    def test_coordinate_extraction_from_dict(self, simple_2d_grid, target_points_dict):
        """Test coordinate extraction from dict target points."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_dict)
        
        # Check that coordinates were extracted correctly
        assert hasattr(interpolator, 'target_crs')
        assert interpolator.target_crs is not None
    
    def test_coordinate_extraction_from_xr_dataset(self, simple_2d_grid, target_points_xr):
        """Test coordinate extraction from xarray Dataset target points."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_xr)
        
        # Check that coordinates were extracted correctly
        assert hasattr(interpolator, 'target_crs')
        assert interpolator.target_crs is not None


class TestPointInterpolatorInterpolationMethods:
    """Test PointInterpolator interpolation methods."""
    
    def test_interpolate_bilinear(self, simple_2d_grid, target_points_df):
        """Test bilinear interpolation."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        assert 'points' in result.dims
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords
    
    def test_interpolate_nearest(self, simple_2d_grid, target_points_df):
        """Test nearest neighbor interpolation."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='nearest')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        assert 'points' in result.dims
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords
    
    def test_interpolate_idw(self, simple_2d_grid, target_points_df):
        """Test IDW interpolation (should fall back to bilinear)."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='idw')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        assert 'points' in result.dims
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords
    
    def test_interpolate_linear(self, simple_2d_grid, target_points_df):
        """Test linear interpolation (should fall back to bilinear)."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='linear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        assert 'points' in result.dims
        assert 'longitude' in result.coords
        assert 'latitude' in result.coords
    
    def test_interpolate_with_dataset_source(self, simple_2d_grid_dataset, target_points_df):
        """Test interpolation with Dataset source."""
        interpolator = PointInterpolator(simple_2d_grid_dataset, target_points_df, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        assert len(result.temperature) == len(target_points_df)
        assert len(result.pressure) == len(target_points_df)


class TestPointInterpolatorCRSTransformation:
    """Test PointInterpolator CRS transformation functionality."""
    
    @patch('pyregrid.core.Transformer')
    def test_crs_transformation_setup(self, mock_transformer, simple_2d_grid, target_points_df):
        """Test CRS transformation setup."""
        interpolator = PointInterpolator(
            simple_2d_grid,
            target_points_df,
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        # Verify transformer was created
        mock_transformer.from_crs.assert_called_once()
        assert interpolator.transformer is not None
    
    def test_crs_transformation_without_crs(self, simple_2d_grid, target_points_df):
        """Test behavior without CRS specification."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df)
        
        # Should not have transformer if no CRS specified
        assert interpolator.transformer is None
    
    def test_crs_transformation_error(self, simple_2d_grid, target_points_df):
        """Test error when CRS transformation fails."""
        interpolator = PointInterpolator(
            simple_2d_grid,
            target_points_df,
            source_crs='EPSG:4326',
            target_crs='EPSG:3857'
        )
        
        # Test that _setup_crs_transformation works when transformer exists
        assert interpolator.transformer is not None
        
        # Test error when source_crs is None
        interpolator.source_crs = None
        with pytest.raises(ValueError, match="Both source_crs and target_crs must be provided"):
            interpolator._setup_crs_transformation()


class TestPointInterpolatorEdgeCases:
    """Test PointInterpolator edge cases and error conditions."""
    
    def test_interpolate_with_nan_values(self, simple_2d_grid_with_nan, target_points_df):
        """Test interpolation with NaN values in source data."""
        interpolator = PointInterpolator(simple_2d_grid_with_nan, target_points_df)
        result = interpolator.interpolate()
        
        # Should handle NaN values gracefully
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
    
    def test_interpolate_with_single_point(self):
        """Test interpolation with single point source grid."""
        # Create single point source grid
        source = xr.DataArray(
            np.array([[42.0]]),
            dims=['lat', 'lon'],
            coords={'lon': [0], 'lat': [0]}
        )
        
        # Create single target point
        target_points = pd.DataFrame({
            'longitude': [0.1],
            'latitude': [0.1]
        })
        
        interpolator = PointInterpolator(source, target_points)
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (1,)
    
    def test_interpolate_with_large_grid(self):
        """Test interpolation with larger grids."""
        # Create larger source grid
        lons = np.linspace(-180, 180, 50)
        lats = np.linspace(-90, 90, 25)
        data = np.random.rand(25, 50)
        
        source = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={'lon': lons, 'lat': lats}
        )
        
        # Create many target points
        target_points = pd.DataFrame({
            'longitude': np.random.uniform(-170, 170, 1000),
            'latitude': np.random.uniform(-80, 80, 1000)
        })
        
        interpolator = PointInterpolator(source, target_points)
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (1000,)
    
    def test_interpolate_with_out_of_bounds_points(self, simple_2d_grid):
        """Test interpolation with target points outside source grid bounds."""
        # Create target points outside source grid bounds
        target_points = pd.DataFrame({
            'longitude': [200, -200],  # Outside typical longitude bounds
            'latitude': [100, -100]    # Outside typical latitude bounds
        })
        
        interpolator = PointInterpolator(simple_2d_grid, target_points)
        result = interpolator.interpolate()
        
        # Should handle out-of-bounds points gracefully
        assert isinstance(result, xr.DataArray)
        assert result.shape == (2,)
        # Should have NaN values for out-of-bounds points
        assert np.isnan(result.values).any()
    
    def test_interpolate_with_sparse_target_points(self, simple_2d_grid):
        """Test interpolation with sparse target points."""
        # Create sparse target points
        target_points = pd.DataFrame({
            'longitude': np.linspace(-170, 170, 5),
            'latitude': np.linspace(-80, 80, 3)
        })
        
        interpolator = PointInterpolator(simple_2d_grid, target_points)
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (15,)  # 5 * 3 = 15 points


class TestPointInterpolatorValidation:
    """Test PointInterpolator validation and error handling."""
    
    def test_invalid_source_type(self, target_points_df):
        """Test initialization with invalid source type."""
        with pytest.raises(TypeError):
            PointInterpolator(Mock(), target_points_df)
    
    def test_invalid_target_type(self, simple_2d_grid):
        """Test initialization with invalid target type."""
        with pytest.raises(TypeError):
            PointInterpolator(simple_2d_grid, "invalid_target")
    
    def test_missing_coordinates_in_source(self, target_points_df):
        """Test initialization with source missing coordinates."""
        # Create source without coordinates
        data = np.random.rand(5, 10)
        source = xr.DataArray(data, dims=['dim1', 'dim2'])
        
        with pytest.raises(ValueError, match="Could not find latitude/longitude coordinates"):
            PointInterpolator(source, target_points_df)
    
    def test_missing_coordinates_in_target(self, simple_2d_grid):
        """Test initialization with target missing coordinates."""
        # Create target without coordinates
        target_points = pd.DataFrame({
            'value1': [1, 2, 3],
            'value2': [4, 5, 6]
        })
        
        with pytest.raises(ValueError, match="Could not find longitude/latitude columns"):
            PointInterpolator(simple_2d_grid, target_points)
    
    def test_interpolate_without_prepare(self, simple_2d_grid, target_points_df):
        """Test interpolation without proper preparation."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df)
        # Manually break the preparation
        interpolator.source_crs = None
        
        with pytest.raises(ValueError, match="Could not find latitude/longitude coordinates"):
            interpolator.interpolate()


class TestPointInterpolatorDaskIntegration:
    """Test PointInterpolator Dask integration."""
    
    def test_interpolate_with_dask_dataarray(self, simple_2d_grid, target_points_df):
        """Test interpolation with Dask DataArray source."""
        # Convert source to Dask array
        import dask.array as da
        dask_data = da.from_array(simple_2d_grid.values, chunks="auto")
        dask_source = xr.DataArray(
            dask_data,
            dims=simple_2d_grid.dims,
            coords=simple_2d_grid.coords
        )
        
        interpolator = PointInterpolator(dask_source, target_points_df, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        # Result should be computed (not a Dask array)
        assert not hasattr(result.data, 'chunks')
    
    def test_interpolate_with_dask_dataset(self, simple_2d_grid_dataset, target_points_df):
        """Test interpolation with Dask Dataset source."""
        # Convert dataset to Dask
        import dask.array as da
        dask_vars = {}
        for var_name, var_data in simple_2d_grid_dataset.data_vars.items():
            dask_vars[var_name] = xr.DataArray(
                da.from_array(var_data.values, chunks="auto"),
                dims=var_data.dims,
                coords=var_data.coords,
                attrs=var_data.attrs
            )
        
        dask_dataset = xr.Dataset(dask_vars, coords=simple_2d_grid_dataset.coords)
        
        interpolator = PointInterpolator(dask_dataset, target_points_df, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.Dataset)
        assert 'temperature' in result.data_vars
        assert 'pressure' in result.data_vars
        # Results should be computed (not Dask arrays)
        for var in result.data_vars.values():
            assert not hasattr(var.data, 'chunks')


class TestPointInterpolatorPerformance:
    """Test PointInterpolator performance characteristics."""
    
    def test_interpolation_performance(self, simple_2d_grid):
        """Test interpolation performance with reasonable data sizes."""
        import time
        
        # Create moderate-sized data
        lons = np.linspace(-180, 180, 100)
        lats = np.linspace(-90, 90, 50)
        data = np.random.rand(50, 100)
        
        source = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={'lon': lons, 'lat': lats}
        )
        
        # Create moderate number of target points
        target_points = pd.DataFrame({
            'longitude': np.random.uniform(-170, 170, 1000),
            'latitude': np.random.uniform(-80, 80, 1000)
        })
        
        interpolator = PointInterpolator(source, target_points, method='bilinear')
        
        # Time the interpolation
        start_time = time.time()
        result = interpolator.interpolate()
        end_time = time.time()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (1000,)
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0


class TestPointInterpolatorGeospatialMethods:
    """Test geospatial-specific interpolation methods."""
    
    def test_interpolate_with_geographic_coordinates(self, simple_2d_grid, target_points_df):
        """Test interpolation with geographic coordinates (lon/lat)."""
        interpolator = PointInterpolator(simple_2d_grid, target_points_df, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (len(target_points_df),)
        # Should handle geographic coordinates correctly
        assert not np.isnan(result.values).all()  # Not all NaN
    
    def test_interpolate_with_projected_coordinates(self, simple_2d_grid, target_points_dict):
        """Test interpolation with projected coordinates (x/y)."""
        # Use projected coordinates
        target_points_dict = {
            'x': np.array([1000, 2000, 3000]),
            'y': np.array([1000, 2000, 3000])
        }
        
        interpolator = PointInterpolator(simple_2d_grid, target_points_dict, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3,)
    
    def test_interpolate_with_mixed_coordinate_names(self, simple_2d_grid):
        """Test interpolation with mixed coordinate names."""
        # Create target points with mixed coordinate names
        target_points = pd.DataFrame({
            'longitude': [1, 2, 3],
            'latitude': [4, 5, 6]
        })
        
        interpolator = PointInterpolator(simple_2d_grid, target_points, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3,)
    
    def test_interpolate_with_wrapped_longitudes(self, simple_2d_grid):
        """Test interpolation with wrapped longitudes (e.g., -180 to 180)."""
        # Create target points with wrapped longitudes
        target_points = pd.DataFrame({
            'longitude': [170, -170, 175, -175],  # Crosses the antimeridian
            'latitude': [0, 0, 10, -10]
        })
        
        interpolator = PointInterpolator(simple_2d_grid, target_points, method='bilinear')
        result = interpolator.interpolate()
        
        assert isinstance(result, xr.DataArray)
        assert result.shape == (4,)
        # Should handle wrapped coordinates gracefully
        assert not np.isnan(result.values).all()