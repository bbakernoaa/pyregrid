"""
Tests for the CRS management functionality in PyRegrid.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from pyproj import CRS

from pyregrid.crs.crs_manager import CRSManager


class TestCRSManager:
    """Test the CRSManager class functionality."""
    
    def test_crs_detection_from_xarray_with_crs_coord(self):
        """Test CRS detection from xarray with CRS coordinate."""
        crs_manager = CRSManager()
        
        # Create test data with CRS coordinate
        lon = np.linspace(-10, 10, 5)
        lat = np.linspace(40, 50, 4)
        data = np.random.random((4, 5))
        
        ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], data)},
            coords={
                'lon': lon,
                'lat': lat,
                'crs': ([], 1)  # dummy CRS coordinate
            }
        )
        ds.coords['crs'].attrs['crs_wkt'] = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]'
        
        detected_crs = crs_manager.parse_crs_from_xarray(ds)
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
    
    def test_crs_detection_from_xarray_with_attrs(self):
        """Test CRS detection from xarray attributes."""
        crs_manager = CRSManager()
        
        # Create test data with CRS in attributes
        lon = np.linspace(-10, 10, 5)
        lat = np.linspace(40, 50, 4)
        data = np.random.random((4, 5))
        
        ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], data)},
            coords={
                'lon': lon,
                'lat': lat
            }
        )
        ds.attrs['crs'] = 'EPSG:4326'
        
        detected_crs = crs_manager.parse_crs_from_xarray(ds)
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
    
    def test_crs_detection_from_dataframe(self):
        """Test CRS detection from DataFrame."""
        crs_manager = CRSManager()
        
        # Create test DataFrame with lat/lon columns
        df = pd.DataFrame({
            'latitude': [40.0, 41.0, 42.0],
            'longitude': [-10.0, -9.0, -8.0],
            'value': [1.0, 2.0, 3.0]
        })
        df.attrs = {'crs': 'EPSG:4326'}
        
        detected_crs = crs_manager.parse_crs_from_dataframe(df)
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
    
    def test_detect_coordinate_system_type(self):
        """Test coordinate system type detection."""
        crs_manager = CRSManager()
        
        # Test geographic CRS
        geographic_crs = CRS.from_epsg(4326)  # WGS 84
        assert crs_manager.detect_coordinate_system_type(geographic_crs) == 'geographic'
        
        # Test projected CRS
        projected_crs = CRS.from_epsg(3857)  # Web Mercator
        assert crs_manager.detect_coordinate_system_type(projected_crs) == 'projected'
        
        # Test with None
        assert crs_manager.detect_coordinate_system_type(None) == 'unknown'
    
    def test_validate_coordinate_arrays(self):
        """Test coordinate array validation."""
        crs_manager = CRSManager()
        
        # Valid coordinates
        x_coords = np.array([0.0, 1.0, 2.0])
        y_coords = np.array([0.0, 1.0, 2.0])
        assert crs_manager.validate_coordinate_arrays(x_coords, y_coords)
        
        # Coordinates with NaN
        x_coords_nan = np.array([0.0, np.nan, 2.0])
        assert not crs_manager.validate_coordinate_arrays(x_coords_nan, y_coords)
        
        # Coordinates with infinite values
        x_coords_inf = np.array([0.0, np.inf, 2.0])
        assert not crs_manager.validate_coordinate_arrays(x_coords_inf, y_coords)
        
        # Mismatched shapes
        x_coords_mismatch = np.array([0.0, 1.0])
        y_coords_mismatch = np.array([0.0, 1.0, 2.0])
        assert not crs_manager.validate_coordinate_arrays(x_coords_mismatch, y_coords_mismatch)
        
        # Valid geographic coordinates
        geographic_crs = CRS.from_epsg(4326)
        lon_coords = np.array([-10.0, 0.0, 10.0])
        lat_coords = np.array([40.0, 45.0, 50.0])
        assert crs_manager.validate_coordinate_arrays(lon_coords, lat_coords, geographic_crs)
        
        # Invalid geographic coordinates (out of range)
        invalid_lon = np.array([-200.0, 0.0, 200.0])
        invalid_lat = np.array([100.0, 0.0, -100.0])
        assert not crs_manager.validate_coordinate_arrays(invalid_lon, invalid_lat, geographic_crs)
    
    def test_detect_crs_from_coordinates(self):
        """Test CRS detection from coordinate names and values."""
        crs_manager = CRSManager()
        
        # Test with lat/lon names and valid geographic values
        x_coords = np.array([-10.0, 0.0, 10.0])
        y_coords = np.array([40.0, 45.0, 50.0])
        
        # Test with 'longitude' and 'latitude' names
        detected_crs = crs_manager.detect_crs_from_coordinates(x_coords, y_coords, 'longitude', 'latitude')
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
        
        # Test with 'lon' and 'lat' names
        detected_crs = crs_manager.detect_crs_from_coordinates(x_coords, y_coords, 'lon', 'lat')
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
        
        # Test with non-geographic names (should return None)
        detected_crs = crs_manager.detect_crs_from_coordinates(x_coords, y_coords, 'x', 'y')
        assert detected_crs is None
    
    def test_transform_coordinates(self):
        """Test coordinate transformation."""
        crs_manager = CRSManager()
        
        # Test transformation from WGS 84 to Web Mercator
        x_coords = np.array([-10.0, 0.0, 10.0])
        y_coords = np.array([40.0, 45.0, 50.0])
        
        source_crs = CRS.from_epsg(4326)  # WGS 84
        target_crs = CRS.from_epsg(3857)  # Web Mercator
        
        transformed_x, transformed_y = crs_manager.transform_coordinates(
            x_coords, y_coords, source_crs, target_crs
        )
        
        # Check that the transformation changed the values
        assert not np.array_equal(transformed_x, x_coords)
        assert not np.array_equal(transformed_y, y_coords)
        
        # Check that the transformed coordinates are in reasonable ranges for Web Mercator
        assert np.all(np.abs(transformed_x) < 2e7)  # Web Mercator X is typically within ±20M meters
        assert np.all(np.abs(transformed_y) < 2e7)  # Web Mercator Y is typically within ±20M meters
    
    def test_get_crs_from_source_with_explicit_crs(self):
        """Test getting CRS from source with explicit CRS information."""
        crs_manager = CRSManager()
        
        # Test with xarray dataset that has explicit CRS
        lon = np.linspace(-10, 10, 5)
        lat = np.linspace(40, 50, 4)
        data = np.random.random((4, 5))
        
        ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], data)},
            coords={
                'lon': lon,
                'lat': lat
            }
        )
        ds.attrs['crs'] = 'EPSG:4326'
        
        x_coords = np.array([-5.0, 0.0, 5.0])
        y_coords = np.array([42.0, 45.0, 48.0])
        
        detected_crs = crs_manager.get_crs_from_source(ds, x_coords, y_coords, 'lon', 'lat')
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
    
    def test_get_crs_from_source_with_lat_lon_names(self):
        """Test getting CRS from source with lat/lon names and geographic values."""
        crs_manager = CRSManager()
        
        # Create a dummy source without explicit CRS
        dummy_source = xr.Dataset()
        
        x_coords = np.array([-10.0, 0.0, 10.0])
        y_coords = np.array([40.0, 45.0, 50.0])
        
        # With lat/lon names and valid geographic values, should assume WGS 84
        detected_crs = crs_manager.get_crs_from_source(dummy_source, x_coords, y_coords, 'longitude', 'latitude')
        assert detected_crs is not None
        assert detected_crs.to_epsg() == 4326
    
    def test_get_crs_from_source_with_ambiguous_coords(self):
        """Test getting CRS from source with ambiguous coordinate names."""
        crs_manager = CRSManager()
        
        # Create a dummy source without explicit CRS
        dummy_source = xr.Dataset()
        
        # Coordinates that look like lat/lon but are outside geographic range
        x_coords = np.array([1000.0, 2000.0, 3000.0])  # Outside geographic range
        y_coords = np.array([1000.0, 2000.0, 3000.0])  # Outside geographic range
        
        # This should raise an error because names suggest lat/lon but values don't match
        with pytest.raises(ValueError):
            crs_manager.get_crs_from_source(dummy_source, x_coords, y_coords, 'longitude', 'latitude')
    
    def test_get_crs_from_source_without_clear_identification(self):
        """Test getting CRS from source without clear coordinate identification."""
        crs_manager = CRSManager()
        
        # Create a dummy source without explicit CRS
        dummy_source = xr.Dataset()
        
        # Coordinates with non-descriptive names and values
        x_coords = np.array([1000.0, 2000.0, 3000.0])
        y_coords = np.array([1000.0, 2000.0, 3000.0])
        
        # This should raise an error because coordinates are not clearly lat/lon
        with pytest.raises(ValueError):
            crs_manager.get_crs_from_source(dummy_source, x_coords, y_coords, 'x', 'y')
    
    def test_ensure_crs_compatibility(self):
        """Test ensuring CRS compatibility."""
        crs_manager = CRSManager()
        
        source_crs = CRS.from_epsg(4326)
        target_crs = CRS.from_epsg(3857)
        
        # Both CRS provided - should work
        result_source, result_target = crs_manager.ensure_crs_compatibility(source_crs, target_crs)
        assert result_source == source_crs
        assert result_target == target_crs
        
        # Missing source CRS - should raise error
        with pytest.raises(ValueError):
            crs_manager.ensure_crs_compatibility(None, target_crs)
        
        # Missing target CRS - should raise error
        with pytest.raises(ValueError):
            crs_manager.ensure_crs_compatibility(source_crs, None)


def test_crs_manager_integration():
    """Test the overall integration of CRS manager functionality."""
    crs_manager = CRSManager()
    
    # Test a complete workflow
    x_coords = np.array([-5.0, 0.0, 5.0])
    y_coords = np.array([42.0, 45.0, 48.0])
    
    # Detect CRS from coordinates that look like lat/lon
    detected_crs = crs_manager.detect_crs_from_coordinates(x_coords, y_coords, 'longitude', 'latitude')
    assert detected_crs is not None
    
    # Validate the coordinates
    is_valid = crs_manager.validate_coordinate_arrays(x_coords, y_coords, detected_crs)
    assert is_valid
    
    # Transform to another CRS
    target_crs = CRS.from_epsg(3857)
    transformed_x, transformed_y = crs_manager.transform_coordinates(
        x_coords, y_coords, detected_crs, target_crs
    )
    
    # Verify transformation occurred
    assert not np.array_equal(transformed_x, x_coords)
    assert not np.array_equal(transformed_y, y_coords)