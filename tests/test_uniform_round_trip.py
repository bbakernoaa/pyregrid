"""
Tests for the Uniform Round-Trip Test.

This test verifies that regridding a uniform field (all values the same)
from a low-resolution grid to a high-resolution grid and back to the
original low-resolution grid preserves the original field.
"""

import pytest
import numpy as np
import xarray as xr
from pyregrid import GridRegridder


def test_uniform_round_trip_basic():
    """
    Test uniform round-trip with basic bilinear interpolation.
    
    This test creates a uniform field (all values the same), regrids from
    a low-resolution grid to a high-resolution grid, and then regrids back
    to the original low-resolution grid. The final result should match the
    original uniform field exactly for a uniform field.
    """
    # Create a uniform field (all values the same)
    original_value = 42.0
    original_values = np.ones((5, 10)) * original_value  # All values are 42.0
    
    # Create original low-resolution grid
    original_lon = np.linspace(-180, 180, 10)
    original_lat = np.linspace(-90, 90, 5)
    
    original_da = xr.DataArray(
        original_values,
        dims=['lat', 'lon'],
        coords={
            'lon': original_lon,
            'lat': original_lat
        },
        name='uniform_field'
    )
    
    # Create high-resolution target grid
    target_lon = np.linspace(-180, 180, 20)  # Higher resolution
    target_lat = np.linspace(-90, 90, 10)    # Higher resolution
    
    target_da = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={
            'lon': target_lon,
            'lat': target_lat
        }
    )
    
    # Regrid from low-res to high-res
    regridder_forward = GridRegridder(
        source_grid=original_da,
        target_grid=target_da,
        method='bilinear'
    )
    
    regridded_high_res = regridder_forward.regrid(original_da)
    
    # Check that the high-res grid has the same uniform value (with small tolerance for floating point errors)
    expected_high_res = np.ones((10, 20)) * original_value
    np.testing.assert_allclose(
        regridded_high_res.values,
        expected_high_res,
        rtol=1e-10,
        atol=1e-10,
        err_msg="High-resolution grid should have same uniform value"
    )
    
    # Regrid back from high-res to low-res
    regridder_backward = GridRegridder(
        source_grid=regridded_high_res,
        target_grid=original_da,
        method='bilinear'
    )
    
    regridded_low_res = regridder_backward.regrid(regridded_high_res)
    
    # Verify that the final result matches the original uniform field exactly
    # For a uniform field, this should be exact with bilinear interpolation
    np.testing.assert_allclose(
        regridded_low_res.values,
        original_da.values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Uniform round-trip test failed: final result does not match original field"
    )


def test_uniform_round_trip_nearest():
    """
    Test uniform round-trip with nearest neighbor interpolation.
    
    Similar to the basic test but using nearest neighbor method.
    """
    # Create a uniform field (all values the same)
    original_value = 100.0
    original_values = np.ones((5, 10)) * original_value  # All values are 100.0
    
    # Create original low-resolution grid
    original_lon = np.linspace(-180, 180, 10)
    original_lat = np.linspace(-90, 90, 5)
    
    original_da = xr.DataArray(
        original_values,
        dims=['lat', 'lon'],
        coords={
            'lon': original_lon,
            'lat': original_lat
        },
        name='uniform_field'
    )
    
    # Create high-resolution target grid
    target_lon = np.linspace(-180, 180, 20)  # Higher resolution
    target_lat = np.linspace(-90, 90, 10)    # Higher resolution
    
    target_da = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={
            'lon': target_lon,
            'lat': target_lat
        }
    )
    
    # Regrid from low-res to high-res
    regridder_forward = GridRegridder(
        source_grid=original_da,
        target_grid=target_da,
        method='nearest'
    )
    
    regridded_high_res = regridder_forward.regrid(original_da)
    
    # Regrid back from high-res to low-res
    regridder_backward = GridRegridder(
        source_grid=regridded_high_res,
        target_grid=original_da,
        method='nearest'
    )
    
    regridded_low_res = regridder_backward.regrid(regridded_high_res)
    
    # Verify that the final result matches the original uniform field
    np.testing.assert_allclose(
        regridded_low_res.values,
        original_da.values,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Uniform round-trip test with nearest neighbor failed: final result does not match original field"
    )


def test_uniform_round_trip_cubic():
    """
    Test uniform round-trip with cubic interpolation.
    """
    # Create a uniform field (all values the same)
    original_value = 55.5
    original_values = np.ones((5, 10)) * original_value  # All values are 55.5
    
    # Create original low-resolution grid
    original_lon = np.linspace(-180, 180, 10)
    original_lat = np.linspace(-90, 90, 5)
    
    original_da = xr.DataArray(
        original_values,
        dims=['lat', 'lon'],
        coords={
            'lon': original_lon,
            'lat': original_lat
        },
        name='uniform_field'
    )
    
    # Create high-resolution target grid
    target_lon = np.linspace(-180, 180, 20)  # Higher resolution
    target_lat = np.linspace(-90, 90, 10)    # Higher resolution
    
    target_da = xr.DataArray(
        np.zeros((10, 20)),
        dims=['lat', 'lon'],
        coords={
            'lon': target_lon,
            'lat': target_lat
        }
    )
    
    # Regrid from low-res to high-res
    regridder_forward = GridRegridder(
        source_grid=original_da,
        target_grid=target_da,
        method='cubic'
    )
    
    regridded_high_res = regridder_forward.regrid(original_da)
    
    # Regrid back from high-res to low-res
    regridder_backward = GridRegridder(
        source_grid=regridded_high_res,
        target_grid=original_da,
        method='cubic'
    )
    
    regridded_low_res = regridder_backward.regrid(regridded_high_res)
    
    # Verify that the final result matches the original uniform field
    # For a uniform field, cubic interpolation should preserve the uniform value
    np.testing.assert_allclose(
        regridded_low_res.values,
        original_da.values,
        rtol=1e-5, # Cubic interpolation might have small numerical errors
        atol=1e-5,
        err_msg="Uniform round-trip test with cubic failed: final result does not match original field"
    )


if __name__ == "__main__":
    # Run the tests if executed directly
    pytest.main([__file__, "-v"])
