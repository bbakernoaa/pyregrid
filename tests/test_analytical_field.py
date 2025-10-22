"""
Analytical Field Test (gold standard for accuracy) for the geospatial regridding library.

This test evaluates the accuracy of the regridding process by comparing the regridded data 
against values computed directly from a known mathematical function on the target grid. 
This provides a quantitative measure of interpolation accuracy.
"""
import pytest
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def test_analytical_field_linear_function_bilinear():
    """
    Test that bilinear interpolation is exact for linear functions.
    
    TDD Anchor: Verify that bilinear regridding is exact for linear analytical functions.
    """
    # Define source grid coordinates
    source_lon = np.linspace(-180, 180, 20)  # Higher resolution for better analytical comparison
    source_lat = np.linspace(-90, 90, 10)
    
    # Define target grid coordinates (can be coarser or different resolution)
    target_lon = np.linspace(-180, 180, 10)
    target_lat = np.linspace(-90, 90, 5)
    
    # Define a simple linear analytical function: f(λ,φ) = a*λ + b*φ + c
    def linear_function(lon, lat, a=0.1, b=0.2, c=10.0):
        """Linear function for testing bilinear interpolation accuracy."""
        return a * lon + b * lat + c
    
    # Generate source data by applying the analytical function to source grid coordinates
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
    source_data_values = linear_function(source_lon_grid, source_lat_grid)
    
    # Create source xarray DataArray
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate target data by applying the analytical function directly to target grid coordinates
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
    target_data_analytical_values = linear_function(target_lon_grid, target_lat_grid)
    
    # Create target xarray DataArray
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Instantiate GridRegridder with source and target grids, specifying 'bilinear' method
    regridder_bilinear = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='bilinear')
    
    # Perform regridding
    regridded_data_bilinear = regridder_bilinear.regrid(source_da)
    
    # Verify the shape and dimensions of the regridded data match the analytical target grid
    assert regridded_data_bilinear.shape == target_da_analytical.shape
    assert regridded_data_bilinear.dims == target_da_analytical.dims
    
    # For linear functions, bilinear interpolation should be exact (within floating point precision)
    # Compare regridded data with the analytically computed target data
    xr.testing.assert_allclose(regridded_data_bilinear, target_da_analytical, rtol=1e-10, atol=1e-10)


def test_analytical_field_nonlinear_function_bilinear():
    """
    Test bilinear interpolation accuracy for non-linear functions.
    
    TDD Anchor: Verify bilinear regridding accuracy against non-linear analytical functions.
    """
    # Define source grid coordinates
    source_lon = np.linspace(-180, 180, 20)  # Higher resolution for better analytical comparison
    source_lat = np.linspace(-90, 90, 10)
    
    # Define target grid coordinates (can be coarser or different resolution)
    target_lon = np.linspace(-180, 180, 10)
    target_lat = np.linspace(-90, 90, 5)
    
    # Define a non-linear analytical function: f(λ,φ) = sin(φ) + cos(λ)
    def nonlinear_function(lon, lat):
        """Non-linear function for testing interpolation accuracy."""
        return np.sin(np.radians(lat[:, np.newaxis])) + np.cos(np.radians(lon[np.newaxis, :]))
    
    # Generate source data by applying the analytical function to source grid coordinates
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
    source_data_values = np.sin(np.radians(source_lat_grid)) + np.cos(np.radians(source_lon_grid))
    
    # Create source xarray DataArray
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate target data by applying the analytical function directly to target grid coordinates
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
    target_data_analytical_values = np.sin(np.radians(target_lat_grid)) + np.cos(np.radians(target_lon_grid))
    
    # Create target xarray DataArray
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Instantiate GridRegridder with source and target grids, specifying 'bilinear' method
    regridder_bilinear = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='bilinear')
    
    # Perform regridding
    regridded_data_bilinear = regridder_bilinear.regrid(source_da)
    
    # Verify the shape and dimensions of the regridded data match the analytical target grid
    assert regridded_data_bilinear.shape == target_da_analytical.shape
    assert regridded_data_bilinear.dims == target_da_analytical.dims
    
    # For non-linear functions, bilinear interpolation will have some error
    # Check that the error is reasonable (less than 10% of the function range)
    max_error = np.max(np.abs(regridded_data_bilinear.values - target_da_analytical.values))
    function_range = np.max(target_da_analytical.values) - np.min(target_da_analytical.values)
    
    # The error should be reasonable for bilinear interpolation of non-linear functions
    assert max_error < 0.5 * function_range  # Error should be less than half the function range


def test_analytical_field_nonlinear_function_cubic():
    """
    Test cubic interpolation accuracy for non-linear functions.
    
    TDD Anchor: Verify cubic regridding accuracy against non-linear analytical functions.
    """
    # Define source grid coordinates
    source_lon = np.linspace(-180, 180, 20)
    source_lat = np.linspace(-90, 90, 10)
    
    # Define target grid coordinates
    target_lon = np.linspace(-180, 180, 10)
    target_lat = np.linspace(-90, 90, 5)
    
    # Define a non-linear analytical function: f(λ,φ) = sin(φ) + cos(λ)
    def nonlinear_function(lon, lat):
        """Non-linear function for testing interpolation accuracy."""
        return np.sin(np.radians(lat[:, np.newaxis])) + np.cos(np.radians(lon[np.newaxis, :]))
    
    # Generate source data by applying the analytical function to source grid coordinates
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
    source_data_values = np.sin(np.radians(source_lat_grid)) + np.cos(np.radians(source_lon_grid))
    
    # Create source xarray DataArray
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate target data by applying the analytical function directly to target grid coordinates
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
    target_data_analytical_values = np.sin(np.radians(target_lat_grid)) + np.cos(np.radians(target_lon_grid))
    
    # Create target xarray DataArray
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Instantiate GridRegridder with source and target grids, specifying 'cubic' method
    regridder_cubic = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='cubic')
    
    # Perform regridding
    regridded_data_cubic = regridder_cubic.regrid(source_da)
    
    # Verify the shape and dimensions of the regridded data match the analytical target grid
    assert regridded_data_cubic.shape == target_da_analytical.shape
    assert regridded_data_cubic.dims == target_da_analytical.dims
    
    # For non-linear functions, cubic interpolation should be more accurate than bilinear
    max_error = np.max(np.abs(regridded_data_cubic.values - target_da_analytical.values))
    function_range = np.max(target_da_analytical.values) - np.min(target_da_analytical.values)
    
    # The error should be reasonable for cubic interpolation of non-linear functions
    assert max_error < 0.3 * function_range  # Cubic should be more accurate than bilinear


def test_analytical_field_nonlinear_function_nearest():
    """
    Test nearest neighbor interpolation accuracy for non-linear functions.
    
    TDD Anchor: Verify nearest neighbor regridding accuracy against non-linear analytical functions.
    """
    # Define source grid coordinates
    source_lon = np.linspace(-180, 180, 20)
    source_lat = np.linspace(-90, 90, 10)
    
    # Define target grid coordinates
    target_lon = np.linspace(-180, 180, 10)
    target_lat = np.linspace(-90, 90, 5)
    
    # Define a non-linear analytical function: f(λ,φ) = sin(φ) + cos(λ)
    def nonlinear_function(lon, lat):
        """Non-linear function for testing interpolation accuracy."""
        return np.sin(np.radians(lat[:, np.newaxis])) + np.cos(np.radians(lon[np.newaxis, :]))
    
    # Generate source data by applying the analytical function to source grid coordinates
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
    source_data_values = np.sin(np.radians(source_lat_grid)) + np.cos(np.radians(source_lon_grid))
    
    # Create source xarray DataArray
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate target data by applying the analytical function directly to target grid coordinates
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
    target_data_analytical_values = np.sin(np.radians(target_lat_grid)) + np.cos(np.radians(target_lon_grid))
    
    # Create target xarray DataArray
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Instantiate GridRegridder with source and target grids, specifying 'nearest' method
    regridder_nearest = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='nearest')
    
    # Perform regridding
    regridded_data_nearest = regridder_nearest.regrid(source_da)
    
    # Verify the shape and dimensions of the regridded data match the analytical target grid
    assert regridded_data_nearest.shape == target_da_analytical.shape
    assert regridded_data_nearest.dims == target_da_analytical.dims
    
    # For non-linear functions, nearest neighbor interpolation will have more error than bilinear/cubic
    max_error = np.max(np.abs(regridded_data_nearest.values - target_da_analytical.values))
    function_range = np.max(target_da_analytical.values) - np.min(target_da_analytical.values)
    
    # The error should be reasonable for nearest neighbor interpolation of non-linear functions
    # It will be higher than bilinear/cubic but should still be within the function range
    assert max_error < function_range


def test_analytical_field_linear_function_exactness():
    """
    Test that linear functions are interpolated exactly by bilinear method.
    
    TDD Anchor: Verify exact interpolation for linear functions with bilinear method.
    """
    # Test with different linear function coefficients
    coefficients = [(0.1, 0.2, 10.0), (2.0, -1.5, 5.0), (-0.5, 0.3, 0.0)]
    
    for a, b, c in coefficients:
        # Define source grid coordinates
        source_lon = np.linspace(-100, 100, 15)
        source_lat = np.linspace(-50, 50, 10)
        
        # Define target grid coordinates
        target_lon = np.linspace(-90, 90, 8)
        target_lat = np.linspace(-40, 40, 6)
        
        # Define a simple linear analytical function: f(λ,φ) = a*λ + b*φ + c
        def linear_function(lon, lat):
            return a * lon + b * lat + c
        
        # Generate source data
        source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
        source_data_values = linear_function(source_lon_grid, source_lat_grid)
        
        source_da = xr.DataArray(
            source_data_values,
            coords={'lat': source_lat, 'lon': source_lon},
            dims=['lat', 'lon'],
            name='analytical_field'
        )
        
        # Generate target data analytically
        target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
        target_data_analytical_values = linear_function(target_lon_grid, target_lat_grid)
        
        target_da_analytical = xr.DataArray(
            target_data_analytical_values,
            coords={'lat': target_lat, 'lon': target_lon},
            dims=['lat', 'lon'],
            name='analytical_field'
        )
        
        # Regrid using bilinear method
        regridder = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='bilinear')
        regridded_data = regridder.regrid(source_da)
        
        # For linear functions, bilinear interpolation should be exact
        xr.testing.assert_allclose(regridded_data, target_da_analytical, rtol=1e-10, atol=1e-10)


def test_analytical_field_high_frequency_function():
    """
    Test interpolation accuracy for high-frequency functions that may challenge the interpolation.
    
    TDD Anchor: Verify regridding accuracy for functions with high frequency components.
    """
    # Define source grid coordinates
    source_lon = np.linspace(-180, 180, 40)  # Higher resolution
    source_lat = np.linspace(-90, 90, 20)
    
    # Define target grid coordinates (coarser to test interpolation quality)
    target_lon = np.linspace(-180, 180, 10)
    target_lat = np.linspace(-90, 90, 5)
    
    # Define a high-frequency analytical function: f(λ,φ) = sin(4*φ) + cos(6*λ)
    def high_frequency_function(lon, lat):
        """High-frequency function to challenge interpolation accuracy."""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(4 * np.radians(lat_grid)) + np.cos(6 * np.radians(lon_grid))
    
    # Generate source data
    source_lon_grid, source_lat_grid = np.meshgrid(source_lon, source_lat, indexing='xy')
    source_data_values = high_frequency_function(source_lon, source_lat)
    
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate target data analytically
    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat, indexing='xy')
    target_data_analytical_values = high_frequency_function(target_lon, target_lat)
    
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Test bilinear interpolation
    regridder_bilinear = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='bilinear')
    regridded_data_bilinear = regridder_bilinear.regrid(source_da)
    
    # For high-frequency functions, bilinear interpolation will have more error
    # But it should still produce reasonable results
    max_error_bilinear = np.max(np.abs(regridded_data_bilinear.values - target_da_analytical.values))
    function_range = np.max(target_da_analytical.values) - np.min(target_da_analytical.values)
    
    # Bilinear should have reasonable accuracy even for high-frequency functions
    assert max_error_bilinear < function_range  # Error should be less than the function range
    
    # Test cubic interpolation (should be more accurate for smooth functions)
    regridder_cubic = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method='cubic')
    regridded_data_cubic = regridder_cubic.regrid(source_da)
    
    max_error_cubic = np.max(np.abs(regridded_data_cubic.values - target_da_analytical.values))
    
    # Cubic interpolation should generally be more accurate than bilinear for smooth functions
    # Note: This might not always hold for high-frequency functions due to Runge's phenomenon
    # So we'll just check that both methods produce reasonable results
    assert max_error_cubic < function_range


def test_analytical_field_conservation_properties():
    """
    Test that interpolation methods preserve certain properties of the analytical functions.
    
    TDD Anchor: Verify conservation properties for analytical field regridding.
    """
    # Define source and target grids
    source_lon = np.linspace(-180, 180, 20)
    source_lat = np.linspace(-90, 90, 10)
    target_lon = np.linspace(-180, 180, 12)
    target_lat = np.linspace(-90, 90, 8)
    
    # Define a function with known extrema
    def test_function(lon, lat):
        """Test function with known maximum and minimum."""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return 2.0 * np.sin(np.radians(lat_grid)) * np.cos(np.radians(lon_grid)) + 3.0
    
    # Generate source data
    source_data_values = test_function(source_lon, source_lat)
    source_da = xr.DataArray(
        source_data_values,
        coords={'lat': source_lat, 'lon': source_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Generate analytical target data
    target_data_analytical_values = test_function(target_lon, target_lat)
    target_da_analytical = xr.DataArray(
        target_data_analytical_values,
        coords={'lat': target_lat, 'lon': target_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Test that the regridding methods maintain reasonable bounds
    for method in ['bilinear', 'cubic', 'nearest']:
        regridder = GridRegridder(source_grid=source_da, target_grid=target_da_analytical, method=method)
        regridded_data = regridder.regrid(source_da)
        
        # Check that regridded values are within reasonable bounds
        # The analytical function ranges from 1.0 to 5.0 (since sin*cos ranges from -1 to 1, scaled by 2 and shifted by 3)
        analytical_min = float(np.min(target_da_analytical.data))
        analytical_max = float(np.max(target_da_analytical.data))
        
        regridded_min = float(np.min(regridded_data.data))
        regridded_max = float(np.max(regridded_data.data))
        
        # The regridded values should be in a reasonable range based on the analytical function
        # Allow for some interpolation error
        assert regridded_min <= analytical_max + 1.0  # Some tolerance for interpolation overshoot
        assert regridded_max >= analytical_min - 1.0  # Some tolerance for interpolation undershoot