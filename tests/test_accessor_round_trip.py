"""
Accessor Round-Trip Test for the geospatial regridding library.

This test evaluates the stability and lossiness of the regridding process through the xarray accessor by:
1. Starting with a low-resolution grid containing analytical data (e.g., f(λ,φ) = sin(φ))
2. Regridding from low-resolution to high-resolution (upsample) using the .pyregrid accessor
3. Regridding the high-resolution data back to the original low-resolution (downsample) using the .pyregrid accessor
4. Comparing the final low-resolution grid with the original low-resolution grid
5. Verifying that the process is stable (doesn't create wild values) and characterizing the diffusive characteristics
6. Testing with different interpolation methods (nearest neighbor, bilinear, cubic)
"""
import pytest
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def test_accessor_round_trip_stability_nearest():
    """
    Test round-trip stability with nearest neighbor interpolation using the xarray accessor.
    
    TDD Anchor: Verify that nearest neighbor regridding through the accessor is stable in round-trip operations.
    """
    # Define low-resolution grid (source)
    low_res_lon = np.linspace(-180, 180, 10)  # 10 points
    low_res_lat = np.linspace(-90, 90, 5)     # 5 points
    
    # Define high-resolution grid (target for upsample)
    high_res_lon = np.linspace(-180, 180, 20)  # 20 points
    high_res_lat = np.linspace(-90, 90, 10)    # 10 points
    
    # Define analytical function: f(λ,φ) = sin(φ)
    def analytical_function(lon, lat):
        """Analytical function for testing: f(λ,φ) = sin(φ)"""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(np.radians(lat_grid))
    
    # Generate low-resolution source data
    low_res_data_values = analytical_function(low_res_lon, low_res_lat)
    low_res_da = xr.DataArray(
        low_res_data_values,
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Create target grids as DataArrays
    high_res_target_da = xr.DataArray(
        np.zeros((len(high_res_lat), len(high_res_lon))),
        coords={'lat': high_res_lat, 'lon': high_res_lon},
        dims=['lat', 'lon']
    )
    
    low_res_target_da = xr.DataArray(
        np.zeros((len(low_res_lat), len(low_res_lon))),
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon']
    )
    
    # Step 1: Regrid from low-resolution to high-resolution (upsample) using nearest neighbor through accessor
    high_res_data = low_res_da.pyregrid.regrid_to(high_res_target_da, method='nearest')
    
    # Step 2: Regrid from high-resolution back to low-resolution (downsample) using nearest neighbor through accessor
    final_low_res_data = high_res_data.pyregrid.regrid_to(low_res_target_da, method='nearest')
    
    # Step 3: Compare final low-resolution grid with original low-resolution grid
    # Check for stability (no wild values)
    original_min = float(np.min(low_res_da.data))
    original_max = float(np.max(low_res_da.data))
    final_min = float(np.min(final_low_res_data.data))
    final_max = float(np.max(final_low_res_data.data))
    
    # Values should remain within reasonable bounds
    assert final_min >= original_min - 0.1, f"Final min {final_min} is too low compared to original min {original_min}"
    assert final_max <= original_max + 0.1, f"Final max {final_max} is too high compared to original max {original_max}"
    
    # Check that the values are not wildly different
    max_difference = np.max(np.abs(final_low_res_data.data - low_res_da.data))
    assert max_difference < 1.0, f"Round-trip difference {max_difference} is too large"
    
    # The shape should match the original
    assert final_low_res_data.shape == low_res_da.shape


def test_accessor_round_trip_stability_bilinear():
    """
    Test round-trip stability with bilinear interpolation using the xarray accessor.
    
    TDD Anchor: Verify that bilinear regridding through the accessor is stable in round-trip operations.
    """
    # Define low-resolution grid (source)
    low_res_lon = np.linspace(-180, 180, 10)  # 10 points
    low_res_lat = np.linspace(-90, 90, 5)     # 5 points
    
    # Define high-resolution grid (target for upsample)
    high_res_lon = np.linspace(-180, 180, 20)  # 20 points
    high_res_lat = np.linspace(-90, 90, 10)    # 10 points
    
    # Define analytical function: f(λ,φ) = sin(φ)
    def analytical_function(lon, lat):
        """Analytical function for testing: f(λ,φ) = sin(φ)"""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(np.radians(lat_grid))
    
    # Generate low-resolution source data
    low_res_data_values = analytical_function(low_res_lon, low_res_lat)
    low_res_da = xr.DataArray(
        low_res_data_values,
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Create target grids as DataArrays
    high_res_target_da = xr.DataArray(
        np.zeros((len(high_res_lat), len(high_res_lon))),
        coords={'lat': high_res_lat, 'lon': high_res_lon},
        dims=['lat', 'lon']
    )
    
    low_res_target_da = xr.DataArray(
        np.zeros((len(low_res_lat), len(low_res_lon))),
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon']
    )
    
    # Step 1: Regrid from low-resolution to high-resolution (upsample) using bilinear through accessor
    high_res_data = low_res_da.pyregrid.regrid_to(high_res_target_da, method='bilinear')
    
    # Step 2: Regrid from high-resolution back to low-resolution (downsample) using bilinear through accessor
    final_low_res_data = high_res_data.pyregrid.regrid_to(low_res_target_da, method='bilinear')
    
    # Step 3: Compare final low-resolution grid with original low-resolution grid
    # Check for stability (no wild values)
    original_min = float(np.min(low_res_da.data))
    original_max = float(np.max(low_res_da.data))
    final_min = float(np.min(final_low_res_data.data))
    final_max = float(np.max(final_low_res_data.data))
    
    # Values should remain within reasonable bounds
    assert final_min >= original_min - 0.1, f"Final min {final_min} is too low compared to original min {original_min}"
    assert final_max <= original_max + 0.1, f"Final max {final_max} is too high compared to original max {original_max}"
    
    # Check that the values are not wildly different
    max_difference = np.max(np.abs(final_low_res_data.data - low_res_da.data))
    assert max_difference < 1.0, f"Round-trip difference {max_difference} is too large"
    
    # The shape should match the original
    assert final_low_res_data.shape == low_res_da.shape


def test_accessor_round_trip_stability_cubic():
    """
    Test round-trip stability with cubic interpolation using the xarray accessor.
    
    TDD Anchor: Verify that cubic regridding through the accessor is stable in round-trip operations.
    """
    # Define low-resolution grid (source)
    low_res_lon = np.linspace(-180, 180, 10) # 10 points
    low_res_lat = np.linspace(-90, 90, 5)     # 5 points
    
    # Define high-resolution grid (target for upsample)
    high_res_lon = np.linspace(-180, 180, 20)  # 20 points
    high_res_lat = np.linspace(-90, 90, 10)    # 10 points
    
    # Define analytical function: f(λ,φ) = sin(φ)
    def analytical_function(lon, lat):
        """Analytical function for testing: f(λ,φ) = sin(φ)"""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(np.radians(lat_grid))
    
    # Generate low-resolution source data
    low_res_data_values = analytical_function(low_res_lon, low_res_lat)
    low_res_da = xr.DataArray(
        low_res_data_values,
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Create target grids as DataArrays
    high_res_target_da = xr.DataArray(
        np.zeros((len(high_res_lat), len(high_res_lon))),
        coords={'lat': high_res_lat, 'lon': high_res_lon},
        dims=['lat', 'lon']
    )
    
    low_res_target_da = xr.DataArray(
        np.zeros((len(low_res_lat), len(low_res_lon))),
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon']
    )
    
    # Step 1: Regrid from low-resolution to high-resolution (upsample) using cubic through accessor
    high_res_data = low_res_da.pyregrid.regrid_to(high_res_target_da, method='cubic')
    
    # Step 2: Regrid from high-resolution back to low-resolution (downsample) using cubic through accessor
    final_low_res_data = high_res_data.pyregrid.regrid_to(low_res_target_da, method='cubic')
    
    # Step 3: Compare final low-resolution grid with original low-resolution grid
    # Check for stability (no wild values)
    original_min = float(np.min(low_res_da.data))
    original_max = float(np.max(low_res_da.data))
    final_min = float(np.min(final_low_res_data.data))
    final_max = float(np.max(final_low_res_data.data))
    
    # Values should remain within reasonable bounds
    assert final_min >= original_min - 0.1, f"Final min {final_min} is too low compared to original min {original_min}"
    assert final_max <= original_max + 0.1, f"Final max {final_max} is too high compared to original max {original_max}"
    
    # Check that the values are not wildly different
    max_difference = np.max(np.abs(final_low_res_data.data - low_res_da.data))
    assert max_difference < 1.0, f"Round-trip difference {max_difference} is too large"
    
    # The shape should match the original
    assert final_low_res_data.shape == low_res_da.shape


def test_accessor_round_trip_characterization():
    """
    Test round-trip diffusive characteristics for different methods using the xarray accessor.
    
    TDD Anchor: Characterize the diffusive behavior of different interpolation methods through the accessor in round-trip operations.
    """
    # Define low-resolution grid (source)
    low_res_lon = np.linspace(-180, 180, 8)   # 8 points
    low_res_lat = np.linspace(-90, 90, 4)     # 4 points
    
    # Define high-resolution grid (target for upsample)
    high_res_lon = np.linspace(-180, 180, 16) # 16 points
    high_res_lat = np.linspace(-90, 90, 8)    # 8 points
    
    # Define analytical function: f(λ,φ) = sin(φ) + cos(λ)
    def analytical_function(lon, lat):
        """Analytical function for testing: f(λ,φ) = sin(φ) + cos(λ)"""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(np.radians(lat_grid)) + np.cos(np.radians(lon_grid))
    
    # Generate low-resolution source data
    low_res_data_values = analytical_function(low_res_lon, low_res_lat)
    low_res_da = xr.DataArray(
        low_res_data_values,
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Create target grids as DataArrays
    high_res_target_da = xr.DataArray(
        np.zeros((len(high_res_lat), len(high_res_lon))),
        coords={'lat': high_res_lat, 'lon': high_res_lon},
        dims=['lat', 'lon']
    )
    
    low_res_target_da = xr.DataArray(
        np.zeros((len(low_res_lat), len(low_res_lon))),
        coords={'lat': low_res_lat, 'lon': low_res_lon},
        dims=['lat', 'lon']
    )
    
    # Test different interpolation methods
    methods = ['nearest', 'bilinear', 'cubic']
    results = {}
    
    for method in methods:
        # Step 1: Regrid from low-resolution to high-resolution (upsample) using accessor
        high_res_data = low_res_da.pyregrid.regrid_to(high_res_target_da, method=method)
        
        # Step 2: Regrid from high-resolution back to low-resolution (downsample) using accessor
        final_low_res_data = high_res_data.pyregrid.regrid_to(low_res_target_da, method=method)
        
        # Step 3: Calculate round-trip error
        round_trip_error = np.abs(final_low_res_data.data - low_res_da.data)
        max_error = np.max(round_trip_error)
        mean_error = np.mean(round_trip_error)
        std_error = np.std(round_trip_error)
        
        results[method] = {
            'max_error': max_error,
            'mean_error': mean_error,
            'std_error': std_error,
            'final_data': final_low_res_data
        }
        
        # Check stability for each method
        original_min = float(np.min(low_res_da.data))
        original_max = float(np.max(low_res_da.data))
        final_min = float(np.min(final_low_res_data.data))
        final_max = float(np.max(final_low_res_data.data))
        
        # Values should remain within reasonable bounds
        assert final_min >= original_min - 0.5, f"{method} final min {final_min} is too low compared to original min {original_min}"
        assert final_max <= original_max + 0.5, f"{method} final max {final_max} is too high compared to original max {original_max}"
        
        # Check that the values are not wildly different
        assert max_error < 2.0, f"{method} round-trip difference {max_error} is too large"
    
    # Compare the diffusive characteristics between methods
    # Generally, nearest neighbor should have higher errors but be more stable
    # Bilinear should have moderate errors
    # Cubic might have more oscillation (higher max error) due to overshoot
    nearest_error = results['nearest']['max_error']
    bilinear_error = results['bilinear']['max_error']
    cubic_error = results['cubic']['max_error']
    
    # All methods should be stable (no extreme values)
    assert nearest_error < 2.0, f"Nearest neighbor max error {nearest_error} is too large"
    assert bilinear_error < 2.0, f"Bilinear max error {bilinear_error} is too large"
    assert cubic_error < 2.0, f"Cubic max error {cubic_error} is too large"


def test_accessor_identity_grid_onto_itself():
    """
    Identity Test (grid onto itself) using the xarray accessor.
    
    This test verifies that when regridding data from a grid onto the same grid
    (same coordinates, same resolution) using the .pyregrid accessor, 
    the output is identical to the input for all interpolation methods.
    
    TDD Anchor: Verify that regridding a grid onto itself through the accessor produces identical results.
    """
    # Define a test grid
    lon = np.linspace(-180, 180, 10)  # 10 longitude points
    lat = np.linspace(-90, 90, 5)     # 5 latitude points
    
    # Define analytical function: f(λ,φ) = sin(φ) + cos(λ)
    def analytical_function(lon, lat):
        """Analytical function for testing: f(λ,φ) = sin(φ) + cos(λ)"""
        lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
        return np.sin(np.radians(lat_grid)) + np.cos(np.radians(lon_grid))
    
    # Generate test data
    data_values = analytical_function(lon, lat)
    original_da = xr.DataArray(
        data_values,
        coords={'lat': lat, 'lon': lon},
        dims=['lat', 'lon'],
        name='analytical_field'
    )
    
    # Test with different interpolation methods
    methods = ['nearest', 'bilinear', 'cubic']
    
    for method in methods:
        # Regrid the data onto itself using the accessor
        regridded_da = original_da.pyregrid.regrid_to(original_da, method=method)
        
        # Verify that the output is identical to the input
        # Check that shapes match
        assert regridded_da.shape == original_da.shape, \
            f"Shape mismatch for {method}: expected {original_da.shape}, got {regridded_da.shape}"
        
        # Check that coordinates match
        np.testing.assert_array_equal(
            regridded_da['lat'].values,
            original_da['lat'].values,
            err_msg=f"Latitude coordinates don't match for {method}"
        )
        np.testing.assert_array_equal(
            regridded_da['lon'].values,
            original_da['lon'].values,
            err_msg=f"Longitude coordinates don't match for {method}"
        )
        
        # Check that the values are nearly identical (allowing for floating-point precision)
        # For identity test, we expect very close to identical values
        np.testing.assert_allclose(
            regridded_da.data,
            original_da.data,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Values don't match for {method} method"
        )
        
        # Check that dimensions are preserved
        assert regridded_da.dims == original_da.dims, \
            f"Dimensions don't match for {method}: expected {original_da.dims}, got {regridded_da.dims}"
        
        # Check that attributes are preserved (name might not be preserved in regridding)
        assert regridded_da.attrs == original_da.attrs, \
            f"Attributes don't match for {method}"


def test_accessor_identity_grid_onto_itself_with_different_data_types():
    """
    Identity Test with different data types using the xarray accessor to ensure no interpolation artifacts.
    
    TDD Anchor: Verify identity test works with different data types through the accessor.
    """
    # Define a test grid
    lon = np.linspace(-180, 180, 8)   # 8 longitude points
    lat = np.linspace(-90, 90, 4)     # 4 latitude points
    
    # Test with different data types
    test_cases = [
        # Float32 data
        {
            'name': 'float32',
            'data': np.random.random((len(lat), len(lon))).astype(np.float32)
        },
        # Float64 data
        {
            'name': 'float64',
            'data': np.random.random((len(lat), len(lon))).astype(np.float64)
        },
        # Integer data (converted to float for interpolation)
        {
            'name': 'integers',
            'data': np.random.randint(0, 100, (len(lat), len(lon))).astype(np.float64)
        },
        # Data with NaN values
        {
            'name': 'with_nan',
            'data': np.random.random((len(lat), len(lon))),
        }
    ]
    
    # Add some NaN values to the last test case
    test_cases[3]['data'][1, 2] = np.nan
    test_cases[3]['data'][3, 5] = np.nan
    
    for test_case in test_cases:
        original_da = xr.DataArray(
            test_case['data'],
            coords={'lat': lat, 'lon': lon},
            dims=['lat', 'lon'],
            name=f"test_data_{test_case['name']}"
        )
        
        # Test with different interpolation methods
        methods = ['nearest', 'bilinear', 'cubic']
        
        for method in methods:
            # Regrid the data onto itself using the accessor
            regridded_da = original_da.pyregrid.regrid_to(original_da, method=method)
            
            # For the identity test, we expect the values to be nearly identical
            # For data with NaN values, different interpolation methods handle them differently:
            # - nearest: should preserve NaN positions exactly and not affect neighboring values
            # - bilinear/cubic: may interpolate NaN values based on neighbors, affecting nearby values too
            # For the identity test, we skip bilinear/cubic with NaN data since they're not expected to preserve identity
            if 'nan' in test_case['name']:
                if method == 'nearest':
                    # For nearest neighbor with NaN, check that NaN positions are preserved exactly
                    original_non_nan = ~np.isnan(original_da.data)
                    regridded_non_nan = ~np.isnan(regridded_da.data)
                    
                    # Check that NaN positions are preserved
                    np.testing.assert_array_equal(
                        original_non_nan,
                        regridded_non_nan,
                        err_msg=f"NaN positions not preserved for {method} with {test_case['name']}"
                    )
                    
                    # Check that non-NaN values are nearly identical
                    np.testing.assert_allclose(
                        original_da.data[original_non_nan],
                        regridded_da.data[regridded_non_nan],
                        rtol=1e-8,
                        atol=1e-8,
                        err_msg=f"Non-NaN values don't match for {method} with {test_case['name']}"
                    )
                else:
                    # For bilinear/cubic with NaN, identity is not expected due to interpolation behavior with missing data
                    # So we skip the detailed comparison for these methods
                    continue  # Skip to the next method/test case
            else:
                # For regular data, check that values are nearly identical
                np.testing.assert_allclose(
                    regridded_da.data,
                    original_da.data,
                    rtol=1e-10,
                    atol=1e-10,
                    err_msg=f"Values don't match for {method} with {test_case['name']}"
                )


def test_accessor_identity_grid_onto_itself_edge_cases():
    """
    Identity Test with edge cases using the xarray accessor to ensure robustness.
    
    TDD Anchor: Verify identity test works with edge cases through the accessor.
    """
    # Test with minimal grid (2x2)
    lon_minimal = np.array([-180, 180])
    lat_minimal = np.array([-90, 90])
    
    data_minimal = np.array([[1.0, 2.0], [3.0, 4.0]])
    minimal_da = xr.DataArray(
        data_minimal,
        coords={'lat': lat_minimal, 'lon': lon_minimal},
        dims=['lat', 'lon'],
        name='minimal_test'
    )
    
    # Test with different interpolation methods
    methods = ['nearest', 'bilinear']
    # Note: cubic may not work well with 2x2 grid, so we skip it for this edge case
    
    for method in methods:
        regridded_da = minimal_da.pyregrid.regrid_to(minimal_da, method=method)
        
        # Check that values are nearly identical
        np.testing.assert_allclose(
            regridded_da.data,
            minimal_da.data,
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"Values don't match for {method} with minimal grid"
        )
    
    # Test with single row/column grids
    lon_single = np.array([0.0, 1.0, 2.0])
    lat_single = np.array([0.0])  # Single latitude
    
    data_single_row = np.array([[1.0, 2.0, 3.0]])
    single_row_da = xr.DataArray(
        data_single_row,
        coords={'lat': lat_single, 'lon': lon_single},
        dims=['lat', 'lon'],
        name='single_row_test'
    )
    
    for method in ['nearest', 'bilinear']:
        regridded_da = single_row_da.pyregrid.regrid_to(single_row_da, method=method)
        
        # Check that values are nearly identical
        np.testing.assert_allclose(
            regridded_da.data,
            single_row_da.data,
            rtol=1e-8,
            atol=1e-8,
            err_msg=f"Values don't match for {method} with single row grid"
        )
    
    # Test with constant data (should remain constant)
    lon_const = np.linspace(-10, 5)
    lat_const = np.linspace(-5, 5, 3)
    
    data_const = np.full((len(lat_const), len(lon_const)), 42.0)  # Constant value
    const_da = xr.DataArray(
        data_const,
        coords={'lat': lat_const, 'lon': lon_const},
        dims=['lat', 'lon'],
        name='constant_test'
    )
    
    for method in ['nearest', 'bilinear', 'cubic']:
        regridded_da = const_da.pyregrid.regrid_to(const_da, method=method)
        
        # For constant data, all methods should preserve the constant value
        np.testing.assert_allclose(
            regridded_da.data,
            const_da.data,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Constant values not preserved for {method}"
        )


def test_accessor_identity_preserves_no_interpolation_artifacts():
    """
    Test that no interpolation artifacts are introduced when source and destination grids are identical using the xarray accessor.
    
    TDD Anchor: Verify that identity regridding through the accessor doesn't introduce interpolation artifacts.
    """
    # Create a grid with specific patterns that could reveal interpolation artifacts
    lon = np.linspace(0, 360, 20, endpoint=False)  # 20 longitude points
    lat = np.linspace(-80, 80, 10)                 # 10 latitude points
    
    # Create data with sharp gradients and patterns
    lon_grid, lat_grid = np.meshgrid(lon, lat, indexing='xy')
    data_sharp = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
    
    # Add some sharp features that might be affected by interpolation
    data_sharp[2, 5] = 10.0  # Sharp peak
    data_sharp[7, 15] = -10.0  # Sharp valley
    
    original_da = xr.DataArray(
        data_sharp,
        coords={'lat': lat, 'lon': lon},
        dims=['lat', 'lon'],
        name='sharp_features_test'
    )
    
    # Test all methods
    methods = ['nearest', 'bilinear', 'cubic']
    
    for method in methods:
        regridded_da = original_da.pyregrid.regrid_to(original_da, method=method)
        
        # Calculate the difference to check for artifacts
        diff = np.abs(regridded_da.data - original_da.data)
        
        # The difference should be very small (numerical precision only)
        max_diff = np.max(diff)
        assert max_diff < 1e-8, \
            f"Interpolation artifacts detected for {method}: max difference = {max_diff}"
        
        # Check that statistics are preserved
        assert abs(np.mean(regridded_da.data) - np.mean(original_da.data)) < 1e-10, \
            f"Mean not preserved for {method}"
        assert abs(np.std(regridded_da.data) - np.std(original_da.data)) < 1e-10, \
            f"Standard deviation not preserved for {method}"
        
        # Verify that extreme values are preserved
        np.testing.assert_allclose(
            np.min(regridded_da.data),
            np.min(original_da.data),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Minimum value not preserved for {method}"
        )
        np.testing.assert_allclose(
            np.max(regridded_da.data),
            np.max(original_da.data),
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Maximum value not preserved for {method}"
        )


def test_accessor_dataset_round_trip():
    """
    Test round-trip functionality with xarray Datasets using the accessor.
    
    TDD Anchor: Verify that the accessor works correctly with xarray Datasets in round-trip operations.
    """
    # Define low-resolution grid (source)
    low_res_lon = np.linspace(-180, 180, 10)  # 10 points
    low_res_lat = np.linspace(-90, 90, 5)     # 5 points
    
    # Define high-resolution grid (target for upsample)
    high_res_lon = np.linspace(-180, 180, 20)  # 20 points
    high_res_lat = np.linspace(-90, 90, 10)    # 10 points
    
    # Generate low-resolution source data for multiple variables
    temp_data = np.random.rand(5, 10)
    pressure_data = np.random.rand(5, 10) * 1000
    
    low_res_ds = xr.Dataset({
        'temperature': (['lat', 'lon'], temp_data),
        'pressure': (['lat', 'lon'], pressure_data)
    }, coords={
        'lon': low_res_lon,
        'lat': low_res_lat
    })
    
    # Create target grids as Datasets
    high_res_target_ds = xr.Dataset(
        coords={'lon': (['lon'], high_res_lon), 'lat': (['lat'], high_res_lat)}
    )
    
    low_res_target_ds = xr.Dataset(
        coords={'lon': (['lon'], low_res_lon), 'lat': (['lat'], low_res_lat)}
    )
    
    # Step 1: Regrid from low-resolution to high-resolution (upsample) using accessor
    high_res_data = low_res_ds.pyregrid.regrid_to(high_res_target_ds, method='bilinear')
    
    # Step 2: Regrid from high-resolution back to low-resolution (downsample) using accessor
    final_low_res_data = high_res_data.pyregrid.regrid_to(low_res_target_ds, method='bilinear')
    
    # Step 3: Compare final low-resolution dataset with original low-resolution dataset
    assert 'temperature' in final_low_res_data.data_vars
    assert 'pressure' in final_low_res_data.data_vars
    
    # Check that the values are not wildly different
    temp_max_difference = np.max(np.abs(final_low_res_data['temperature'].data - low_res_ds['temperature'].data))
    pressure_max_difference = np.max(np.abs(final_low_res_data['pressure'].data - low_res_ds['pressure'].data))
    
    assert temp_max_difference < 1.0, f"Temperature round-trip difference {temp_max_difference} is too large"
    assert pressure_max_difference < 1000.0, f"Pressure round-trip difference {pressure_max_difference} is too large"
    
    # The shapes should match the original
    assert final_low_res_data['temperature'].shape == low_res_ds['temperature'].shape
    assert final_low_res_data['pressure'].shape == low_res_ds['pressure'].shape