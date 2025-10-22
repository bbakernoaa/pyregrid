"""
Test to verify identity regridding works correctly with curvilinear grid coordinates.

This test specifically focuses on datasets where latitude and longitude are 
2-dimensional arrays (curvilinear grids) rather than 1-dimensional arrays 
(rectilinear grids). It verifies that the identity transformation preserves
the curvilinear coordinate structure, maintains proper coordinate mapping
between the 2D lat/lon arrays and the data dimensions, and correctly handles
the non-rectangular grid geometry.
"""
import pytest
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def create_curvilinear_dataset():
    """
    Create a synthetic curvilinear dataset for testing.
    
    Returns:
        xr.Dataset: A dataset with 2D latitude and longitude coordinates
    """
    # Define the grid dimensions
    nlat = 8
    nlon = 12
    
    # Create base coordinate arrays
    base_lat = np.linspace(-60, 60, nlat)
    base_lon = np.linspace(-180, 180, nlon)
    
    # Create curvilinear coordinates by adding some curvature
    # This simulates a non-rectangular grid (e.g., rotated pole, stretched grid)
    lon_2d, lat_2d = np.meshgrid(base_lon, base_lat, indexing='xy')
    
    # Add curvature to make it truly curvilinear
    # Example: Add a sinusoidal variation to latitude and longitude
    curvature_factor = 0.2
    lat_curvilinear = lat_2d + curvature_factor * np.sin(np.radians(lon_2d))
    lon_curvilinear = lon_2d + curvature_factor * np.cos(np.radians(lat_2d))
    
    # Create some test data that varies with position
    # Use a function that depends on both latitude and longitude
    def test_function(lat, lon):
        """Test function for curvilinear grid."""
        return (np.sin(np.radians(lat)) * np.cos(np.radians(lon)) + 
                0.5 * np.sin(2 * np.radians(lat)) * np.sin(2 * np.radians(lon)))
    
    data_values = test_function(lat_curvilinear, lon_curvilinear)
    
    # Add some noise to make it more realistic
    np.random.seed(42)  # For reproducible tests
    noise = 0.1 * np.random.normal(size=data_values.shape)
    data_values += noise
    
    # Create the dataset
    ds = xr.Dataset(
        {
            'temperature': (['lat', 'lon'], data_values),
            'pressure': (['lat', 'lon'], data_values * 0.8 + 5),
        },
        coords={
            'lat': (['lat', 'lon'], lat_curvilinear),
            'lon': (['lat', 'lon'], lon_curvilinear),
            # Add 1D coordinate labels for reference
            'lat_index': (['lat'], np.arange(nlat)),
            'lon_index': (['lon'], np.arange(nlon)),
        },
        attrs={
            'description': 'Synthetic curvilinear grid dataset for testing',
            'grid_type': 'curvilinear',
            'curvature_factor': curvature_factor,
        }
    )
    
    # Add coordinate attributes
    ds['lat'].attrs = {
        'long_name': 'latitude',
        'units': 'degrees_north',
        'standard_name': 'latitude',
        'axis': 'Y',
    }
    
    ds['lon'].attrs = {
        'long_name': 'longitude', 
        'units': 'degrees_east',
        'standard_name': 'longitude',
        'axis': 'X',
    }
    
    ds['temperature'].attrs = {
        'long_name': 'air_temperature',
        'units': 'K',
        'standard_name': 'air_temperature',
    }
    
    ds['pressure'].attrs = {
        'long_name': 'air_pressure',
        'units': 'hPa',
        'standard_name': 'air_pressure',
    }
    
    return ds


def test_identity_regridding_with_curvilinear_grid():
    """
    Test identity regridding with curvilinear grid coordinates.
    
    This test verifies that when regridding a curvilinear dataset to itself using
    nearest neighbor interpolation, the result should be identical to
    the original dataset (within floating point precision).
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds,
        target_grid=ds,
        method='nearest'
    )
    
    # Test with both data variables
    for var_name in ['temperature', 'pressure']:
        regridded_data = regridder.regrid(ds[var_name])
        
        # For identity regridding with nearest neighbor, the result should be
        # very close to the original (allowing for minor floating point differences)
        try:
            xr.testing.assert_allclose(
                regridded_data,
                ds[var_name],
                rtol=1e-10,
                atol=1e-10
            )
        except Exception as e:
            # If the test fails, provide more detailed error information
            print(f"Variable: {var_name}")
            print(f"Original shape: {ds[var_name].shape}")
            print(f"Regridded shape: {regridded_data.shape}")
            print(f"Original coords: {ds[var_name].coords}")
            print(f"Regridded coords: {regridded_data.coords}")
            raise e


def test_identity_regridding_conserves_curvilinear_shape_and_coordinates():
    """
    Test that identity regridding conserves shape and coordinates for curvilinear grids.
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds,
        target_grid=ds,
        method='nearest'
    )
    
    # Test with temperature data
    regridded_ds = regridder.regrid(ds['temperature'])
    
    # Shape should be identical
    assert regridded_ds.shape == ds['temperature'].shape, \
        f"Shape mismatch: expected {ds['temperature'].shape}, got {regridded_ds.shape}"
    
    # 2D coordinate arrays should be identical
    np.testing.assert_array_equal(
        regridded_ds['lat'].values,
        ds['lat'].values,
        err_msg="Latitude coordinates don't match after identity regridding"
    )
    
    np.testing.assert_array_equal(
        regridded_ds['lon'].values,
        ds['lon'].values,
        err_msg="Longitude coordinates don't match after identity regridding"
    )
    
    # Coordinate attributes should be preserved
    assert regridded_ds['lat'].attrs == ds['lat'].attrs, \
        "Latitude attributes don't match after identity regridding"
    
    assert regridded_ds['lon'].attrs == ds['lon'].attrs, \
        "Longitude attributes don't match after identity regridding"


def test_curvilinear_coordinate_mapping_preserved():
    """
    Test that coordinate mapping between 2D lat/lon arrays and data dimensions is preserved.
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds,
        target_grid=ds,
        method='nearest'
    )
    
    regridded_ds = regridder.regrid(ds['temperature'])
    
    # Check that coordinate dimensions are correctly mapped
    assert 'lat' in regridded_ds.dims, "Latitude dimension missing from regridded data"
    assert 'lon' in regridded_ds.dims, "Longitude dimension missing from regridded data"
    
    # Check that coordinate arrays have the correct shape
    assert regridded_ds['lat'].shape == regridded_ds.shape, \
        "Latitude coordinate shape doesn't match data shape"

    assert regridded_ds['lon'].shape == regridded_ds.shape, \
        "Longitude coordinate shape doesn't match data shape"
    
    # Verify that the coordinate mapping is consistent
    # Each data point should have corresponding lat/lon coordinates
    for i in range(regridded_ds.shape[0]):  # latitude dimension
        for j in range(regridded_ds.shape[1]):  # longitude dimension
            # Check that the coordinate values are consistent
            assert not np.isnan(regridded_ds['lat'].values[i, j]), \
                f"NaN found in latitude coordinate at ({i}, {j})"
            assert not np.isnan(regridded_ds['lon'].values[i, j]), \
                f"NaN found in longitude coordinate at ({i}, {j})"


def test_spatial_relationships_preserved():
    """
    Test that spatial relationships between grid points are preserved.
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds,
        target_grid=ds,
        method='nearest'
    )
    
    regridded_ds = regridder.regrid(ds['temperature'])
    
    # Calculate spatial distances between adjacent points in original grid
    lat_orig = ds['lat'].values
    lon_orig = ds['lon'].values
    
    # Calculate spatial distances between adjacent points in regridded grid
    lat_regrid = regridded_ds['lat'].values
    lon_regrid = regridded_ds['lon'].values
    
    # Check that the coordinate ranges are preserved
    orig_lat_range = [np.min(lat_orig), np.max(lat_orig)]
    regrid_lat_range = [np.min(lat_regrid), np.max(lat_regrid)]
    
    orig_lon_range = [np.min(lon_orig), np.max(lon_orig)]
    regrid_lon_range = [np.min(lon_regrid), np.max(lon_regrid)]
    
    np.testing.assert_allclose(
        orig_lat_range, regrid_lat_range, rtol=1e-10, atol=1e-10,
        err_msg="Latitude range not preserved"
    )
    
    np.testing.assert_allclose(
        orig_lon_range, regrid_lon_range, rtol=1e-10, atol=1e-10,
        err_msg="Longitude range not preserved"
    )
    
    # Check that the grid topology is preserved (monotonicity)
    # For each latitude row, longitude should be monotonic
    for i in range(lat_orig.shape[0]):
        orig_lon_row = lon_orig[i, :]
        regrid_lon_row = lon_regrid[i, :]
        
        # Check monotonicity (allowing for small floating point differences)
        orig_diff = np.diff(orig_lon_row)
        regrid_diff = np.diff(regrid_lon_row)
        
        # The sign of differences should be the same (preserving monotonicity)
        assert np.all(np.sign(orig_diff) == np.sign(regrid_diff)), \
            f"Longitude monotonicity not preserved in row {i}"


def test_curvilinear_grid_with_different_interpolation_methods():
    """
    Test identity regridding with different interpolation methods on curvilinear grids.
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Test different interpolation methods
    methods = ['nearest', 'bilinear', 'cubic']
    
    for method in methods:
        regridder = GridRegridder(
            source_grid=ds,
            target_grid=ds,
            method=method
        )
        
        regridded_ds = regridder.regrid(ds['temperature'])
        
        # For identity regridding, coordinates should be exactly preserved
        np.testing.assert_array_equal(
            regridded_ds['lat'].values,
            ds['lat'].values,
            err_msg=f"Latitude coordinates don't match for {method} method"
        )
        
        np.testing.assert_array_equal(
            regridded_ds['lon'].values,
            ds['lon'].values,
            err_msg=f"Longitude coordinates don't match for {method} method"
        )
        
        # Data should be very close (allowing for method-specific precision)
        if method == 'nearest':
            # Nearest neighbor should be exact for identity regridding
            try:
                xr.testing.assert_allclose(
                    regridded_ds,
                    ds['temperature'],
                    rtol=1e-12,
                    atol=1e-12
                )
            except AssertionError:
                raise AssertionError(f"Identity regridding failed for {method} method")
        else:
            # Bilinear and cubic may have tiny floating point differences
            try:
                xr.testing.assert_allclose(
                    regridded_ds,
                    ds['temperature'],
                    rtol=1e-10,
                    atol=1e-10
                )
            except AssertionError:
                raise AssertionError(f"Identity regridding failed for {method} method")


def test_curvilinear_grid_coordinate_attributes_preserved():
    """
    Test that coordinate attributes are properly maintained in curvilinear grids.
    """
    # Create curvilinear dataset
    ds = create_curvilinear_dataset()
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds,
        target_grid=ds,
        method='nearest'
    )
    
    regridded_ds = regridder.regrid(ds['temperature'])
    
    # Check that all coordinate attributes are preserved
    for coord_name in ['lat', 'lon']:
        assert regridded_ds[coord_name].attrs == ds[coord_name].attrs, \
            f"Attributes for {coord_name} not preserved"
    
    # Check that the data variable attributes are preserved
    assert regridded_ds.attrs == ds['temperature'].attrs, \
        "Data variable attributes not preserved"


if __name__ == "__main__":
    pytest.main([__file__])