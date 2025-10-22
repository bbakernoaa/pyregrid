"""
Test to verify identity regridding works correctly with xarray toy datasets.
"""
import pytest
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def test_identity_regridding_with_xarray_toy_dataset():
    """
    Test identity regridding with xarray toy dataset.
    
    This test verifies that when regridding a dataset to itself using
    nearest neighbor interpolation, the result should be identical to
    the original dataset (within floating point precision).
    """
    # Load xarray tutorial dataset
    ds = xr.tutorial.open_dataset("air_temperature")
    
    # Select a single time slice for testing
    ds_single_time = ds.isel(time=0)
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds_single_time,
        target_grid=ds_single_time,
        method='nearest'
    )
    
    regridded_ds = regridder.regrid(ds_single_time['air'])
    
    # For identity regridding with nearest neighbor, the result should be
    # very close to the original (allowing for minor floating point differences)
    xr.testing.assert_allclose(
        regridded_ds,
        ds_single_time['air'],
        rtol=1e-10,
        atol=1e-10
    )


def test_identity_regridding_conserves_shape_and_coordinates():
    """
    Test that identity regridding conserves shape and coordinates.
    """
    # Load xarray tutorial dataset
    ds = xr.tutorial.open_dataset("air_temperature")
    
    # Select a single time slice for testing
    ds_single_time = ds.isel(time=0)
    
    # Regrid to itself using nearest neighbor method
    regridder = GridRegridder(
        source_grid=ds_single_time,
        target_grid=ds_single_time,
        method='nearest'
    )
    
    regridded_ds = regridder.regrid(ds_single_time['air'])
    
    # Shape should be identical
    assert regridded_ds.shape == ds_single_time['air'].shape
    
    # Coordinates should be identical
    np.testing.assert_array_equal(
        regridded_ds['lat'].values,
        ds_single_time['lat'].values
    )
    np.testing.assert_array_equal(
        regridded_ds['lon'].values,
        ds_single_time['lon'].values
    )


if __name__ == "__main__":
    pytest.main([__file__])