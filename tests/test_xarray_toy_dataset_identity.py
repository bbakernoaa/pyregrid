"""
Test to reproduce the identity issue with xarray toy dataset.

This test reproduces the issue where regridding an xarray toy dataset to itself
using the accessor doesn't provide the same results.
"""
import pytest
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def test_xarray_toy_dataset_identity_issue():
    """
    Test to reproduce the identity issue with xarray toy dataset.
    
    This test reproduces the reported issue where:
    source_da.pyregrid.regrid_to(source_da, method='nearest') == source_da
    returns False for all values instead of True.
    """
    # Load the xarray toy dataset
    ds = xr.tutorial.open_dataset('air_temperature').isel(time=0)
    source_da = ds.air
    
    print(f"Original shape: {source_da.shape}")
    print(f"Original coordinates: {source_da.coords}")
    
    # Test regridding to itself using the accessor
    regridded_da = source_da.pyregrid.regrid_to(source_da, method='nearest')
    
    print(f"Regridded shape: {regridded_da.shape}")
    print(f"Regridded coordinates: {regridded_da.coords}")
    
    # Check if they are equal
    are_equal = regridded_da.equals(source_da)
    print(f"Are equal using .equals(): {are_equal}")
    
    # Check element-wise equality
    element_equal = regridded_da == source_da
    print(f"All elements equal: {element_equal.all().values}")
    
    # Check if values are close (allowing for floating point precision)
    values_close = np.allclose(regridded_da.data, source_da.data, rtol=1e-10, atol=1e-10)
    print(f"Values close: {values_close}")
    
    # Print some statistics
    print(f"Max difference: {np.max(np.abs(regridded_da.data - source_da.data))}")
    print(f"Mean difference: {np.mean(np.abs(regridded_da.data - source_da.data))}")
    
    # The test should pass if the regridded data is nearly identical to the original
    assert values_close, "Regridded data should be nearly identical to original data"


def test_xarray_toy_dataset_identity_with_gridregridder():
    """
    Test the same scenario but using GridRegridder directly to see if the issue is specific to the accessor.
    """
    # Load the xarray toy dataset
    ds = xr.tutorial.open_dataset('air_temperature').isel(time=0)
    source_da = ds.air
    
    print(f"Original shape: {source_da.shape}")
    print(f"Original coordinates: {source_da.coords}")
    
    # Test regridding to itself using GridRegridder directly
    regridder = GridRegridder(source_grid=source_da, target_grid=source_da, method='nearest')
    regridded_da = regridder.regrid(source_da)
    
    print(f"Regridded shape: {regridded_da.shape}")
    print(f"Regridded coordinates: {regridded_da.coords}")
    
    # Check if they are equal
    are_equal = regridded_da.equals(source_da)
    print(f"Are equal using .equals(): {are_equal}")
    
    # Check if values are close (allowing for floating point precision)
    values_close = np.allclose(regridded_da.data, source_da.data, rtol=1e-10, atol=1e-10)
    print(f"Values close: {values_close}")
    
    # Print some statistics
    print(f"Max difference: {np.max(np.abs(regridded_da.data - source_da.data))}")
    print(f"Mean difference: {np.mean(np.abs(regridded_da.data - source_da.data))}")
    
    # The test should pass if the regridded data is nearly identical to the original
    assert values_close, "Regridded data should be nearly identical to original data"


def test_xarray_toy_dataset_identity_comparison():
    """
    Compare the results between using the accessor and using GridRegridder directly.
    """
    # Load the xarray toy dataset
    ds = xr.tutorial.open_dataset('air_temperature').isel(time=0)
    source_da = ds.air
    
    # Using the accessor
    regridded_accessor = source_da.pyregrid.regrid_to(source_da, method='nearest')
    
    # Using GridRegridder directly
    regridder = GridRegridder(source_grid=source_da, target_grid=source_da, method='nearest')
    regridded_direct = regridder.regrid(source_da)
    
    # Compare the results
    accessor_close = np.allclose(regridded_accessor.data, source_da.data, rtol=1e-10, atol=1e-10)
    direct_close = np.allclose(regridded_direct.data, source_da.data, rtol=1e-10, atol=1e-10)
    accessor_vs_direct = np.allclose(regridded_accessor.data, regridded_direct.data, rtol=1e-10, atol=1e-10)
    
    print(f"Accessor result close to original: {accessor_close}")
    print(f"Direct result close to original: {direct_close}")
    print(f"Accessor vs Direct results close: {accessor_vs_direct}")
    
    print(f"Accessor max diff from original: {np.max(np.abs(regridded_accessor.data - source_da.data))}")
    print(f"Direct max diff from original: {np.max(np.abs(regridded_direct.data - source_da.data))}")
    print(f"Accessor vs Direct max diff: {np.max(np.abs(regridded_accessor.data - regridded_direct.data))}")
    
    # Both should produce similar results
    assert accessor_vs_direct, "Accessor and direct methods should produce similar results"