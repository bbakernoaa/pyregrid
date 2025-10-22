"""
Test to understand the coordinate indices issue in GridRegridder.
"""
import numpy as np
import xarray as xr
from pyregrid.core import GridRegridder


def test_coordinate_indices_identical_grids():
    """
    Test coordinate index calculation when source and target grids are identical.
    """
    # Create identical grids
    source_lon = np.linspace(-180, 180, 53)  # From the xarray toy dataset
    source_lat = np.linspace(15, 75, 25)     # From the xarray toy dataset
    
    # Create xarray DataArrays with these coordinates
    source_da = xr.DataArray(
        np.random.rand(25, 53),  # Random data
        dims=['lat', 'lon'],
        coords={
            'lon': source_lon,
            'lat': source_lat
        }
    )
    
    # Create target grid identical to source
    target_da = xr.DataArray(
        np.zeros((25, 53)),  # Zero data
        dims=['lat', 'lon'],
        coords={
            'lon': source_lon,
            'lat': source_lat
        }
    )
    
    print(f"Source lon range: {source_lon[0]} to {source_lon[-1]}, step: {source_lon[1] - source_lon[0]}")
    print(f"Source lat range: {source_lat[0]} to {source_lat[-1]}, step: {source_lat[1] - source_lat[0]}")
    print(f"Target and source are identical: {np.array_equal(source_lon, target_da.lon.values) and np.array_equal(source_lat, target_da.lat.values)}")
    
    # Create GridRegridder
    regridder = GridRegridder(source_grid=source_da, target_grid=target_da, method='nearest')
    
    # Check the calculated weights
    print(f"Lon indices shape: {regridder.weights['lon_indices'].shape}")
    print(f"Lat indices shape: {regridder.weights['lat_indices'].shape}")
    
    # Check if indices are correctly mapping to themselves
    expected_lon_indices = np.arange(len(source_lon))
    expected_lat_indices = np.arange(len(source_lat))
    
    # Create expected index grids
    expected_lon_idx_grid, expected_lat_idx_grid = np.meshgrid(expected_lon_indices, expected_lat_indices, indexing='xy')
    
    print(f"Expected lon indices shape: {expected_lon_idx_grid.shape}")
    print(f"Expected lat indices shape: {expected_lat_idx_grid.shape}")
    
    print(f"Actual lon indices shape: {regridder.weights['lon_indices'].shape}")
    print(f"Actual lat indices shape: {regridder.weights['lat_indices'].shape}")
    
    # Create proper index grids with the correct shapes
    # For the target grid, we need to create index grids that match the target shape
    target_lon_idx = np.arange(len(target_da.lon))
    target_lat_idx = np.arange(len(target_da.lat))
    expected_lon_idx_grid, expected_lat_idx_grid = np.meshgrid(target_lon_idx, target_lat_idx, indexing='xy')
    
    # Compare actual vs expected
    print(f"Lon indices close to expected: {np.allclose(regridder.weights['lon_indices'], expected_lon_idx_grid, rtol=1e-5, atol=1e-5)}")
    print(f"Lat indices close to expected: {np.allclose(regridder.weights['lat_indices'], expected_lat_idx_grid, rtol=1e-5, atol=1e-5)}")
    
    print(f"Max difference in lon indices: {np.max(np.abs(regridder.weights['lon_indices'] - expected_lon_idx_grid))}")
    print(f"Max difference in lat indices: {np.max(np.abs(regridder.weights['lat_indices'] - expected_lat_idx_grid))}")
    
    # Test the actual regridding
    result = regridder.regrid(source_da)
    print(f"Regridding result close to original: {np.allclose(result.data, source_da.data, rtol=1e-5, atol=1e-5)}")
    print(f"Max difference in regridded data: {np.max(np.abs(result.data - source_da.data))}")


def test_coordinate_indices_simple_case():
    """
    Test with a simple, regular grid to understand the issue better.
    """
    # Create simple regular grids
    source_lon = np.array([0, 1, 2, 3, 4])
    source_lat = np.array([0, 1, 2])
    
    target_lon = np.array([0, 1, 2, 3, 4])  # Identical
    target_lat = np.array([0, 1, 2])        # Identical
    
    source_da = xr.DataArray(
        np.random.rand(3, 5),
        dims=['lat', 'lon'],
        coords={
            'lon': source_lon,
            'lat': source_lat
        }
    )
    
    target_da = xr.DataArray(
        np.zeros((3, 5)),
        dims=['lat', 'lon'],
        coords={
            'lon': target_lon,
            'lat': target_lat
        }
    )
    
    print("Simple case:")
    print(f"Source lon: {source_lon}")
    print(f"Source lat: {source_lat}")
    print(f"Target lon: {target_lon}")
    print(f"Target lat: {target_lat}")
    
    # Create GridRegridder
    regridder = GridRegridder(source_grid=source_da, target_grid=target_da, method='nearest')
    
    # Check indices
    expected_lon_indices = np.arange(len(source_lon))
    expected_lat_indices = np.arange(len(source_lat))
    
    expected_lon_idx_grid, expected_lat_idx_grid = np.meshgrid(expected_lon_indices, expected_lat_indices, indexing='xy')
    
    print(f"Expected lon indices:\n{expected_lon_idx_grid}")
    print(f"Actual lon indices:\n{regridder.weights['lon_indices']}")
    print(f"Expected lat indices:\n{expected_lat_idx_grid}")
    print(f"Actual lat indices:\n{regridder.weights['lat_indices']}")
    
    print(f"Lon indices close to expected: {np.allclose(regridder.weights['lon_indices'], expected_lon_idx_grid)}")
    print(f"Lat indices close to expected: {np.allclose(regridder.weights['lat_indices'], expected_lat_idx_grid)}")
    
    # Test the actual regridding
    result = regridder.regrid(source_da)
    print(f"Regridding result close to original: {np.allclose(result.data, source_da.data)}")
    print(f"Max difference in regridded data: {np.max(np.abs(result.data - source_da.data))}")


if __name__ == "__main__":
    print("=== Testing coordinate indices with identical grids ===")
    test_coordinate_indices_identical_grids()
    
    print("\n=== Testing coordinate indices with simple case ===")
    test_coordinate_indices_simple_case()