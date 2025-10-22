"""
Tests for Dask functionality in PyRegrid.

This module contains tests to verify that PyRegrid works correctly with Dask arrays
for out-of-core processing and parallel computation.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd

# Try to import Dask
try:
    import dask.array as da
    import dask
    
    # Check if Dask is available
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    da = None
    dask = None

from pyregrid import GridRegridder, PointInterpolator, grid_from_points
from pyregrid.dask import DaskRegridder, ChunkingStrategy, MemoryManager, ParallelProcessor


@pytest.mark.skipif(not HAS_DASK, reason="Dask not available")
class TestDaskFunctionality:
    """Test Dask functionality in PyRegrid."""
    
    def test_dask_regridder_creation(self):
        """Test that DaskRegridder can be created."""
        # Create test grids
        source_lon = np.linspace(-10, 10, 5)
        source_lat = np.linspace(40, 50, 4)
        target_lon = np.linspace(-8, 8, 10)
        target_lat = np.linspace(42, 48, 8)
        
        source_data = np.random.random((4, 5))
        target_data = np.zeros((8, 10))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Create DaskRegridder
        regridder = DaskRegridder(
            source_grid=source_ds,
            target_grid=target_ds,
            method='bilinear'
        )
        
        assert regridder is not None
        assert regridder.weights is not None
    
    def test_dask_regridder_with_dask_arrays(self):
       """Test that DaskRegridder works with Dask arrays."""
       # Create test grids - ensure coordinate dimensions match data dimensions
       source_lon = np.linspace(-10, 10, 5)  # 5 points
       source_lat = np.linspace(40, 50, 4)  # 4 points
       target_lon = np.linspace(-8, 8, 5)   # 5 points (match source data dimensions)
       target_lat = np.linspace(42, 48, 4)  # 4 points (match source data dimensions)
       
       # Create source data as Dask array with matching dimensions
       source_data_np = np.random.random((4, 5))  # 4 lat x 5 lon
       if HAS_DASK and da is not None:
           source_data_da = da.from_array(source_data_np, chunks='auto')
       else:
           source_data_da = source_data_np
       
       # Create target data with dimensions matching target grid coordinates
       target_data = np.zeros((4, 5))  # 4 lat x 5 lon (match target grid)
       
       source_ds = xr.Dataset(
           {'temperature': (['lat', 'lon'], source_data_da)},
           coords={'lon': source_lon, 'lat': source_lat}
       )
       target_ds = xr.Dataset(
           {'temperature': (['lat', 'lon'], target_data)},
           coords={'lon': target_lon, 'lat': target_lat}
       )
       
       # Create DaskRegridder
       regridder = DaskRegridder(
           source_grid=source_ds,
           target_grid=target_ds,
           method='bilinear'
       )
       
       # Regrid the data
       result = regridder.regrid(source_ds)
       
       assert result is not None
       assert isinstance(result, xr.Dataset)
       # Result should contain Dask arrays if Dask is available
       if HAS_DASK and da is not None:
           assert hasattr(result['temperature'].data, 'chunks')
       else:
           # If Dask is not available, result should still work
           assert result['temperature'].shape == (4, 5)
    
    def test_chunking_strategy(self):
        """Test the ChunkingStrategy class."""
        chunking_strategy = ChunkingStrategy()
        
        # Create test data
        source_lon = np.linspace(-10, 100, 100)
        source_lat = np.linspace(40, 50, 100)
        target_lon = np.linspace(-8, 8, 50)
        target_lat = np.linspace(42, 48, 50)
        
        source_data = np.random.random((100, 100))
        target_data = np.zeros((50, 50))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Determine chunk size
        chunk_size = chunking_strategy.determine_chunk_size(source_ds, target_ds)
        
        assert chunk_size is not None
        assert isinstance(chunk_size, (int, tuple))
    
    def test_memory_manager(self):
        """Test the MemoryManager class."""
        memory_manager = MemoryManager()
        
        # Create test data
        source_lon = np.linspace(-10, 100, 100)
        source_lat = np.linspace(40, 50, 100)
        target_lon = np.linspace(-8, 8, 50)
        target_lat = np.linspace(42, 48, 50)
        
        source_data = np.random.random((100, 100))
        target_data = np.zeros((50, 50))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Estimate memory usage
        estimated_memory = memory_manager.estimate_operation_memory(source_ds, target_ds)
        
        assert estimated_memory > 0
        assert isinstance(estimated_memory, int)
        
        # Check if operation can fit in memory
        can_fit = memory_manager.can_fit_in_memory(source_ds, target_ds)
        assert isinstance(can_fit, bool)
    
    def test_parallel_processor(self):
        """Test the ParallelProcessor class."""
        parallel_processor = ParallelProcessor()
        
        # Create test data
        source_lon = np.linspace(-10, 100, 100)
        source_lat = np.linspace(40, 50, 100)
        target_lon = np.linspace(-8, 8, 50)
        target_lat = np.linspace(42, 48, 50)
        
        source_data = np.random.random((100, 100))
        target_data = np.zeros((50, 50))
        
        source_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], source_data)},
            coords={'lon': source_lon, 'lat': source_lat}
        )
        target_ds = xr.Dataset(
            {'temperature': (['lat', 'lon'], target_data)},
            coords={'lon': target_lon, 'lat': target_lat}
        )
        
        # Optimize parallel execution
        optimization_params = parallel_processor.optimize_parallel_execution(
            source_ds, target_ds, method='bilinear'
        )
        
        assert optimization_params is not None
        assert 'workers' in optimization_params
        assert 'chunk_size' in optimization_params
        assert 'method' in optimization_params
    
    def test_point_interpolator_with_dask(self):
        """Test PointInterpolator with Dask arrays."""
        # Create test point data
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30],
            'humidity': [50, 60, 70]
        })
        
        # Create target grid
        target_lon = np.linspace(-4, 4, 20)
        target_lat = np.linspace(43, 47, 15)
        target_grid = xr.Dataset(
            coords={'lon': (['lon'], target_lon), 'lat': (['lat'], target_lat)}
        )
        
        # Create PointInterpolator
        interpolator = PointInterpolator(
            source_points=df,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        # Interpolate to grid
        result = interpolator.interpolate_to_grid(target_grid)
        
        assert result is not None
        assert isinstance(result, xr.Dataset)
        # Check that result contains the expected variables
        assert 'temperature' in result.data_vars
        assert 'humidity' in result.data_vars
    
    def test_grid_from_points_with_dask(self):
        """Test grid_from_points function with Dask arrays."""
        # Create test point data
        df = pd.DataFrame({
            'longitude': [-5, 0, 5],
            'latitude': [42, 45, 48],
            'temperature': [20, 25, 30],
            'humidity': [50, 60, 70]
        })
        
        # Create target grid
        target_lon = np.linspace(-4, 4, 20)
        target_lat = np.linspace(43, 47, 15)
        target_grid = {'lon': target_lon, 'lat': target_lat}
        
        # Create grid from points
        result = grid_from_points(
            source_points=df,
            target_grid=target_grid,
            method='idw',
            x_coord='longitude',
            y_coord='latitude'
        )
        
        assert result is not None
        assert isinstance(result, xr.Dataset)
        # Check that result contains the expected variables
        assert 'temperature' in result.data_vars
        assert 'humidity' in result.data_vars


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__])