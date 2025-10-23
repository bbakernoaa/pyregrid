"""
Simple integration tests for the unified benchmarking system.

This module tests basic integration between the unified benchmarking interface
and the existing PyRegrid functionality.
"""
import pytest
import numpy as np
import xarray as xr
import tempfile
import os
from pathlib import Path

# Import the unified benchmarking components
from benchmarks.unified_benchmark import UnifiedBenchmarkRunner, BenchmarkConfiguration, run_standard_benchmark
from benchmarks.performance_metrics import HighResolutionBenchmark
from benchmarks.accuracy_validation import AccuracyBenchmark

# Import PyRegrid core functionality
from pyregrid import GridRegridder, PointInterpolator


class TestSimpleIntegration:
    """Test basic integration between unified benchmarking and PyRegrid core functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create a simple 2D grid
        lon = np.linspace(-180, 180, 20)
        lat = np.linspace(-90, 90, 10)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create some sample data (simple sine wave pattern)
        data = np.sin(np.radians(lon_grid)) * np.cos(np.radians(lat_grid))
        
        # Create xarray DataArray
        da = xr.DataArray(
            data,
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
            name='temperature'
        )
        
        return da
    
    @pytest.fixture
    def sample_target_grid(self):
        """Create a target grid for testing."""
        # Create a coarser target grid
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 5)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create xarray DataArray for target grid
        target_da = xr.DataArray(
            np.zeros((len(lat), len(lon))),  # Placeholder values
            dims=['lat', 'lon'],
            coords={'lat': lat, 'lon': lon},
            name='target_temperature'
        )
        
        return target_da
    
    def test_unified_benchmark_runner_initialization(self):
        """Test that UnifiedBenchmarkRunner initializes correctly."""
        config = BenchmarkConfiguration(
            use_dask=False,  # Disable Dask for simpler testing
            output_dir="./test_results"
        )
        
        runner = UnifiedBenchmarkRunner(config)
        
        assert runner.config.use_dask == False
        assert runner.config.output_dir == "./test_results"
        assert runner.performance_benchmark is not None
        assert runner.accuracy_benchmark is not None
        assert runner.scalability_benchmark is not None
    
    def test_benchmark_configuration_defaults(self):
        """Test that BenchmarkConfiguration has sensible defaults."""
        config = BenchmarkConfiguration()
        
        assert config.use_dask == True
        assert config.output_dir == "./benchmark_results"
        assert config.performance_threshold == 10.0
        assert config.accuracy_threshold == 1e-4
        assert config.scalability_threshold == 0.7
        assert config.iterations == 3
        assert config.methods == ['bilinear', 'nearest']
    
    def test_grid_regridder_compatibility(self, sample_data, sample_target_grid):
        """Test that GridRegridder works with benchmarking data."""
        # Create a GridRegridder instance
        regridder = GridRegridder(
            source_grid=sample_data,
            target_grid=sample_target_grid,
            method='bilinear'
        )
        
        # Test that regridding works
        regridded_data = regridder.regrid(sample_data)
        
        assert regridded_data is not None
        assert regridded_data.shape == sample_target_grid.shape
        assert regridded_data.dims == sample_target_grid.dims
    
    def test_high_resolution_benchmark_simple(self, sample_data, sample_target_grid):
        """Test that HighResolutionBenchmark works with PyRegrid data."""
        benchmark = HighResolutionBenchmark(use_dask=False)
        
        # Test that benchmark can work with PyRegrid DataArrays
        # Convert world coordinates to array indices for map_coordinates
        height, width = sample_data.shape
        target_lon = sample_target_grid.lon.values
        target_lat = sample_target_grid.lat.values
        
        # Create coordinate arrays for map_coordinates
        # map_coordinates expects (row_coords, col_coords) where each has the same length
        # We need to create a meshgrid and then flatten it
        target_y, target_x = np.meshgrid(target_lat, target_lon, indexing='ij')
        
        # Flatten and create coordinate arrays
        y_coords = target_y.ravel()
        x_coords = target_x.ravel()
        
        # Convert world coordinates to array indices
        # Assuming the source data covers -180 to 180 in longitude and -90 to 90 in latitude
        x_indices = ((x_coords + 180) / 360 * (width - 1)).astype(int)
        y_indices = ((y_coords + 90) / 180 * (height - 1)).astype(int)
        
        result = benchmark.benchmark_regridding_operation(
            source_data=sample_data,
            target_coords=(y_indices, x_indices),
            method='bilinear',
            name='test_integration'
        )
        
        assert result is not None
        assert result.name == 'test_integration'
        assert result.execution_time > 0
        assert result.throughput > 0
    
    def test_accuracy_benchmark_simple(self, sample_data):
        """Test that AccuracyBenchmark works with PyRegrid data."""
        benchmark = AccuracyBenchmark(threshold=1e-3)
        
        # Test that benchmark can work with PyRegrid DataArrays
        result, metrics = benchmark.benchmark_interpolation_accuracy(
            source_resolution=(sample_data.shape[0], sample_data.shape[1]),
            target_resolution=(sample_data.shape[0], sample_data.shape[1]),
            method='bilinear'
        )
        
        assert result is not None
        assert metrics is not None
        assert hasattr(metrics, 'rmse')  # AccuracyMetrics object has attributes
        assert hasattr(result, 'accuracy_error')  # BenchmarkResult has accuracy_error
        assert hasattr(result, 'execution_time')  # BenchmarkResult has execution_time
    
    def test_unified_benchmark_simple_run(self, sample_data):
        """Test a simple run of the unified benchmark."""
        # Create a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            config = BenchmarkConfiguration(
                use_dask=False,
                output_dir=temp_dir,
                iterations=1,  # Single iteration for faster testing
                methods=['bilinear']
            )
            
            runner = UnifiedBenchmarkRunner(config)
            
            # Define test resolutions
            test_resolutions = [(sample_data.shape[0], sample_data.shape[1])]
            
            # Run comprehensive benchmark
            results = runner.run_comprehensive_benchmark(test_resolutions)
            
            # Verify results structure
            assert 'timestamp' in results
            assert 'performance' in results
            assert 'accuracy' in results
            assert 'summary' in results
            
            # Verify performance results
            assert 'individual_results' in results['performance']
            assert len(results['performance']['individual_results']) > 0
            
            # Verify accuracy results
            assert 'individual_results' in results['accuracy']
            assert len(results['accuracy']['individual_results']) > 0
            
            # Verify summary
            assert 'total_performance_tests' in results['summary']
            assert 'total_accuracy_tests' in results['summary']
            assert 'config' in results['summary']
            
            # Test saving results
            results_path = runner.save_results()
            assert os.path.exists(results_path)
            
            # Verify the saved file contains expected data
            import json
            with open(results_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'timestamp' in saved_data
            assert 'performance' in saved_data
            assert 'accuracy' in saved_data
    
    def test_run_standard_benchmark_simple(self, sample_data):
        """Test the convenience function for running standard benchmarks."""
        # Create a temporary directory for results
        with tempfile.TemporaryDirectory() as temp_dir:
            # Run standard benchmark
            results_path = run_standard_benchmark(
                resolutions=[(sample_data.shape[0], sample_data.shape[1])],
                methods=['bilinear'],
                output_dir=temp_dir,
                use_dask=False
            )
            
            # Verify results file exists
            assert os.path.exists(results_path)
            
            # Verify the saved file contains expected data
            import json
            with open(results_path, 'r') as f:
                saved_data = json.load(f)
            
            assert 'timestamp' in saved_data
            assert 'performance' in saved_data
            assert 'accuracy' in saved_data
    
    def test_error_handling(self, sample_data):
        """Test error handling in benchmarking components."""
        # Test with invalid method
        benchmark = HighResolutionBenchmark(use_dask=False)
        
        with pytest.raises(ValueError):
            benchmark.benchmark_regridding_operation(
                source_data=sample_data,
                target_coords=(np.array([-90, 0, 90]), np.array([-45, 0, 45])),
                method='invalid_method',
                name='test_error'
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])