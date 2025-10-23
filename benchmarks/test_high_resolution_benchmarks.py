"""
High-resolution benchmark tests for PyRegrid.

This module contains specialized tests for high-resolution scenarios,
such as a 3km global grid which would be extremely large.
"""
import pytest
import numpy as np
import dask.array as da
from typing import Tuple
import time
import math

from .benchmark_base import BenchmarkResult
from .performance_metrics import HighResolutionBenchmark
from .accuracy_validation import AccuracyBenchmark
from .scalability_testing import ScalabilityBenchmark


def estimate_3km_grid_size():
    """
    Estimate the size of a 3km global grid.
    
    A 3km global grid would have approximately:
    - 12001 x 6001 points (36.1 million points) for a global grid
    - With 0.03 degree spacing (approximate for 3km at equator)
    """
    # For 3km resolution: about 0.03 degrees per cell at equator
    # Global coverage: -180 to 180 longitude, -90 to 90 latitude
    # This gives us approximately 12001 x 6001 points
    lon_points = 12001  # ~0.03 degree spacing in longitude
    lat_points = 6001   # ~0.03 degree spacing in latitude
    return lat_points, lon_points


@pytest.mark.benchmark
class Test3kmGridBenchmarks:
    """Benchmarks for 3km global grid scenarios."""
    
    def test_3km_grid_memory_efficiency(self, dask_client):
        """Test memory efficiency with 3km grid equivalent data."""
        if dask_client is None:
            pytest.skip("Dask client required for large grid testing")
        
        # Use a smaller proxy for the 3km grid to avoid memory issues in testing
        # In real usage, we'd use the full 3km grid size
        proxy_height, proxy_width = 600, 1200  # 1/10th scale for testing
        
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create proxy test data
        source_data = benchmark._create_test_data(proxy_height, proxy_width, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, proxy_width//2),
            np.linspace(-90, 90, proxy_height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Run benchmark
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='3km_proxy_memory_efficiency'
        )
        
        assert isinstance(result, BenchmarkResult)
        print(f"3km proxy ({proxy_height}x{proxy_width}): Memory={result.memory_usage:.2f}MB, "
              f"Time={result.execution_time:.4f}s")
        
        # Memory should be reasonable for the proxy size
        expected_memory_mb = (proxy_height * proxy_width * 8) / (1024 * 1024)  # 8 bytes per element
        assert result.memory_usage < expected_memory_mb * 5  # Allow 5x overhead
    
    def test_3km_grid_chunking_strategy(self, dask_client):
        """Test optimal chunking strategies for 3km grid data."""
        if dask_client is None:
            pytest.skip("Dask client required for chunking tests")
        
        # Test different chunking strategies with proxy data
        proxy_height, proxy_width = 600, 1200
        
        # Create data with different chunk sizes
        base_data = np.random.random((proxy_height, proxy_width))
        
        chunk_strategies = [
            (50, 100),   # Small chunks
            (100, 200),  # Medium chunks  
            (20, 400),  # Large chunks
            'auto'       # Auto chunks
        ]
        
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        results = {}
        
        for strategy in chunk_strategies:
            if strategy == 'auto':
                dask_data = da.from_array(base_data, chunks='auto')
            else:
                chunk_h, chunk_w = strategy
                dask_data = da.from_array(base_data, chunks=(chunk_h, chunk_w))
            
            target_coords = (
                np.linspace(-180, 180, proxy_width//2),
                np.linspace(-90, 90, proxy_height//2)
            )
            target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
            target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
            
            result = benchmark.benchmark_regridding_operation(
                source_data=dask_data,
                target_coords=(target_points[0, :], target_points[1, :]),
                method='bilinear',
                name=f'chunking_strategy_{strategy}'
            )
            
            results[str(strategy)] = result
            print(f"Chunking {strategy}: Time={result.execution_time:.4f}s, "
                  f"Memory={result.memory_usage:.2f}MB")
        
        # Verify all strategies completed successfully
        assert len(results) == len(chunk_strategies)
        for strategy, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert result.execution_time >= 0
    
    def test_3km_grid_parallel_efficiency(self, dask_client):
        """Test parallel processing efficiency with 3km grid equivalent."""
        if dask_client is None:
            pytest.skip("Dask client required for parallel efficiency tests")
        
        proxy_height, proxy_width = 600, 1200
        scalability_benchmark = ScalabilityBenchmark()
        
        # Test scalability with proxy size
        metrics_list = scalability_benchmark.test_worker_scalability(
            resolution=(proxy_height, proxy_width),
            max_workers=4,
            method='bilinear',
            dask_client=dask_client
        )
        
        assert len(metrics_list) == 4  # 1 to 4 workers
        for i, metrics in enumerate(metrics_list):
            print(f"Workers {metrics.workers_used}: Time={metrics.execution_time:.4f}s, "
                  f"Speedup={metrics.speedup:.2f}x, Efficiency={metrics.efficiency:.2f}")
            assert metrics.data_size == proxy_height * proxy_width
            assert metrics.execution_time >= 0
            assert metrics.workers_used == i + 1


@pytest.mark.benchmark
class TestHighResolutionPatterns:
    """Tests for high-resolution patterns and behaviors."""
    
    @pytest.mark.parametrize("resolution_factor", [1, 2, 4, 8])
    def test_scaling_patterns(self, resolution_factor, dask_client):
        """Test how performance scales with resolution."""
        base_height, base_width = 100, 200
        height = base_height * resolution_factor
        width = base_width * resolution_factor
        
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create test data (use smaller chunks for higher resolutions to manage memory)
        source_data = benchmark._create_test_data(height, width, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, width//2),
            np.linspace(-90, 90, height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        start_time = time.time()
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name=f'scaling_test_{height}x{width}'
        )
        end_time = time.time()
        
        actual_time = end_time - start_time
        
        # Log performance metrics
        data_size = height * width
        print(f"Resolution {height}x{width} ({data_size/1e6:.1f}M elements): "
              f"Time={result.execution_time:.4f}s, "
              f"Throughput={data_size/result.execution_time/1e6:.2f}M elements/s")
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time >= 0
    
    def test_memory_vs_resolution_relationship(self, dask_client):
        """Test the relationship between resolution and memory usage."""
        resolutions = [(50, 100), (100, 200), (200, 400), (400, 800)]
        memory_usage = []
        
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        for height, width in resolutions:
            source_data = benchmark._create_test_data(height, width, use_dask=True)
            target_coords = (
                np.linspace(-180, 180, width//2),
                np.linspace(-90, 90, height//2)
            )
            target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
            target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
            
            result = benchmark.benchmark_regridding_operation(
                source_data=source_data,
                target_coords=(target_points[0, :], target_points[1, :]),
                method='bilinear',
                name=f'memory_test_{height}x{width}'
            )
            
            memory_usage.append(result.memory_usage)
            data_size = height * width
            expected_mb = (data_size * 8) / (1024 * 1024)  # 8 bytes per element
            
            print(f"Resolution {height}x{width}: Memory={result.memory_usage:.2f}MB, "
                  f"Expected~={expected_mb:.2f}MB")
        
        # Verify all results are valid
        assert len(memory_usage) == len(resolutions)
        for mem in memory_usage:
            assert mem >= 0


@pytest.mark.benchmark
class TestLargeGridOptimizations:
    """Tests for optimizations with large grids."""
    
    def test_lazy_evaluation_benefits(self, dask_client):
        """Test benefits of lazy evaluation with large grids."""
        if dask_client is None:
            pytest.skip("Dask client required for lazy evaluation tests")
        
        # Create a moderately large grid
        height, width = 500, 1000
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create dask array (lazy)
        source_data_lazy = benchmark._create_test_data(height, width, use_dask=True)
        
        # Create numpy array (eager)
        source_data_eager = benchmark._create_test_data(height, width, use_dask=False)
        
        target_coords = (
            np.linspace(-180, 180, width//2),
            np.linspace(-90, 90, height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Test lazy evaluation
        start_time = time.time()
        result_lazy = benchmark.benchmark_regridding_operation(
            source_data=source_data_lazy,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='lazy_evaluation_test'
        )
        time_lazy = time.time() - start_time
        
        # Test eager evaluation (for comparison, though this might be slow)
        start_time = time.time()
        result_eager = benchmark.benchmark_regridding_operation(
            source_data=source_data_eager,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='eager_evaluation_test'
        )
        time_eager = time.time() - start_time
        
        print(f"Lazy evaluation: {result_lazy.execution_time:.4f}s")
        print(f"Eager evaluation: {result_eager.execution_time:.4f}s")
        
        # Both should complete successfully
        assert isinstance(result_lazy, BenchmarkResult)
        assert isinstance(result_eager, BenchmarkResult)
    
    def test_chunk_boundary_effects(self, dask_client):
        """Test performance at chunk boundaries."""
        if dask_client is None:
            pytest.skip("Dask client required for chunk boundary tests")
        
        height, width = 400, 800
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create data with different chunk sizes to test boundary effects
        base_data = np.random.random((height, width))
        
        # Test with chunk size that divides evenly
        chunked_data_even = da.from_array(base_data, chunks=(100, 200))
        
        # Test with chunk size that doesn't divide evenly
        chunked_data_odd = da.from_array(base_data, chunks=(123, 247))
        
        target_coords = (
            np.linspace(-180, 180, width//2),
            np.linspace(-90, 90, height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        result_even = benchmark.benchmark_regridding_operation(
            source_data=chunked_data_even,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='chunk_even_division'
        )
        
        result_odd = benchmark.benchmark_regridding_operation(
            source_data=chunked_data_odd,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='chunk_odd_division'
        )
        
        print(f"Even chunks: {result_even.execution_time:.4f}s")
        print(f"Odd chunks: {result_odd.execution_time:.4f}s")
        
        # Both should complete successfully
        assert isinstance(result_even, BenchmarkResult)
        assert isinstance(result_odd, BenchmarkResult)


@pytest.mark.benchmark
def test_realistic_3km_scenario(dask_client):
    """Test a realistic scenario approximating 3km grid processing."""
    if dask_client is None:
        pytest.skip("Dask client required for realistic scenario test")
    
    # Use proxy dimensions that are manageable but still represent high-resolution challenges
    height, width = 800, 1600  # This is about 1/27th of a true 3km grid in each dimension
    
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    # Create high-resolution test data
    source_data = benchmark._create_test_data(height, width, use_dask=True)
    target_coords = (
        np.linspace(-180, 180, width//4),  # Target at 1/4 resolution
        np.linspace(-90, 90, height//4)
    )
    target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
    target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
    
    # Run the high-resolution benchmark
    result = benchmark.benchmark_regridding_operation(
        source_data=source_data,
        target_coords=(target_points[0, :], target_points[1, :]),
        method='bilinear',
        name='realistic_high_res_scenario'
    )
    
    print(f"Realistic high-res test ({height}x{width}): "
          f"Time={result.execution_time:.4f}s, "
          f"Memory={result.memory_usage:.2f}MB")
    
    # Verify the result
    assert isinstance(result, BenchmarkResult)
    assert result.execution_time >= 0
    assert result.memory_usage >= 0


if __name__ == "__main__":
    # This allows running the high-resolution benchmarks directly with Python
    pytest.main([__file__, "-v", "--benchmark"])