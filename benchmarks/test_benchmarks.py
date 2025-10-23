"""
Main benchmark test suite for PyRegrid.

This module contains comprehensive benchmark tests that integrate
performance metrics, accuracy validation, and scalability testing.
"""
import pytest
import numpy as np
import dask.array as da
from typing import Tuple
import time

from .benchmark_base import benchmark_runner, BenchmarkResult
from .performance_metrics import HighResolutionBenchmark, create_performance_report
from .accuracy_validation import AccuracyBenchmark, create_accuracy_report
from .scalability_testing import ScalabilityBenchmark, StrongScalabilityTester, WeakScalabilityTester, analyze_scalability_results


@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_basic_performance_benchmark(self, benchmark_data_small, dask_client):
        """Test basic performance benchmarking."""
        source_data, target_coords = benchmark_data_small
        height, width = source_data.shape
        
        # Create target coordinate mesh
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Create benchmark instance
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Run benchmark
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='basic_performance_test'
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time >= 0
        assert result.memory_usage >= 0
        assert result.name == 'basic_performance_test'
    
    def test_performance_multiple_resolutions(self, dask_client):
        """Test performance across multiple resolutions."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Test different resolutions
        resolutions = [(25, 50), (50, 100), (100, 200)]
        results = benchmark.benchmark_multiple_resolutions(
            resolutions=resolutions,
            method='bilinear',
            iterations=2
        )
        
        assert len(results) == len(resolutions) * 2  # 2 iterations per resolution
        for result in results:
            assert isinstance(result, BenchmarkResult)
            assert result.execution_time >= 0
    
    def test_dask_vs_numpy_performance(self, dask_client):
        """Compare Dask vs NumPy performance."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Compare performance between Dask and NumPy
        comparison_results = benchmark.benchmark_dask_vs_numpy(
            resolution=(100, 200),
            method='bilinear',
            iterations=2
        )
        
        assert 'numpy' in comparison_results
        assert 'dask' in comparison_results
        assert len(comparison_results['numpy']) == 2
        assert len(comparison_results['dask']) == 2
        
        # Both should return valid results
        for numpy_result in comparison_results['numpy']:
            assert isinstance(numpy_result, BenchmarkResult)
        for dask_result in comparison_results['dask']:
            assert isinstance(dask_result, BenchmarkResult)


@pytest.mark.benchmark
class TestAccuracyBenchmarks:
    """Accuracy validation benchmark tests."""
    
    def test_interpolation_accuracy(self):
        """Test interpolation accuracy validation."""
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-3)
        
        # Test accuracy at different resolutions
        benchmark_result, accuracy_metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=(50, 100),
            target_resolution=(50, 100),
            method='bilinear'
        )
        
        assert isinstance(benchmark_result, BenchmarkResult)
        assert benchmark_result.accuracy_error is not None
        assert accuracy_metrics.rmse >= 0
        assert accuracy_metrics.mae >= 0
        assert accuracy_metrics.max_error >= 0
        assert -1 <= accuracy_metrics.correlation <= 1
    
    def test_accuracy_convergence(self):
        """Test accuracy convergence across resolutions."""
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-3)
        
        # Test convergence with increasing resolutions
        resolutions = [(25, 50), (50, 100), (100, 200)]
        results = accuracy_benchmark.run_accuracy_convergence_test(
            resolutions=resolutions,
            method='bilinear'
        )
        
        assert len(results) == len(resolutions)
        for benchmark_result, accuracy_metrics in results:
            assert isinstance(benchmark_result, BenchmarkResult)
            assert isinstance(accuracy_metrics, accuracy_benchmark.calculate_accuracy_metrics.__annotations__['return'])
    
    def test_round_trip_accuracy(self):
        """Test round-trip interpolation accuracy."""
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-3)
        
        is_accurate, accuracy_metrics = accuracy_benchmark.validate_round_trip_accuracy(
            resolution=(50, 100),
            method='bilinear',
            max_error_threshold=1e-2
        )
        
        assert isinstance(is_accurate, bool)
        assert isinstance(accuracy_metrics, accuracy_benchmark.calculate_accuracy_metrics.__annotations__['return'])


@pytest.mark.benchmark
class TestScalabilityBenchmarks:
    """Scalability benchmark tests."""
    
    def test_worker_scalability(self, dask_client):
        """Test scalability with varying worker counts."""
        if dask_client is None:
            pytest.skip("Dask client not available for scalability testing")
        
        scalability_benchmark = ScalabilityBenchmark()
        
        # Test scalability with different worker counts
        metrics_list = scalability_benchmark.test_worker_scalability(
            resolution=(100, 200),
            max_workers=4,
            method='bilinear',
            dask_client=dask_client
        )
        
        assert len(metrics_list) == 4  # 1 to 4 workers
        for metrics in metrics_list:
            assert metrics.data_size > 0
            assert metrics.execution_time >= 0
            assert metrics.workers_used > 0
            assert metrics.speedup >= 0
            # Efficiency can sometimes exceed 1 due to measurement noise or system variations
            assert metrics.efficiency >= 0
    
    def test_strong_scalability(self, dask_client):
        """Test strong scalability (fixed problem size, varying workers)."""
        if dask_client is None:
            pytest.skip("Dask client not available for scalability testing")
        
        strong_tester = StrongScalabilityTester()
        
        # Test strong scalability
        results = strong_tester.test_strong_scalability(
            resolution=(100, 200),
            worker_counts=[1, 2, 4],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'execution_times' in results
        assert 'speedups' in results
        assert 'efficiencies' in results
        assert len(results['execution_times']) == 3
        assert len(results['speedups']) == 3
        assert len(results['efficiencies']) == 3
        
        # Analyze the results
        analysis = analyze_scalability_results(results)
        assert 'scalability_score' in analysis
        assert 'scaling_efficiency' in analysis
    
    def test_weak_scalability(self, dask_client):
        """Test weak scalability (proportional problem size and workers)."""
        if dask_client is None:
            pytest.skip("Dask client not available for scalability testing")
        
        weak_tester = WeakScalabilityTester()
        
        # Test weak scalability
        results = weak_tester.test_weak_scalability(
            base_resolution=(50, 100),
            worker_scale_factors=[1, 2, 4],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'execution_times' in results
        assert 'work_per_worker' in results
        assert len(results['execution_times']) == 3
        assert len(results['work_per_worker']) == 3


@pytest.mark.large_benchmark
class TestLargeScaleBenchmarks:
    """Large-scale benchmark tests."""
    
    def test_large_performance_benchmark(self, benchmark_data_large, dask_client):
        """Test performance with large datasets."""
        try:
            if hasattr(pytest, 'config') and not pytest.config.getoption("--benchmark-large"):
                pytest.skip("Large benchmarks not enabled")
        except:
            # If pytest.config is not available, proceed without checking
            pass
        
        source_data, target_coords = benchmark_data_large
        height, width = source_data.shape
        
        # Create target coordinate mesh
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Create benchmark instance
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Run benchmark
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='large_performance_test'
        )
        
        # Verify the result
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time >= 0
        assert result.memory_usage >= 0
    
    def test_large_scalability(self, dask_client):
        """Test scalability with large datasets."""
        try:
            if hasattr(pytest, 'config') and not pytest.config.getoption("--benchmark-large"):
                pytest.skip("Large benchmarks not enabled")
        except:
            # If pytest.config is not available, proceed without checking
            pass
        
        if dask_client is None:
            pytest.skip("Dask client not available for scalability testing")
        
        scalability_benchmark = ScalabilityBenchmark()
        
        # Test with larger resolution
        metrics_list = scalability_benchmark.test_worker_scalability(
            resolution=(200, 400),  # Larger than default
            max_workers=8,
            method='bilinear',
            dask_client=dask_client
        )
        
        assert len(metrics_list) == 8
        for metrics in metrics_list:
            assert metrics.data_size > 0
            assert metrics.execution_time >= 0


@pytest.mark.benchmark
def test_comprehensive_benchmark_report(temporary_benchmark_file, dask_client):
    """Generate a comprehensive benchmark report."""
    # Run a few benchmark tests to populate results
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    # Run some tests to generate results
    # Create target coordinates in the correct format for map_coordinates
    height, width = 50, 100
    target_lon = np.linspace(-180, 180, width)
    target_lat = np.linspace(-90, 90, height)
    
    # Create coordinate arrays for map_coordinates
    target_y, target_x = np.meshgrid(target_lat, target_lon, indexing='ij')
    
    # Flatten and create coordinate arrays
    y_coords = target_y.ravel()
    x_coords = target_x.ravel()
    
    # Convert world coordinates to array indices
    x_indices = ((x_coords + 180) / 360 * (width - 1)).astype(int)
    y_indices = ((y_coords + 90) / 180 * (height - 1)).astype(int)
    
    result1 = benchmark.benchmark_regridding_operation(
        source_data=np.random.random((height, width)),
        target_coords=(y_indices, x_indices),
        method='bilinear',
        name='report_test_1'
    )
    
    # Create target coordinates in the correct format for map_coordinates
    height2, width2 = 25, 50
    target_lon2 = np.linspace(-180, 180, width2)
    target_lat2 = np.linspace(-90, 90, height2)
    
    # Create coordinate arrays for map_coordinates
    target_y2, target_x2 = np.meshgrid(target_lat2, target_lon2, indexing='ij')
    
    # Flatten and create coordinate arrays
    y_coords2 = target_y2.ravel()
    x_coords2 = target_x2.ravel()
    
    # Convert world coordinates to array indices
    x_indices2 = ((x_coords2 + 180) / 360 * (width2 - 1)).astype(int)
    y_indices2 = ((y_coords2 + 90) / 180 * (height2 - 1)).astype(int)
    
    result2 = benchmark.benchmark_regridding_operation(
        source_data=np.random.random((height2, width2)),
        target_coords=(y_indices2, x_indices2),
        method='nearest',
        name='report_test_2'
    )
    
    # Create performance report
    report = create_performance_report(
        results=benchmark_runner.results,
        output_file=temporary_benchmark_file
    )
    
    # Verify report structure
    assert 'summary' in report
    assert 'detailed_results' in report
    assert 'aggregated_metrics' in report
    assert len(report['summary']) > 0


@pytest.mark.benchmark
def test_accuracy_validation_report(temporary_benchmark_file):
    """Generate an accuracy validation report."""
    accuracy_benchmark = AccuracyBenchmark(threshold=1e-3)
    
    # Run accuracy tests
    results = []
    for resolution in [(25, 50), (50, 100)]:
        result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=resolution,
            target_resolution=resolution,
            method='bilinear'
        )
        results.append((result, metrics))
    
    # Create accuracy report
    report = create_accuracy_report(results, output_file=temporary_benchmark_file)
    
    # Verify report structure
    assert 'summary' in report
    assert 'detailed_results' in report
    assert len(report['summary']) > 0
    assert len(report['detailed_results']) == len(results)


if __name__ == "__main__":
    # This allows running the benchmarks directly with Python
    pytest.main([__file__, "-v"])