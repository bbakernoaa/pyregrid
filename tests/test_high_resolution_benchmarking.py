"""
Comprehensive TDD test cases for the high-resolution benchmarking system.

This module contains test cases covering performance metrics, accuracy validation,
and scalability testing following TDD principles.
"""
import pytest
import numpy as np
import dask
import dask.array as da
from typing import Tuple, Dict, List
import tempfile
import os
import time
import math

try:
    from dask.distributed import Client, LocalCluster
    HAS_DASK_DISTRIBUTED = True
except ImportError:
    HAS_DASK_DISTRIBUTED = False

from benchmarks.performance_metrics import (
    PerformanceMetrics, PerformanceCollector, HighResolutionBenchmark
)
from benchmarks.accuracy_validation import (
    AccuracyValidator, AccuracyMetrics, AnalyticalFieldGenerator, AccuracyBenchmark
)
from benchmarks.scalability_testing import (
    ScalabilityMetrics, ScalabilityBenchmark, StrongScalabilityTester, WeakScalabilityTester
)
from benchmarks.benchmark_base import BenchmarkResult


@pytest.fixture(scope="session")
def dask_client():
    """
    Create a Dask client for benchmark tests.
    
    This fixture creates a local Dask cluster for parallel processing during benchmarks.
    """
    # Create a local cluster for benchmarks
    cluster = LocalCluster(
        n_workers=1,  # Start with 1 worker for tests
        threads_per_worker=2,
        processes=False,  # Use threads instead of processes for better memory sharing
        dashboard_address=None  # Disable dashboard to reduce overhead
    )
    
    client = Client(cluster)
    
    try:
        yield client
    finally:
        client.close()
        cluster.close()


class TestPerformanceMetricsTDD:
    """TDD tests for performance metrics collection."""
    
    def test_performance_metrics_initialization(self):
        """Test that PerformanceMetrics can be initialized with required fields."""
        # This should initially fail since we need to create the implementation
        metrics = PerformanceMetrics(
            execution_time=1.0,
            memory_usage_peak=100.0,
            memory_usage_avg=50.0,
            cpu_percent=50.0,
            throughput=10.0,
            io_read_bytes=1024.0,
            io_write_bytes=2048.0
        )
        
        assert metrics.execution_time == 1.0
        assert metrics.memory_usage_peak == 100.0
        assert metrics.memory_usage_avg == 50.0
        assert metrics.cpu_percent == 50.0
        assert metrics.throughput == 10.0
        assert metrics.io_read_bytes == 1024.0
        assert metrics.io_write_bytes == 2048.0
    
    def test_performance_collector_initialization(self):
        """Test that PerformanceCollector can be initialized."""
        collector = PerformanceCollector()
        assert len(collector.metrics_history) == 0
    
    def test_performance_collection_context_manager(self):
        """Test that performance collection context manager works."""
        collector = PerformanceCollector()
        
        with collector.collect_performance("test_operation"):
            # Simulate some work
            time.sleep(0.01)
        
        assert len(collector.metrics_history) == 1
        metrics = collector.metrics_history[0]
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.execution_time >= 0
        assert metrics.memory_usage_peak >= 0
    
    def test_aggregated_metrics_calculation(self):
        """Test that aggregated metrics are calculated correctly."""
        collector = PerformanceCollector()
        
        # Add some sample metrics
        for i in range(3):
            with collector.collect_performance(f"test_op_{i}"):
                time.sleep(0.01)
        
        aggregated = collector.get_aggregated_metrics()
        
        assert 'execution_time_mean' in aggregated
        assert 'execution_time_median' in aggregated
        assert 'memory_usage_peak_mean' in aggregated
        assert aggregated['execution_time_mean'] >= 0
    
    def test_high_resolution_benchmark_initialization(self):
        """Test that HighResolutionBenchmark can be initialized."""
        benchmark = HighResolutionBenchmark(use_dask=True)
        assert benchmark.use_dask is True
        assert benchmark.performance_collector is not None
    
    def test_benchmark_regridding_operation(self, dask_client):
        """Test the benchmark_regridding_operation method."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create test data
        source_data = np.random.random((50, 100))
        target_coords = (np.linspace(-180, 180, 50), np.linspace(-90, 90, 100))
        
        # Create target coordinate mesh and flatten to points
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # This should initially fail since the implementation may not exist
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='test_benchmark'
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.name == 'test_benchmark'
        assert result.execution_time >= 0
    
    def test_benchmark_multiple_resolutions(self, dask_client):
        """Test the benchmark_multiple_resolutions method."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        resolutions = [(25, 50), (50, 100)]
        results = benchmark.benchmark_multiple_resolutions(
            resolutions=resolutions,
            method='bilinear',
            iterations=2
        )
        
        assert len(results) == len(resolutions) * 2
        for result in results:
            assert isinstance(result, BenchmarkResult)
    
    def test_benchmark_dask_vs_numpy(self, dask_client):
        """Test the benchmark_dask_vs_numpy method."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        results = benchmark.benchmark_dask_vs_numpy(
            resolution=(50, 100),
            method='bilinear',
            iterations=1
        )
        
        assert 'numpy' in results
        assert 'dask' in results
        assert len(results['numpy']) == 1
        assert len(results['dask']) == 1


class TestAccuracyValidationTDD:
    """TDD tests for accuracy validation."""
    
    def test_accuracy_validator_initialization(self):
        """Test that AccuracyValidator can be initialized."""
        validator = AccuracyValidator()
        assert validator is not None
    
    def test_rmse_calculation(self):
        """Test RMSE calculation with identical arrays."""
        validator = AccuracyValidator()
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        
        rmse = validator.calculate_rmse(arr1, arr2)
        assert rmse == 0.0  # Identical arrays should have 0 RMSE
    
    def test_rmse_calculation_different_arrays(self):
        """Test RMSE calculation with different arrays."""
        validator = AccuracyValidator()
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([2.0, 3.0, 4.0])
        
        rmse = validator.calculate_rmse(arr1, arr2)
        assert rmse > 0.0
    
    def test_mae_calculation(self):
        """Test MAE calculation."""
        validator = AccuracyValidator()
        arr1 = np.array([1.0, 2.0, 3.0])
        arr2 = np.array([2.0, 3.0, 4.0])
        
        mae = validator.calculate_mae(arr1, arr2)
        assert mae == 1.0  # Each element differs by 1.0
    
    def test_correlation_calculation(self):
        """Test correlation calculation."""
        validator = AccuracyValidator()
        arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        arr2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # Perfect correlation
        
        corr = validator.calculate_correlation(arr1, arr2)
        assert abs(corr - 1.0) < 1e-10 # Should be very close to 1.0
    
    def test_bias_calculation(self):
        """Test bias calculation."""
        validator = AccuracyValidator()
        arr1 = np.array([1.0, 2.0, 3.0])  # actual values
        arr2 = np.array([1.5, 2.5, 3.5])  # expected values
        
        bias = validator.calculate_bias(arr1, arr2)
        # The bias is calculated as mean(actual - expected) = mean([1.0,2.0,3.0] - [1.5,2.5,3.5])
        # = mean([-0.5, -0.5, -0.5]) = -0.5
        expected_bias = -0.5  # actual is 0.5 lower than expected on average
        assert abs(bias - expected_bias) < 1e-10
    
    def test_accuracy_metrics_initialization(self):
        """Test that AccuracyMetrics can be initialized."""
        metrics = AccuracyMetrics(
            rmse=0.1,
            mae=0.05,
            max_error=0.2,
            mean_error=0.01,
            std_error=0.02,
            correlation=0.95,
            bias=0.001,
            relative_rmse=0.01,
            n_valid_points=100,
            n_total_points=100,
            accuracy_threshold=1e-3
        )
        
        assert metrics.rmse == 0.1
        assert metrics.mae == 0.05
        assert metrics.correlation == 0.95
    
    def test_analytical_field_generator_sine_wave(self):
        """Test sine wave field generation."""
        generator = AnalyticalFieldGenerator()
        field = generator.sine_wave_field(10, 20)
        
        assert field.shape == (10, 20)
        assert not np.all(field == 0)  # Should not be all zeros
    
    def test_analytical_field_generator_gaussian_bump(self):
        """Test Gaussian bump field generation."""
        generator = AnalyticalFieldGenerator()
        field = generator.gaussian_bump_field(10, 20)
        
        assert field.shape == (10, 20)
        assert not np.all(field == 0)  # Should not be all zeros
    
    def test_analytical_field_generator_polynomial(self):
        """Test polynomial field generation."""
        generator = AnalyticalFieldGenerator()
        field = generator.polynomial_field(10, 20)
        
        assert field.shape == (10, 20)
        assert not np.all(field == 0)  # Should not be all zeros
    
    def test_accuracy_benchmark_initialization(self):
        """Test that AccuracyBenchmark can be initialized."""
        benchmark = AccuracyBenchmark(threshold=1e-6)
        assert benchmark.threshold == 1e-6
        assert benchmark.validator is not None
        assert benchmark.analytical_generator is not None
    
    def test_accuracy_benchmark_interpolation_accuracy(self):
        """Test interpolation accuracy benchmark."""
        benchmark = AccuracyBenchmark(threshold=1e-3)
        
        result, metrics = benchmark.benchmark_interpolation_accuracy(
            source_resolution=(20, 40),
            target_resolution=(20, 40),
            method='bilinear'
        )
        
        assert isinstance(result, BenchmarkResult)
        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
    
    def test_calculate_accuracy_metrics(self):
        """Test calculation of comprehensive accuracy metrics."""
        benchmark = AccuracyBenchmark(threshold=1e-3)
        
        actual = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = np.array([[1.1, 2.1], [3.1, 4.1]])  # Slightly different
        
        metrics = benchmark.calculate_accuracy_metrics(actual, expected, 1e-3)
        
        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.rmse >= 0
        assert metrics.mae >= 0
        assert metrics.n_valid_points == 4  # All points are valid
        assert metrics.n_total_points == 4
    
    def test_run_accuracy_convergence_test(self):
        """Test accuracy convergence test."""
        benchmark = AccuracyBenchmark(threshold=1e-3)
        
        resolutions = [(10, 20), (20, 40)]
        results = benchmark.run_accuracy_convergence_test(
            resolutions=resolutions,
            method='bilinear'
        )
        
        assert len(results) == len(resolutions)
        for result, metrics in results:
            assert isinstance(result, BenchmarkResult)
            assert isinstance(metrics, AccuracyMetrics)
    
    def test_validate_round_trip_accuracy(self):
        """Test round-trip accuracy validation."""
        benchmark = AccuracyBenchmark(threshold=1e-3)
        
        is_accurate, metrics = benchmark.validate_round_trip_accuracy(
            resolution=(20, 40),
            method='bilinear'
        )
        
        assert isinstance(is_accurate, bool)
        assert isinstance(metrics, AccuracyMetrics)


class TestScalabilityTestingTDD:
    """TDD tests for scalability testing."""
    
    def test_scalability_metrics_initialization(self):
        """Test that ScalabilityMetrics can be initialized."""
        metrics = ScalabilityMetrics(
            data_size=1000,
            execution_time=1.0,
            memory_usage=100.0,
            workers_used=2,
            threads_per_worker=4,
            tasks_completed=10,
            speedup=2.0,
            efficiency=0.8,
            throughput=1000.0,
            overhead=0.1
        )
        
        assert metrics.data_size == 1000
        assert metrics.execution_time == 1.0
        assert metrics.speedup == 2.0
        assert metrics.efficiency == 0.8
    
    def test_scalability_benchmark_initialization(self):
        """Test that ScalabilityBenchmark can be initialized."""
        benchmark = ScalabilityBenchmark(baseline_workers=1)
        assert benchmark.baseline_workers == 1
        assert len(benchmark.scalability_results) == 0
    
    def test_scalability_benchmark_worker_scalability(self, dask_client):
        """Test worker scalability benchmark."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        benchmark = ScalabilityBenchmark()
        
        metrics_list = benchmark.test_worker_scalability(
            resolution=(50, 100),
            max_workers=2,
            method='bilinear',
            dask_client=dask_client
        )
        
        assert len(metrics_list) == 2
        for metrics in metrics_list:
            assert isinstance(metrics, ScalabilityMetrics)
            assert metrics.data_size == 50 * 100
            assert metrics.workers_used > 0
    
    def test_scalability_benchmark_data_size_scalability(self, dask_client):
        """Test data size scalability benchmark."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        benchmark = ScalabilityBenchmark()
        
        sizes = [(25, 50), (50, 100)]
        n_workers_list = [1, 2]
        
        results = benchmark.test_data_size_scalability(
            sizes=sizes,
            n_workers_list=n_workers_list,
            method='bilinear',
            dask_client=dask_client
        )
        
        assert len(results) == len(sizes)
        for size_key, metrics_list in results.items():
            assert len(metrics_list) == len(n_workers_list)
            for metrics in metrics_list:
                assert isinstance(metrics, ScalabilityMetrics)
    
    def test_strong_scalability_tester_initialization(self):
        """Test that StrongScalabilityTester can be initialized."""
        tester = StrongScalabilityTester(baseline_workers=1)
        assert tester.baseline_workers == 1
    
    def test_strong_scalability_test(self, dask_client):
        """Test strong scalability analysis."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        tester = StrongScalabilityTester()
        
        results = tester.test_strong_scalability(
            resolution=(50, 100),
            worker_counts=[1, 2],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'execution_times' in results
        assert 'speedups' in results
        assert 'efficiencies' in results
        assert len(results['worker_counts']) == 2
        assert len(results['execution_times']) == 2
        assert len(results['speedups']) == 2
        assert len(results['efficiencies']) == 2
    
    def test_weak_scalability_tester_initialization(self):
        """Test that WeakScalabilityTester can be initialized."""
        tester = WeakScalabilityTester()
        assert tester is not None
    
    def test_weak_scalability_test(self, dask_client):
        """Test weak scalability analysis."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        tester = WeakScalabilityTester()
        
        results = tester.test_weak_scalability(
            base_resolution=(25, 50),
            worker_scale_factors=[1, 2],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'execution_times' in results
        assert 'work_per_worker' in results
        assert len(results['worker_scale_factors']) == 2
        assert len(results['execution_times']) == 2
        assert len(results['work_per_worker']) == 2


class TestIntegrationTDD:
    """TDD tests for integration of all benchmarking components."""
    
    def test_complete_benchmark_workflow(self, dask_client):
        """Test a complete workflow combining performance, accuracy, and scalability."""
        # Performance benchmark
        perf_benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        source_data = np.random.random((50, 100))
        target_coords = (np.linspace(-180, 180, 50), np.linspace(-90, 90, 100))
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        perf_result = perf_benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name='integration_perf_test'
        )
        
        # Accuracy benchmark
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
        acc_result, acc_metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=(50, 100),
            target_resolution=(50, 100),
            method='bilinear'
        )
        
        # Scalability benchmark
        if dask_client is not None:
            scalability_benchmark = ScalabilityBenchmark()
            scalability_results = scalability_benchmark.test_worker_scalability(
                resolution=(50, 100),
                max_workers=2,
                method='bilinear',
                dask_client=dask_client
            )
        
        # Verify all components work together
        assert isinstance(perf_result, BenchmarkResult)
        assert isinstance(acc_result, BenchmarkResult)
        assert isinstance(acc_metrics, AccuracyMetrics)
        
        if dask_client is not None:
            assert len(scalability_results) == 2
            for metrics in scalability_results:
                assert isinstance(metrics, ScalabilityMetrics)
    
    def test_error_handling_in_benchmarks(self, dask_client):
        """Test error handling in benchmark components."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Test with invalid method
        with pytest.raises(ValueError):
            benchmark.benchmark_regridding_operation(
                source_data=np.random.random((10, 10)),
                target_coords=(np.linspace(-180, 180, 10), np.linspace(-90, 90, 10)),
                method='invalid_method',
                name='error_test'
            )
    
    def test_edge_cases_in_accuracy_validation(self):
        """Test edge cases in accuracy validation."""
        validator = AccuracyValidator()
        
        # Test with NaN arrays
        arr1 = np.array([1.0, np.nan, 3.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        
        rmse = validator.calculate_rmse(arr1, arr2)
        assert rmse >= 0  # Should handle NaN gracefully
        
        # Test with arrays of different shapes
        arr1 = np.array([1.0, 2.0])
        arr2 = np.array([1.0, 2.0, 3.0])
        
        with pytest.raises(ValueError):
            validator.calculate_rmse(arr1, arr2)


# Parametrized tests for various scenarios
@pytest.mark.parametrize("resolution", [(10, 20), (20, 40), (50, 100)])
def test_performance_at_different_resolutions(resolution, dask_client):
    """Parametrized test for performance at different resolutions."""
    height, width = resolution
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    source_data = np.random.random((height, width))
    target_coords = (np.linspace(-180, 180, height), np.linspace(-90, 90, width))
    target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
    target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
    
    result = benchmark.benchmark_regridding_operation(
        source_data=source_data,
        target_coords=(target_points[0, :], target_points[1, :]),
        method='bilinear',
        name=f'param_test_{height}x{width}'
    )
    
    assert isinstance(result, BenchmarkResult)
    assert result.execution_time >= 0


@pytest.mark.parametrize("method", ['bilinear', 'nearest'])
def test_accuracy_with_different_methods(method):
    """Parametrized test for accuracy with different interpolation methods."""
    benchmark = AccuracyBenchmark(threshold=1e-3)
    
    result, metrics = benchmark.benchmark_interpolation_accuracy(
        source_resolution=(20, 40),
        target_resolution=(20, 40),
        method=method
    )
    
    assert isinstance(result, BenchmarkResult)
    assert isinstance(metrics, AccuracyMetrics)


@pytest.mark.parametrize("field_type", ['sine_wave', 'gaussian_bump', 'polynomial'])
def test_accuracy_with_different_field_types(field_type):
    """Parametrized test for accuracy with different analytical field types."""
    benchmark = AccuracyBenchmark(threshold=1e-3)
    generator = AnalyticalFieldGenerator()
    
    # Generate field based on type
    if field_type == 'sine_wave':
        field = generator.sine_wave_field(20, 40)
    elif field_type == 'gaussian_bump':
        field = generator.gaussian_bump_field(20, 40)
    elif field_type == 'polynomial':
        field = generator.polynomial_field(20, 40)
    
    assert field.shape == (20, 40)
    assert not np.all(field == 0)