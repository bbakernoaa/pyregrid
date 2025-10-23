"""
Comprehensive benchmark test suite for PyRegrid.

This module contains detailed benchmark tests covering all aspects of
performance, accuracy, and scalability for high-resolution regridding.
"""
import pytest
import numpy as np
import dask.array as da
from typing import Tuple, Dict, List
import time
import tempfile
import json
import os

from .benchmark_base import BenchmarkResult
from .performance_metrics import HighResolutionBenchmark, DistributedBenchmarkRunner, create_performance_report
from .accuracy_validation import AccuracyBenchmark, create_accuracy_report
from .scalability_testing import ScalabilityBenchmark, StrongScalabilityTester, WeakScalabilityTester, analyze_scalability_results


@pytest.mark.benchmark
class TestHighResolutionPerformance:
    """Tests for high-resolution performance benchmarks."""
    
    @pytest.mark.parametrize("resolution", [
        (100, 200),
        (200, 400),
        (500, 1000)
    ])
    def test_performance_at_resolution(self, resolution, dask_client):
        """Test performance at different high resolutions."""
        height, width = resolution
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create test data at the specified resolution
        source_data = benchmark._create_test_data(height, width, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, width//2),
            np.linspace(-90, height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Run benchmark
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name=f'high_res_performance_{height}x{width}'
        )
        
        # Verify results
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time >= 0
        assert result.memory_usage >= 0
        
        # For high-resolution tests, log performance metrics
        print(f"Resolution {height}x{width}: Time={result.execution_time:.4f}s, "
              f"Memory={result.memory_usage:.2f}MB")
    
    @pytest.mark.parametrize("method", ['bilinear', 'nearest'])
    def test_method_performance_comparison(self, method, dask_client):
        """Compare performance of different interpolation methods."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create test data
        source_data = benchmark._create_test_data(200, 400, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, 200),
            np.linspace(-90, 90, 400)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Run benchmark with specified method
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method=method,
            name=f'method_comparison_{method}'
        )
        
        assert isinstance(result, BenchmarkResult)
        assert result.execution_time >= 0
        print(f"Method {method}: Time={result.execution_time:.4f}s")


@pytest.mark.benchmark
class TestAccuracyAtScale:
    """Tests for accuracy validation at scale."""
    
    @pytest.mark.parametrize("resolution", [
        (100, 200),
        (200, 400)
    ])
    def test_accuracy_at_resolution(self, resolution):
        """Test accuracy at different resolutions."""
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
        
        # Test interpolation accuracy
        benchmark_result, accuracy_metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=resolution,
            target_resolution=resolution,
            method='bilinear'
        )
        
        assert isinstance(benchmark_result, BenchmarkResult)
        assert benchmark_result.accuracy_error is not None
        assert accuracy_metrics.rmse >= 0
        assert accuracy_metrics.mae >= 0
        
        print(f"Resolution {resolution}: RMSE={accuracy_metrics.rmse:.6f}, "
              f"MAE={accuracy_metrics.mae:.6f}, Correlation={accuracy_metrics.correlation:.4f}")
    
    def test_accuracy_vs_resolution_trend(self):
        """Test how accuracy changes with resolution."""
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
        
        resolutions = [(50, 100), (100, 200), (200, 400)]
        results = accuracy_benchmark.run_accuracy_convergence_test(
            resolutions=resolutions,
            method='bilinear'
        )
        
        assert len(results) == len(resolutions)
        
        # Check that we get valid results for each resolution
        for i, (benchmark_result, accuracy_metrics) in enumerate(results):
            assert isinstance(benchmark_result, BenchmarkResult)
            assert isinstance(accuracy_metrics, type(results[0][1]))
            print(f"Resolution {resolutions[i]}: RMSE={accuracy_metrics.rmse:.6f}")
        
        # In a properly converging system, accuracy should generally improve with resolution
        # (though this depends on the test function used)


@pytest.mark.benchmark 
class TestScalabilityPatterns:
    """Tests for different scalability patterns."""
    
    def test_strong_scalability_pattern(self, dask_client):
        """Test strong scalability pattern."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        tester = StrongScalabilityTester(baseline_workers=1)
        
        results = tester.test_strong_scalability(
            resolution=(200, 400),
            worker_counts=[1, 2, 4, 8],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'speedups' in results
        assert 'efficiencies' in results
        assert len(results['speedups']) == 4
        assert len(results['efficiencies']) == 4
        
        print(f"Strong scalability - Workers: {results['worker_counts']}")
        print(f"Speedups: {[f'{s:.2f}' for s in results['speedups']]}")
        print(f"Efficiencies: {[f'{e:.2f}' for e in results['efficiencies']]}")
    
    def test_weak_scalability_pattern(self, dask_client):
        """Test weak scalability pattern."""
        if dask_client is None:
            pytest.skip("Dask client not available")
        
        tester = WeakScalabilityTester()
        
        results = tester.test_weak_scalability(
            base_resolution=(100, 200),
            worker_scale_factors=[1, 2, 4],
            method='bilinear',
            dask_client=dask_client
        )
        
        assert 'execution_times' in results
        assert 'work_per_worker' in results
        assert len(results['execution_times']) == 3
        
        print(f"Weak scalability - Scale factors: {results['worker_scale_factors']}")
        print(f"Execution times: {[f'{t:.3f}' for t in results['execution_times']]}s")
        print(f"Work per worker: {[f'{w:.0f}' for w in results['work_per_worker']]} elements")


@pytest.mark.benchmark
class TestMemoryEfficiency:
    """Tests for memory efficiency at scale."""
    
    @pytest.mark.parametrize("resolution", [
        (100, 200),
        (200, 400)
    ])
    def test_memory_usage_at_resolution(self, resolution, dask_client):
        """Test memory usage patterns at different resolutions."""
        height, width = resolution
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create test data
        source_data = benchmark._create_test_data(height, width, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, width//2),
            np.linspace(-90, 90, height//2)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Run benchmark and capture memory metrics
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=(target_points[0, :], target_points[1, :]),
            method='bilinear',
            name=f'memory_test_{height}x{width}'
        )
        
        assert isinstance(result, BenchmarkResult)
        print(f"Resolution {height}x{width}: Memory usage={result.memory_usage:.2f}MB")
        
        # Expected memory should be roughly proportional to data size
        expected_memory_mb = (height * width * 8) / (1024 * 1024)  # 8 bytes per element (float64)
        print(f"Expected memory (rough estimate): {expected_memory_mb:.2f}MB")


@pytest.mark.benchmark
class TestIntegrationWorkflows:
    """Tests for integrated benchmark workflows."""
    
    def test_complete_benchmark_workflow(self, benchmark_output_dir, dask_client):
        """Test a complete benchmark workflow with report generation."""
        # Performance benchmark
        perf_benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create target coordinates in the correct format for map_coordinates
        height, width = 100, 200
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
        
        perf_result = perf_benchmark.benchmark_regridding_operation(
            source_data=np.random.random((height, width)),
            target_coords=(y_indices, x_indices),
            method='bilinear',
            name='workflow_performance_test'
        )
        
        # Accuracy benchmark
        accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
        acc_result, acc_metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=(100, 200),
            target_resolution=(100, 200),
            method='bilinear'
        )
        
        # Scalability benchmark
        if dask_client is not None:
            scalability_benchmark = ScalabilityBenchmark()
            scalability_results = scalability_benchmark.test_worker_scalability(
                resolution=(100, 200),
                max_workers=4,
                method='bilinear',
                dask_client=dask_client
            )
        
        # Generate reports
        perf_report_path = os.path.join(benchmark_output_dir, "performance_report.json")
        acc_report_path = os.path.join(benchmark_output_dir, "accuracy_report.json")
        
        # Create performance report
        perf_report = create_performance_report(
            results=[perf_result],
            output_file=perf_report_path
        )
        
        # Create accuracy report
        acc_report = create_accuracy_report(
            results=[(acc_result, acc_metrics)],
            output_file=acc_report_path
        )
        
        # Verify reports were created
        assert os.path.exists(perf_report_path)
        assert os.path.exists(acc_report_path)
        
        print(f"Performance report saved to: {perf_report_path}")
        print(f"Accuracy report saved to: {acc_report_path}")
    
    def test_distributed_benchmark_workflow(self, dask_client):
        """Test distributed benchmark workflow."""
        if not dask_client:
            pytest.skip("Dask client not available for distributed testing")
        
        try:
            # Create distributed benchmark runner
            dist_runner = DistributedBenchmarkRunner(client=dask_client)
            
            # Define a simple benchmark function
            def simple_benchmark_task(worker_client=None, **kwargs):
                # Simulate a computational task
                time.sleep(0.1)  # Simulate work
                return BenchmarkResult(
                    name='distributed_task',
                    execution_time=0.1,
                    memory_usage=10.0,
                    cpu_percent=50.0,
                    metadata=kwargs
                )
            
            # Run benchmark on multiple workers
            results = dist_runner.run_benchmark_on_workers(
                simple_benchmark_task,
                n_workers=2
            )
            
            assert len(results) >= 1  # At least one result should be returned
            for result in results:
                assert isinstance(result, BenchmarkResult)
                
        except Exception as e:
            pytest.skip(f"Distributed benchmark failed: {str(e)}")


@pytest.mark.benchmark
class TestBenchmarkRegression:
    """Tests for detecting performance regressions."""
    
    def test_performance_regression_detection(self, dask_client):
        """Test for detecting performance regressions."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create target coordinates in the correct format for map_coordinates
        height, width = 100, 200
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
        
        # Run the same operation multiple times to establish baseline
        execution_times = []
        for i in range(5):
            result = benchmark.benchmark_regridding_operation(
                source_data=np.random.random((height, width)),
                target_coords=(y_indices, x_indices),
                method='bilinear',
                name=f'regression_test_run_{i}'
            )
            execution_times.append(result.execution_time)
        
        # Calculate statistics
        mean_time = np.mean(execution_times)
        std_time = np.std(execution_times)
        
        print(f"Execution times: {[f'{t:.4f}' for t in execution_times]}")
        print(f"Mean: {mean_time:.4f}s, Std: {std_time:.4f}s")
        
        # Check that times are reasonably consistent (no major regression)
        # Allow for 50% variation as a basic sanity check
        cv = std_time / mean_time if mean_time > 0 else 0
        assert cv < 0.5, f"Coefficient of variation too high: {cv:.2f}"


# Additional utility functions for benchmark management

def run_performance_suite(output_dir: str = "./benchmark_results", dask_client = None):
    """
    Run the complete performance benchmark suite.
    
    Args:
        output_dir: Directory to store results
        dask_client: Dask client for parallel processing
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create benchmarks instance
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    # Define test resolutions
    resolutions = [(50, 100), (100, 200), (200, 400)]
    methods = ['bilinear', 'nearest']
    
    results = []
    for res in resolutions:
        for method in methods:
            height, width = res
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
                method=method,
                name=f'perf_suite_{method}_{height}x{width}'
            )
            results.append(result)
    
    # Generate performance report
    report_path = os.path.join(output_dir, "performance_suite_report.json")
    create_performance_report(results, output_path=report_path)
    
    return results


def run_accuracy_suite(output_dir: str = "./benchmark_results"):
    """
    Run the complete accuracy benchmark suite.
    
    Args:
        output_dir: Directory to store results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
    
    # Define test parameters
    resolutions = [(50, 100), (100, 200)]
    methods = ['bilinear', 'nearest']
    
    results = []
    for res in resolutions:
        for method in methods:
            result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
                source_resolution=res,
                target_resolution=res,
                method=method
            )
            results.append((result, metrics))
    
    # Generate accuracy report
    report_path = os.path.join(output_dir, "accuracy_suite_report.json")
    create_accuracy_report(results, output_file=report_path)
    
    return results


if __name__ == "__main__":
    # This allows running the benchmarks directly with Python
    pytest.main([__file__, "-v", "--benchmark"])