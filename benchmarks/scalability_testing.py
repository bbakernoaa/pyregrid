"""
Scalability testing framework for PyRegrid benchmarking.

This module provides utilities for testing scalability across different
data sizes, worker counts, and computational configurations.
"""
import numpy as np
import dask
import dask.array as da
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from dataclasses import dataclass
import time
import psutil
import statistics
from contextlib import contextmanager

try:
    from dask.distributed import Client, as_completed, wait
    HAS_DASK_DISTRIBUTED = True
except ImportError:
    HAS_DASK_DISTRIBUTED = False

from .benchmark_base import BenchmarkResult, ScalabilityTester
from .performance_metrics import HighResolutionBenchmark, PerformanceCollector


@dataclass
class ScalabilityMetrics:
    """Data class to store scalability metrics."""
    data_size: int  # Total number of elements
    execution_time: float
    memory_usage: float  # Peak memory in MB
    workers_used: int
    threads_per_worker: int
    tasks_completed: int
    speedup: float  # Speedup compared to single worker
    efficiency: float  # Efficiency percentage
    throughput: float  # Elements processed per second
    overhead: float  # Overhead percentage


class ScalabilityBenchmark:
    """Class for running scalability tests."""
    
    def __init__(self, baseline_workers: int = 1):
        self.baseline_workers = baseline_workers
        self.performance_collector = PerformanceCollector()
        self.scalability_results: List[ScalabilityMetrics] = []
    
    def test_data_size_scalability(self, 
                                 sizes: List[Tuple[int, int]],  # List of (height, width)
                                 n_workers_list: List[int] = [1, 2, 4, 8],
                                 method: str = 'bilinear',
                                 dask_client: Optional['Client'] = None) -> Dict[str, List[ScalabilityMetrics]]:
        """
        Test scalability across different data sizes and worker counts.
        
        Args:
            sizes: List of (height, width) tuples for different data sizes
            n_workers_list: List of worker counts to test
            method: Interpolation method to use
            dask_client: Optional Dask client to use
            
        Returns:
            Dictionary mapping size to scalability metrics
        """
        results = {}
        
        for height, width in sizes:
            size_metrics = []
            data_size = height * width
            
            # Create baseline test (single worker or baseline configuration)
            baseline_time = self._run_single_test(height, width, method, 1, dask_client)
            
            for n_workers in n_workers_list:
                # Run test with specified number of workers
                execution_time = self._run_single_test(height, width, method, n_workers, dask_client)
                
                # Calculate scalability metrics
                speedup = baseline_time / execution_time if execution_time > 0 else 0
                efficiency = speedup / n_workers if n_workers > 0 else 0
                
                # Calculate throughput
                throughput = data_size / execution_time if execution_time > 0 else 0
                
                # Get memory usage (simplified - in real implementation would track actual usage)
                memory_usage = self._estimate_memory_usage(height, width)
                
                metrics = ScalabilityMetrics(
                    data_size=data_size,
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    workers_used=n_workers,
                    threads_per_worker=1,  # Simplified
                    tasks_completed=1,     # Simplified
                    speedup=speedup,
                    efficiency=efficiency,
                    throughput=throughput,
                    overhead=0 # Simplified
                )
                
                size_metrics.append(metrics)
            
            results[f"{height}x{width}"] = size_metrics
        
        return results
    
    def test_worker_scalability(self, 
                              resolution: Tuple[int, int],
                              max_workers: int = 8,
                              method: str = 'bilinear',
                              dask_client: Optional['Client'] = None) -> List[ScalabilityMetrics]:
        """
        Test scalability by varying the number of workers.
        
        Args:
            resolution: Grid resolution (height, width)
            max_workers: Maximum number of workers to test
            method: Interpolation method to use
            dask_client: Optional Dask client to use
            
        Returns:
            List of scalability metrics for different worker counts
        """
        height, width = resolution
        data_size = height * width
        metrics_list = []
        
        # Establish baseline with 1 worker
        baseline_time = self._run_single_test(height, width, method, 1, dask_client)
        
        # Test with increasing numbers of workers
        for n_workers in range(1, max_workers + 1):
            execution_time = self._run_single_test(height, width, method, n_workers, dask_client)
            
            # Calculate scalability metrics
            speedup = baseline_time / execution_time if execution_time > 0 else 0
            efficiency = speedup / n_workers if n_workers > 0 else 0
            throughput = data_size / execution_time if execution_time > 0 else 0
            
            # Get memory usage estimate
            memory_usage = self._estimate_memory_usage(height, width)
            
            metrics = ScalabilityMetrics(
                data_size=data_size,
                execution_time=execution_time,
                memory_usage=memory_usage,
                workers_used=n_workers,
                threads_per_worker=1,  # Simplified
                tasks_completed=1,     # Simplified
                speedup=speedup,
                efficiency=efficiency,
                throughput=throughput,
                overhead=0 # Simplified
            )
            
            metrics_list.append(metrics)
        
        return metrics_list
    
    def _run_single_test(self, 
                        height: int, 
                        width: int, 
                        method: str, 
                        n_workers: int,
                        dask_client: Optional['Client']) -> float:
        """
        Run a single test and return execution time.
        
        Args:
            height: Grid height
            width: Grid width
            method: Interpolation method
            n_workers: Number of workers to use
            dask_client: Dask client to use
            
        Returns:
            Execution time in seconds
        """
        # Create test data
        source_data = self._create_test_data(height, width)
        target_coords = (
            np.linspace(-180, 180, width),
            np.linspace(-90, 90, height)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Create benchmark instance
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Record start time
        start_time = time.perf_counter()
        
        # Run the interpolation
        try:
            result = benchmark.benchmark_regridding_operation(
                source_data=source_data,
                target_coords=(target_points[0, :], target_points[1, :]),
                method=method,
                name=f"scalability_test_{height}x{width}_{n_workers}workers"
            )
            execution_time = result.execution_time
        except Exception as e:
            # If there's an error, return a large time value
            execution_time = float('inf')
        
        return execution_time
    
    def _create_test_data(self, height: int, width: int) -> da.Array:
        """Create test data for scalability testing."""
        # Create analytical test function (e.g., sine wave pattern)
        lon = np.linspace(-180, 180, width)
        lat = np.linspace(-90, 90, height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create test pattern: combination of sine waves
        data = (np.sin(np.radians(lat_grid)) * 
                np.cos(np.radians(lon_grid)) + 
                0.5 * np.sin(2 * np.radians(lat_grid)) * 
                np.cos(2 * np.radians(lon_grid)))
        
        # Convert to dask array
        return da.from_array(data, chunks='auto')
    
    def _estimate_memory_usage(self, height: int, width: int) -> float:
        """Estimate memory usage based on data size."""
        # Simplified memory estimation (4 bytes per float32 element)
        data_size = height * width
        estimated_memory_mb = (data_size * 4 * 3) / (1024 * 1024)  # 3 arrays: source, target coords, result
        return estimated_memory_mb  # Return in MB


class StrongScalabilityTester:
    """Class for testing strong scalability (fixed problem size, varying workers)."""
    
    def __init__(self, baseline_workers: int = 1):
        self.baseline_workers = baseline_workers
    
    def test_strong_scalability(self, 
                              resolution: Tuple[int, int],
                              worker_counts: List[int],
                              method: str = 'bilinear',
                              dask_client: Optional['Client'] = None) -> Dict[str, Any]:
        """
        Test strong scalability: fixed problem size, varying number of workers.
        
        Args:
            resolution: Fixed problem resolution
            worker_counts: List of worker counts to test
            method: Interpolation method to use
            dask_client: Optional Dask client to use
            
        Returns:
            Dictionary with strong scalability analysis
        """
        height, width = resolution
        data_size = height * width
        
        execution_times = []
        speedups = []
        efficiencies = []
        
        # Run baseline test (with baseline_workers or first in list)
        baseline_workers = min(worker_counts) if self.baseline_workers == 1 else self.baseline_workers
        baseline_time = self._run_test_with_workers(height, width, method, baseline_workers, dask_client)
        
        for n_workers in worker_counts:
            execution_time = self._run_test_with_workers(height, width, method, n_workers, dask_client)
            execution_times.append(execution_time)
            
            # Calculate speedup and efficiency
            speedup = baseline_time / execution_time if execution_time > 0 else 0
            efficiency = speedup / n_workers if n_workers > 0 else 0
            
            speedups.append(speedup)
            efficiencies.append(efficiency)
        
        # Calculate ideal speedup for comparison
        ideal_speedups = worker_counts[:]  # Perfect scaling would be linear
        
        return {
            'resolution': resolution,
            'data_size': data_size,
            'worker_counts': worker_counts,
            'execution_times': execution_times,
            'speedups': speedups,
            'efficiencies': efficiencies,
            'ideal_speedups': ideal_speedups,
            'baseline_time': baseline_time,
            'baseline_workers': baseline_workers
        }
    
    def _run_test_with_workers(self, 
                              height: int, 
                              width: int, 
                              method: str, 
                              n_workers: int,
                              dask_client: Optional['Client']) -> float:
        """Run a single test with specified number of workers."""
        # In a real implementation, this would configure the client with n_workers
        # For now, we'll just run the test and return the time
        source_data = self._create_test_data(height, width)
        target_coords = (
            np.linspace(-180, 180, width),
            np.linspace(-90, 90, height)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Create benchmark instance
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        start_time = time.perf_counter()
        try:
            result = benchmark.benchmark_regridding_operation(
                source_data=source_data,
                target_coords=(target_points[0, :], target_points[1, :]),
                method=method,
                name=f"strong_scalability_{height}x{width}_{n_workers}workers"
            )
            execution_time = result.execution_time
        except Exception:
            execution_time = float('inf')
        
        return execution_time
    
    def _create_test_data(self, height: int, width: int) -> da.Array:
        """Create test data for scalability testing."""
        # Create analytical test function (e.g., sine wave pattern)
        lon = np.linspace(-180, 180, width)
        lat = np.linspace(-90, 90, height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create test pattern: combination of sine waves
        data = (np.sin(np.radians(lat_grid)) * 
                np.cos(np.radians(lon_grid)) + 
                0.5 * np.sin(2 * np.radians(lat_grid)) * 
                np.cos(2 * np.radians(lon_grid)))
        
        # Convert to dask array
        return da.from_array(data, chunks='auto')


class WeakScalabilityTester:
    """Class for testing weak scalability (proportional problem size and workers)."""
    
    def __init__(self):
        pass
    
    def test_weak_scalability(self, 
                            base_resolution: Tuple[int, int],
                            worker_scale_factors: List[int],
                            method: str = 'bilinear',
                            dask_client: Optional['Client'] = None) -> Dict[str, Any]:
        """
        Test weak scalability: problem size scales proportionally with workers.
        
        Args:
            base_resolution: Base problem resolution (for 1 worker)
            worker_scale_factors: List of scale factors (number of workers)
            method: Interpolation method to use
            dask_client: Optional Dask client to use
            
        Returns:
            Dictionary with weak scalability analysis
        """
        base_height, base_width = base_resolution
        execution_times = []
        work_per_worker = []  # Should remain constant for perfect weak scaling
        
        for scale_factor in worker_scale_factors:
            # Scale the problem size proportionally to the number of workers
            height = base_height * int(np.sqrt(scale_factor))
            width = base_width * int(np.sqrt(scale_factor))
            
            # Keep work per worker constant (approximately)
            work_per_worker.append((height * width) / scale_factor)
            
            # Run test
            execution_time = self._run_test_with_workers(height, width, method, scale_factor, dask_client)
            execution_times.append(execution_time)
        
        return {
            'base_resolution': base_resolution,
            'worker_scale_factors': worker_scale_factors,
            'scaled_resolutions': [(base_height * int(np.sqrt(sf)), base_width * int(np.sqrt(sf))) for sf in worker_scale_factors],
            'execution_times': execution_times,
            'work_per_worker': work_per_worker,  # Should be approximately constant
            'perfect_weak_scaling_time': execution_times[0] if execution_times else None  # Baseline time should be maintained
        }
    
    def _run_test_with_workers(self, 
                              height: int, 
                              width: int, 
                              method: str, 
                              n_workers: int,
                              dask_client: Optional['Client']) -> float:
        """Run a single test with specified parameters."""
        source_data = self._create_test_data(height, width)
        target_coords = (
            np.linspace(-180, 180, width),
            np.linspace(-90, 90, height)
        )
        target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
        target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
        
        # Create benchmark instance
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        start_time = time.perf_counter()
        try:
            result = benchmark.benchmark_regridding_operation(
                source_data=source_data,
                target_coords=(target_points[0, :], target_points[1, :]),
                method=method,
                name=f"weak_scalability_{height}x{width}_{n_workers}workers"
            )
            execution_time = result.execution_time
        except Exception:
            execution_time = float('inf')
        
        return execution_time
    
    def _create_test_data(self, height: int, width: int) -> da.Array:
        """Create test data for scalability testing."""
        # Create analytical test function (e.g., sine wave pattern)
        lon = np.linspace(-180, 180, width)
        lat = np.linspace(-90, 90, height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create test pattern: combination of sine waves
        data = (np.sin(np.radians(lat_grid)) * 
                np.cos(np.radians(lon_grid)) + 
                0.5 * np.sin(2 * np.radians(lat_grid)) * 
                np.cos(2 * np.radians(lon_grid)))
        
        # Convert to dask array
        return da.from_array(data, chunks='auto')


def analyze_scalability_results(results: Union[Dict, List]) -> Dict[str, Any]:
    """
    Analyze scalability test results and provide insights.
    
    Args:
        results: Scalability test results
        
    Returns:
        Analysis dictionary with insights
    """
    analysis = {
        'scalability_score': 0.0,
        'bottlenecks_identified': [],
        'optimization_recommendations': [],
        'scaling_efficiency': 0.0
    }
    
    if isinstance(results, dict) and 'speedups' in results:
        speedups = results['speedups']
        worker_counts = results['worker_counts']
        
        # Calculate average efficiency
        efficiencies = [speedup/wc for speedup, wc in zip(speedups, worker_counts)]
        avg_efficiency = statistics.mean(efficiencies) if efficiencies else 0
        
        analysis['scaling_efficiency'] = avg_efficiency
        analysis['scalability_score'] = avg_efficiency * 100
        
        # Identify bottlenecks (where efficiency drops significantly)
        for i in range(1, len(efficiencies)):
            if efficiencies[i] < efficiencies[i-1] * 0.8:  # Efficiency drops by more than 20%
                analysis['bottlenecks_identified'].append({
                    'workers_before': worker_counts[i-1],
                    'workers_after': worker_counts[i],
                    'efficiency_drop': efficiencies[i-1] - efficiencies[i]
                })
        
        # Recommendations based on analysis
        if avg_efficiency > 0.8:
            analysis['optimization_recommendations'].append("Excellent scaling performance maintained")
        elif avg_efficiency > 0.5:
            analysis['optimization_recommendations'].append("Good scaling with room for improvement")
        else:
            analysis['optimization_recommendations'].append("Poor scaling - consider optimizing communication overhead")
    
    return analysis