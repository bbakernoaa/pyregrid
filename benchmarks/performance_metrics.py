"""
Performance metrics collection utilities for PyRegrid benchmarking.

This module provides utilities for collecting and analyzing performance metrics
during benchmarking of regridding operations.
"""
import time
import psutil
import numpy as np
import dask
import dask.array as da
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from contextlib import contextmanager
import statistics
import gc
from dataclasses import dataclass, asdict
import json

try:
    from dask.distributed import Client, as_completed
    HAS_DASK_DISTRIBUTED = True
except ImportError:
    HAS_DASK_DISTRIBUTED = False

from .benchmark_base import BenchmarkResult, benchmark_runner


@dataclass
class PerformanceMetrics:
    """Data class to store detailed performance metrics."""
    execution_time: float
    memory_usage_peak: float # in MB
    memory_usage_avg: float   # in MB
    cpu_percent: float
    throughput: float # operations per second
    io_read_bytes: float
    io_write_bytes: float
    cache_hits: Optional[int] = None
    cache_misses: Optional[int] = None
    disk_reads: Optional[int] = None
    disk_writes: Optional[int] = None
    network_bytes_sent: Optional[int] = None
    network_bytes_recv: Optional[int] = None
    dask_tasks_completed: Optional[int] = None
    dask_memory_spill: Optional[float] = None  # spilled to disk in MB
    dask_workers: Optional[int] = None
    dask_threads_per_worker: Optional[int] = None


class PerformanceCollector:
    """Class for collecting detailed performance metrics."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self._process = psutil.Process()
    
    @contextmanager
    def collect_performance(self, name: str = "operation", **metadata):
        """Context manager to collect detailed performance metrics."""
        # Initial metrics
        try:
            initial_io = self._process.io_counters()
            initial_read_bytes = initial_io.read_bytes
            initial_write_bytes = initial_io.write_bytes
        except:
            initial_read_bytes = 0
            initial_write_bytes = 0
        initial_cpu = self._process.cpu_percent()
        initial_memory = self._process.memory_info().rss / 1024 / 1024
        initial_time = time.perf_counter()
        
        # For I/O stats
        # Already handled above
        
        # Collect initial Dask metrics if available
        initial_dask_metrics = self._get_dask_metrics()
        
        try:
            yield
        finally:
            # Final metrics
            final_time = time.perf_counter()
            final_memory = self._process.memory_info().rss / 1024 / 1024
            final_cpu = self._process.cpu_percent()
            
            try:
                final_io = self._process.io_counters()
                final_read_bytes = final_io.read_bytes
                final_write_bytes = final_io.write_bytes
            except:
                final_read_bytes = 0
                final_write_bytes = 0
            # Already handled above
            
            final_dask_metrics = self._get_dask_metrics()
            
            # Calculate metrics
            execution_time = final_time - initial_time
            memory_delta = final_memory - initial_memory
            # Ensure memory usage is non-negative
            memory_usage = max(0, memory_delta)
            avg_memory = (initial_memory + final_memory) / 2
            io_read_delta = final_read_bytes - initial_read_bytes
            io_write_delta = final_write_bytes - initial_write_bytes
            
            # Calculate throughput (simplified)
            throughput = self._calculate_throughput(execution_time)
            
            # Calculate Dask-specific metrics
            dask_tasks_completed = None
            dask_memory_spill = None
            dask_workers = None
            dask_threads_per_worker = None
            if initial_dask_metrics and final_dask_metrics:
                dask_tasks_completed = final_dask_metrics.get('tasks_completed', 0) - initial_dask_metrics.get('tasks_completed', 0)
                dask_memory_spill = (final_dask_metrics.get('memory_spill', 0) - initial_dask_metrics.get('memory_spill', 0)) / (1024 * 1024)  # Convert to MB
                dask_workers = final_dask_metrics.get('workers', 0)
                dask_threads_per_worker = final_dask_metrics.get('threads_per_worker', 0)
            
            metrics = PerformanceMetrics(
                execution_time=execution_time,
                memory_usage_peak=final_memory,
                memory_usage_avg=avg_memory,
                cpu_percent=final_cpu,
                throughput=throughput,
                io_read_bytes=io_read_delta,
                io_write_bytes=io_write_delta,
                dask_tasks_completed=dask_tasks_completed,
                dask_memory_spill=dask_memory_spill,
                dask_workers=dask_workers,
                dask_threads_per_worker=dask_threads_per_worker
            )
            
            self.metrics_history.append(metrics)
            
            # Store in benchmark runner as well
            benchmark_result = BenchmarkResult(
                name=name,
                execution_time=execution_time,
                memory_usage=memory_usage,
                cpu_percent=final_cpu,
                throughput=throughput,
                metadata=metadata
            )
            benchmark_runner.results.append(benchmark_result)
    
    def _get_dask_metrics(self) -> Optional[Dict[str, Any]]:
        """Get current Dask metrics if available."""
        try:
            # Try to get Dask client info if available
            if HAS_DASK_DISTRIBUTED:
                try:
                    from dask.distributed import get_client
                    client = get_client()
                    if client:
                        scheduler_info = client.scheduler_info()
                        worker_count = len(scheduler_info.get('workers', {}))
                        if scheduler_info.get('workers', {}):
                            threads_per_worker = list(scheduler_info['workers'].values())[0]['nthreads']
                        else:
                            threads_per_worker = 1
                        return {
                            'tasks_completed': 0,  # Placeholder - actual implementation would track tasks
                            'memory_spill': 0,     # Placeholder - actual implementation would track spill
                            'workers': worker_count,
                            'threads_per_worker': threads_per_worker
                        }
                except:
                    pass  # Client not available or other error
            # If no distributed client, return basic info
            return {
                'tasks_completed': 0,
                'memory_spill': 0,
                'workers': 0,
                'threads_per_worker': 1
            }
        except:
            return None
    
    def _calculate_throughput(self, execution_time: float) -> float:
        """Calculate throughput based on execution time."""
        if execution_time <= 0:
            return 0.0
        return 1.0 / execution_time  # Simplified throughput calculation
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Get aggregated performance metrics from history."""
        if not self.metrics_history:
            return {}
        
        metrics = {
            'execution_time_mean': statistics.mean([m.execution_time for m in self.metrics_history]),
            'execution_time_median': statistics.median([m.execution_time for m in self.metrics_history]),
            'execution_time_stdev': statistics.stdev([m.execution_time for m in self.metrics_history]) if len(self.metrics_history) > 1 else 0,
            'execution_time_min': min([m.execution_time for m in self.metrics_history]),
            'execution_time_max': max([m.execution_time for m in self.metrics_history]),
            'memory_usage_peak_mean': statistics.mean([m.memory_usage_peak for m in self.metrics_history]),
            'memory_usage_avg_mean': statistics.mean([m.memory_usage_avg for m in self.metrics_history]),
            'cpu_percent_mean': statistics.mean([m.cpu_percent for m in self.metrics_history]),
            'throughput_mean': statistics.mean([m.throughput for m in self.metrics_history]),
            'io_read_bytes_mean': statistics.mean([m.io_read_bytes for m in self.metrics_history]),
            'io_write_bytes_mean': statistics.mean([m.io_write_bytes for m in self.metrics_history]),
        }
        
        # Add Dask-specific metrics if available
        dask_completed = [m.dask_tasks_completed for m in self.metrics_history if m.dask_tasks_completed is not None]
        if dask_completed:
            metrics['dask_tasks_completed_mean'] = statistics.mean(dask_completed)
        
        dask_spill = [m.dask_memory_spill for m in self.metrics_history if m.dask_memory_spill is not None]
        if dask_spill:
            metrics['dask_memory_spill_mean'] = statistics.mean(dask_spill)
        
        dask_workers = [m.dask_workers for m in self.metrics_history if m.dask_workers is not None]
        if dask_workers:
            metrics['dask_workers_mean'] = statistics.mean(dask_workers)
        
        dask_threads = [m.dask_threads_per_worker for m in self.metrics_history if m.dask_threads_per_worker is not None]
        if dask_threads:
            metrics['dask_threads_per_worker_mean'] = statistics.mean(dask_threads)
        
        return metrics
    
    def reset(self):
        """Reset the metrics history."""
        self.metrics_history = []


class HighResolutionBenchmark:
    """Class for running high-resolution regridding benchmarks."""
    
    def __init__(self, use_dask: bool = True, dask_client: Optional['Client'] = None):
        self.use_dask = use_dask
        self.performance_collector = PerformanceCollector()
        self.client = dask_client
    
    def benchmark_regridding_operation(self, 
                                    source_data: Union[np.ndarray, da.Array],
                                    target_coords: Tuple[np.ndarray, np.ndarray],
                                    method: str = 'bilinear',
                                    name: str = 'regridding_benchmark',
                                    **kwargs) -> BenchmarkResult:
        """
        Benchmark a regridding operation with performance metrics.
        
        Args:
            source_data: Source data array (numpy or dask)
            target_coords: Target coordinates (lon, lat)
            method: Regridding method to use
            name: Name for the benchmark
            **kwargs: Additional parameters for regridding
            
        Returns:
            BenchmarkResult with performance metrics
        """
        # Import here to avoid circular dependencies
        from pyregrid.algorithms.interpolators import BilinearInterpolator, NearestInterpolator
        
        # Select interpolator based on method
        interpolator_map = {
            'bilinear': BilinearInterpolator,
            'nearest': NearestInterpolator,
        }
        
        if method not in interpolator_map:
            raise ValueError(f"Unsupported method: {method}")
        
        interpolator = interpolator_map[method]()
        
        def run_regridding():
            # Perform the regridding operation
            result = interpolator.interpolate(source_data, target_coords)
            
            # If using dask, compute the result to measure actual execution time
            if isinstance(result, da.Array):
                if self.client:
                    # Use the provided client for computation
                    result = result.compute(scheduler=self.client)
                else:
                    # Use default scheduler
                    result = result.compute()
            
            return result
        
        # Run the benchmark with performance collection
        with self.performance_collector.collect_performance(name, method=method, **kwargs):
            result = run_regridding()
        
        # Return the last benchmark result
        return benchmark_runner.results[-1]
    
    def benchmark_multiple_resolutions(self, 
                                     resolutions: List[Tuple[int, int]],
                                     method: str = 'bilinear',
                                     iterations: int = 3) -> List[BenchmarkResult]:
        """
        Benchmark regridding at multiple resolutions.
        
        Args:
            resolutions: List of (height, width) tuples for different resolutions
            method: Regridding method to use
            iterations: Number of iterations for each resolution
            
        Returns:
            List of benchmark results for each resolution
        """
        results = []
        
        for height, width in resolutions:
            for i in range(iterations):
                # Create test data at the specified resolution
                source_data = self._create_test_data(height, width)
                
                # Create target coordinates in the correct format for map_coordinates
                # map_coordinates expects coordinates as (y_coords, x_coords) where:
                # y_coords are the row indices (latitude) and x_coords are the column indices (longitude)
                target_lon = np.linspace(-180, 180, width)
                target_lat = np.linspace(-90, 90, height)
                
                # Create coordinate arrays for map_coordinates
                # For a 2D array, map_coordinates expects (row_coords, col_coords)
                # where row_coords corresponds to the first dimension (height/latitude)
                # and col_coords corresponds to the second dimension (width/longitude)
                target_y, target_x = np.meshgrid(target_lat, target_lon, indexing='ij')
                
                # Flatten and create coordinate arrays
                y_coords = target_y.ravel()
                x_coords = target_x.ravel()
                
                # Convert world coordinates to array indices
                # Assuming the source data covers -180 to 180 in longitude and -90 to 90 in latitude
                x_indices = ((x_coords + 180) / 360 * (width - 1)).astype(int)
                y_indices = ((y_coords + 90) / 180 * (height - 1)).astype(int)
                
                benchmark_name = f"resolution_{height}x{width}_iteration_{i}"
                
                result = self.benchmark_regridding_operation(
                    source_data=source_data,
                    target_coords=(y_indices, x_indices),  # Correct format for interpolator
                    method=method,
                    name=benchmark_name,
                    resolution=(height, width)
                )
                
                results.append(result)
        
        return results
    
    def benchmark_dask_vs_numpy(self, 
                              resolution: Tuple[int, int],
                              method: str = 'bilinear',
                              iterations: int = 3) -> Dict[str, List[BenchmarkResult]]:
        """
        Benchmark comparison between Dask and NumPy implementations.
        
        Args:
            resolution: Resolution tuple (height, width)
            method: Regridding method to use
            iterations: Number of iterations for each approach
            
        Returns:
            Dictionary with results for each approach
        """
        results = {
            'numpy': [],
            'dask': []
        }
        
        height, width = resolution
        
        # Create base data
        base_data = self._create_test_data(height, width, use_dask=False)
        
        # Create target coordinates in the correct format for map_coordinates
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
        
        # Benchmark NumPy approach
        for i in range(iterations):
            numpy_data = base_data  # Use the same data
            result = self.benchmark_regridding_operation(
                source_data=numpy_data,
                target_coords=(y_indices, x_indices),
                method=method,
                name=f"numpy_{resolution[0]}x{resolution[1]}_iteration_{i}",
                approach='numpy',
                resolution=resolution
            )
            results['numpy'].append(result)
        
        # Benchmark Dask approach
        for i in range(iterations):
            chunk_size_h = max(1, height//4)
            chunk_size_w = max(1, width//4)
            dask_data = da.from_array(base_data, chunks='auto')
            result = self.benchmark_regridding_operation(
                source_data=dask_data,
                target_coords=(y_indices, x_indices),
                method=method,
                name=f"dask_{resolution[0]}x{resolution[1]}_iteration_{i}",
                approach='dask',
                resolution=resolution
            )
            results['dask'].append(result)
        
        return results
    
    def _create_test_data(self, height: int, width: int, use_dask: Optional[bool] = None) -> Union[np.ndarray, da.Array]:
        """Create test data at specified resolution."""
        # Create analytical test function (e.g., sine wave pattern)
        lon = np.linspace(-180, 180, width)
        lat = np.linspace(-90, 90, height)  # Fixed the missing value
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create test pattern: combination of sine waves
        data = (np.sin(np.radians(lat_grid)) * 
                np.cos(np.radians(lon_grid)) + 
                0.5 * np.sin(2 * np.radians(lat_grid)) * 
                np.cos(2 * np.radians(lon_grid)))
        
        # Determine whether to use dask
        use_dask_final = self.use_dask if use_dask is None else use_dask
        
        # Convert to dask array if requested
        if use_dask_final:
            chunk_size_h = max(1, height//4)
            chunk_size_w = max(1, width//4)
            return da.from_array(data, chunks='auto')
        else:
            return data


def create_performance_report(results: List[BenchmarkResult], 
                            output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a performance report from benchmark results.
    
    Args:
        results: List of benchmark results
        output_file: Optional file path to save the report
        
    Returns:
        Dictionary containing the performance report
    """
    if not results:
        return {"error": "No results to report"}
    
    # Group results by name
    grouped_results = {}
    for result in results:
        if result.name not in grouped_results:
            grouped_results[result.name] = []
        grouped_results[result.name].append(result)
    
    report = {
        "summary": {
            "total_benchmarks": len(results),
            "unique_names": list(grouped_results.keys()),
            "total_execution_time": sum(r.execution_time for r in results),
        },
        "detailed_results": {},
        "aggregated_metrics": {}
    }
    
    # Calculate metrics for each group
    for name, group_results in grouped_results.items():
        execution_times = [r.execution_time for r in group_results]
        memory_usages = [r.memory_usage for r in group_results]
        cpu_percents = [r.cpu_percent for r in group_results]
        
        report["detailed_results"][name] = {
            "count": len(group_results),
            "execution_time_mean": statistics.mean(execution_times),
            "execution_time_stdev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            "execution_time_min": min(execution_times),
            "execution_time_max": max(execution_times),
            "memory_usage_mean": statistics.mean(memory_usages),
            "cpu_percent_mean": statistics.mean(cpu_percents),
        }
    
    # Calculate overall aggregated metrics
    all_execution_times = [r.execution_time for r in results]
    all_memory_usages = [r.memory_usage for r in results]
    all_cpu_percents = [r.cpu_percent for r in results]
    
    report["aggregated_metrics"] = {
        "overall_execution_time_mean": statistics.mean(all_execution_times),
        "overall_execution_time_stdev": statistics.stdev(all_execution_times) if len(all_execution_times) > 1 else 0,
        "overall_execution_time_min": min(all_execution_times),
        "overall_execution_time_max": max(all_execution_times),
        "overall_memory_usage_mean": statistics.mean(all_memory_usages),
        "overall_cpu_percent_mean": statistics.mean(all_cpu_percents),
    }
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report


class DistributedBenchmarkRunner:
    """Class for running benchmarks in a distributed Dask environment."""
    
    def __init__(self, client: Optional['Client'] = None, 
                 scheduler_address: Optional[str] = None):
        """
        Initialize the distributed benchmark runner.
        
        Args:
            client: Optional Dask client to use
            scheduler_address: Optional address of Dask scheduler
        """
        if client:
            self.client = client
        elif scheduler_address:
            if HAS_DASK_DISTRIBUTED:
                from dask.distributed import Client
                self.client = Client(scheduler_address)
            else:
                raise RuntimeError("Dask distributed is required for distributed benchmarks")
        elif HAS_DASK_DISTRIBUTED:
            from dask.distributed import Client
            self.client = Client()  # Create local cluster
        else:
            raise RuntimeError("Dask distributed is required for distributed benchmarks")
    
    def run_benchmark_on_workers(self, 
                                func: Callable, 
                                *args,
                                n_workers: int = 4,
                                **kwargs) -> List[BenchmarkResult]:
        """
        Run benchmark function on multiple workers in parallel.
        
        Args:
            func: Function to benchmark
            *args: Arguments to pass to the function
            n_workers: Number of workers to use
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            List of benchmark results from each worker
        """
        futures = []
        
        # Create tasks for each worker
        for i in range(n_workers):
            future = self.client.submit(
                func, 
                *args, 
                worker_client=self.client,  # Pass client to worker
                **kwargs
            )
            futures.append(future)
        
        # Collect results
        results = []
        from dask.distributed import as_completed
        for future in as_completed(futures):
            result = future.result()
            if isinstance(result, BenchmarkResult):
                results.append(result)
        
        return results
    
    def benchmark_scalability(self, 
                            resolutions: List[Tuple[int, int]],
                            n_workers_list: List[int] = [1, 2, 4, 8]) -> Dict[int, List[BenchmarkResult]]:
        """
        Benchmark scalability across different numbers of workers.
        
        Args:
            resolutions: List of resolutions to test
            n_workers_list: List of worker counts to test
            
        Returns:
            Dictionary mapping worker count to benchmark results
        """
        results = {}
        
        for n_workers in n_workers_list:
            worker_results = []
            
            for height, width in resolutions:
                # Create benchmark task
                benchmark_task = HighResolutionBenchmark(
                    use_dask=True, 
                    dask_client=self.client
                )
                
                # Create test data
                source_data = benchmark_task._create_test_data(height, width)
                
                # Create target coordinates in the correct format for map_coordinates
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
                
                # Run benchmark
                result = benchmark_task.benchmark_regridding_operation(
                    source_data=source_data,
                    target_coords=(y_indices, x_indices),
                    name=f"scalability_{n_workers}workers_{height}x{width}",
                    n_workers=n_workers
                )
                
                worker_results.append(result)
            
            results[n_workers] = worker_results
        
        return results