"""
Base module for benchmarking utilities in PyRegrid.

This module provides foundational classes and utilities for performance
benchmarking, accuracy validation, and scalability testing.
"""
import time
import psutil
import numpy as np
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
from contextlib import contextmanager
import statistics


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    name: str
    execution_time: float
    memory_usage: float  # in MB
    cpu_percent: float
    accuracy_error: Optional[float] = None  # for accuracy validation
    throughput: Optional[float] = None  # operations per second
    scalability_factor: Optional[float] = None  # speedup compared to baseline
    metadata: Optional[Dict[str, Any]] = None


class BenchmarkRunner:
    """Base class for running benchmarks with performance metrics collection."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    @contextmanager
    def benchmark_context(self, name: str, **metadata):
        """Context manager to measure performance metrics for a code block."""
        # Record initial state
        initial_memory = self._get_memory_usage()
        initial_time = time.perf_counter()
        initial_cpu = psutil.cpu_percent(interval=None)
        
        yield
        
        # Record final state
        final_time = time.perf_counter()
        final_memory = self._get_memory_usage()
        final_cpu = psutil.cpu_percent(interval=None)
        
        # Calculate metrics
        execution_time = final_time - initial_time
        memory_delta = final_memory - initial_memory
        avg_cpu = (initial_cpu + final_cpu) / 2
        
        # Create and store result
        result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_percent=avg_cpu,
            metadata=metadata or {}
        )
        self.results.append(result)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run a single benchmark function and return results."""
        with self.benchmark_context(name):
            result = func(*args, **kwargs)
        return self.results[-1]  # Return the last result
    
    def run_multiple_iterations(self, name: str, func: Callable, iterations: int = 5, 
                               *args, **kwargs) -> List[BenchmarkResult]:
        """Run benchmark multiple times and return statistics."""
        results = []
        for i in range(iterations):
            with self.benchmark_context(f"{name}_iteration_{i}"):
                func(*args, **kwargs)
            results.append(self.results[-1])
        
        return results
    
    def get_statistics(self, results: List[BenchmarkResult]) -> Dict[str, float]:
        """Calculate statistics from multiple benchmark results."""
        if not results:
            return {}
        
        execution_times = [r.execution_time for r in results]
        memory_usages = [r.memory_usage for r in results]
        cpu_percents = [r.cpu_percent for r in results]
        
        stats = {
            'execution_time_mean': statistics.mean(execution_times),
            'execution_time_median': statistics.median(execution_times),
            'execution_time_stdev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'execution_time_min': min(execution_times),
            'execution_time_max': max(execution_times),
            'memory_usage_mean': statistics.mean(memory_usages),
            'cpu_percent_mean': statistics.mean(cpu_percents),
        }
        
        return stats


class AccuracyValidator:
    """Class for validating accuracy of regridding operations."""
    
    @staticmethod
    def calculate_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate Root Mean Square Error between actual and expected arrays."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return float('inf')  # If all values are NaN, return infinity error
        
        diff = actual[mask] - expected[mask]
        mse = np.mean(diff ** 2)
        return np.sqrt(mse)
    
    @staticmethod
    def calculate_mae(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate Mean Absolute Error between actual and expected arrays."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return float('inf')  # If all values are NaN, return infinity error
        
        diff = np.abs(actual[mask] - expected[mask])
        return float(np.mean(diff))
    
    @staticmethod
    def calculate_max_error(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate Maximum Absolute Error between actual and expected arrays."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return float('inf')  # If all values are NaN, return infinity error
        
        diff = np.abs(actual[mask] - expected[mask])
        return np.max(diff)


class ScalabilityTester:
    """Class for testing scalability across different data sizes and configurations."""
    
    def __init__(self):
        self.scalability_results: List[BenchmarkResult] = []
    
    def test_scalability(self, func: Callable, data_sizes: List[Tuple], 
                        name: str = "scalability_test") -> List[BenchmarkResult]:
        """Test scalability across different data sizes."""
        results = []
        
        for i, size_params in enumerate(data_sizes):
            test_name = f"{name}_size_{i}"
            with self._benchmark_scalability_context(test_name, size_params):
                if isinstance(size_params, tuple):
                    result = func(*size_params)
                else:
                    result = func(size_params)
            results.append(self.scalability_results[-1])
        
        return results
    
    @contextmanager
    def _benchmark_scalability_context(self, name: str, size_params: Any):
        """Context manager for scalability benchmarking."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        initial_time = time.perf_counter()
        
        yield
        
        final_time = time.perf_counter()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        execution_time = final_time - initial_time
        memory_delta = final_memory - initial_memory
        
        # Calculate throughput (operations per second)
        # This is a simplified calculation - actual throughput calculation may vary
        throughput = self._calculate_throughput(size_params, execution_time)
        
        result = BenchmarkResult(
            name=name,
            execution_time=execution_time,
            memory_usage=memory_delta,
            cpu_percent=psutil.cpu_percent(),
            throughput=throughput,
            metadata={'size_params': size_params}
        )
        self.scalability_results.append(result)
    
    def _calculate_throughput(self, size_params: Any, execution_time: float) -> Optional[float]:
        """Calculate throughput based on size parameters and execution time."""
        if execution_time == 0:
            return None
        
        # Extract total operations from size parameters (this is a simplified approach)
        # In practice, this would depend on the specific operation being tested
        try:
            if isinstance(size_params, tuple) and len(size_params) >= 2:
                # Assume first two elements are dimensions
                total_elements = np.prod(size_params[:2])
            elif isinstance(size_params, (int, float)):
                total_elements = size_params
            else:
                total_elements = 1
        except:
            total_elements = 1
        
        return float(total_elements / execution_time) if execution_time > 0 else None


# Global benchmark runner instance
benchmark_runner = BenchmarkRunner()