# High-Resolution Benchmarking

This guide explains how to use PyRegrid's comprehensive benchmarking system to evaluate performance, accuracy, and scalability of regridding operations, particularly for high-resolution scenarios like 3km global grids.

## Overview

The benchmarking system provides tools to:

- **Performance Analysis**: Measure execution time, memory usage, CPU utilization, and throughput
- **Accuracy Validation**: Validate interpolation accuracy against analytical solutions
- **Scalability Testing**: Test performance scaling across different data sizes and worker counts
- **High-Resolution Optimization**: Optimize for large grid scenarios with memory-efficient strategies

## System Components

### Performance Metrics

The [`HighResolutionBenchmark`](benchmarks/performance_metrics.py:227) class provides comprehensive performance measurement:

```python
from benchmarks import HighResolutionBenchmark

# Create benchmark instance
benchmark = HighResolutionBenchmark(use_dask=True)

# Run a single regridding operation
result = benchmark.benchmark_regridding_operation(
    source_data=your_data,
    target_coords=(target_lons, target_lats),
    method='bilinear',
    name='my_benchmark'
)

print(f"Execution time: {result.execution_time:.4f}s")
print(f"Memory usage: {result.memory_usage:.2f}MB")
print(f"CPU usage: {result.cpu_percent:.1f}%")
```

### Accuracy Validation

The [`AccuracyBenchmark`](benchmarks/accuracy_validation.py:193) class validates interpolation accuracy:

```python
from benchmarks import AccuracyBenchmark

accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)

# Test accuracy against analytical solution
result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
    source_resolution=(100, 200),
    target_resolution=(100, 200),
    method='bilinear'
)

print(f"RMSE: {metrics.rmse:.6f}")
print(f"MAE: {metrics.mae:.6f}")
print(f"Correlation: {metrics.correlation:.4f}")
```

### Scalability Testing

The [`ScalabilityBenchmark`](benchmarks/scalability_testing.py:42) class tests performance scaling:

```python
from benchmarks import ScalabilityBenchmark

scalability_benchmark = ScalabilityBenchmark()

# Test strong scalability (fixed problem size, varying workers)
metrics_list = scalability_benchmark.test_worker_scalability(
    resolution=(200, 400),
    max_workers=8,
    method='bilinear'
)

for metrics in metrics_list:
    print(f"Workers {metrics.workers_used}: Speedup={metrics.speedup:.2f}x, "
          f"Efficiency={metrics.efficiency:.2f}")
```

## Setup and Configuration

### Prerequisites

Install required dependencies:

```bash
pip install pyregrid[dask-benchmarking]
```

### Basic Configuration

```python
import dask
from dask.distributed import Client
from benchmarks import HighResolutionBenchmark, AccuracyBenchmark

# Configure Dask for optimal performance
dask.config.set(scheduler='threads', num_workers=4)

# Create Dask client for distributed processing
client = Client(n_workers=4, threads_per_worker=2)

# Initialize benchmarks
performance_benchmark = HighResolutionBenchmark(use_dask=True, dask_client=client)
accuracy_benchmark = AccuracyBenchmark(threshold=1e-6)
```

### High-Resolution Configuration

For 3km global grid scenarios (approximately 36M points):

```python
# Use proxy data for testing (1/10th scale for 3km grid)
proxy_height, proxy_width = 600, 1200

# Configure memory-efficient chunking
source_data = da.from_array(large_array, chunks=(100, 200))

# Enable lazy evaluation
benchmark = HighResolutionBenchmark(use_dask=True, dask_client=client)
```

## Usage Examples

### Performance Benchmarking

#### Single Operation Benchmarking

```python
import numpy as np
import dask.array as da
from benchmarks import HighResolutionBenchmark

# Create test data
height, width = 500, 1000
source_data = da.random.random((height, width), chunks='auto')

target_coords = (
    np.linspace(-180, 180, width//2),
    np.linspace(-90, 90, height//2)
)

# Run benchmark
benchmark = HighResolutionBenchmark(use_dask=True)
result = benchmark.benchmark_regridding_operation(
    source_data=source_data,
    target_coords=target_coords,
    method='bilinear',
    name='performance_test'
)

print(f"Performance metrics:")
print(f"  Time: {result.execution_time:.4f}s")
print(f"  Memory: {result.memory_usage:.2f}MB")
print(f"  Throughput: {height*width/result.execution_time/1e6:.2f}M elements/s")
```

#### Multi-Resolution Analysis

```python
# Test performance across different resolutions
resolutions = [(100, 200), (200, 400), (500, 1000), (1000, 2000)]
results = []

for height, width in resolutions:
    source_data = benchmark._create_test_data(height, width, use_dask=True)
    target_coords = (
        np.linspace(-180, 180, width//2),
        np.linspace(-90, 90, height//2)
    )
    
    result = benchmark.benchmark_regridding_operation(
        source_data=source_data,
        target_coords=target_coords,
        method='bilinear',
        name=f'resolution_{height}x{width}'
    )
    results.append(result)

# Analyze scaling patterns
for i, result in enumerate(results):
    data_size = resolutions[i][0] * resolutions[i][1]
    print(f"Resolution {resolutions[i]}: {result.execution_time:.4f}s, "
          f"{data_size/result.execution_time/1e6:.2f}M elements/s")
```

### Accuracy Validation

#### Basic Accuracy Testing

```python
from benchmarks import AccuracyBenchmark

accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)

# Test bilinear interpolation accuracy
result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
    source_resolution=(200, 400),
    target_resolution=(200, 400),
    method='bilinear',
    field_type='sine_wave'
)

print(f"Accuracy metrics:")
print(f"  RMSE: {metrics.rmse:.6f}")
print(f"  MAE: {metrics.mae:.6f}")
print(f"  Max error: {metrics.max_error:.6f}")
print(f"  Correlation: {metrics.correlation:.4f}")
print(f"  Pass threshold: {metrics.rmse <= metrics.accuracy_threshold}")
```

#### Convergence Testing

```python
# Test accuracy convergence across resolutions
resolutions = [(50, 100), (100, 200), (200, 400), (400, 800)]
convergence_results = accuracy_benchmark.run_accuracy_convergence_test(
    resolutions=resolutions,
    method='bilinear'
)

print("Convergence analysis:")
for i, (result, metrics) in enumerate(convergence_results):
    print(f"Resolution {resolutions[i]}: RMSE = {metrics.rmse:.6f}")
```

### Scalability Testing

#### Strong Scalability

```python
from benchmarks import StrongScalabilityTester

tester = StrongScalabilityTester(baseline_workers=1)

# Test strong scalability: fixed problem size, varying workers
results = tester.test_strong_scalability(
    resolution=(400, 800),
    worker_counts=[1, 2, 4, 8],
    method='bilinear'
)

print("Strong scalability analysis:")
print(f"{'Workers':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
for i, n_workers in enumerate(results['worker_counts']):
    time_s = results['execution_times'][i]
    speedup = results['speedups'][i]
    efficiency = results['efficiencies'][i]
    print(f"{n_workers:<8} {time_s:<12.4f} {speedup:<10.2f} {efficiency:<12.2f}")
```

#### Weak Scalability

```python
from benchmarks import WeakScalabilityTester

tester = WeakScalabilityTester()

# Test weak scalability: problem size scales with workers
results = tester.test_weak_scalability(
    base_resolution=(200, 400),
    worker_scale_factors=[1, 2, 4, 8],
    method='bilinear'
)

print("Weak scalability analysis:")
print(f"{'Workers':<8} {'Resolution':<15} {'Time (s)':<12} {'Work/Worker':<15}")
for i, scale_factor in enumerate(results['worker_scale_factors']):
    res = results['scaled_resolutions'][i]
    time_s = results['execution_times'][i]
    work_per_worker = results['work_per_worker'][i]
    print(f"{scale_factor:<8} {str(res):<15} {time_s:<12.4f} {work_per_worker:<15.0f}")
```

## Integration with Pytest

### Running Benchmarks with Pytest

The benchmarking system integrates seamlessly with pytest:

```bash
# Run all benchmark tests
python -m pytest benchmarks/ -v --benchmark

# Run with custom output directory
python -m pytest benchmarks/ -v --benchmark --benchmark-output-dir ./results

# Run large-scale benchmarks only
python -m pytest benchmarks/ -v --benchmark --benchmark-large
```

### Custom Benchmark Tests

```python
import pytest
from benchmarks import HighResolutionBenchmark

@pytest.mark.benchmark
class TestMyBenchmarks:
    
    def test_performance_at_scale(self, dask_client):
        """Test performance at high resolution."""
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        
        # Create high-resolution test data
        source_data = benchmark._create_test_data(1000, 2000, use_dask=True)
        target_coords = (
            np.linspace(-180, 180, 1000),
            np.linspace(-90, 90, 2000)
        )
        
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=target_coords,
            method='bilinear',
            name='high_res_test'
        )
        
        # Assert performance requirements
        assert result.execution_time < 10.0  # Should complete in under 10 seconds
        assert result.memory_usage < 1000    # Should use less than 1GB memory
```

## Integration with Dask

### Distributed Benchmarking

```python
from dask.distributed import Client
from benchmarks import DistributedBenchmarkRunner

# Create distributed client
client = Client('tcp://scheduler:8786')

# Create distributed benchmark runner
dist_runner = DistributedBenchmarkRunner(client=client)

# Define benchmark function
def distributed_benchmark_task(source_data, target_coords, method='bilinear'):
    from benchmarks import HighResolutionBenchmark
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=client)
    return benchmark.benchmark_regridding_operation(
        source_data=source_data,
        target_coords=target_coords,
        method=method
    )

# Run benchmark across multiple workers
results = dist_runner.run_benchmark_on_workers(
    distributed_benchmark_task,
    source_data=large_dask_array,
    target_coords=target_coords,
    n_workers=4
)

# Aggregate results
all_times = [r.execution_time for r in results]
print(f"Average time: {np.mean(all_times):.4f}s")
print(f"Time std: {np.std(all_times):.4f}s")
```

### Memory Management for Large Grids

```python
import dask
from benchmarks import HighResolutionBenchmark

# Configure memory limits
dask.config.set({'array.chunk-size': '128MB'})

# Create benchmark with memory constraints
benchmark = HighResolutionBenchmark(use_dask=True)

# Use chunked data for large grids
large_data = da.random.random((3600, 7200), chunks=(100, 200))

# Run with memory monitoring
result = benchmark.benchmark_regridding_operation(
    source_data=large_data,
    target_coords=target_coords,
    method='bilinear',
    name='large_grid_memory_test'
)

print(f"Memory efficiency: {large_data.nbytes/result.memory_usage:.1f}x")
```

## Result Interpretation

### Performance Metrics

Key performance indicators to monitor:

- **Execution Time**: Total time for the regridding operation
- **Memory Usage**: Peak memory consumption during operation
- **Throughput**: Elements processed per second
- **CPU Utilization**: Average CPU usage during computation

```python
# Comprehensive performance analysis
def analyze_performance_results(results):
    """Analyze benchmark performance results."""
    if not results:
        return {}
    
    execution_times = [r.execution_time for r in results]
    memory_usages = [r.memory_usage for r in results]
    
    analysis = {
        'performance_summary': {
            'total_operations': len(results),
            'avg_execution_time': np.mean(execution_times),
            'median_execution_time': np.median(execution_times),
            'execution_time_std': np.std(execution_times),
            'avg_memory_usage': np.mean(memory_usages),
            'memory_efficiency_score': np.mean([r.throughput/r.memory_usage for r in results if r.throughput])
        }
    }
    
    return analysis
```

### Accuracy Metrics

Key accuracy indicators:

- **RMSE (Root Mean Square Error)**: Overall error magnitude
- **MAE (Mean Absolute Error)**: Average absolute error
- **Correlation**: Linear correlation with expected values
- **Bias**: Systematic error in the results

```python
def analyze_accuracy_results(results):
    """Analyze accuracy validation results."""
    passing_tests = 0
    total_tests = len(results)
    
    for result, metrics in results:
        if metrics.rmse <= metrics.accuracy_threshold:
            passing_tests += 1
    
    accuracy_summary = {
        'total_tests': total_tests,
        'passing_tests': passing_tests,
        'pass_rate': passing_tests / total_tests if total_tests > 0 else 0,
        'avg_rmse': np.mean([m.rmse for _, m in results]),
        'avg_correlation': np.mean([m.correlation for _, m in results])
    }
    
    return accuracy_summary
```

### Scalability Analysis

Key scalability indicators:

- **Speedup**: Performance improvement with additional workers
- **Efficiency**: How well performance scales relative to ideal scaling
- **Bottlenecks**: Points where scaling efficiency drops significantly

```python
def analyze_scalability_efficiency(results):
    """Analyze scalability efficiency."""
    if 'speedups' not in results:
        return {}
    
    speedups = results['speedups']
    worker_counts = results['worker_counts']
    efficiencies = [speedup/wc for speedup, wc in zip(speedups, worker_counts)]
    
    # Identify scaling bottlenecks
    bottlenecks = []
    for i in range(1, len(efficiencies)):
        if efficiencies[i] < efficiencies[i-1] * 0.8:  # >20% efficiency drop
            bottlenecks.append({
                'worker_transition': f"{worker_counts[i-1]}→{worker_counts[i]}",
                'efficiency_drop': efficiencies[i-1] - efficiencies[i]
            })
    
    analysis = {
        'scaling_efficiency': {
            'avg_efficiency': np.mean(efficiencies),
            'max_efficiency': max(efficiencies),
            'min_efficiency': min(efficiencies),
            'efficiency_std': np.std(efficiencies),
            'bottlenecks': bottlenecks,
            'optimal_worker_count': worker_counts[np.argmax(efficiencies)]
        }
    }
    
    return analysis
```

## Advanced Configuration

### Custom Benchmark Scenarios

```python
from benchmarks import HighResolutionBenchmark, AccuracyBenchmark

# Create specialized benchmark for climate data
class ClimateDataBenchmark:
    def __init__(self, dask_client=None):
        self.performance_benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
        self.accuracy_benchmark = AccuracyBenchmark(threshold=1e-5)
    
    def benchmark_climate_regridding(self, source_data, target_grid, methods=['bilinear', 'nearest']):
        """Benchmark climate data regridding with multiple methods."""
        results = {}
        
        for method in methods:
            # Performance benchmark
            perf_result = self.performance_benchmark.benchmark_regridding_operation(
                source_data=source_data,
                target_coords=(target_grid['lon'], target_grid['lat']),
                method=method,
                name=f'climate_{method}'
            )
            
            # Accuracy benchmark (if analytical solution available)
            acc_result, acc_metrics = self.accuracy_benchmark.benchmark_interpolation_accuracy(
                source_resolution=source_data.shape,
                target_resolution=(len(target_grid['lat']), len(target_grid['lon'])),
                method=method
            )
            
            results[method] = {
                'performance': perf_result,
                'accuracy': acc_metrics
            }
        
        return results
```

### Automated Benchmarking Pipeline

```python
import json
import os
from datetime import datetime

class BenchmarkingPipeline:
    def __init__(self, output_dir="./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def run_comprehensive_benchmark(self, data_scenarios, methods):
        """Run comprehensive benchmark across all scenarios and methods."""
        from benchmarks import HighResolutionBenchmark, AccuracyBenchmark
        
        benchmark = HighResolutionBenchmark(use_dask=True)
        accuracy_benchmark = AccuracyBenchmark()
        
        all_results = []
        
        for scenario_name, scenario_data in data_scenarios.items():
            for method in methods:
                print(f"Running {scenario_name} with {method}...")
                
                # Performance benchmark
                perf_result = benchmark.benchmark_regridding_operation(
                    source_data=scenario_data['source'],
                    target_coords=scenario_data['target'],
                    method=method,
                    name=f"{scenario_name}_{method}"
                )
                
                # Accuracy benchmark (if analytical solution available)
                if 'analytical_solution' in scenario_data:
                    acc_result, acc_metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
                        source_resolution=scenario_data['source'].shape,
                        target_resolution=scenario_data['target_resolution'],
                        method=method
                    )
                    all_results.append({
                        'scenario': scenario_name,
                        'method': method,
                        'performance': perf_result.__dict__,
                        'accuracy': acc_metrics.__dict__
                    })
                else:
                    all_results.append({
                        'scenario': scenario_name,
                        'method': method,
                        'performance': perf_result.__dict__
                    })
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(self.output_dir, f"comprehensive_benchmark_{timestamp}.json")
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"Results saved to: {output_file}")
        return all_results
```

## Best Practices

### Performance Optimization

1. **Optimal Chunking**: Choose chunk sizes that balance memory usage and parallelization
2. **Lazy Evaluation**: Keep operations lazy until final computation
3. **Memory Management**: Monitor memory usage and adjust chunking accordingly
4. **Scheduler Selection**: Choose between threads and processes based on your workload

```python
# Optimal chunking strategy for regridding
def optimize_chunking(data_shape, target_memory_mb=128):
    """Calculate optimal chunk sizes for regridding operations."""
    # Estimate element size (assuming float64 = 8 bytes)
    element_size_bytes = 8
    target_elements = (target_memory_mb * 1024 * 1024) // element_size_bytes
    
    # Calculate chunk dimensions
    height, width = data_shape
    chunk_area = min(target_elements, height * width)
    
    # Use square-ish chunks for better cache performance
    chunk_size = int(np.sqrt(chunk_area))
    
    return (min(chunk_size, height), min(chunk_size, width))
```

### Accuracy Validation

1. **Multiple Test Fields**: Use different analytical functions for comprehensive validation
2. **Resolution Convergence**: Test at multiple resolutions to ensure convergence
3. **Round-trip Testing**: Validate that interpolation is reversible
4. **Error Analysis**: Examine error patterns to identify systematic issues

### Scalability Testing

1. **Strong Scaling**: Test with fixed problem size and varying workers
2. **Weak Scaling**: Test with proportional problem size and worker scaling
3. **Memory Scaling**: Monitor memory usage scaling with problem size
4. **Bottleneck Identification**: Identify where scaling efficiency drops

## Troubleshooting

### Common Issues

#### Memory Errors with Large Grids

```python
# Solution: Use smaller chunks or distributed processing
if memory_error:
    # Reduce chunk size
    smaller_chunks = optimize_chunking(data_shape, target_memory_mb=64)
    data = data.rechunk(smaller_chunks)
    
    # Or use distributed processing
    from dask.distributed import Client
    client = Client(n_workers=4, threads_per_worker=1)
```

#### Poor Scaling Performance

```python
# Solution: Optimize chunking and scheduler
if poor_scaling:
    # Try different chunk sizes
    data = data.rechunk('auto')  # Let Dask choose optimal chunks
    
    # Try different scheduler
    dask.config.set(scheduler='processes')  # May help for CPU-bound tasks
    
    # Check for communication overhead
    print("Dask dashboard:", client.dashboard_link)
```

#### Accuracy Validation Failures

```python
# Solution: Check interpolation method and field characteristics
if accuracy_failure:
    # Try different interpolation methods
    methods = ['bilinear', 'nearest', 'cubic']
    
    # Test with simpler analytical fields
    field_types = ['sine_wave', 'gaussian_bump', 'polynomial']
    
    # Increase tolerance for complex fields
    accuracy_benchmark = AccuracyBenchmark(threshold=1e-3)
```

## Conclusion

The high-resolution benchmarking system provides comprehensive tools for evaluating and optimizing PyRegrid performance across different scenarios. By following the guidelines in this document, you can:

- Establish performance baselines for your specific use cases
- Validate accuracy against analytical solutions
- Optimize scaling for large datasets
- Identify and resolve performance bottlenecks
- Ensure consistent performance across different computational environments

For more detailed information about individual benchmarking components, refer to the API documentation and the source code in the `benchmarks/` directory.