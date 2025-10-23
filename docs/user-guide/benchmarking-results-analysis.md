# Benchmark Results Analysis and Best Practices

This guide explains how to interpret benchmark results, analyze performance patterns, and implement best practices for PyRegrid's high-resolution benchmarking system.

## Understanding Benchmark Results

### Performance Metrics Breakdown

The benchmarking system collects comprehensive performance metrics:

```python
from benchmarks import HighResolutionBenchmark

# Run a benchmark
result = benchmark.benchmark_regridding_operation(
    source_data=source_data,
    target_coords=target_coords,
    method='bilinear'
)

# Result contains:
print(f"Execution time: {result.execution_time:.4f}s")
print(f"Memory usage: {result.memory_usage:.2f}MB")
print(f"CPU utilization: {result.cpu_percent:.1f}%")
print(f"Throughput: {result.throughput:.2f}M elements/s")
print(f"Metadata: {result.metadata}")
```

#### Key Performance Indicators

1. **Execution Time**: Total time for the regridding operation
   - **Target**: Should scale linearly with data size
   - **Warning**: Non-linear scaling may indicate bottlenecks

2. **Memory Usage**: Peak memory consumption
   - **Target**: Should scale linearly with data size
   - **Warning**: Memory leaks or inefficient data structures

3. **CPU Utilization**: Average CPU usage during computation
   - **Target**: Should be high for CPU-bound operations
   - **Warning**: Low CPU usage may indicate I/O bottlenecks

4. **Throughput**: Elements processed per second
   - **Target**: Should remain relatively constant across similar operations
   - **Warning**: Decreasing throughput may indicate scaling issues

### Accuracy Metrics Interpretation

```python
from benchmarks import AccuracyBenchmark

# Run accuracy benchmark
result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
    source_resolution=(200, 400),
    target_resolution=(200, 400),
    method='bilinear'
)

# Metrics contain:
print(f"RMSE: {metrics.rmse:.6f}")
print(f"MAE: {metrics.mae:.6f}")
print(f"Max error: {metrics.max_error:.6f}")
print(f"Correlation: {metrics.correlation:.4f}")
print(f"Bias: {metrics.bias:.6f}")
```

#### Accuracy Interpretation Guidelines

1. **RMSE (Root Mean Square Error)**:
   - **Good**: < 1e-4 for analytical functions
   - **Acceptable**: < 1e-3 for complex fields
   - **Poor**: > 1e-2 indicates significant issues

2. **MAE (Mean Absolute Error)**:
   - Should be proportional to RMSE
   - MAE typically 60-80% of RMSE for normal distributions

3. **Correlation**:
   - **Good**: > 0.99 for analytical solutions
   - **Acceptable**: > 0.95 for complex fields
   - **Poor**: < 0.9 indicates systematic errors

4. **Bias**:
   - Should be close to zero for unbiased methods
   - Significant bias indicates systematic error

### Scalability Analysis

```python
from benchmarks import StrongScalabilityTester

# Test strong scalability
tester = StrongScalabilityTester(baseline_workers=1)
results = tester.test_strong_scalability(
    resolution=(400, 800),
    worker_counts=[1, 2, 4, 8],
    method='bilinear'
)

# Analyze scaling efficiency
for i, n_workers in enumerate(results['worker_counts']):
    speedup = results['speedups'][i]
    efficiency = results['efficiencies'][i]
    print(f"Workers {n_workers}: Speedup={speedup:.2f}x, Efficiency={efficiency:.2f}")
```

#### Scalability Interpretation

1. **Ideal Scaling**: Speedup = number of workers, Efficiency = 1.0
2. **Good Scaling**: Efficiency > 0.7
3. **Acceptable Scaling**: Efficiency > 0.5
4. **Poor Scaling**: Efficiency < 0.5 indicates bottlenecks

## Performance Analysis Patterns

### Scaling Pattern Analysis

```python
def analyze_scaling_patterns(results):
    """Analyze performance scaling patterns."""
    if 'execution_times' not in results:
        return {}
    
    execution_times = results['execution_times']
    worker_counts = results['worker_counts']
    
    # Calculate scaling efficiency
    baseline_time = execution_times[0]
    speedups = [baseline_time / t for t in execution_times]
    efficiencies = [s / wc for s, wc in zip(speedups, worker_counts)]
    
    # Identify scaling phases
    scaling_phases = []
    for i in range(1, len(efficiencies)):
        if efficiencies[i] < efficiencies[i-1] * 0.8:
            scaling_phases.append({
                'transition': f"{worker_counts[i-1]}→{worker_counts[i]}",
                'efficiency_drop': efficiencies[i-1] - efficiencies[i]
            })
    
    analysis = {
        'scaling_efficiency': {
            'avg_efficiency': np.mean(efficiencies),
            'max_efficiency': max(efficiencies),
            'min_efficiency': min(efficiencies),
            'efficiency_std': np.std(efficiencies),
            'scaling_phases': scaling_phases,
            'optimal_workers': worker_counts[np.argmax(efficiencies)]
        },
        'performance_characteristics': {
            'strong_scaling_efficiency': np.mean(efficiencies),
            'scaling_bottlenecks': len(scaling_phases),
            'diminishing_returns_start': worker_counts[next((i for i, e in enumerate(efficiencies) if e < 0.5), len(efficiencies))]
        }
    }
    
    return analysis
```

### Memory Efficiency Analysis

```python
def analyze_memory_efficiency(results):
    """Analyze memory usage patterns."""
    memory_usages = [r.memory_usage for r in results]
    data_sizes = [r.metadata.get('data_size', 0) for r in results]
    
    # Calculate memory efficiency
    memory_efficiency = []
    for i, (mem, size) in enumerate(zip(memory_usages, data_sizes)):
        if size > 0:
            # Expected memory for float64 data
            expected_mb = (size * 8) / (1024 * 1024)
            efficiency = expected_mb / mem if mem > 0 else 0
            memory_efficiency.append(efficiency)
    
    analysis = {
        'memory_usage': {
            'avg_memory': np.mean(memory_usages),
            'max_memory': max(memory_usages),
            'memory_std': np.std(memory_usages),
            'memory_efficiency': np.mean(memory_efficiency) if memory_efficiency else 0
        },
        'memory_patterns': {
            'scaling_efficiency': np.mean(memory_efficiency) if memory_efficiency else 0,
            'overhead_factor': np.mean([mem / (size * 8 / (1024 * 1024)) 
                                       for mem, size in zip(memory_usages, data_sizes) if size > 0])
        }
    }
    
    return analysis
```

### Method Comparison Analysis

```python
def compare_interpolation_methods(results):
    """Compare performance across different interpolation methods."""
    method_results = {}
    
    # Group results by method
    for result in results:
        method = result.metadata.get('method', 'unknown')
        if method not in method_results:
            method_results[method] = []
        method_results[method].append(result)
    
    # Calculate statistics for each method
    comparison = {}
    for method, method_result_list in method_results.items():
        execution_times = [r.execution_time for r in method_result_list]
        memory_usages = [r.memory_usage for r in method_result_list]
        
        comparison[method] = {
            'avg_time': np.mean(execution_times),
            'time_std': np.std(execution_times),
            'avg_memory': np.mean(memory_usages),
            'memory_std': np.std(memory_usages),
            'avg_throughput': np.mean([r.throughput for r in method_result_list])
        }
    
    # Find best performing method
    best_method = min(comparison.items(), 
                    key=lambda x: x[1]['avg_time'])
    
    return {
        'method_comparison': comparison,
        'best_method': best_method[0],
        'best_performance': best_method[1]
    }
```

## Best Practices for High-Resolution Benchmarking

### Performance Optimization Strategies

#### 1. Optimal Chunking

```python
def calculate_optimal_chunk_size(data_shape, target_workers=4, memory_limit_mb=1024):
    """Calculate optimal chunk size for high-resolution data."""
    height, width = data_shape
    total_elements = height * width
    
    # Calculate elements per worker
    elements_per_worker = total_elements // target_workers
    
    # Account for memory overhead (assume 2x for intermediate data)
    max_elements_per_worker = (memory_limit_mb * 1024 * 1024) // (8 * 2)  # 8 bytes per float64
    
    # Use the smaller of the two calculations
    elements_per_worker = min(elements_per_worker, max_elements_per_worker)
    
    # Calculate chunk dimensions (aim for square-ish chunks)
    chunk_size = int(np.sqrt(elements_per_worker))
    
    # Ensure reasonable minimum chunk size
    chunk_size = max(chunk_size, 100)
    
    # Don't exceed original dimensions
    chunk_height = min(chunk_size, height)
    chunk_width = min(chunk_size, width)
    
    return (chunk_height, chunk_width)

# Usage example
optimal_chunks = calculate_optimal_chunk_size((2000, 4000), target_workers=4)
print(f"Optimal chunk size: {optimal_chunks}")
```

#### 2. Memory Management

```python
def configure_memory_for_benchmarking():
    """Configure memory settings for optimal benchmarking."""
    import dask
    
    # Configure Dask memory management
    dask.config.set({
        'array.chunk-size': '128MB',
        'pool.threads': 8,
        'scheduler': 'threads',
        'optimization.fuse.active': True
    })
    
    # Set memory limits
    from dask.distributed import performance_report
    
    # Use memory context manager for monitoring
    with performance_report(filename='memory_usage.html'):
        # Run benchmark operations here
        pass

# Alternative: Memory-aware chunking
def memory_aware_chunking(data, target_memory_mb=512):
    """Create chunks that respect memory limits."""
    element_size_bytes = 8  # float64
    max_elements = int(target_memory_mb * 1024 * 1024 / element_size_bytes)
    
    # Calculate chunk dimensions
    height, width = data.shape
    chunk_area = min(max_elements, height * width)
    
    # Use square-ish chunks
    chunk_size = int(np.sqrt(chunk_area))
    
    return (min(chunk_size, height), min(chunk_size, width))
```

#### 3. Parallel Processing Optimization

```python
def optimize_parallel_processing():
    """Configure Dask for optimal parallel processing."""
    from dask.distributed import Client, LocalCluster
    
    # Create optimized cluster
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        processes=False,  # Use threads for better memory sharing
        memory_limit='8GB',
        dashboard_address=None,  # Disable dashboard for cleaner output
        silence_logs=False
    )
    
    client = Client(cluster)
    
    # Configure global settings
    import dask
    dask.config.set({
        'scheduler': 'threads',
        'pool.threads': 8,
        'array.chunk-size': '128MB',
        'optimization.fuse.active': True,
        'optimization.fuse.width': 4
    })
    
    return client, cluster
```

### Accuracy Validation Best Practices

#### 1. Multiple Test Fields

```python
def create_analytical_test_fields():
    """Create multiple analytical test fields for comprehensive validation."""
    import numpy as np
    
    def sine_wave_field(lon, lat):
        """Simple sine wave field."""
        return np.sin(np.radians(lat)) * np.cos(np.radians(lon))
    
    def gaussian_bump_field(lon, lat):
        """Gaussian bump field."""
        lon_center, lat_center = 0, 0
        sigma = 30  # degrees
        return np.exp(-((lon - lon_center)**2 + (lat - lat_center)**2) / (2 * sigma**2))
    
    def polynomial_field(lon, lat):
        """Polynomial field for testing interpolation accuracy."""
        return (lon/180)**2 + (lat/90)**2 + 0.5 * (lon/180) * (lat/90)
    
    def complex_wave_field(lon, lat):
        """Complex wave field for testing edge cases."""
        return (np.sin(2 * np.radians(lat)) * np.cos(3 * np.radians(lon)) + 
                0.5 * np.sin(5 * np.radians(lat)) * np.cos(2 * np.radians(lon)))
    
    return {
        'sine_wave': sine_wave_field,
        'gaussian_bump': gaussian_bump_field,
        'polynomial': polynomial_field,
        'complex_wave': complex_wave_field
    }

# Usage
test_fields = create_analytical_test_fields()
for field_name, field_func in test_fields.items():
    print(f"Testing with {field_name} field...")
    # Run accuracy benchmark with this field
```

#### 2. Convergence Testing

```python
def run_convergence_analysis():
    """Run convergence analysis across multiple resolutions."""
    from benchmarks import AccuracyBenchmark
    
    accuracy_benchmark = AccuracyBenchmark(threshold=1e-5)
    
    # Test resolutions spanning multiple orders of magnitude
    resolutions = [
        (50, 100),    # Low resolution
        (100, 200),   # Medium resolution
        (200, 400),   # High resolution
        (400, 800),   # Very high resolution
        (800, 1600)   # Ultra high resolution
    ]
    
    convergence_results = {}
    
    for resolution in resolutions:
        result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
            source_resolution=resolution,
            target_resolution=resolution,
            method='bilinear',
            field_type='sine_wave'
        )
        
        convergence_results[str(resolution)] = {
            'rmse': metrics.rmse,
            'mae': metrics.mae,
            'correlation': metrics.correlation,
            'resolution': resolution
        }
    
    # Analyze convergence rate
    resolutions_list = list(convergence_results.keys())
    rmse_values = [convergence_results[r]['rmse'] for r in resolutions_list]
    
    # Calculate convergence rate (should be O(h^2) for bilinear interpolation)
    convergence_rates = []
    for i in range(1, len(rmse_values)):
        rate = np.log(rmse_values[i-1] / rmse_values[i]) / np.log(2)
        convergence_rates.append(rate)
    
    return {
        'convergence_results': convergence_results,
        'convergence_rates': convergence_rates,
        'avg_convergence_rate': np.mean(convergence_rates)
    }
```

### Scalability Testing Best Practices

#### 1. Strong vs Weak Scaling

```python
def compare_scaling_strategies():
    """Compare strong and weak scaling strategies."""
    from benchmarks import StrongScalabilityTester, WeakScalabilityTester
    
    # Strong scaling: fixed problem size, varying workers
    strong_scaler = StrongScalabilityTester(baseline_workers=1)
    strong_results = strong_scaler.test_strong_scalability(
        resolution=(400, 800),
        worker_counts=[1, 2, 4, 8],
        method='bilinear'
    )
    
    # Weak scaling: problem size scales with workers
    weak_scaler = WeakScalabilityTester()
    weak_results = weak_scaler.test_weak_scalability(
        base_resolution=(200, 400),
        worker_scale_factors=[1, 2, 4, 8],
        method='bilinear'
    )
    
    # Compare scaling strategies
    strong_efficiency = np.mean(strong_results['efficiencies'])
    weak_efficiency = np.mean([t / weak_results['execution_times'][0] 
                             for t in weak_results['execution_times']])
    
    analysis = {
        'strong_scaling': {
            'avg_efficiency': strong_efficiency,
            'max_workers': max(strong_results['worker_counts']),
            'efficiency_std': np.std(strong_results['efficiencies'])
        },
        'weak_scaling': {
            'avg_efficiency': weak_efficiency,
            'scale_factors': weak_results['worker_scale_factors'],
            'execution_times': weak_results['execution_times']
        },
        'comparison': {
            'strong_vs_weak_ratio': strong_efficiency / weak_efficiency,
            'optimal_strategy': 'strong' if strong_efficiency > weak_efficiency else 'weak'
        }
    }
    
    return analysis
```

#### 2. Bottleneck Identification

```python
def identify_scaling_bottlenecks(results):
    """Identify scaling bottlenecks in benchmark results."""
    if 'efficiencies' not in results:
        return {}
    
    efficiencies = results['efficiencies']
    worker_counts = results['worker_counts']
    
    bottlenecks = []
    significant_drops = []
    
    # Find efficiency drops
    for i in range(1, len(efficiencies)):
        drop = efficiencies[i-1] - efficiencies[i]
        if drop > 0.1:  # Significant drop (>10%)
            significant_drops.append({
                'transition': f"{worker_counts[i-1]}→{worker_counts[i]}",
                'drop': drop,
                'efficiency_before': efficiencies[i-1],
                'efficiency_after': efficiencies[i]
            })
        
        if efficiencies[i] < 0.5:  # Below 50% efficiency
            bottlenecks.append({
                'worker_count': worker_counts[i],
                'efficiency': efficiencies[i],
                'severity': 'severe' if efficiencies[i] < 0.3 else 'moderate'
            })
    
    # Analyze bottleneck patterns
    bottleneck_analysis = {
        'bottlenecks': bottlenecks,
        'significant_drops': significant_drops,
        'bottleneck_count': len(bottlenecks),
        'drop_count': len(significant_drops),
        'max_efficiency': max(efficiencies),
        'min_efficiency': min(efficiencies),
        'efficiency_degradation': efficiencies[0] - efficiencies[-1]
    }
    
    return bottleneck_analysis
```

## Advanced Analysis Techniques

### Performance Regression Detection

```python
class PerformanceRegressionDetector:
    """Detect performance regressions in benchmark results."""
    
    def __init__(self, baseline_results):
        self.baseline = baseline_results
    
    def detect_regressions(self, current_results, threshold=0.2):
        """Detect performance regressions against baseline."""
        regressions = []
        improvements = []
        
        # Compare performance metrics
        for test_name, current_metrics in current_results.items():
            if test_name in self.baseline:
                baseline_metrics = self.baseline[test_name]
                
                # Check execution time regression
                baseline_time = baseline_metrics['execution_time']
                current_time = current_metrics['execution_time']
                
                if current_time > baseline_time * (1 + threshold):
                    regressions.append({
                        'test': test_name,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'degradation_percent': ((current_time - baseline_time) / baseline_time) * 100,
                        'severity': 'critical' if current_time > baseline_time * 1.5 else 'moderate'
                    })
                
                # Check for improvements
                elif current_time < baseline_time * (1 - threshold):
                    improvements.append({
                        'test': test_name,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'improvement_percent': ((baseline_time - current_time) / baseline_time) * 100
                    })
        
        return {
            'regressions': regressions,
            'improvements': improvements,
            'regression_count': len(regressions),
            'improvement_count': len(improvements),
            'has_critical_regressions': any(r['severity'] == 'critical' for r in regressions)
        }
```

### Statistical Analysis

```python
import scipy.stats as stats

def statistical_analysis_of_benchmarks(results):
    """Perform statistical analysis of benchmark results."""
    if not results:
        return {}
    
    # Extract execution times
    execution_times = [r.execution_time for r in results]
    memory_usages = [r.memory_usage for r in results]
    
    # Basic statistics
    time_stats = {
        'mean': np.mean(execution_times),
        'median': np.median(execution_times),
        'std': np.std(execution_times),
        'min': np.min(execution_times),
        'max': np.max(execution_times),
        'cv': np.std(execution_times) / np.mean(execution_times) if np.mean(execution_times) > 0 else 0
    }
    
    memory_stats = {
        'mean': np.mean(memory_usages),
        'median': np.median(memory_usages),
        'std': np.std(memory_usages),
        'min': np.min(memory_usages),
        'max': np.max(memory_usages)
    }
    
    # Normality test
    _, time_p_value = stats.normaltest(execution_times)
    _, memory_p_value = stats.normaltest(memory_usages)
    
    # Outlier detection using IQR method
    time_q1, time_q3 = np.percentile(execution_times, [25, 75])
    time_iqr = time_q3 - time_q1
    time_outliers = [t for t in execution_times if t < time_q1 - 1.5 * time_iqr or t > time_q3 + 1.5 * time_iqr]
    
    memory_q1, memory_q3 = np.percentile(memory_usages, [25, 75])
    memory_iqr = memory_q3 - memory_q1
    memory_outliers = [m for m in memory_usages if m < memory_q1 - 1.5 * memory_iqr or m > memory_q3 + 1.5 * memory_iqr]
    
    analysis = {
        'execution_time': time_stats,
        'memory_usage': memory_stats,
        'statistical_tests': {
            'time_normality_p_value': time_p_value,
            'memory_normality_p_value': memory_p_value,
            'time_is_normal': time_p_value > 0.05,
            'memory_is_normal': memory_p_value > 0.05
        },
        'outliers': {
            'time_outliers': time_outliers,
            'memory_outliers': memory_outliers,
            'time_outlier_count': len(time_outliers),
            'memory_outlier_count': len(memory_outliers)
        }
    }
    
    return analysis
```

## Visualization and Reporting

### Performance Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_benchmark_results(results, output_file='benchmark_results.png'):
    """Create visualization of benchmark results."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Execution time vs resolution
    resolutions = [r.metadata.get('resolution', 'unknown') for r in results]
    times = [r.execution_time for r in results]
    
    axes[0, 0].scatter(range(len(times)), times)
    axes[0, 0].set_title('Execution Time vs Test Index')
    axes[0, 0].set_xlabel('Test Index')
    axes[0, 0].set_ylabel('Execution Time (s)')
    
    # 2. Memory usage vs resolution
    memory = [r.memory_usage for r in results]
    axes[0, 1].scatter(range(len(memory)), memory)
    axes[0, 1].set_title('Memory Usage vs Test Index')
    axes[0, 1].set_xlabel('Test Index')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    
    # 3. Throughput analysis
    throughput = [r.throughput for r in results]
    axes[1, 0].scatter(range(len(throughput)), throughput)
    axes[1, 0].set_title('Throughput vs Test Index')
    axes[1, 0].set_xlabel('Test Index')
    axes[1, 0].set_ylabel('Throughput (M elements/s)')
    
    # 4. Correlation matrix
    metrics_df = pd.DataFrame({
        'execution_time': times,
        'memory_usage': memory,
        'throughput': throughput
    })
    
    correlation_matrix = metrics_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1])
    axes[1, 1].set_title('Metric Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_file}")
```

### Automated Report Generation

```python
def generate_comprehensive_report(results, output_file='benchmark_report.md'):
    """Generate comprehensive benchmark report."""
    report_lines = [
        "# PyRegrid Benchmark Report\n",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"Total tests: {len(results)}\n\n"
    ]
    
    # Performance summary
    report_lines.append("## Performance Summary\n")
    execution_times = [r.execution_time for r in results]
    memory_usages = [r.memory_usage for r in results]
    
    report_lines.append(f"- Average execution time: {np.mean(execution_times):.4f}s ± {np.std(execution_times):.4f}s\n")
    report_lines.append(f"- Average memory usage: {np.mean(memory_usages):.2f}MB ± {np.std(memory_usages):.2f}MB\n")
    report_lines.append(f"- Average throughput: {np.mean([r.throughput for r in results]):.2f}M elements/s\n\n")
    
    # Method comparison
    methods = {}
    for r in results:
        method = r.metadata.get('method', 'unknown')
        if method not in methods:
            methods[method] = []
        methods[method].append(r)
    
    report_lines.append("## Method Comparison\n")
    for method, method_results in methods.items():
        method_times = [r.execution_time for r in method_results]
        method_memory = [r.memory_usage for r in method_results]
        report_lines.append(f"### {method}\n")
        report_lines.append(f"- Average time: {np.mean(method_times):.4f}s\n")
        report_lines.append(f"- Average memory: {np.mean(method_memory):.2f}MB\n")
        report_lines.append(f"- Test count: {len(method_results)}\n\n")
    
    # Detailed results table
    report_lines.append("## Detailed Results\n")
    report_lines.append("| Test | Method | Time (s) | Memory (MB) | Throughput (M/s) |\n")
    report_lines.append("|------|--------|----------|-------------|------------------|\n")
    
    for r in results:
        method = r.metadata.get('method', 'unknown')
        resolution = r.metadata.get('resolution', 'unknown')
        report_lines.append(f"| {resolution} | {method} | {r.execution_time:.4f} | "
                          f"{r.memory_usage:.2f} | {r.throughput:.2f} |\n")
    
    # Write report
    with open(output_file, 'w') as f:
        f.writelines(report_lines)
    
    print(f"Report saved to: {output_file}")
```

## Conclusion

This guide has provided comprehensive techniques for analyzing benchmark results and implementing best practices for PyRegrid's high-resolution benchmarking system. By following these patterns, you can:

- **Interpret** complex benchmark results with statistical rigor
- **Identify** performance bottlenecks and scaling issues
- **Optimize** chunking and memory management strategies
- **Validate** accuracy across multiple test scenarios
- **Detect** performance regressions early
- **Generate** comprehensive reports and visualizations

These analysis techniques enable data-driven optimization of high-resolution regridding operations and ensure consistent performance across different computational environments.