# Benchmarking Integration and Advanced Workflows

This guide covers advanced integration patterns, pytest configuration, and comprehensive benchmarking workflows for PyRegrid's high-resolution benchmarking system.

## Pytest Integration

### Configuration and Setup

The benchmarking system integrates seamlessly with pytest through the [`conftest.py`](benchmarks/conftest.py:1) configuration module:

```python
# benchmarks/conftest.py
import pytest
import numpy as np
import dask
from dask.distributed import Client, LocalCluster

def pytest_addoption(parser):
    """Add command-line options for benchmark tests."""
    parser.addoption(
        "--benchmark",
        action="store_true", 
        default=False,
        help="Run benchmark tests"
    )
    parser.addoption(
        "--benchmark-output-dir",
        action="store",
        default="./benchmark_results",
        help="Directory to store benchmark results"
    )
    parser.addoption(
        "--benchmark-large",
        action="store_true", 
        default=False,
        help="Run large-scale benchmark tests"
    )
```

### Dask Client Fixture

The [`dask_client`](benchmarks/conftest.py:47) fixture provides automatic Dask cluster management:

```python
@pytest.fixture(scope="session")
def dask_client(request) -> Generator[Client, None, None]:
    """Create a Dask client for benchmark tests."""
    if not request.config.getoption("--benchmark"):
        # For non-benchmark tests, use default dask scheduler
        with dask.config.set(scheduler='threads'):
            yield None
        return
    
    # Create a local cluster for benchmarks
    cluster = LocalCluster(
        n_workers=2,  # Start with 2 workers for benchmarks
        threads_per_worker=2,
        processes=False,  # Use threads for better memory sharing
        dashboard_address=None  # Disable dashboard to reduce overhead
    )
    
    client = Client(cluster)
    
    try:
        yield client
    finally:
        client.close()
        cluster.close()
```

### Benchmark Data Fixtures

Pre-configured test data fixtures for different scenarios:

```python
@pytest.fixture(scope="function")
def benchmark_data_small() -> tuple:
    """Create small benchmark data for quick tests."""
    height, width = 50, 100
    
    # Create analytical test function
    lon = np.linspace(-180, 180, width)
    lat = np.linspace(-90, 90, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Create test pattern: combination of sine waves
    source_data = (np.sin(np.radians(lat_grid)) * 
                  np.cos(np.radians(lon_grid)) + 
                  0.5 * np.sin(2 * np.radians(lat_grid)) * 
                  np.cos(2 * np.radians(lon_grid)))
    
    target_coords = (
        np.linspace(-180, 180, width//2),  # Target is half resolution
        np.linspace(-90, 90, height//2)
    )
    
    return source_data, target_coords

@pytest.fixture(scope="function") 
def benchmark_data_large() -> tuple:
    """Create large benchmark data for comprehensive tests."""
    if not pytest.config.getoption("--benchmark-large"):
        # Return small data if large benchmarks are not requested
        height, width = 50, 100
    else:
        # Large resolution for comprehensive tests
        height, width = 200, 400
    
    # Create analytical test function
    lon = np.linspace(-180, 180, width)
    lat = np.linspace(-90, 90, height)
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    
    # Create test pattern: combination of sine waves
    source_data = (np.sin(np.radians(lat_grid)) * 
                  np.cos(np.radians(lon_grid)) + 
                  0.5 * np.sin(2 * np.radians(lat_grid)) * 
                  np.cos(2 * np.radians(lon_grid)))
    
    target_coords = (
        np.linspace(-180, 180, width//2),  # Target is half resolution
        np.linspace(-90, 90, height//2)
    )
    
    return source_data, target_coords
```

### Running Benchmark Tests

#### Command Line Usage

```bash
# Run all benchmark tests
python -m pytest benchmarks/ -v --benchmark

# Run with custom output directory
python -m pytest benchmarks/ -v --benchmark --benchmark-output-dir ./results

# Run large-scale benchmarks only
python -m pytest benchmarks/ -v --benchmark --benchmark-large

# Run specific benchmark test file
python -m pytest benchmarks/test_high_resolution_benchmarks.py -v --benchmark

# Run with benchmark markers
python -m pytest benchmarks/ -v --benchmark -m benchmark
```

#### Test Markers

The system uses pytest markers to categorize different types of benchmarks:

```python
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Standard performance benchmarks."""
    
    @pytest.mark.parametrize("resolution", [(100, 200), (200, 400)])
    def test_performance_at_resolution(self, resolution, dask_client):
        """Test performance at different resolutions."""
        # Test implementation

@pytest.mark.large_benchmark  
class TestLargeScaleBenchmarks:
    """Large-scale benchmarks requiring significant resources."""
    
    def test_3km_grid_equivalent(self, dask_client):
        """Test with 3km grid equivalent data."""
        # Test implementation
```

### Custom Benchmark Test Patterns

#### Performance Regression Testing

```python
@pytest.mark.benchmark
def test_performance_regression_detection(dask_client):
    """Test for detecting performance regressions."""
    from benchmarks import HighResolutionBenchmark
    
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    # Run the same operation multiple times to establish baseline
    execution_times = []
    for i in range(5):
        result = benchmark.benchmark_regridding_operation(
            source_data=np.random.random((100, 200)),
            target_coords=(np.linspace(-180, 180, 100), np.linspace(-90, 90, 200)),
            method='bilinear',
            name=f'regression_test_run_{i}'
        )
        execution_times.append(result.execution_time)
    
    # Calculate statistics
    mean_time = np.mean(execution_times)
    std_time = np.std(execution_times)
    
    # Check that times are reasonably consistent (no major regression)
    cv = std_time / mean_time if mean_time > 0 else 0
    assert cv < 0.5, f"Coefficient of variation too high: {cv:.2f}"
```

#### Multi-Method Comparison

```python
@pytest.mark.benchmark
@pytest.mark.parametrize("method", ['bilinear', 'nearest', 'conservative'])
def test_method_comparison(method, dask_client):
    """Compare performance of different interpolation methods."""
    from benchmarks import HighResolutionBenchmark
    
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=dask_client)
    
    # Create test data
    source_data = benchmark._create_test_data(200, 400, use_dask=True)
    target_coords = (
        np.linspace(-180, 180, 200),
        np.linspace(-90, 90, 400)
    )
    
    # Run benchmark with specified method
    result = benchmark.benchmark_regridding_operation(
        source_data=source_data,
        target_coords=target_coords,
        method=method,
        name=f'method_comparison_{method}'
    )
    
    # Verify results and log performance
    assert isinstance(result, BenchmarkResult)
    assert result.execution_time >= 0
    print(f"Method {method}: Time={result.execution_time:.4f}s, Memory={result.memory_usage:.2f}MB")
```

#### Scalability Testing

```python
@pytest.mark.benchmark
def test_strong_scalability_pattern(dask_client):
    """Test strong scalability pattern."""
    from benchmarks import StrongScalabilityTester
    
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
    
    # Verify scaling efficiency doesn't drop too much
    avg_efficiency = np.mean(results['efficiencies'])
    assert avg_efficiency > 0.5, f"Average efficiency too low: {avg_efficiency:.2f}"
```

## Advanced Dask Integration

### Distributed Benchmarking Workflows

#### Multi-Worker Benchmarking

```python
from dask.distributed import Client
from benchmarks import DistributedBenchmarkRunner

def run_distributed_benchmark():
    """Run benchmarks across multiple Dask workers."""
    # Create distributed client
    client = Client('tcp://scheduler:8786')
    
    # Create distributed benchmark runner
    dist_runner = DistributedBenchmarkRunner(client=client)
    
    # Define benchmark function that can run on workers
    def worker_benchmark_task(source_data, target_coords, method='bilinear'):
        """Benchmark function that runs on Dask workers."""
        from benchmarks import HighResolutionBenchmark
        
        # Create benchmark instance on worker
        benchmark = HighResolutionBenchmark(use_dask=True, dask_client=client)
        
        # Run the benchmark
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=target_coords,
            method=method,
            name=f'distributed_{method}'
        )
        
        return result
    
    # Prepare test data
    source_data = da.random.random((1000, 2000), chunks=(200, 400))
    target_coords = (
        np.linspace(-180, 180, 1000),
        np.linspace(-90, 90, 2000)
    )
    
    # Run benchmark on multiple workers
    results = dist_runner.run_benchmark_on_workers(
        worker_benchmark_task,
        source_data=source_data,
        target_coords=target_coords,
        n_workers=4
    )
    
    # Analyze distributed results
    execution_times = [r.execution_time for r in results]
    print(f"Distributed benchmark results:")
    print(f"  Average time: {np.mean(execution_times):.4f}s")
    print(f"  Time std: {np.std(execution_times):.4f}s")
    print(f"  Speedup vs single worker: {np.min(execution_times)/np.max(execution_times):.2f}x")
    
    return results
```

#### Memory-Aware Chunking

```python
def optimize_chunking_for_benchmarking(data_shape, target_workers=4):
    """Optimize chunking for distributed benchmarking."""
    height, width = data_shape
    
    # Calculate optimal chunk size based on target workers
    total_elements = height * width
    elements_per_worker = total_elements // target_workers
    
    # Aim for chunks that are roughly square
    chunk_size = int(np.sqrt(elements_per_worker))
    
    # Ensure chunks are reasonable size (not too small)
    chunk_size = max(chunk_size, 100)
    
    # Don't exceed original dimensions
    chunk_height = min(chunk_size, height)
    chunk_width = min(chunk_size, width)
    
    return (chunk_height, chunk_width)

# Usage example
large_data = da.random.random((2000, 4000))
optimized_chunks = optimize_chunking_for_benchmarking(large_data.shape, target_workers=4)
large_data = large_data.rechunk(optimized_chunks)
```

### Advanced Dask Configuration

#### Scheduler Optimization

```python
import dask
from dask.distributed import Client

def configure_dask_for_benchmarking():
    """Configure Dask for optimal benchmarking performance."""
    
    # Configure global Dask settings
    dask.config.set({
        'scheduler': 'threads',  # Use threads for better memory sharing
        'pool.threads': 8,       # Number of threads for thread pool
        'array.chunk-size': '128MB',  # Optimal chunk size for regridding
        'optimization.fuse.active': True,  # Enable fusion optimization
    })
    
    # Create local cluster with optimized settings
    cluster = LocalCluster(
        n_workers=4,
        threads_per_worker=2,
        processes=False,  # Use threads for better memory sharing
        memory_limit='4GB',
        dashboard_address=None,  # Disable dashboard for cleaner output
        silence_logs=False  # Keep logs for debugging
    )
    
    client = Client(cluster)
    
    return client, cluster

# Usage
client, cluster = configure_dask_for_benchmarking()
```

#### Memory Management

```python
from dask.distributed import performance_report

def benchmark_with_memory_monitoring():
    """Benchmark with detailed memory monitoring."""
    from benchmarks import HighResolutionBenchmark
    
    benchmark = HighResolutionBenchmark(use_dask=True, dask_client=client)
    
    # Create large test data
    source_data = da.random.random((1000, 2000), chunks=(200, 400))
    target_coords = (
        np.linspace(-180, 180, 1000),
        np.linspace(-90, 90, 2000)
    )
    
    # Run benchmark with memory monitoring
    with performance_report(filename='memory_report.html'):
        result = benchmark.benchmark_regridding_operation(
            source_data=source_data,
            target_coords=target_coords,
            method='bilinear',
            name='memory_monitored_benchmark'
        )
    
    print(f"Benchmark completed with memory monitoring")
    print(f"Execution time: {result.execution_time:.4f}s")
    print(f"Memory usage: {result.memory_usage:.2f}MB")
    
    return result
```

## Comprehensive Benchmarking Workflows

### Automated Benchmark Suite

```python
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class ComprehensiveBenchmarkSuite:
    """Automated benchmark suite for comprehensive testing."""
    
    def __init__(self, output_dir: str = "./benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize benchmark components
        from benchmarks import (
            HighResolutionBenchmark, 
            AccuracyBenchmark, 
            ScalabilityBenchmark,
            StrongScalabilityTester,
            WeakScalabilityTester
        )
        
        self.performance_benchmark = HighResolutionBenchmark()
        self.accuracy_benchmark = AccuracyBenchmark()
        self.scalability_benchmark = ScalabilityBenchmark()
        self.strong_scalability = StrongScalabilityTester()
        self.weak_scalability = WeakScalabilityTester()
    
    def run_full_suite(self, dask_client=None) -> Dict[str, Any]:
        """Run the complete benchmark suite."""
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'dask_workers': dask_client.cluster.workers if dask_client else 1,
                'total_tests': 0
            },
            'performance': {},
            'accuracy': {},
            'scalability': {}
        }
        
        # Performance benchmarks
        results['performance'] = self._run_performance_benchmarks(dask_client)
        results['metadata']['total_tests'] += len(results['performance'])
        
        # Accuracy benchmarks  
        results['accuracy'] = self._run_accuracy_benchmarks()
        results['metadata']['total_tests'] += len(results['accuracy'])
        
        # Scalability benchmarks
        if dask_client:
            results['scalability'] = self._run_scalability_benchmarks(dask_client)
            results['metadata']['total_tests'] += len(results['scalability'])
        
        # Save comprehensive results
        self._save_results(results)
        
        return results
    
    def _run_performance_benchmarks(self, dask_client) -> Dict[str, Any]:
        """Run performance benchmarks."""
        results = {}
        
        # Test different resolutions
        resolutions = [(100, 200), (200, 400), (500, 1000)]
        methods = ['bilinear', 'nearest']
        
        for height, width in resolutions:
            for method in methods:
                source_data = self.performance_benchmark._create_test_data(
                    height, width, use_dask=True
                )
                target_coords = (
                    np.linspace(-180, 180, width//2),
                    np.linspace(-90, 90, height//2)
                )
                
                result = self.performance_benchmark.benchmark_regridding_operation(
                    source_data=source_data,
                    target_coords=target_coords,
                    method=method,
                    name=f'perf_{method}_{height}x{width}'
                )
                
                key = f'{method}_{height}x{width}'
                results[key] = {
                    'execution_time': result.execution_time,
                    'memory_usage': result.memory_usage,
                    'cpu_percent': result.cpu_percent,
                    'throughput': (height * width) / result.execution_time if result.execution_time > 0 else 0
                }
        
        return results
    
    def _run_accuracy_benchmarks(self) -> Dict[str, Any]:
        """Run accuracy benchmarks."""
        results = {}
        
        resolutions = [(100, 200), (200, 400)]
        methods = ['bilinear', 'nearest']
        
        for height, width in resolutions:
            for method in methods:
                result, metrics = self.accuracy_benchmark.benchmark_interpolation_accuracy(
                    source_resolution=(height, width),
                    target_resolution=(height, width),
                    method=method
                )
                
                key = f'{method}_{height}x{width}'
                results[key] = {
                    'rmse': metrics.rmse,
                    'mae': metrics.mae,
                    'max_error': metrics.max_error,
                    'correlation': metrics.correlation,
                    'bias': metrics.bias,
                    'passes_threshold': metrics.rmse <= metrics.accuracy_threshold
                }
        
        return results
    
    def _run_scalability_benchmarks(self, dask_client) -> Dict[str, Any]:
        """Run scalability benchmarks."""
        results = {}
        
        # Strong scalability
        strong_results = self.strong_scalability.test_strong_scalability(
            resolution=(200, 400),
            worker_counts=[1, 2, 4, 8],
            method='bilinear',
            dask_client=dask_client
        )
        
        results['strong_scalability'] = {
            'worker_counts': strong_results['worker_counts'],
            'speedups': strong_results['speedups'],
            'efficiencies': strong_results['efficiencies'],
            'avg_efficiency': np.mean(strong_results['efficiencies'])
        }
        
        # Weak scalability
        weak_results = self.weak_scalability.test_weak_scalability(
            base_resolution=(100, 200),
            worker_scale_factors=[1, 2, 4],
            method='bilinear',
            dask_client=dask_client
        )
        
        results['weak_scalability'] = {
            'scale_factors': weak_results['worker_scale_factors'],
            'execution_times': weak_results['execution_times'],
            'work_per_worker': weak_results['work_per_worker']
        }
        
        return results
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_benchmark_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Comprehensive benchmark results saved to: {filepath}")
        
        # Generate summary report
        self._generate_summary_report(results)
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate a human-readable summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_summary_{timestamp}.md"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("# PyRegrid Benchmark Summary\n\n")
            f.write(f"**Timestamp:** {results['metadata']['timestamp']}\n")
            f.write(f"**Dask Workers:** {results['metadata']['dask_workers']}\n")
            f.write(f"**Total Tests:** {results['metadata']['total_tests']}\n\n")
            
            # Performance summary
            f.write("## Performance Benchmarks\n\n")
            f.write("| Method | Resolution | Time (s) | Memory (MB) | Throughput (M/s) |\n")
            f.write("|--------|------------|----------|-------------|------------------|\n")
            
            for key, metrics in results['performance'].items():
                method, resolution = key.split('_', 1)
                f.write(f"| {method} | {resolution} | {metrics['execution_time']:.4f} | "
                       f"{metrics['memory_usage']:.2f} | {metrics['throughput']:.2f} |\n")
            
            # Accuracy summary
            f.write("\n## Accuracy Benchmarks\n\n")
            f.write("| Method | Resolution | RMSE | MAE | Passes Threshold |\n")
            f.write("|--------|------------|------|-----|------------------|\n")
            
            for key, metrics in results['accuracy'].items():
                method, resolution = key.split('_', 1)
                f.write(f"| {method} | {resolution} | {metrics['rmse']:.6f} | "
                       f"{metrics['mae']:.6f} | {metrics['passes_threshold']} |\n")
            
            # Scalability summary
            if 'scalability' in results:
                f.write("\n## Scalability Benchmarks\n\n")
                
                if 'strong_scalability' in results['scalability']:
                    strong = results['scalability']['strong_scalability']
                    f.write(f"**Strong Scalability (avg efficiency):** {strong['avg_efficiency']:.2f}\n\n")
                    f.write("| Workers | Speedup | Efficiency |\n")
                    f.write("|---------|---------|------------|\n")
                    
                    for i, workers in enumerate(strong['worker_counts']):
                        f.write(f"| {workers} | {strong['speedups'][i]:.2f} | "
                               f"{strong['efficiencies'][i]:.2f} |\n")
        
        print(f"Summary report saved to: {filepath}")
```

### Regression Testing Pipeline

```python
class BenchmarkRegressionTester:
    """Pipeline for detecting performance regressions."""
    
    def __init__(self, baseline_results_file: str):
        self.baseline_results = self._load_baseline_results(baseline_results_file)
    
    def _load_baseline_results(self, filepath: str) -> Dict[str, Any]:
        """Load baseline benchmark results."""
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def run_regression_tests(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run regression tests against baseline."""
        regression_report = {
            'baseline_timestamp': self.baseline_results['metadata']['timestamp'],
            'current_timestamp': current_results['metadata']['timestamp'],
            'regressions_detected': [],
            'improvements_detected': [],
            'no_change': []
        }
        
        # Compare performance metrics
        baseline_perf = self.baseline_results['performance']
        current_perf = current_results['performance']
        
        for key in baseline_perf:
            if key in current_perf:
                baseline_time = baseline_perf[key]['execution_time']
                current_time = current_perf[key]['execution_time']
                
                # Allow 20% variation as noise
                threshold = baseline_time * 0.2
                
                if current_time > baseline_time + threshold:
                    regression_report['regressions_detected'].append({
                        'test': key,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'degradation_percent': ((current_time - baseline_time) / baseline_time) * 100
                    })
                elif current_time < baseline_time - threshold:
                    regression_report['improvements_detected'].append({
                        'test': key,
                        'baseline_time': baseline_time,
                        'current_time': current_time,
                        'improvement_percent': ((baseline_time - current_time) / baseline_time) * 100
                    })
                else:
                    regression_report['no_change'].append(key)
        
        return regression_report
    
    def generate_regression_report(self, regression_report: Dict[str, Any]) -> str:
        """Generate human-readable regression report."""
        report_lines = [
            "# Performance Regression Report\n",
            f"**Baseline:** {regression_report['baseline_timestamp']}\n",
            f"**Current:** {regression_report['current_timestamp']}\n",
            f"**Total Tests Compared:** {len(regression_report['regressions_detected']) + len(regression_report['improvements_detected']) + len(regression_report['no_change'])}\n\n"
        ]
        
        # Regressions
        if regression_report['regressions_detected']:
            report_lines.append("## Regressions Detected\n\n")
            report_lines.append("| Test | Baseline (s) | Current (s) | Degradation |\n")
            report_lines.append("|------|--------------|-------------|-------------|\n")
            
            for reg in regression_report['regressions_detected']:
                report_lines.append(f"| {reg['test']} | {reg['baseline_time']:.4f} | "
                                   f"{reg['current_time']:.4f} | {reg['degradation_percent']:.1f}% |\n")
        
        # Improvements
        if regression_report['improvements_detected']:
            report_lines.append("\n## Improvements Detected\n\n")
            report_lines.append("| Test | Baseline (s) | Current (s) | Improvement |\n")
            report_lines.append("|------|--------------|-------------|-------------|\n")
            
            for imp in regression_report['improvements_detected']:
                report_lines.append(f"| {imp['test']} | {imp['baseline_time']:.4f} | "
                                   f"{imp['current_time']:.4f} | {imp['improvement_percent']:.1f}% |\n")
        
        # No change
        if regression_report['no_change']:
            report_lines.append(f"\n## No Significant Change ({len(regression_report['no_change'])} tests)\n")
        
        return ''.join(report_lines)
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/benchmark.yml
name: Benchmark Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -e .[dask-benchmarking]
        pip install pytest pytest-benchmark pytest-xdist
    
    - name: Start Dask cluster
      run: |
        dask-scheduler --port 8786 &
        dask-worker tcp://localhost:8786 --n-workers 2 --threads-per-worker 2 &
        sleep 5  # Wait for cluster to start
    
    - name: Run benchmark suite
      run: |
        python -m pytest benchmarks/ -v --benchmark-only --benchmark-json=benchmark_results.json
    
    - name: Upload benchmark results
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-results
        path: benchmark_results.json
    
    - name: Generate benchmark report
      run: |
        python -c "
        import json
        with open('benchmark_results.json') as f:
            results = json.load(f)
        
        # Generate simple report
        report = '# Benchmark Results\\n\\n'
        for result in results['benchmarks']:
            report += f'## {result['name']}\\n'
            report += f'- Execution time: {result['stats']['mean']:.4f}s ± {result['stats']['stddev']:.4f}s\\n'
            report += f'- Memory: {result['stats']['mem_usage']:.2f}MB\\n\\n'
        
        with open('benchmark_report.md', 'w') as f:
            f.write(report)
        "
    
    - name: Upload benchmark report
      uses: actions/upload-artifact@v3
      with:
        name: benchmark-report
        path: benchmark_report.md
```

### Performance Gate Checks

```python
# benchmark_gate_checker.py
import json
import sys
from typing import Dict, Any

class BenchmarkGateChecker:
    """Check if benchmark results meet performance gates."""
    
    def __init__(self, gates_config: Dict[str, Any]):
        self.gates = gates_config
    
    def check_performance_gates(self, benchmark_results: Dict[str, Any]) -> bool:
        """Check if all benchmark results meet performance gates."""
        all_passed = True
        
        for test_name, metrics in benchmark_results.items():
            if test_name in self.gates:
                gate = self.gates[test_name]
                
                # Check execution time gate
                if 'max_execution_time' in gate:
                    if metrics['execution_time'] > gate['max_execution_time']:
                        print(f"❌ {test_name}: Execution time {metrics['execution_time']:.4f}s exceeds gate {gate['max_execution_time']:.4f}s")
                        all_passed = False
                    else:
                        print(f"✅ {test_name}: Execution time {metrics['execution_time']:.4f}s within gate")
                
                # Check memory usage gate
                if 'max_memory_usage' in gate:
                    if metrics['memory_usage'] > gate['max_memory_usage']:
                        print(f"❌ {test_name}: Memory usage {metrics['memory_usage']:.2f}MB exceeds gate {gate['max_memory_usage']:.2f}MB")
                        all_passed = False
                    else:
                        print(f"✅ {test_name}: Memory usage {metrics['memory_usage']:.2f}MB within gate")
        
        return all_passed

# Usage example
if __name__ == "__main__":
    # Load benchmark results
    with open('benchmark_results.json', 'r') as f:
        results = json.load(f)
    
    # Define performance gates
    performance_gates = {
        'perf_bilinear_100x200': {
            'max_execution_time': 1.0,
            'max_memory_usage': 100.0
        },
        'perf_bilinear_200x400': {
            'max_execution_time': 4.0,
            'max_memory_usage': 400.0
        },
        'perf_bilinear_500x1000': {
            'max_execution_time': 20.0,
            'max_memory_usage': 2000.0
        }
    }
    
    # Check gates
    gate_checker = BenchmarkGateChecker(performance_gates)
    passed = gate_checker.check_performance_gates(results['benchmarks'])
    
    if passed:
        print("🎉 All performance gates passed!")
        sys.exit(0)
    else:
        print("💥 Some performance gates failed!")
        sys.exit(1)
```

## Conclusion

This guide has covered advanced integration patterns and comprehensive workflows for PyRegrid's benchmarking system. By implementing these patterns, you can:

- **Seamlessly integrate** with pytest for automated testing
- **Leverage Dask** for distributed benchmarking at scale
- **Automate comprehensive** benchmark suites with regression detection
- **Integrate with CI/CD** for continuous performance monitoring
- **Implement performance gates** to ensure code quality

These advanced patterns enable robust, automated performance monitoring and optimization workflows for high-resolution regridding operations.