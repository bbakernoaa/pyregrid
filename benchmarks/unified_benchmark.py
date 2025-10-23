
"""
Unified benchmarking interface for PyRegrid.

This module provides a high-level interface that orchestrates all benchmarking components:
- Performance metrics
- Accuracy validation
- Scalability testing
- Report generation
"""
import time
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

try:
    from dask.distributed import Client
    HAS_DASK_DISTRIBUTED = True
except ImportError:
    HAS_DASK_DISTRIBUTED = False

from .benchmark_base import BenchmarkResult
from .performance_metrics import HighResolutionBenchmark, create_performance_report
from .accuracy_validation import AccuracyBenchmark, create_accuracy_report
from .scalability_testing import ScalabilityBenchmark, StrongScalabilityTester, WeakScalabilityTester, analyze_scalability_results


@dataclass
class BenchmarkConfiguration:
    """Configuration for benchmark runs."""
    use_dask: bool = True
    dask_client: Optional['Client'] = None
    output_dir: str = "./benchmark_results"
    performance_threshold: float = 10.0  # seconds
    accuracy_threshold: float = 1e-4
    scalability_threshold: float = 0.7 # efficiency percentage
    iterations: int = 3
    methods: Optional[List[str]] = None  # Default to ['bilinear', 'nearest']
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ['bilinear', 'nearest']


class UnifiedBenchmarkRunner:
    """
    Unified benchmarking runner that coordinates performance, accuracy, and scalability tests.
    
    This class provides a single interface to run comprehensive benchmarking across all
    dimensions of PyRegrid functionality.
    """
    
    def __init__(self, config: Optional[BenchmarkConfiguration] = None):
        self.config = config or BenchmarkConfiguration()
        self.performance_benchmark = HighResolutionBenchmark(
            use_dask=self.config.use_dask,
            dask_client=self.config.dask_client
        )
        self.accuracy_benchmark = AccuracyBenchmark(
            threshold=self.config.accuracy_threshold
        )
        self.scalability_benchmark = ScalabilityBenchmark()
        self.strong_scalability_tester = StrongScalabilityTester()
        self.weak_scalability_tester = WeakScalabilityTester()
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Store all results
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'performance': {},
            'accuracy': {},
            'scalability': {},
            'summary': {}
        }
    
    def run_comprehensive_benchmark(self,
                                  resolutions: List[Tuple[int, int]],
                                  methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a comprehensive benchmark across all dimensions.
        
        Args:
            resolutions: List of (height, width) tuples to test
            methods: List of interpolation methods to test
        
        Returns:
            Dictionary containing all benchmark results
        """
        methods = methods or self.config.methods or ['bilinear', 'nearest']
        
        print(f"Starting comprehensive benchmark with {len(resolutions)} resolutions and {len(methods)} methods")
        
        # Run performance benchmarks
        self.results['performance'] = self._run_performance_benchmarks(resolutions, methods)
        
        # Run accuracy benchmarks
        self.results['accuracy'] = self._run_accuracy_benchmarks(resolutions, methods)
        
        # Run scalability benchmarks (if using Dask)
        if self.config.use_dask and self.config.dask_client:
            self.results['scalability'] = self._run_scalability_benchmarks(resolutions, methods)
        
        # Generate summary
        self.results['summary'] = self._generate_summary()
        
        return self.results
    
    def _run_performance_benchmarks(self, 
                                  resolutions: List[Tuple[int, int]], 
                                  methods: List[str]) -> Dict[str, Any]:
        """Run performance benchmarks across resolutions and methods."""
        print("Running performance benchmarks...")
        
        performance_results = {
            'resolutions': resolutions,
            'methods': methods,
            'individual_results': [],
            'reports': {}
        }
        
        for resolution in resolutions:
            height, width = resolution
            for method in methods:
                print(f"  Testing performance: {method} at {height}x{width}")
                
                # Run multiple iterations for statistical significance
                iteration_results = []
                for i in range(self.config.iterations):
                    source_data = self.performance_benchmark._create_test_data(height, width, use_dask=self.config.use_dask)
                    target_coords = (
                        np.linspace(-180, 180, width//2),
                        np.linspace(-90, 90, height//2)
                    )
                    target_lon, target_lat = np.meshgrid(target_coords[0], target_coords[1])
                    target_points = np.column_stack([target_lon.ravel(), target_lat.ravel()])
                    
                    result = self.performance_benchmark.benchmark_regridding_operation(
                        source_data=source_data,
                        target_coords=(target_points[:, 1], target_points[:, 0]),  # Correct format: (latitudes, longitudes)
                        method=method,
                        name=f'perf_{method}_{height}x{width}_iter_{i}'
                    )
                    iteration_results.append(result)
                
                performance_results['individual_results'].extend(iteration_results)
        
        # Generate performance report
        report = create_performance_report(
            performance_results['individual_results'],
            output_file=os.path.join(self.config.output_dir, "performance_report.json")
        )
        performance_results['reports']['detailed'] = report
        
        return performance_results
    
    def _run_accuracy_benchmarks(self, 
                               resolutions: List[Tuple[int, int]], 
                               methods: List[str]) -> Dict[str, Any]:
        """Run accuracy benchmarks across resolutions and methods."""
        print("Running accuracy benchmarks...")
        
        accuracy_results = {
            'resolutions': resolutions,
            'methods': methods,
            'individual_results': [],
            'reports': {}
        }
        
        for resolution in resolutions:
            height, width = resolution
            for method in methods:
                print(f"  Testing accuracy: {method} at {height}x{width}")
                
                # Run accuracy test
                result, metrics = self.accuracy_benchmark.benchmark_interpolation_accuracy(
                    source_resolution=resolution,
                    target_resolution=resolution,
                    method=method
                )
                
                accuracy_results['individual_results'].append((result, metrics))
        
        # Generate accuracy report
        report = create_accuracy_report(
            accuracy_results['individual_results'],
            output_file=os.path.join(self.config.output_dir, "accuracy_report.json")
        )
        accuracy_results['reports']['detailed'] = report
        
        return accuracy_results
    
    def _run_scalability_benchmarks(self, 
                                  resolutions: List[Tuple[int, int]], 
                                  methods: List[str]) -> Dict[str, Any]:
        """Run scalability benchmarks across resolutions and methods."""
        print("Running scalability benchmarks...")
        
        scalability_results = {
            'resolutions': resolutions,
            'methods': methods,
            'strong_scalability': {},
            'weak_scalability': {},
            'reports': {}
        }
        
        for resolution in resolutions[:2]:  # Limit scalability tests to first 2 resolutions to avoid excessive time
            height, width = resolution
            for method in methods[:1]:  # Test scalability with just one method to reduce time
                print(f" Testing scalability: {method} at {height}x{width}")
                
                # Strong scalability test
                strong_results = self.strong_scalability_tester.test_strong_scalability(
                    resolution=resolution,
                    worker_counts=[1, 2, 4],
                    method=method,
                    dask_client=self.config.dask_client
                )
                scalability_results['strong_scalability'][f'{method}_{height}x{width}'] = strong_results
                
                # Weak scalability test
                weak_results = self.weak_scalability_tester.test_weak_scalability(
                    base_resolution=resolution,
                    worker_scale_factors=[1, 2, 4],
                    method=method,
                    dask_client=self.config.dask_client
                )
                scalability_results['weak_scalability'][f'{method}_{height}x{width}'] = weak_results
        
        # Generate scalability analysis
        for test_type in ['strong_scalability', 'weak_scalability']:
            for test_name, test_data in scalability_results[test_type].items():
                analysis = analyze_scalability_results(test_data)
                scalability_results['reports'][test_name] = analysis
        
        return scalability_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate a summary of all benchmark results."""
        summary = {
            'total_performance_tests': len(self.results['performance'].get('individual_results', [])),
            'total_accuracy_tests': len(self.results['accuracy'].get('individual_results', [])),
            'total_scalability_tests': len(self.results['scalability'].get('strong_scalability', {})) + 
                                       len(self.results['scalability'].get('weak_scalability', {})) if 'scalability' in self.results else 0,
            'timestamp': self.results['timestamp'],
            'config': asdict(self.config)
        }
        
        # Add performance summary
        if 'performance' in self.results and 'reports' in self.results['performance']:
            perf_report = self.results['performance']['reports'].get('detailed', {})
            if 'aggregated_metrics' in perf_report:
                summary['performance_summary'] = perf_report['aggregated_metrics']
        
        # Add accuracy summary
        if 'accuracy' in self.results and 'reports' in self.results['accuracy']:
            acc_report = self.results['accuracy']['reports'].get('detailed', {})
            if 'summary' in acc_report:
                summary['accuracy_summary'] = acc_report['summary']
        
        # Add scalability summary
        if 'scalability' in self.results and 'reports' in self.results['scalability']:
            scalability_analysis = {}
            for test_name, analysis in self.results['scalability']['reports'].items():
                if 'scalability_score' in analysis:
                    scalability_analysis[test_name] = analysis['scalability_score']
            summary['scalability_summary'] = scalability_analysis
        
        return summary
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """Save benchmark results to a JSON file."""
        if filename is None:
            filename = f"unified_benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.config.output_dir, filename)
        
        # Convert dataclass instances to dictionaries for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"Benchmark results saved to: {filepath}")
        return filepath
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_serializable(item) for item in obj)
        elif hasattr(obj, '__dataclass_fields__'):
            # Handle dataclass instances
            return {key: self._make_serializable(getattr(obj, key)) 
                    for key in obj.__dataclass_fields__}
        elif isinstance(obj, BenchmarkResult):
            # Convert BenchmarkResult to dictionary
            return {
                'name': obj.name,
                'execution_time': obj.execution_time,
                'memory_usage': obj.memory_usage,
                'cpu_percent': obj.cpu_percent,
                'accuracy_error': obj.accuracy_error,
                'throughput': obj.throughput,
                'scalability_factor': obj.scalability_factor,
                'metadata': self._make_serializable(obj.metadata) if obj.metadata else None
            }
        elif isinstance(obj, np.ndarray):
            # Convert numpy arrays to lists
            return obj.tolist()
        else:
            # For other types, try to convert to string as fallback
            try:
                json.dumps(obj)  # Test if it's already serializable
                return obj
            except (TypeError, ValueError):
                return str(obj)


# Convenience function for running standard benchmark workflows
def run_standard_benchmark(resolutions: Optional[List[Tuple[int, int]]] = None,
                          methods: Optional[List[str]] = None,
                          output_dir: str = "./benchmark_results",
                          use_dask: bool = True,
                          dask_client: Optional['Client'] = None) -> str:
    """
    Run a standard benchmark workflow with common configurations.
    
    Args:
        resolutions: List of (height, width) tuples to test
        methods: List of interpolation methods to test
        output_dir: Directory to save results
        use_dask: Whether to use Dask for parallel processing
        dask_client: Dask client to use (if None, will create one)
    
    Returns:
        Path to the saved results file
    """
    # Set default resolutions if not provided
    if resolutions is None:
        resolutions = [(50, 100), (100, 200), (200, 400)]
    
    # Set default methods if not provided
    if methods is None:
        methods = ['bilinear', 'nearest']
    
    # Create configuration
    config = BenchmarkConfiguration(
        use_dask=use_dask,
        dask_client=dask_client,
        output_dir=output_dir
    )
    
    # Create and run benchmark
    runner = UnifiedBenchmarkRunner(config)
    runner.run_comprehensive_benchmark(resolutions, methods)
    
    # Save results
    results_path = runner.save_results()
    
    return results_path


if __name__ == "__main__":
    # Example usage
    # Define test resolutions
    test_resolutions = [(25, 50), (50, 100)]  # Small for testing purposes
    
    # Run standard benchmark
    results_file = run_standard_benchmark(
        resolutions=test_resolutions,
        methods=['bilinear', 'nearest'],
        output_dir="./benchmark_results"
    )
    
    print(f"Benchmark completed. Results saved to: {results_file}")