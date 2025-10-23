"""
Benchmarking module for PyRegrid.

This module provides comprehensive benchmarking capabilities for
performance, accuracy, and scalability testing of regridding operations.
"""
from .benchmark_base import BenchmarkResult, BenchmarkRunner, AccuracyValidator, ScalabilityTester
from .performance_metrics import (
    PerformanceMetrics,
    PerformanceCollector,
    HighResolutionBenchmark,
    DistributedBenchmarkRunner,
    create_performance_report
)
from .accuracy_validation import (
    AccuracyMetrics,
    AccuracyBenchmark,
    AnalyticalFieldGenerator,
    create_accuracy_report
)
from .scalability_testing import (
    ScalabilityMetrics,
    ScalabilityBenchmark,
    StrongScalabilityTester,
    WeakScalabilityTester,
    analyze_scalability_results
)
from .unified_benchmark import (
    BenchmarkConfiguration,
    UnifiedBenchmarkRunner,
    run_standard_benchmark
)

__all__ = [
    # Base benchmarking
    'BenchmarkResult', 'BenchmarkRunner', 'AccuracyValidator', 'ScalabilityTester',
    
    # Performance metrics
    'PerformanceMetrics', 'PerformanceCollector', 'HighResolutionBenchmark',
    'DistributedBenchmarkRunner', 'create_performance_report',
    
    # Accuracy validation
    'AccuracyMetrics', 'AccuracyBenchmark', 'AnalyticalFieldGenerator', 'create_accuracy_report',
    
    # Scalability testing
    'ScalabilityMetrics', 'ScalabilityBenchmark', 'StrongScalabilityTester',
    'WeakScalabilityTester', 'analyze_scalability_results',
    
    # Unified benchmarking
    'BenchmarkConfiguration', 'UnifiedBenchmarkRunner', 'run_standard_benchmark',
]