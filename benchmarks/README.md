# PyRegrid Benchmarking Suite

This directory contains the comprehensive benchmarking system for PyRegrid, designed to evaluate performance, accuracy, and scalability of regridding operations, particularly for high-resolution grids such as 3km global grids.

## Overview

The benchmarking suite includes:

- **Performance Metrics**: Execution time, memory usage, CPU utilization, throughput
- **Accuracy Validation**: RMSE, MAE, correlation, bias, and other accuracy metrics
- **Scalability Testing**: Strong and weak scalability analysis
- **High-Resolution Testing**: Specialized tests for large grid scenarios

## Components

### 1. Performance Metrics (`performance_metrics.py`)

Provides utilities for measuring performance characteristics:

- `HighResolutionBenchmark`: Main class for performance benchmarking
- `PerformanceCollector`: Collects detailed performance metrics
- `DistributedBenchmarkRunner`: Runs benchmarks across distributed workers
- `create_performance_report`: Generates performance reports

### 2. Accuracy Validation (`accuracy_validation.py`)

Validates the accuracy of regridding operations:

- `AccuracyBenchmark`: Tests interpolation accuracy
- `AnalyticalFieldGenerator`: Creates test fields with known solutions
- `create_accuracy_report`: Generates accuracy validation reports

### 3. Scalability Testing (`scalability_testing.py`)

Tests how performance scales with resources:

- `ScalabilityBenchmark`: General scalability testing
- `StrongScalabilityTester`: Fixed problem size, varying workers
- `WeakScalabilityTester`: Proportional problem size and workers
- `analyze_scalability_results`: Analyzes scalability data

## Running Benchmarks

### Basic Usage

```bash
# Run all benchmarks
python -m pytest benchmarks/ -v --benchmark

# Run with specific options
python -m pytest benchmarks/ -v --benchmark --benchmark-large

# Run with custom output directory
python -m pytest benchmarks/ -v --benchmark --benchmark-output-dir ./results
```

### Direct Python Usage

```python
from benchmarks import HighResolutionBenchmark, AccuracyBenchmark

# Performance benchmarking
benchmark = HighResolutionBenchmark(use_dask=True)
result = benchmark.benchmark_regridding_operation(
    source_data=your_data,
    target_coords=your_coords,
    method='bilinear'
)

# Accuracy validation
accuracy_benchmark = AccuracyBenchmark(threshold=1e-4)
result, metrics = accuracy_benchmark.benchmark_interpolation_accuracy(
    source_resolution=(100, 200),
    target_resolution=(100, 200),
    method='bilinear'
)
```

## Configuration Options

The benchmark suite supports several configuration options:

- `--benchmark`: Enable benchmark tests
- `--benchmark-large`: Run large-scale benchmark tests
- `--benchmark-output-dir`: Directory for benchmark results

## High-Resolution Testing

Special attention has been given to testing high-resolution scenarios like 3km global grids:

- Memory-efficient chunking strategies
- Lazy evaluation benefits
- Parallel processing optimization
- Scalability analysis for large grids

The `test_high_resolution_benchmarks.py` file contains specialized tests for these scenarios using proxy data to avoid memory issues during testing while maintaining realistic characteristics.

## Reports

The system generates detailed reports for analysis:

- Performance reports with execution times, memory usage, and throughput
- Accuracy reports with error metrics and validation results  
- Scalability reports with speedup and efficiency analysis

## Integration with Dask

The benchmarking system is fully integrated with Dask for parallel processing:

- Distributed benchmarking across multiple workers
- Memory management for large datasets
- Chunking strategy optimization
- Lazy evaluation support