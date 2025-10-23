"""
Configuration for benchmark tests using pytest.

This module provides fixtures and configuration for running benchmark tests.
"""
import pytest
import numpy as np
import dask
from dask.distributed import Client, LocalCluster
import tempfile
import os
from typing import Generator, Optional


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


def pytest_configure(config):
    """Configure pytest for benchmark tests."""
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )
    config.addinivalue_line(
        "markers", "large_benchmark: mark test as a large-scale benchmark"
    )


@pytest.fixture(scope="session")
def dask_client(request) -> Generator[Client, None, None]:
    """
    Create a Dask client for benchmark tests.
    
    This fixture creates a local Dask cluster for parallel processing during benchmarks.
    """
    # Check if we're running benchmark tests
    if not request.config.getoption("--benchmark"):
        # For non-benchmark tests, use default dask scheduler
        with dask.config.set(scheduler='threads'):
            yield None
        return
    
    # Create a local cluster for benchmarks
    cluster = LocalCluster(
        n_workers=2,  # Start with 2 workers for benchmarks
        threads_per_worker=2,
        processes=False,  # Use threads instead of processes for better memory sharing
        dashboard_address=None  # Disable dashboard to reduce overhead
    )
    
    client = Client(cluster)
    
    try:
        yield client
    finally:
        client.close()
        cluster.close()


@pytest.fixture(scope="function")
def benchmark_data_small() -> tuple:
    """
    Create small benchmark data for quick tests.
    
    Returns:
        Tuple of (source_data, target_coords) for testing
    """
    # Small resolution for quick tests
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
    """
    Create large benchmark data for comprehensive tests.
    
    Returns:
        Tuple of (source_data, target_coords) for testing
    """
    # Check if large benchmarks are enabled
    try:
        if hasattr(pytest, 'config') and not pytest.config.getoption("--benchmark-large"):
            # Return small data if large benchmarks are not requested
            height, width = 50, 100
        else:
            # Large resolution for comprehensive tests
            height, width = 200, 400
    except:
        # If pytest.config is not available, use default size
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
    
    # Create target coordinates in the correct format for map_coordinates
    target_lon = np.linspace(-180, 180, width//2)  # Target is half resolution
    target_lat = np.linspace(-90, 90, height//2)
    
    # Create coordinate arrays for map_coordinates
    target_y, target_x = np.meshgrid(target_lat, target_lon, indexing='ij')
    
    # Flatten and create coordinate arrays
    y_coords = target_y.ravel()
    x_coords = target_x.ravel()
    
    # Convert world coordinates to array indices
    x_indices = ((x_coords + 180) / 360 * (width//2 - 1)).astype(int)
    y_indices = ((y_coords + 90) / 180 * (height//2 - 1)).astype(int)
    
    return source_data, (y_indices, x_indices)


@pytest.fixture(scope="session")
def benchmark_output_dir(request) -> str:
    """
    Create a directory for storing benchmark results.
    
    Returns:
        Path to the benchmark results directory
    """
    output_dir = request.config.getoption("--benchmark-output-dir")
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    return output_dir


@pytest.fixture(scope="function")
def temporary_benchmark_file(benchmark_output_dir) -> Generator[str, None, None]:
    """
    Create a temporary file for benchmark results.
    
    Args:
        benchmark_output_dir: Directory for benchmark results
        
    Yields:
        Path to temporary file
    """
    temp_file = os.path.join(
        benchmark_output_dir, 
        f"temp_benchmark_{os.getpid()}_{id(tempfile)}.json"
    )
    
    try:
        yield temp_file
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)