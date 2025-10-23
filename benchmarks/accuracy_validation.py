"""
Accuracy validation utilities for PyRegrid benchmarking.

This module provides utilities for validating the accuracy of regridding operations
by comparing results against known analytical solutions or reference implementations.
"""
import numpy as np
import dask.array as da
from typing import Tuple, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
import warnings

from .benchmark_base import BenchmarkResult
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .benchmark_base import AccuracyValidator


class AccuracyValidator:
    """Enhanced class for validating accuracy of regridding operations."""
    
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
        return float(np.sqrt(mse))
    
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
        return float(np.max(diff))
    
    @staticmethod
    def calculate_correlation(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate correlation coefficient between actual and expected arrays."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return 0.0  # If all values are NaN, return zero correlation
        
        actual_clean = actual[mask]
        expected_clean = expected[mask]
        
        if len(actual_clean) < 2:
            return 0.0  # Need at least 2 points for correlation
        
        # Calculate correlation using numpy
        correlation_matrix = np.corrcoef(actual_clean, expected_clean)
        return float(correlation_matrix[0, 1])
    
    @staticmethod
    def calculate_bias(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate bias (mean error) between actual and expected arrays."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return 0.0  # If all values are NaN, return zero bias
        
        diff = actual[mask] - expected[mask]
        return float(np.mean(diff))
    
    @staticmethod
    def calculate_relative_rmse(actual: np.ndarray, expected: np.ndarray) -> float:
        """Calculate relative RMSE normalized by the range of expected values."""
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch: {actual.shape} vs {expected.shape}")
        
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        if not np.any(mask):
            return float('inf')  # If all values are NaN, return infinity error
        
        rmse = AccuracyValidator.calculate_rmse(actual, expected)
        expected_range = np.max(expected[mask]) - np.min(expected[mask])
        
        if expected_range == 0:
            return float('inf')  # Avoid division by zero
        
        return rmse / expected_range


@dataclass
class AccuracyMetrics:
    """Data class to store accuracy validation metrics."""
    rmse: float
    mae: float
    max_error: float
    mean_error: float
    std_error: float
    correlation: float
    bias: float
    relative_rmse: float  # RMSE normalized by range of values
    n_valid_points: int
    n_total_points: int
    accuracy_threshold: float


class AnalyticalFieldGenerator:
    """Class for generating analytical test fields with known solutions."""
    
    @staticmethod
    def sine_wave_field(height: int, width: int, 
                       lon_range: Tuple[float, float] = (-180, 180),
                       lat_range: Tuple[float, float] = (-90, 90)) -> np.ndarray:
        """Generate a test field with sine wave patterns."""
        lon = np.linspace(lon_range[0], lon_range[1], width)
        lat = np.linspace(lat_range[0], lat_range[1], height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create test pattern: combination of sine waves
        field = (np.sin(np.radians(lat_grid)) * 
                np.cos(np.radians(lon_grid)) + 
                0.5 * np.sin(2 * np.radians(lat_grid)) * 
                np.cos(2 * np.radians(lon_grid)))
        
        return field
    
    @staticmethod
    def gaussian_bump_field(height: int, width: int,
                           lon_range: Tuple[float, float] = (-180, 180),
                           lat_range: Tuple[float, float] = (-90, 90),
                           center: Tuple[float, float] = (0, 0),
                           sigma: float = 30.0) -> np.ndarray:
        """Generate a test field with a Gaussian bump."""
        lon = np.linspace(lon_range[0], lon_range[1], width)
        lat = np.linspace(lat_range[0], lat_range[1], height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Calculate distance from center
        dist_sq = (lon_grid - center[0])**2 + (lat_grid - center[1])**2
        field = np.exp(-dist_sq / (2 * sigma**2))
        
        return field
    
    @staticmethod
    def polynomial_field(height: int, width: int,
                        lon_range: Tuple[float, float] = (-180, 180),
                        lat_range: Tuple[float, float] = (-90, 90)) -> np.ndarray:
        """Generate a test field with polynomial patterns."""
        lon = np.linspace(lon_range[0], lon_range[1], width)
        lat = np.linspace(lat_range[0], lat_range[1], height)
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        
        # Create polynomial test pattern
        field = (0.1 * lon_grid**2 + 
                0.05 * lat_grid**2 + 
                0.01 * lon_grid * lat_grid + 
                0.3 * np.sin(0.1 * lon_grid) + 
                0.2 * np.cos(0.1 * lat_grid))
        
        return field


class AccuracyBenchmark:
    """Class for running accuracy validation benchmarks."""
    
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold
        self.validator = AccuracyValidator()
        self.analytical_generator = AnalyticalFieldGenerator()
    
    def benchmark_interpolation_accuracy(self, 
                                       source_resolution: Tuple[int, int],
                                       target_resolution: Tuple[int, int],
                                       method: str = 'bilinear',
                                       field_type: str = 'sine_wave',
                                       **kwargs) -> Tuple[BenchmarkResult, AccuracyMetrics]:
        """
        Benchmark interpolation accuracy by comparing against analytical solution.
        
        Args:
            source_resolution: Source grid resolution (height, width)
            target_resolution: Target grid resolution (height, width)
            method: Interpolation method to test
            field_type: Type of analytical field to use
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (BenchmarkResult, AccuracyMetrics)
        """
        from pyregrid.algorithms.interpolators import BilinearInterpolator, NearestInterpolator
        
        # Generate analytical field at high resolution
        source_field = self.analytical_generator.sine_wave_field(
            source_resolution[0], source_resolution[1]
        )
        
        # Generate the same field at target resolution (true solution)
        true_field = self.analytical_generator.sine_wave_field(
            target_resolution[0], target_resolution[1]
        )
        
        # Get interpolator
        interpolator_map = {
            'bilinear': BilinearInterpolator,
            'nearest': NearestInterpolator,
        }
        
        if method not in interpolator_map:
            raise ValueError(f"Unsupported method: {method}")
        
        interpolator = interpolator_map[method]()
        
        # Create target coordinates
        target_lon = np.linspace(-180, 180, target_resolution[1])
        target_lat = np.linspace(-90, 90, target_resolution[0])
        target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
        
        # Convert world coordinates to array indices for map_coordinates
        # map_coordinates expects array indices, not world coordinates
        x_indices = ((target_lon_grid + 180) / 360 * (source_field.shape[1] - 1)).astype(int)
        y_indices = ((target_lat_grid + 90) / 180 * (source_field.shape[0] - 1)).astype(int)
        
        # Interpolate from source to target using array indices
        interpolated_field = interpolator.interpolate(
            source_field,
            (y_indices, x_indices)
        )
        
        if isinstance(interpolated_field, da.Array):
            interpolated_field = interpolated_field.compute()
        
        # Reshape to match target grid
        interpolated_field = interpolated_field.reshape(target_resolution)
        
        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics(
            interpolated_field, 
            true_field,
            self.threshold
        )
        
        # Create benchmark result
        benchmark_result = BenchmarkResult(
            name=f"accuracy_{method}_{field_type}_{source_resolution[0]}x{source_resolution[1]}_to_{target_resolution[0]}x{target_resolution[1]}",
            execution_time=0.0,  # Accuracy validation doesn't measure time
            memory_usage=0.0,
            cpu_percent=0.0,
            accuracy_error=accuracy_metrics.rmse,
            metadata={
                'method': method,
                'field_type': field_type,
                'source_resolution': source_resolution,
                'target_resolution': target_resolution,
                'accuracy_metrics': accuracy_metrics
            }
        )
        
        return benchmark_result, accuracy_metrics
    
    def calculate_accuracy_metrics(self, actual: np.ndarray, expected: np.ndarray, 
                                 threshold: float = 1e-6) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics."""
        # Handle NaN values by ignoring them in the calculation
        mask = ~(np.isnan(actual) | np.isnan(expected))
        
        n_total_points = actual.size
        n_valid_points = np.sum(mask) if mask.any() else 0
        
        if n_valid_points == 0:
            # If no valid points, return infinite errors
            return AccuracyMetrics(
                rmse=float('inf'),
                mae=float('inf'),
                max_error=float('inf'),
                mean_error=float('inf'),
                std_error=float('inf'),
                correlation=0.0,
                bias=0.0,
                relative_rmse=float('inf'),
                n_valid_points=0,
                n_total_points=n_total_points,
                accuracy_threshold=threshold
            )
        
        actual_valid = actual[mask]
        expected_valid = expected[mask]
        
        # Calculate individual metrics
        rmse = self.validator.calculate_rmse(actual, expected)
        mae = self.validator.calculate_mae(actual, expected)
        max_error = self.validator.calculate_max_error(actual, expected)
        correlation = self.validator.calculate_correlation(actual, expected)
        bias = self.validator.calculate_bias(actual, expected)
        relative_rmse = self.validator.calculate_relative_rmse(actual, expected)
        
        # Calculate mean and std of errors
        errors = actual_valid - expected_valid
        mean_error = float(np.mean(errors))
        std_error = float(np.std(errors))
        
        return AccuracyMetrics(
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            mean_error=mean_error,
            std_error=std_error,
            correlation=correlation,
            bias=bias,
            relative_rmse=relative_rmse,
            n_valid_points=n_valid_points,
            n_total_points=n_total_points,
            accuracy_threshold=threshold
        )
    
    def run_accuracy_convergence_test(self, 
                                    resolutions: List[Tuple[int, int]],
                                    method: str = 'bilinear',
                                    field_type: str = 'sine_wave') -> List[Tuple[BenchmarkResult, AccuracyMetrics]]:
        """
        Run accuracy convergence test across multiple resolutions.
        
        Args:
            resolutions: List of (height, width) tuples for different resolutions
            method: Interpolation method to test
            field_type: Type of analytical field to use
            
        Returns:
            List of (BenchmarkResult, AccuracyMetrics) tuples
        """
        results = []
        
        # Use the highest resolution as the reference
        max_height = max(r[0] for r in resolutions)
        max_width = max(r[1] for r in resolutions)
        reference_field = self.analytical_generator.sine_wave_field(max_height, max_width)
        
        for height, width in resolutions:
            # Interpolate reference field to current resolution
            target_lon = np.linspace(-180, 180, width)
            target_lat = np.linspace(-90, 90, height)
            target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
            
            # Convert world coordinates to array indices for map_coordinates
            # map_coordinates expects array indices, not world coordinates
            x_indices = ((target_lon_grid + 180) / 360 * (reference_field.shape[1] - 1)).astype(int)
            y_indices = ((target_lat_grid + 90) / 180 * (reference_field.shape[0] - 1)).astype(int)
            
            # Create interpolator and interpolate using array indices
            from pyregrid.algorithms.interpolators import BilinearInterpolator, NearestInterpolator
            interpolator_map = {
                'bilinear': BilinearInterpolator,
                'nearest': NearestInterpolator,
            }
            
            if method not in interpolator_map:
                raise ValueError(f"Unsupported method: {method}")
            
            interpolator = interpolator_map[method]()
            interpolated_field = interpolator.interpolate(
                reference_field,
                (y_indices, x_indices)
            )
            
            if isinstance(interpolated_field, da.Array):
                interpolated_field = interpolated_field.compute()
            
            interpolated_field = interpolated_field.reshape((height, width))
            
            # Calculate accuracy against the analytical solution at this resolution
            analytical_field = self.analytical_generator.sine_wave_field(height, width)
            accuracy_metrics = self.calculate_accuracy_metrics(
                interpolated_field, 
                analytical_field,
                self.threshold
            )
            
            benchmark_result = BenchmarkResult(
                name=f"convergence_{method}_{field_type}_{height}x{width}",
                execution_time=0.0,
                memory_usage=0.0,
                cpu_percent=0.0,
                accuracy_error=accuracy_metrics.rmse,
                metadata={
                    'method': method,
                    'field_type': field_type,
                    'resolution': (height, width),
                    'accuracy_metrics': accuracy_metrics
                }
            )
            
            results.append((benchmark_result, accuracy_metrics))
        
        return results
    
    def validate_round_trip_accuracy(self, 
                                   resolution: Tuple[int, int],
                                   method: str = 'bilinear',
                                   max_error_threshold: float = 1e-3) -> Tuple[bool, AccuracyMetrics]:
        """
        Validate round-trip accuracy by interpolating to a different grid and back.
        
        Args:
            resolution: Grid resolution (height, width)
            method: Interpolation method to test
            max_error_threshold: Maximum acceptable error threshold
            
        Returns:
            Tuple of (is_accurate, AccuracyMetrics)
        """
        # Generate test field
        original_field = self.analytical_generator.sine_wave_field(
            resolution[0], resolution[1]
        )
        
        # Create intermediate grid with different resolution
        intermediate_resolution = (resolution[0] * 2, resolution[1] * 2)
        intermediate_lon = np.linspace(-180, 180, intermediate_resolution[1])
        intermediate_lat = np.linspace(-90, 90, intermediate_resolution[0])
        int_lon_grid, int_lat_grid = np.meshgrid(intermediate_lon, intermediate_lat)
        
        # Convert world coordinates to array indices for map_coordinates
        # map_coordinates expects array indices, not world coordinates
        int_x_indices = ((int_lon_grid + 180) / 360 * (original_field.shape[1] - 1)).astype(int)
        int_y_indices = ((int_lat_grid + 90) / 180 * (original_field.shape[0] - 1)).astype(int)
        
        # Create interpolator
        from pyregrid.algorithms.interpolators import BilinearInterpolator, NearestInterpolator
        interpolator_map = {
            'bilinear': BilinearInterpolator,
            'nearest': NearestInterpolator,
        }
        
        if method not in interpolator_map:
            raise ValueError(f"Unsupported method: {method}")
        
        interpolator = interpolator_map[method]()
        
        # Interpolate to intermediate grid using array indices
        intermediate_field = interpolator.interpolate(
            original_field,
            (int_y_indices, int_x_indices)
        )
        
        if isinstance(intermediate_field, da.Array):
            intermediate_field = intermediate_field.compute()
        
        intermediate_field = intermediate_field.reshape(intermediate_resolution)
        
        # Create points for interpolation back to original grid
        orig_lon = np.linspace(-180, 180, resolution[1])
        orig_lat = np.linspace(-90, 90, resolution[0])
        orig_lon_grid, orig_lat_grid = np.meshgrid(orig_lon, orig_lat)
        
        # Convert world coordinates to array indices for map_coordinates
        # map_coordinates expects array indices, not world coordinates
        orig_x_indices = ((orig_lon_grid + 180) / 360 * (intermediate_field.shape[1] - 1)).astype(int)
        orig_y_indices = ((orig_lat_grid + 90) / 180 * (intermediate_field.shape[0] - 1)).astype(int)
        
        # Interpolate back to original grid using array indices
        round_trip_field = interpolator.interpolate(
            intermediate_field,
            (orig_y_indices, orig_x_indices)
        )
        
        if isinstance(round_trip_field, da.Array):
            round_trip_field = round_trip_field.compute()
        
        round_trip_field = round_trip_field.reshape(resolution)
        
        # Calculate accuracy metrics
        accuracy_metrics = self.calculate_accuracy_metrics(
            round_trip_field,
            original_field,
            max_error_threshold
        )
        
        # Check if accuracy is within threshold
        is_accurate = accuracy_metrics.rmse <= max_error_threshold
        
        return is_accurate, accuracy_metrics


def create_accuracy_report(results: List[Tuple[BenchmarkResult, AccuracyMetrics]], 
                         output_file: Optional[str] = None) -> Dict[str, any]:
    """
    Create an accuracy validation report from benchmark results.
    
    Args:
        results: List of (BenchmarkResult, AccuracyMetrics) tuples
        output_file: Optional file path to save the report
        
    Returns:
        Dictionary containing the accuracy report
    """
    if not results:
        return {"error": "No results to report"}
    
    report = {
        "summary": {
            "total_tests": len(results),
            "passing_tests": 0,
            "failing_tests": 0,
            "average_rmse": 0.0,
            "min_rmse": float('inf'),
            "max_rmse": 0.0,
        },
        "detailed_results": [],
        "accuracy_trends": {}
    }
    
    rmses = []
    for benchmark_result, accuracy_metrics in results:
        rmses.append(accuracy_metrics.rmse)
        
        # Check if test passes threshold
        passes = accuracy_metrics.rmse <= accuracy_metrics.accuracy_threshold
        if passes:
            report["summary"]["passing_tests"] += 1
        else:
            report["summary"]["failing_tests"] += 1
        
        result_entry = {
            "name": benchmark_result.name,
            "passes": passes,
            "accuracy_metrics": {
                "rmse": accuracy_metrics.rmse,
                "mae": accuracy_metrics.mae,
                "max_error": accuracy_metrics.max_error,
                "mean_error": accuracy_metrics.mean_error,
                "std_error": accuracy_metrics.std_error,
                "correlation": accuracy_metrics.correlation,
                "bias": accuracy_metrics.bias,
                "relative_rmse": accuracy_metrics.relative_rmse,
                "n_valid_points": accuracy_metrics.n_valid_points,
                "n_total_points": accuracy_metrics.n_total_points,
            },
            "metadata": benchmark_result.metadata
        }
        report["detailed_results"].append(result_entry)
    
    if rmses:
        report["summary"]["average_rmse"] = float(np.mean(rmses))
        report["summary"]["min_rmse"] = float(np.min(rmses))
        report["summary"]["max_rmse"] = float(np.max(rmses))
    
    # Identify accuracy trends if metadata contains resolution information
    resolutions = []
    resolution_rmses = []
    for benchmark_result, accuracy_metrics in results:
        if 'resolution' in benchmark_result.metadata:
            resolutions.append(benchmark_result.metadata['resolution'])
            resolution_rmses.append(accuracy_metrics.rmse)
    
    if resolutions and resolution_rmses:
        report["accuracy_trends"] = {
            "resolutions": resolutions,
            "rmses": resolution_rmses,
            "convergence_observed": len(resolutions) > 1 and resolution_rmses[-1] < resolution_rmses[0] if len(resolution_rmses) > 1 else None
        }
    
    # Save to file if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    return report