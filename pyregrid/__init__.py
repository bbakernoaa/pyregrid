"""
PyRegrid: A modern, minimal dependency library for geospatial interpolation and regridding.

This library provides xarray-native methods for:
- Grid-to-grid regridding (bilinear, cubic, nearest neighbor)
- Grid-to-point interpolation 
- Point-to-grid interpolation (IDW and other methods)
- Dask integration for out-of-core processing

The primary interface is accessed via the .pyregrid accessor on xarray objects.
"""

__version__ = "0.1.0"

# Import core classes and functions
from .core import GridRegridder # noqa: F401
from .point_interpolator import PointInterpolator  # noqa: F401
from .accessors import PyRegridAccessor  # noqa: F401
from .utils.grid_from_points import grid_from_points  # noqa: F401
from .scattered_interpolation import (
    NeighborBasedInterpolator,
    TriangulationBasedInterpolator,
    HybridSpatialIndex,
    idw_interpolation,
    moving_average_interpolation,
    gaussian_interpolation,
    exponential_interpolation,
    linear_interpolation
)  # noqa: F401

# Register the accessor automatically when the package is imported
import xarray as xr  # noqa: E402

# Import and expose Dask functionality if available
try:
    from .dask import DaskRegridder, ChunkingStrategy, MemoryManager, ParallelProcessor
    HAS_DASK = True
except ImportError:
    # Dask is optional, so if it's not available, provide placeholder classes
    class DaskRegridder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "DaskRegridder requires Dask to be installed. "
                "Install with `pip install pyregrid[dask]`"
            )

    class ChunkingStrategy:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChunkingStrategy requires Dask to be installed. "
                "Install with `pip install pyregrid[dask]`"
            )

    class MemoryManager:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MemoryManager requires Dask to be installed. "
                "Install with `pip install pyregrid[dask]`"
            )

    class ParallelProcessor:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ParallelProcessor requires Dask to be installed. "
                "Install with `pip install pyregrid[dask]`"
            )

    HAS_DASK = False

# Public API
__all__ = [
    "GridRegridder",
    "PointInterpolator",
    "PyRegridAccessor",
    "grid_from_points",
    "DaskRegridder",
    "ChunkingStrategy",
    "MemoryManager",
    "ParallelProcessor",
    "HAS_DASK",
    "NeighborBasedInterpolator",
    "TriangulationBasedInterpolator",
    "HybridSpatialIndex",
    "idw_interpolation",
    "moving_average_interpolation",
    "gaussian_interpolation",
    "exponential_interpolation",
    "linear_interpolation"
]

# Make the accessor available on xarray objects
# The accessor is already registered in the accessor.py module with decorators