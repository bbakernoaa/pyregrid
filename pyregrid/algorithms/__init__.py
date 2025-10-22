"""
Algorithms module for PyRegrid.

This module contains implementations of various interpolation and regridding algorithms:
- Gridded data regridding (bilinear, cubic, nearest neighbor, conservative)
- Scattered data interpolation (IDW, linear, etc.)
"""

from .interpolators import (
    BaseInterpolator,
    BilinearInterpolator,
    CubicInterpolator,
    NearestInterpolator
)

__all__ = [
    'BaseInterpolator',
    'BilinearInterpolator',
    'CubicInterpolator',
    'NearestInterpolator'
]