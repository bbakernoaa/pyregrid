"""
Coordinate Reference System (CRS) management module for PyRegrid.

This module handles all CRS parsing, coordinate transformation,
and geospatial operations using pyproj as the sole dependency.
"""

from .crs_manager import CRSManager, crs_manager

__all__ = ['CRSManager', 'crs_manager']