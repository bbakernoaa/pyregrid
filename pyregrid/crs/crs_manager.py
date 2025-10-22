"""
Coordinate Reference System (CRS) management for PyRegrid.

This module provides comprehensive CRS handling functionality including:
- CRS detection and parsing from various sources
- Coordinate transformation between different CRS
- WGS 84 assumption policy implementation
- Error handling for ambiguous coordinate systems
- Coordinate validation and type detection
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union, cast

import numpy as np
import pyproj
import xarray as xr
from pyproj import CRS, Transformer
from pyproj.exceptions import CRSError


class CRSManager:
    """
    A class that handles all Coordinate Reference System operations for PyRegrid.
    
    This class implements a "strict but helpful" policy for coordinate systems:
    - Explicit CRS information is always prioritized
    - WGS 84 is assumed for lat/lon coordinates without explicit CRS
    - Errors are raised for ambiguous coordinate systems
    """
    
    def __init__(self):
        """Initialize the CRSManager."""
        self.wgs84_crs = CRS.from_epsg(4326)  # WGS 84 geographic coordinate system
        
    def detect_coordinate_system_type(self, crs: Optional[CRS]) -> str:
        """
        Detect if the coordinate system is geographic or projected.
        
        Args:
            crs: The coordinate reference system to analyze
            
        Returns:
            'geographic' or 'projected'
        """
        if crs is None:
            # If no CRS is provided, we can't determine the type
            # This should be handled by the calling function
            return "unknown"
        
        if crs.is_geographic:
            return "geographic"
        elif crs.is_projected:
            return "projected"
        else:
            # Could be other types like vertical CRS, compound CRS, etc.
            return "other"
    
    def parse_crs_from_xarray(self, ds: Union[xr.Dataset, xr.DataArray]) -> Optional[CRS]:
        """
        Parse CRS information from xarray objects.
        
        Args:
            ds: xarray Dataset or DataArray with potential CRS information
            
        Returns:
            Parsed CRS object or None if no CRS is found
        """
        # Check for CRS in various common locations
        # 1. Check for crs coordinate
        if hasattr(ds, 'coords') and 'crs' in ds.coords:
            crs_coord = ds.coords['crs']
            if hasattr(crs_coord, 'attrs') and 'crs_wkt' in crs_coord.attrs:
                try:
                    return CRS.from_wkt(crs_coord.attrs['crs_wkt'])
                except CRSError:
                    pass
            if hasattr(crs_coord, 'attrs') and 'epsg' in crs_coord.attrs:
                try:
                    return CRS.from_epsg(crs_coord.attrs['epsg'])
                except CRSError:
                    pass
        
        # 2. Check attributes of the dataset/array
        for attr_name in ['crs', 'grid_mapping', 'crs_wkt', 'spatial_ref']:
            if hasattr(ds, 'attrs') and attr_name in ds.attrs:
                try:
                    attr_value = ds.attrs[attr_name]
                    if isinstance(attr_value, str):
                        return CRS.from_string(attr_value)
                    elif hasattr(attr_value, 'attrs') and 'crs_wkt' in attr_value.attrs:
                        return CRS.from_wkt(attr_value.attrs['crs_wkt'])
                except (CRSError, TypeError):
                    continue
        
        # 3. Check for grid_mapping variable
        if hasattr(ds, 'attrs') and 'grid_mapping' in ds.attrs:
            grid_mapping_name = ds.attrs['grid_mapping']
            if hasattr(ds, 'coords') and grid_mapping_name in ds.coords:
                grid_mapping_var = ds.coords[grid_mapping_name]
                try:
                    return CRS.from_cf(grid_mapping_var.attrs)
                except CRSError:
                    pass
        
        return None
    
    def parse_crs_from_dataframe(self, df) -> Optional[CRS]:
        """
        Parse CRS information from pandas DataFrame.
        
        Args:
            df: pandas DataFrame with potential CRS information
            
        Returns:
            Parsed CRS object or None if no CRS is found
        """
        # Check common DataFrame attributes or metadata
        if hasattr(df, 'attrs') and 'crs' in df.attrs:
            try:
                return CRS.from_string(df.attrs['crs'])
            except (CRSError, TypeError):
                pass
        
        # Check for common coordinate column names
        # This is a heuristic approach based on common column names
        lat_cols = [col for col in df.columns if 'lat' in col.lower() or 'latitude' in col.lower()]
        lon_cols = [col for col in df.columns if 'lon' in col.lower() or 'lng' in col.lower() or 'longitude' in col.lower()]
        
        if lat_cols and lon_cols:
            # If we have latitude and longitude columns, assume WGS 84
            # but issue a warning since no explicit CRS was provided
            return self.wgs84_crs
        
        return None
    
    def validate_coordinate_arrays(self, 
                                 x_coords: np.ndarray, 
                                 y_coords: np.ndarray, 
                                 crs: Optional[CRS] = None) -> bool:
        """
        Validate coordinate arrays and detect potential issues.
        
        Args:
            x_coords: X coordinate array (longitude or easting)
            y_coords: Y coordinate array (latitude or northing)
            crs: Optional CRS to validate against
            
        Returns:
            True if coordinates appear valid, False otherwise
        """
        # Check for NaN or infinite values
        if np.any(np.isnan(x_coords)) or np.any(np.isnan(y_coords)):
            return False
        if np.any(np.isinf(x_coords)) or np.any(np.isinf(y_coords)):
            return False
        
        # Check coordinate ranges if we know it's geographic
        if crs and crs.is_geographic:
            # For geographic coordinates, check typical ranges
            # Note: these are typical but not absolute bounds
            if np.any(x_coords < -360) or np.any(x_coords > 360):
                return False
            if np.any(y_coords < -90) or np.any(y_coords > 90):
                return False
        
        # Check if arrays have the same shape
        if x_coords.shape != y_coords.shape:
            return False
        
        return True
    
    def detect_crs_from_coordinates(self,
                                  x_coords: np.ndarray,
                                  y_coords: np.ndarray,
                                  x_name: str = 'x',
                                  y_name: str = 'y') -> Optional[CRS]:
        """
        Attempt to detect CRS from coordinate names and values.
        
        Args:
            x_coords: X coordinate array
            y_coords: Y coordinate array
            x_name: Name of the x coordinate variable
            y_name: Name of the y coordinate variable
            
        Returns:
            Detected CRS or None if uncertain
        """
        # Check if coordinate names suggest geographic coordinates
        # Only consider the specific geographic names, not generic 'x' and 'y'
        lat_names = ['lat', 'latitude', 'ycoords']  # Exclude 'y' to be more strict
        lon_names = ['lon', 'longitude', 'lng', 'xcoords']  # Exclude 'x' to be more strict
        
        is_lat_lon = (x_name.lower() in lon_names and y_name.lower() in lat_names) or \
                     (x_name.lower() in lat_names and y_name.lower() in lon_names)
        
        if is_lat_lon:
            # Check if coordinate values are within typical geographic ranges
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            # If values are within geographic ranges, assume WGS 84
            if -360 <= x_min <= 360 and -360 <= x_max <= 360 and \
               -90 <= y_min <= 90 and -90 <= y_max <= 90:
                # Issue a warning about the assumption
                warnings.warn(
                    f"Coordinates named '{x_name}' and '{y_name}' appear to be "
                    f"geographic (lat/lon) but no explicit CRS was provided. "
                    f"Assuming WGS 84 (EPSG:4326) coordinate system.",
                    UserWarning
                )
                return self.wgs84_crs
        
        return None
    
    def transform_coordinates(self, 
                            x_coords: np.ndarray, 
                            y_coords: np.ndarray, 
                            source_crs: Union[CRS, str], 
                            target_crs: Union[CRS, str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform coordinates from one CRS to another.
        
        Args:
            x_coords: X coordinate array
            y_coords: Y coordinate array
            source_crs: Source coordinate reference system
            target_crs: Target coordinate reference system
            
        Returns:
            Tuple of (transformed_x, transformed_y) coordinate arrays
        """
        # Ensure CRS objects are properly created
        if isinstance(source_crs, str):
            source_crs = CRS.from_string(source_crs)
        if isinstance(target_crs, str):
            target_crs = CRS.from_string(target_crs)
        
        # Create transformer
        transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
        
        # Perform transformation
        x_transformed, y_transformed = transformer.transform(x_coords, y_coords)
        
        return x_transformed, y_transformed
    
    def get_crs_from_source(self,
                           source: Union[xr.Dataset, xr.DataArray, Any],
                           x_coords: np.ndarray,
                           y_coords: np.ndarray,
                           x_name: str = 'x',
                           y_name: str = 'y') -> Optional[CRS]:
        """
        Get CRS from various source types with the "strict but helpful" policy.
        
        Args:
            source: The data source (xarray Dataset/DataArray, DataFrame, or other)
            x_coords: X coordinate array
            y_coords: Y coordinate array
            x_name: Name of the x coordinate variable
            y_name: Name of the y coordinate variable
            
        Returns:
            Detected or provided CRS, or None if uncertain
            
        Raises:
            ValueError: If CRS is ambiguous and cannot be determined safely
        """
        # Try to parse CRS from the source object first
        if isinstance(source, (xr.Dataset, xr.DataArray)):
            crs = self.parse_crs_from_xarray(source)
            if crs is not None:
                return crs
        # For pandas DataFrame - check if it has 'columns' attribute
        elif hasattr(source, 'columns') and hasattr(source, 'attrs'):
            crs = self.parse_crs_from_dataframe(source)
            if crs is not None:
                return crs
        
        # If no explicit CRS found, try to detect from coordinates and names
        detected_crs = self.detect_crs_from_coordinates(x_coords, y_coords, x_name, y_name)
        if detected_crs is not None:
            return detected_crs
        
        # Check if coordinate names suggest lat/lon and values match typical ranges
        lat_names = ['lat', 'latitude', 'y']
        lon_names = ['lon', 'longitude', 'lng', 'x']
        
        is_lat_lon = (x_name.lower() in lon_names and y_name.lower() in lat_names) or \
                     (x_name.lower() in lat_names and y_name.lower() in lon_names)
        
        if is_lat_lon:
            # Check if coordinate values are within typical geographic ranges
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            if -360 <= x_min <= 360 and -360 <= x_max <= 360 and \
               -90 <= y_min <= 90 and -90 <= y_max <= 90:
                # Values are within geographic ranges, assume WGS 84
                warnings.warn(
                    f"Coordinates named '{x_name}' and '{y_name}' appear to be "
                    f"geographic (lat/lon) but no explicit CRS was provided. "
                    f"Assuming WGS 84 (EPSG:4326) coordinate system.",
                    UserWarning
                )
                return self.wgs84_crs
            else:
                # Values are outside geographic range, raise error
                raise ValueError(
                    f"Coordinate variables named '{x_name}' and '{y_name}' suggest "
                    f"geographic coordinates (lat/lon), but the coordinate values "
                    f"({x_min:.6f} to {x_max:.6f}, {y_min:.6f} to {y_max:.6f}) are "
                    f"outside the typical geographic range. Please provide an explicit "
                    f"coordinate reference system (CRS) to clarify the coordinate system."
                )
        
        # If coordinates are not clearly lat/lon and no CRS is provided, raise an error
        # This implements the "strict" part of the "strict but helpful" policy
        raise ValueError(
            f"No coordinate reference system (CRS) information found for coordinates "
            f"'{x_name}' and '{y_name}'. Coordinate names do not clearly indicate "
            f"geographic coordinates (latitude/longitude). Please provide explicit "
            f"CRS information to avoid incorrect assumptions about the coordinate system."
        )
    
    def ensure_crs_compatibility(self, 
                               source_crs: Optional[CRS], 
                               target_crs: Optional[CRS]) -> Tuple[CRS, CRS]:
        """
        Ensure both source and target CRS are defined and compatible for transformation.
        
        Args:
            source_crs: Source CRS or None
            target_crs: Target CRS or None
            
        Returns:
            Tuple of (source_crs, target_crs) both as valid CRS objects
        """
        if source_crs is None:
            raise ValueError("Source CRS must be defined for coordinate transformation")
        
        if target_crs is None:
            raise ValueError("Target CRS must be defined for coordinate transformation")
        
        return source_crs, target_crs


# Global instance for convenience
crs_manager = CRSManager()