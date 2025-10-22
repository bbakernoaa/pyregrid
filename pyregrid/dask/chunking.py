"""
Chunking strategies for Dask-based regridding operations.

This module provides utilities for determining optimal chunk sizes and strategies
for different data types and sizes when processing with Dask.
"""
import xarray as xr
import numpy as np
from typing import Union, Tuple, Optional, Dict, Any
import math


class ChunkingStrategy:
    """
    A utility class for determining optimal chunking strategies for Dask arrays.
    """
    
    def __init__(self):
        self.default_chunk_size = 1000000  # 1M elements per chunk by default
        self.max_chunk_size = 10000000    # 10M elements max per chunk
        self.min_chunk_size = 10000       # 10K elements min per chunk
    
    def determine_chunk_size(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "auto"
    ) -> Union[int, Tuple[int, ...]]:
        """
        Determine the optimal chunk size for regridding operations.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid for regridding
        method : str, optional
            The method for determining chunk size ('auto', 'memory', 'performance')
            
        Returns
        -------
        int or tuple of ints
            The optimal chunk size(s)
        """
        if method == "auto":
            return self._auto_chunk_size(data, target_grid)
        elif method == "memory":
            return self._memory_based_chunk_size(data, target_grid)
        elif method == "performance":
            return self._performance_based_chunk_size(data, target_grid)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _auto_chunk_size(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray]
    ) -> Union[int, Tuple[int, ...]]:
        """
        Automatically determine chunk size based on data characteristics.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid for regridding
            
        Returns
        -------
        int or tuple of ints
            The optimal chunk size(s)
        """
        # Calculate the size of the source grid
        source_size = self._calculate_grid_size(data)
        target_size = self._calculate_grid_size(target_grid)
        
        # Use a heuristic to determine chunk size based on the smaller grid
        base_size = min(source_size, target_size)
        
        # Calculate chunk size to keep it within reasonable bounds
        chunk_size = int(math.sqrt(min(self.max_chunk_size, max(self.min_chunk_size, base_size))))
        
        # Return as tuple for spatial dimensions
        return (chunk_size, chunk_size)
    
    def _memory_based_chunk_size(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray]
    ) -> Union[int, Tuple[int, ...]]:
        """
        Determine chunk size based on memory constraints.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid for regridding
            
        Returns
        -------
        int or tuple of ints
            The optimal chunk size(s)
        """
        # Estimate memory usage based on data size
        data_size = self._estimate_memory_usage(data)
        
        # Assume we want to keep chunks under 100MB for safety
        target_chunk_memory = 100 * 1024 * 1024  # 100 MB in bytes
        
        # Calculate appropriate chunk size
        if data_size > 0:
            elements_per_chunk = int(target_chunk_memory / (data_size * np.dtype(data.dtype).itemsize))
            chunk_size = int(math.sqrt(max(self.min_chunk_size, min(self.max_chunk_size, elements_per_chunk))))
        else:
            chunk_size = int(math.sqrt(self.default_chunk_size))
        
        return (chunk_size, chunk_size)
    
    def _performance_based_chunk_size(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray]
    ) -> Union[int, Tuple[int, ...]]:
        """
        Determine chunk size based on performance considerations.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid for regridding
            
        Returns
        -------
        int or tuple of ints
            The optimal chunk size(s)
        """
        # For performance, we want larger chunks to reduce overhead
        # but not so large that they cause memory issues
        source_size = self._calculate_grid_size(data)
        
        # Use larger chunks for performance, but cap at max_chunk_size
        chunk_size = min(int(math.sqrt(source_size * 2)), int(math.sqrt(self.max_chunk_size)))
        chunk_size = max(chunk_size, int(math.sqrt(self.min_chunk_size)))
        
        return (chunk_size, chunk_size)
    
    def _calculate_grid_size(self, data: Union[xr.Dataset, xr.DataArray]) -> int:
        """
        Calculate the effective size of a grid.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The grid data
            
        Returns
        -------
        int
            The calculated grid size
        """
        if isinstance(data, xr.DataArray):
            # For DataArray, return the product of spatial dimensions
            spatial_dims = [dim for dim in data.dims if 'x' in str(dim).lower() or 'y' in str(dim).lower() or
                           'lon' in str(dim).lower() or 'lat' in str(dim).lower()]
            if spatial_dims:
                size = 1
                for dim in spatial_dims:
                    size *= data.sizes[dim]
                return size
            else:
                # If no spatial dims identified, return total size
                return data.size
        elif isinstance(data, xr.Dataset):
            # For Dataset, consider the first data variable
            for var_name, var_data in data.data_vars.items():
                spatial_dims = [dim for dim in var_data.dims if 'x' in str(dim).lower() or 'y' in str(dim).lower() or
                               'lon' in str(dim).lower() or 'lat' in str(dim).lower()]
                if spatial_dims:
                    size = 1
                    for dim in spatial_dims:
                        size *= var_data.sizes[dim]
                    return size
            # If no spatial dims found in any variable, return size of first variable
            if data.data_vars:
                first_var = next(iter(data.data_vars.values()))
                return first_var.size
            else:
                return 0
        else:
            return 0
    
    def _estimate_memory_usage(self, data: Union[xr.Dataset, xr.DataArray]) -> int:
        """
        Estimate the memory usage of the data.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to estimate memory usage for
            
        Returns
        -------
        int
            Estimated memory usage in bytes
        """
        if isinstance(data, xr.DataArray):
            return data.nbytes
        elif isinstance(data, xr.Dataset):
            total_bytes = 0
            for var_name, var_data in data.data_vars.items():
                total_bytes += var_data.nbytes
            return total_bytes
        else:
            return 0
    
    def apply_chunking(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        chunk_size: Union[int, Tuple[int, ...], Dict[str, int]]
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Apply the specified chunking to the data.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to chunk
        chunk_size : int, tuple of ints, or dict
            The chunk size specification
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The chunked data
        """
        if isinstance(data, xr.DataArray):
            return data.chunk(chunk_size)
        elif isinstance(data, xr.Dataset):
            return data.chunk(chunk_size)
        else:
            raise TypeError(f"Expected xr.DataArray or xr.Dataset, got {type(data)}")