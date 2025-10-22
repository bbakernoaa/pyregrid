"""
Memory management utilities for Dask-based regridding operations.

This module provides utilities for efficient memory usage during large-scale
regridding operations with Dask.
"""
import xarray as xr
import numpy as np
from typing import Union, Optional, Dict, Any, List
import psutil
import gc
from contextlib import contextmanager


class MemoryManager:
    """
    A utility class for managing memory during Dask-based regridding operations.
    """
    
    def __init__(self):
        self.max_memory_fraction = 0.8  # Use up to 80% of available memory
        self.current_memory_usage = 0
    
    def get_available_memory(self) -> int:
        """
        Get the amount of available system memory in bytes.
        
        Returns
        -------
        int
            Available memory in bytes
        """
        memory = psutil.virtual_memory()
        return int(memory.available * self.max_memory_fraction)
    
    def estimate_operation_memory(
        self,
        source_data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear"
    ) -> int:
        """
        Estimate the memory required for a regridding operation.
        
        Parameters
        ----------
        source_data : xr.Dataset or xr.DataArray
            The source data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid
        method : str, optional
            The regridding method to be used
            
        Returns
        -------
        int
            Estimated memory usage in bytes
        """
        # Estimate memory for source data
        source_memory = self._estimate_xarray_memory(source_data)
        
        # Estimate memory for target data
        target_memory = self._estimate_xarray_memory(target_grid)
        
        # Estimate memory for intermediate arrays during regridding
        # This depends on the method and grid sizes
        method_factor = self._get_method_memory_factor(method)
        
        # Calculate grid size factors
        source_size = self._calculate_grid_size(source_data)
        target_size = self._calculate_grid_size(target_grid)
        
        # Estimate intermediate memory usage (coordinates, weights, etc.)
        intermediate_memory = (source_size + target_size) * 8  # 8 bytes per coordinate/index
        
        # Total estimated memory
        total_memory = source_memory + target_memory + (intermediate_memory * method_factor)
        
        return int(total_memory)
    
    def _estimate_xarray_memory(self, data: Union[xr.Dataset, xr.DataArray]) -> int:
        """
        Estimate memory usage of xarray data structure.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The xarray data to estimate memory for
            
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
    
    def _get_method_memory_factor(self, method: str) -> float:
        """
        Get a memory factor based on the regridding method.
        
        Parameters
        ----------
        method : str
            The regridding method
            
        Returns
        -------
        float
            Memory factor multiplier
        """
        method_factors = {
            'bilinear': 1.0,
            'cubic': 1.5,
            'nearest': 0.8,
            'conservative': 2.0  # Conservative methods typically require more memory
        }
        return method_factors.get(method, 1.0)
    
    def can_fit_in_memory(
        self,
        source_data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear",
        chunk_size: Optional[Union[int, tuple]] = None
    ) -> bool:
        """
        Check if a regridding operation can fit in available memory.
        
        Parameters
        ----------
        source_data : xr.Dataset or xr.DataArray
            The source data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid
        method : str, optional
            The regridding method to be used
        chunk_size : int or tuple, optional
            The chunk size to be used (if chunking)
            
        Returns
        -------
        bool
            True if the operation can fit in memory, False otherwise
        """
        estimated_memory = self.estimate_operation_memory(source_data, target_grid, method)
        
        # If chunking is specified, adjust the estimate
        if chunk_size is not None:
            if isinstance(chunk_size, (tuple, list)):
                chunk_elements = np.prod(chunk_size)
            else:
                chunk_elements = chunk_size * chunk_size  # assume square chunks
            
            # Calculate how many chunks we'll have
            source_size = self._calculate_grid_size(source_data)
            if source_size > 0:
                num_chunks = max(1, source_size // chunk_elements)
                # Adjust estimate based on number of chunks (we process one at a time)
                estimated_memory = estimated_memory // num_chunks
        
        available_memory = self.get_available_memory()
        return bool(estimated_memory <= available_memory)
    
    def optimize_chunking(
        self,
        source_data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear",
        max_chunk_size: Optional[Union[int, tuple]] = None,
        min_chunk_size: int = 10
    ) -> Optional[Union[int, tuple]]:
        """
        Determine optimal chunking to fit within memory constraints.
        
        Parameters
        ----------
        source_data : xr.Dataset or xr.DataArray
            The source data to be regridded
        target_grid : xr.Dataset or xr.DataArray
            The target grid
        method : str, optional
            The regridding method to be used
        max_chunk_size : int or tuple, optional
            Maximum chunk size to consider. If None, uses data dimensions
        min_chunk_size : int, optional
            Minimum chunk size to consider (default: 10)
            
        Returns
        -------
        int or tuple or None
            Optimal chunk size, or None if data fits in memory without chunking
        """
        if self.can_fit_in_memory(source_data, target_grid, method):
            return None  # No chunking needed
        
        # Get source grid dimensions for chunking guidance
        source_size = self._calculate_grid_size(source_data)
        if isinstance(source_data, xr.DataArray):
            dims = source_data.dims
        elif isinstance(source_data, xr.Dataset):
            # Use the first data variable's dimensions
            first_var = next(iter(source_data.data_vars.values()))
            dims = first_var.dims
        
        # Determine maximum chunk size based on data dimensions
        if max_chunk_size is None:
            if len(dims) >= 2:
                # Use 25% of each dimension as starting point
                max_chunk_size = tuple(max(min_chunk_size, int(source_data.sizes[dim] * 0.25)) for dim in dims[-2:])
            else:
                max_chunk_size = min_chunk_size * 4
        
        # Start with a reasonable chunk size and adjust based on memory
        if isinstance(max_chunk_size, tuple):
            base_chunk_size = min(max_chunk_size)
        else:
            base_chunk_size = max_chunk_size
        
        # Try different chunk sizes from largest to smallest
        while base_chunk_size >= min_chunk_size:
            if isinstance(max_chunk_size, tuple):
                # For 2D data, try square chunks first
                chunk_size = (base_chunk_size, base_chunk_size)
                
                # If that doesn't work, try rectangular chunks
                if not self.can_fit_in_memory(source_data, target_grid, method, chunk_size):
                    # Try chunks that match the aspect ratio of the data
                    if len(dims) >= 2:
                        dim1_size = source_data.sizes[dims[-2]]
                        dim2_size = source_data.sizes[dims[-1]]
                        aspect_ratio = dim1_size / dim2_size
                        
                        # Adjust chunk size based on aspect ratio
                        if aspect_ratio > 1:
                            # Wider than tall
                            chunk_size = (base_chunk_size, int(base_chunk_size / aspect_ratio))
                        else:
                            # Taller than wide
                            chunk_size = (int(base_chunk_size * aspect_ratio), base_chunk_size)
                
                if self.can_fit_in_memory(source_data, target_grid, method, chunk_size):
                    return chunk_size
            else:
                # For 1D data
                chunk_size = base_chunk_size
                if self.can_fit_in_memory(source_data, target_grid, method, chunk_size):
                    return chunk_size
            
            base_chunk_size = max(min_chunk_size, base_chunk_size // 2)
        
        # If we can't fit even small chunks, provide a more informative error
        estimated_memory = self.estimate_operation_memory(source_data, target_grid, method)
        available_memory = self.get_available_memory()
        
        # Try to suggest a solution
        if isinstance(source_data, (xr.DataArray, xr.Dataset)):
            if hasattr(source_data, 'chunks') and source_data.chunks:
                # Data is already chunked, suggest reducing chunk size
                if hasattr(source_data, 'chunks') and source_data.chunks:
                    # Get the chunk sizes for each dimension
                    chunks_info = source_data.chunks
                    if isinstance(chunks_info, dict):
                        # For dictionary format, extract the values
                        chunk_sizes = list(chunks_info.values())
                        # If values are tuples (which they usually are for each dimension), get the first element
                        chunk_sizes = [c[0] if isinstance(c, tuple) else c for c in chunk_sizes]
                    elif isinstance(chunks_info, tuple):
                        # For tuple format, each element might be a tuple of chunk sizes for that dimension
                        chunk_sizes = [c[0] if isinstance(c, tuple) else c for c in chunks_info]
                    else:
                        # Fallback
                        chunk_sizes = [min_chunk_size, min_chunk_size]
                    
                    # Reduce each chunk size by half
                    suggested_chunk = tuple(max(1, int(c / 2)) for c in chunk_sizes[-2:])
                else:
                    suggested_chunk = (min_chunk_size, min_chunk_size)
                raise MemoryError(
                    f"Operation requires {estimated_memory:,} bytes but only "
                    f"{available_memory:,} bytes available. "
                    f"Consider reducing chunk size to {suggested_chunk} or smaller."
                )
            else:
                # Data is not chunked, suggest chunking
                if len(dims) >= 2:
                    suggested_chunk = (min_chunk_size, min_chunk_size)
                    raise MemoryError(
                        f"Operation requires {estimated_memory:,} bytes but only "
                        f"{available_memory:,} bytes available. "
                        f"Consider chunking your data with chunks={suggested_chunk} or smaller."
                    )
        
        raise MemoryError(
            f"Operation requires {estimated_memory:,} bytes but only "
            f"{available_memory:,} bytes available. "
            "Consider using a machine with more memory or reducing data size."
        )
    
    @contextmanager
    def memory_monitor(self, operation_name: str = "Operation"):
        """
        Context manager to monitor memory usage during an operation.
        
        Parameters
        ----------
        operation_name : str
            Name of the operation for logging
        """
        initial_memory = self.get_available_memory()
        print(f"Starting {operation_name} with {initial_memory:,} bytes available")
        
        try:
            yield
        finally:
            gc.collect()  # Force garbage collection
            final_memory = self.get_available_memory()
            memory_change = final_memory - initial_memory
            print(f"Completed {operation_name}, memory change: {memory_change:+,} bytes")
    
    def clear_memory(self):
        """
        Clear any cached memory information and force garbage collection.
        """
        gc.collect()
        self.current_memory_usage = 0