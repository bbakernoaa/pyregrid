"""
Parallel processing utilities for Dask-based regridding operations.

This module provides utilities for leveraging Dask's distributed computing
capabilities for improved performance during regridding operations.
"""
import xarray as xr
import numpy as np
from typing import Union, Optional, Dict, Any, Tuple, Callable
from dask.distributed import Client, as_completed
from dask.delayed import delayed
import dask.array as da


class ParallelProcessor:
    """
    A utility class for parallel processing of regridding operations using Dask.
    """
    
    def __init__(self, client: Optional[Client] = None):
        """
        Initialize the parallel processor.
        
        Parameters
        ----------
        client : dask.distributed.Client, optional
            Dask client for distributed computing. If None, uses default scheduler.
        """
        self.client = client
    
    def regrid_in_parallel(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        regrid_function: Callable,
        chunks: Optional[Union[int, Tuple[int, ...], Dict[str, int]]] = None,
        **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Perform regridding in parallel using Dask.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to regrid
        regrid_function : callable
            The function to apply for regridding
        chunks : int, tuple, dict or None
            Chunk specification for parallel processing
        **kwargs
            Additional arguments to pass to the regrid function
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The regridded data
        """
        # Chunk the data appropriately for parallel processing
        if chunks is not None:
            data = data.chunk(chunks)
        
        # Apply the regridding function in parallel
        if isinstance(data, xr.DataArray):
            result = self._process_dataarray_parallel(data, regrid_function, **kwargs)
        elif isinstance(data, xr.Dataset):
            result = self._process_dataset_parallel(data, regrid_function, **kwargs)
        else:
            raise TypeError(f"Expected xr.DataArray or xr.Dataset, got {type(data)}")
        
        return result
    
    def _process_dataarray_parallel(
        self,
        data: xr.DataArray,
        regrid_function: Callable,
        **kwargs
    ) -> xr.DataArray:
        """
        Process a DataArray in parallel.
        
        Parameters
        ----------
        data : xr.DataArray
            The DataArray to process
        regrid_function : callable
            The function to apply
        **kwargs
            Additional arguments
            
        Returns
        -------
        xr.DataArray
            The processed DataArray
        """
        # Apply the regrid function to each chunk of the data array
        # For now, we'll use dask's map_blocks functionality
        result_data = data.data.map_blocks(
            self._apply_regrid_chunk,
            dtype=data.dtype,
            drop_axis=[],  # Don't drop any axes
            meta=np.array((), dtype=data.dtype),
            regrid_function=regrid_function,
            **kwargs
        )
        
        # Create result DataArray with the same coordinates and attributes
        result = xr.DataArray(
            result_data,
            dims=data.dims,
            coords=data.coords,
            attrs=data.attrs
        )
        
        return result
    
    def _process_dataset_parallel(
        self,
        data: xr.Dataset,
        regrid_function: Callable,
        **kwargs
    ) -> xr.Dataset:
        """
        Process a Dataset in parallel.
        
        Parameters
        ----------
        data : xr.Dataset
            The Dataset to process
        regrid_function : callable
            The function to apply
        **kwargs
            Additional arguments
            
        Returns
        -------
        xr.Dataset
            The processed Dataset
        """
        # Process each data variable in parallel
        processed_vars = {}
        for var_name, var_data in data.data_vars.items():
            processed_vars[var_name] = self._process_dataarray_parallel(
                var_data, regrid_function, **kwargs
            )
        
        # Create result Dataset with the same coordinates
        result = xr.Dataset(
            processed_vars,
            coords=data.coords,
            attrs=data.attrs
        )
        
        return result
    
    @staticmethod
    def _apply_regrid_chunk(chunk, regrid_function: Callable, **kwargs):
        """
        Apply the regridding function to a chunk of data.
        
        Parameters
        ----------
        chunk : array-like
            A chunk of the data array
        regrid_function : callable
            The regridding function to apply
        **kwargs
            Additional arguments
            
        Returns
        -------
        array-like
            The regridded chunk
        """
        # Convert the chunk to a DataArray temporarily to work with it
        # This is a simplified approach - in practice, this would need to handle
        # coordinate transformations appropriately for each chunk
        return regrid_function(chunk, **kwargs)
    
    def optimize_parallel_execution(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        target_grid: Union[xr.Dataset, xr.DataArray],
        method: str = "bilinear"
    ) -> Dict[str, Any]:
        """
        Optimize parallel execution parameters based on data characteristics.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The input data
        target_grid : xr.Dataset or xr.DataArray
            The target grid
        method : str
            The regridding method
            
        Returns
        -------
        dict
            Dictionary of optimized execution parameters
        """
        # Calculate optimal number of workers based on data size
        data_size = self._estimate_data_size(data)
        target_size = self._estimate_data_size(target_grid)
        
        # Suggest parallelism based on data size
        optimal_workers = min(8, max(1, data_size // 1000000))  # 1 worker per 1M elements, max 8
        
        # Determine optimal chunk size
        optimal_chunk_size = max(1000, min(10000, int(np.sqrt(data_size // optimal_workers))))
        
        return {
            'workers': optimal_workers,
            'chunk_size': (optimal_chunk_size, optimal_chunk_size),
            'method': method
        }
    
    def _estimate_data_size(self, data: Union[xr.Dataset, xr.DataArray]) -> int:
        """
        Estimate the size of the data in elements.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to estimate size for
            
        Returns
        -------
        int
            Estimated number of elements
        """
        if isinstance(data, xr.DataArray):
            return data.size
        elif isinstance(data, xr.Dataset):
            total_size = 0
            for var_name, var_data in data.data_vars.items():
                total_size += var_data.size
            return total_size // len(data.data_vars) if data.data_vars else 0
        else:
            return 0
    
    def execute_with_scheduler(
        self,
        data: Union[xr.Dataset, xr.DataArray],
        regrid_function: Callable,
        scheduler: str = "threads",
        **kwargs
    ) -> Union[xr.Dataset, xr.DataArray]:
        """
        Execute regridding with a specific Dask scheduler.
        
        Parameters
        ----------
        data : xr.Dataset or xr.DataArray
            The data to regrid
        regrid_function : callable
            The function to apply
        scheduler : str
            The Dask scheduler to use ('threads', 'processes', 'synchronous', or client)
        **kwargs
            Additional arguments
            
        Returns
        -------
        xr.Dataset or xr.DataArray
            The regridded data
        """
        # Apply the regridding function
        result = regrid_function(data, **kwargs)
        
        # Compute the result with the specified scheduler
        if isinstance(result, xr.DataArray):
            if hasattr(result.data, 'compute'):
                result = result.copy(data=result.data.compute(scheduler=scheduler))
        elif isinstance(result, xr.Dataset):
            for var_name in result.data_vars:
                if hasattr(result[var_name].data, 'compute'):
                    result[var_name].values = result[var_name].data.compute(scheduler=scheduler)
        
        return result