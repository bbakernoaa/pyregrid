# Dask Integration

This guide explains how PyRegrid integrates with Dask for parallel and out-of-core processing of large datasets.

## Overview

PyRegrid provides seamless integration with Dask, allowing you to:

- Process datasets larger than memory
- Distribute computations across multiple cores
- Maintain lazy evaluation until computation is needed
- Handle chunked xarray datasets efficiently

## Basic Dask Usage

PyRegrid automatically works with Dask-backed xarray objects:

```python
import xarray as xr
import pyregrid

# Load data with Dask chunking
source_data = xr.open_dataset('large_file.nc', chunks={'time': 10, 'lat': 50, 'lon': 50})
target_grid = xr.open_dataset('target_grid.nc')

# Create regridder (works the same way)
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=target_grid,
    method='bilinear'
)

# Result is a Dask array - no computation happens yet
result = regridder.regrid(source_data['temperature'])

# Computation happens when you call compute()
final_result = result.compute()
```

## Controlling Chunking

Proper chunking is essential for performance:

```python
# Optimize chunking for your computation
source_data = xr.open_dataset('data.nc', chunks={'time': 20, 'lat': 100, 'lon': 10})

# Or rechunk after loading
source_data = source_data.chunk({'time': 15, 'lat': 80, 'lon': 80})
```

### Chunking Best Practices:

- **Time dimension**: Usually best to have larger time chunks if processing time series
- **Spatial dimensions**: Balance between memory usage and parallelization
- **Consistent chunks**: Ensure all variables have compatible chunking

## Dask-Specific Configuration

PyRegrid provides options for optimizing Dask operations:

```python
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=target_grid,
    method='bilinear',
    # Use multiple processes
    dask_chunks='auto',  # or specify chunk sizes
    # Memory management options
    max_memory_usage=0.8  # Use up to 80% of available memory
)
```

## Memory Management

For very large datasets, PyRegrid includes memory management features:

```python
from pyregrid.dask import memory_management

# Configure memory usage limits
memory_management.set_memory_limit('8GB')

# Or as a fraction of available memory
memory_management.set_memory_fraction(0.7)
```

## Parallel Processing

PyRegrid can leverage multiple cores automatically:

```python
import dask
# Configure Dask for optimal performance
dask.config.set(scheduler='threads', num_workers=4)

# Or use processes (may be better for CPU-intensive tasks)
dask.config.set(scheduler='processes', num_workers=4)
```

## Performance Tips

### 1. Optimize Chunk Sizes
```python
# For regridding operations, spatial chunk size affects performance significantly
# Too small: overhead from many small operations
# Too large: memory constraints
optimal_chunks = {'time': 10, 'lat': 200, 'lon': 200}
```

### 2. Use Appropriate Storage Format
```python
# For large datasets, consider using Zarr instead of NetCDF
import zarr
result.to_zarr('output.zarr', mode='w')
```

### 3. Persist Intermediate Results
```python
# For multi-step operations, persist intermediate results
intermediate = regridder.regrid(data['var1']).persist()
result2 = regridder.regrid(data['var2'])
```

## Common Patterns

### Processing Multiple Variables
```python
# Process multiple variables efficiently
variables_to_regrid = ['temperature', 'humidity', 'pressure']
regridded_data = {}

for var in variables_to_regrid:
    regridded_data[var] = regridder.regrid(source_data[var])

# Compute all at once for efficiency
import dask
results = dask.compute(*regridded_data.values())
```

### Time-Series Processing
```python
# For time-series analysis
time_chunks = source_data.chunk({'time': 30})  # Process 30 time steps at a time
result = regridder.regrid(time_chunks['temperature'])
```

## Troubleshooting

### Memory Issues
- Reduce chunk sizes
- Use `rechunk` to optimize chunking
- Monitor memory usage during computation

### Performance Issues
- Experiment with different chunk sizes
- Consider the scheduler type (threads vs processes)
- Profile your computation to identify bottlenecks