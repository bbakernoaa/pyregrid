# Performance Tips

This guide provides optimization strategies to maximize the performance of your PyRegrid operations.

## Overview

Optimizing PyRegrid performance involves several aspects:

- Choosing appropriate interpolation methods
- Optimizing memory usage
- Leveraging parallel processing
- Efficient data I/O
- Proper chunking strategies

## Method Selection

Different interpolation methods have varying performance characteristics:

### Fastest Methods
- **Nearest neighbor**: Fastest for any data type
- **Bilinear**: Good balance of speed and quality for continuous fields

### Moderate Performance
- **Conservative**: Slower but necessary for flux conservation

### More Computationally Intensive
- **Patch recovery**: Highest accuracy but slowest
- **Bicubic**: Better gradients but more computation

## Memory Optimization

### Chunking Strategy
```python
# Optimize chunk sizes for your system
import psutil
available_memory = psutil.virtual_memory().available
# Use approximately 70% of available memory for operations
optimal_chunk_size = int(0.7 * available_memory / (data_item_size * num_arrays))
```

### Memory Management
```python
from pyregrid.dask import memory_management

# Set memory fraction to avoid system slowdown
memory_management.set_memory_fraction(0.8)

# Or set absolute limit
memory_management.set_memory_limit('4GB')
```

## Dask Configuration

### Optimal Scheduler
```python
import dask

# For CPU-bound regridding operations
dask.config.set(scheduler='threads', num_workers=psutil.cpu_count())

# For I/O-bound operations
dask.config.set(scheduler='threads', num_workers=2*psutil.cpu_count())
```

### Chunk Size Optimization
```python
# For regridding, spatial chunk size is critical
# Balance: larger chunks = less overhead, but more memory usage
optimal_chunks = {
    'time': 20,      # Adjust based on your time series length
    'lat': 100,      # Adjust based on grid size
    'lon': 10       # Adjust based on grid size
}
```

## Data Format Optimization

### Use Efficient Formats
```python
# For large datasets, use Zarr instead of NetCDF
import xarray as xr

# Zarr is more efficient for chunked access
ds = xr.open_zarr('data.zarr')  # More efficient for chunked operations

# When saving results
result.to_zarr('output.zarr', mode='w', 
               encoding={'variable': {'compressor': zarr.Blosc()}})
```

### Optimize Compression
```python
# When saving intermediate results
encoding = {
    'temperature': {
        'zlib': True,
        'complevel': 1,  # Lower compression for faster I/O
        'dtype': 'float32'  # Use appropriate precision
    }
}
result.to_netcdf('output.nc', encoding=encoding)
```

## Preprocessing for Performance

### Optimize Grid Preparation
```python
# Precompute regridding weights if using the same grids repeatedly
regridder = pyregrid.GridRegridder(
    source_grid=source_grid,
    destination_grid=target_grid,
    method='bilinear',
    save_weights=True # Save weights for reuse
)

# Reuse the regridder for multiple variables
for var in ['temp', 'humidity', 'pressure']:
    result = regridder.regrid(source_data[var])
```

### Subset Data When Possible
```python
# Only load the spatial region you need
subset = full_data.sel(
    lat=slice(min_lat, max_lat),
    lon=slice(min_lon, max_lon)
)
result = regridder.regrid(subset)
```

## Parallel Processing Strategies

### Multiple Variables
```python
# Process multiple variables in parallel
import dask

variables = ['temp', 'humidity', 'pressure']
results = []

for var in variables:
    results.append(regridder.regrid(source_data[var]))

# Compute all at once
computed_results = dask.compute(*results)
```

### Multiple Time Periods
```python
# Process time chunks in parallel
time_chunks = source_data.chunk({'time': 30})
result = regridder.regrid(time_chunks['temperature'])
```

## Profiling and Monitoring

### Monitor Performance
```python
import time
import dask

# Time your operations
start_time = time.time()
result = regridder.regrid(data).compute()
end_time = time.time()

print(f"Regridding took {end_time - start_time:.2f} seconds")
```

### Memory Usage
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.2f} MB")
```

## Common Performance Pitfalls

### Avoid These Patterns
1. **Too small chunks**: Creates excessive overhead
2. **Too large chunks**: Causes memory issues
3. **Inconsistent chunking**: Leads to inefficient rechunking
4. **Loading unnecessary data**: Always subset when possible

### Best Practices Summary
- Profile your specific use case
- Start with moderate chunk sizes and adjust
- Use appropriate data formats (Zarr for large datasets)
- Reuse regridders when processing multiple variables
- Monitor memory usage during operations
- Consider the trade-off between compression and speed