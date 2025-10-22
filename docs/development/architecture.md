# Architecture

This document describes the architecture of the PyRegrid library and its design principles.

## Overview

PyRegrid is designed with a modular architecture that separates concerns while maintaining high performance and usability. The library is organized into several key modules, each responsible for specific functionality.

## Core Architecture

### Module Structure

```
pyregrid/
├── __init__.py          # Public API
├── core.py              # Main regridding classes
├── interpolation.py     # Interpolation algorithms
├── algorithms/          # Specific interpolation implementations
├── crs/                 # Coordinate reference system handling
├── dask/                # Dask integration
├── utils/               # Utility functions
├── accessors/           # Xarray accessor extensions
└── point_interpolator.py # Point-to-grid interpolation
```

### Core Components

#### GridRegridder
The main class for grid-to-grid regridding operations. It handles:
- Source and destination grid definition
- Interpolation method selection
- Weight computation and caching
- Actual regridding operations

#### PointInterpolator
Handles interpolation from scattered points to regular grids, supporting:
- Inverse distance weighting
- Kriging methods
- Natural neighbor interpolation

#### Coordinate Reference System (CRS) Management
The `crs` module provides:
- Coordinate system detection
- Transformation between different CRS
- Proper handling of geographic coordinates

## Design Principles

### Modularity
Each module has a clear, well-defined responsibility and can be used independently where appropriate.

### Extensibility
The architecture supports adding new interpolation methods and features without breaking existing functionality.

### Performance
Critical paths are optimized for performance while maintaining usability.

### Xarray Integration
Deep integration with xarray provides a familiar interface for geospatial data operations.

## Integration Points

### Dask Integration
The `dask` module provides:
- Parallel processing capabilities
- Memory management for large datasets
- Chunking strategies for optimal performance

### Xarray Accessors
The `accessors` module extends xarray with PyRegrid functionality, allowing intuitive usage like:
```python
result = dataset.regrid(target_grid, method='bilinear')
```

## Future Considerations

The architecture is designed to accommodate:
- Additional interpolation algorithms
- New coordinate systems
- Performance optimizations
- Extended data format support