# Regridding Methods

This guide covers the different regridding methods available in PyRegrid and when to use each one.

## Bilinear Interpolation

Bilinear interpolation is suitable for continuous fields like temperature, pressure, or precipitation rates. It provides smooth transitions between grid points.

```python
import pyregrid

regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=dest_grid,
    method='bilinear'
)
```

### When to use:
- Continuous physical fields
- When smooth output is desired
- For fields without sharp gradients

## Nearest Neighbor

Nearest neighbor interpolation preserves original data values and is suitable for categorical data or when conservation is not critical.

```python
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=dest_grid,
    method='nearest'
)
```

### When to use:
- Categorical data (land use, soil type)
- When preserving original values is important
- For quick, approximate regridding

## Conservative Remapping

Conservative methods preserve the integral of the field, making them essential for flux calculations and mass/volume conservation.

```python
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=dest_grid,
    method='conservative'
)
```

### When to use:
- Flux calculations
- Mass/volume conservation requirements
- Precipitation totals
- Other conserved quantities

## Higher-Order Methods

Higher-order methods like patch recovery provide more accurate results for smooth fields at the cost of increased computational complexity.

```python
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=dest_grid,
    method='patch'
)
```

### When to use:
- Smooth, high-quality fields
- When accuracy is more important than speed
- For applications requiring high-order accuracy

## Custom Weights

You can also provide precomputed interpolation weights:

```python
regridder = pyregrid.GridRegridder(
    source_grid=source_data,
    destination_grid=dest_grid,
    method='weights',
    weights=precomputed_weights
)
```

### When to use:
- Repeated regridding with same grids
- When using weights from external tools
- For specialized interpolation approaches