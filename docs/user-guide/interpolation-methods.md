# Interpolation Methods

This guide details the various interpolation methods available in PyRegrid and their appropriate use cases.

## Overview

Interpolation is a critical component of the regridding process. The choice of interpolation method can significantly impact both the accuracy and performance of your regridding operations.

## Continuous Methods

### Bilinear Interpolation

Bilinear interpolation estimates values at new grid points using a weighted average of the four nearest source points. This method produces smooth results suitable for continuous fields.

**Characteristics:**
- Produces smooth output
- Suitable for continuous fields (temperature, pressure)
- Computationally efficient
- May introduce slight smoothing of sharp gradients

### Bicubic Interpolation

Bicubic interpolation uses 16 neighboring points to create a smoother interpolation than bilinear, with continuous derivatives.

**Characteristics:**
- Very smooth output
- Better preservation of gradients
- Higher computational cost
- Potential for overshoot in regions with high gradients

## Discontinuous Methods

### Nearest Neighbor

Nearest neighbor interpolation simply assigns the value of the closest source point to each destination point.

**Characteristics:**
- Preserves original data values
- Suitable for categorical data
- Fastest method
- Produces blocky output

### Nearest S2D (Source to Destination)

This method finds the closest source point for each destination point, accounting for great-circle distances on a sphere.

**Characteristics:**
- Properly handles spherical distances
- Good for sparse data
- Preserves original values

## Conservative Methods

### First-Order Conservative

Conservative methods ensure that the total value is preserved during regridding, which is crucial for flux calculations.

**Characteristics:**
- Preserves integrals
- Essential for mass/volume conservation
- More complex to compute
- Suitable for flux quantities

### Second-Order Conservative

Second-order conservative methods add gradient reconstruction to improve accuracy while maintaining conservation properties.

**Characteristics:**
- Better accuracy than first-order
- Maintains conservation
- More computationally expensive
- Reduces numerical artifacts

## Specialized Methods

### Patch Recovery

Patch recovery uses local polynomial fitting to achieve higher-order accuracy for smooth fields.

**Characteristics:**
- High-order accuracy
- Good for smooth fields
- More computationally intensive
- May have stability issues with noisy data

### Inverse Distance Weighting (IDW)

IDW weights nearby points more heavily than distant points, with the influence decreasing as distance increases.

**Characteristics:**
- Good for scattered data
- Parameterizable smoothness
- Can handle irregular spacing
- May have artifacts with clustered data

## Method Selection Guidelines

When choosing an interpolation method, consider:

1. **Data type**: Continuous vs. categorical
2. **Conservation requirements**: Whether integrals must be preserved
3. **Accuracy needs**: Smoothness vs. fidelity requirements
4. **Computational constraints**: Speed vs. accuracy trade-offs
5. **Data characteristics**: Smooth vs. noisy, regular vs. irregular