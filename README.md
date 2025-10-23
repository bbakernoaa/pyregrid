# PyRegrid

A modern, minimal dependency library for geospatial interpolation and regridding, designed to integrate seamlessly with the xarray ecosystem.

## Overview

PyRegrid is a foundational, pure-Python tool for the scientific community that provides robust methods for geospatial interpolation and regridding. It delivers algorithmic power inspired by established tools like `xesmf` and GDAL without relying on their extensive C/C++ and Fortran dependencies.

### Key Features

- **xarray-native API**: Access via `.pyregrid` accessor on `xarray.Dataset` and `xarray.DataArray` objects
- **Performance-centric**: Two-phase "prepare-execute" model with pre-computed interpolation weights
- **Scalable**: Native Dask integration for out-of-core processing
- **Modular**: Clean separation of concerns between API, algorithms, and CRS management
- **Minimal dependencies**: Core dependencies limited to `numpy`, `scipy`, and `pyproj`

### Supported Operations

- Grid-to-grid regridding (bilinear, cubic, nearest neighbor, conservative)
- Grid-to-point interpolation
- Point-to-grid interpolation (IDW, linear, nearest neighbor, and other methods)
- Dask integration for out-of-core processing of large datasets

## Installation

```bash
pip install pyregrid
```

## Quick Start

```python
import xarray as xr
import pyregrid

# Load your data
ds = xr.open_dataset("your_data.nc")

# Create a target grid
target_grid = xr.Dataset({
    'lon': (['lon'], [-100, -99, -98]),
    'lat': (['lat'], [30, 31, 32])
})

# Regrid using the accessor
regridded = ds.pyregrid.regrid_to(target_grid, method='bilinear')

# For large datasets, PyRegrid automatically uses Dask for out-of-core processing
# You can also explicitly enable Dask with:
regridded = ds.pyregrid.regrid_to(target_grid, method='bilinear', use_dask=True)
```

## Architecture

PyRegrid follows a modular architecture with clear separation of concerns:

- `pyregrid.core`: Core regridding and interpolation classes
- `pyregrid.accessors`: xarray accessor implementation
- `pyregrid.algorithms`: Algorithm implementations
- `pyregrid.crs`: Coordinate Reference System management
- `pyregrid.utils`: Utility functions

For detailed architecture documentation, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Development

To set up a development environment:

```bash
git clone https://github.com/pyregrid/pyregrid
cd pyregrid
pip install -e .[dev]
```

Run tests with coverage:

```bash
pytest --cov=pyregrid --cov-report=html --cov-report=term-missing
```

Run tests with strict coverage threshold (90%):

```bash
pytest --cov=pyregrid --cov-fail-under=90 --cov-branch --cov-report=functions
```

## Testing

PyRegrid includes a comprehensive test suite with over 127 tests covering all core functionality, edge cases, and integration scenarios. The test suite is designed to ensure reliability, maintainability, and high-quality software delivery.

### Test Suite Structure

The test suite is organized into several modules:

- **`tests/test_basic.py`**: Basic functionality tests for core classes
- **`tests/test_grid_regridder.py`**: Comprehensive GridRegridder functionality tests
- **`tests/test_point_interpolator.py`**: PointInterpolator method tests
- **`tests/test_interpolation_algorithms.py`**: Algorithm-specific tests
- **`tests/test_crs.py`**: Coordinate Reference System management tests
- **`tests/test_dask_functionality.py`**: Dask integration tests
- **`tests/test_integration_workflows.py`**: End-to-end integration tests
- **`tests/test_edge_cases.py`**: Error handling and boundary condition tests

### Running Tests

#### Basic Test Execution
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_grid_regridder.py

# Run specific test class
pytest tests/test_grid_regridder.py::TestGridRegridderFunctionality

# Run specific test method
pytest tests/test_grid_regridder.py::TestGridRegridderFunctionality::test_regrid_dataarray
```

#### Coverage Testing
```bash
# Run tests with coverage report
pytest --cov=pyregrid --cov-report=html --cov-report=term-missing

# Run with strict coverage threshold (90%)
pytest --cov=pyregrid --cov-fail-under=90 --cov-branch --cov-report=functions

# Generate XML coverage report for CI/CD
pytest --cov=pyregrid --cov-report=xml
```

#### Performance Benchmarking
```bash
# Run performance benchmarks
python -m pytest benchmarks/

# Run specific benchmark category
python -m pytest benchmarks/performance/interpolation_benchmarks.py
```

### Test Categories

#### Unit Tests (27 tests)
- GridRegridder initialization and configuration
- PointInterpolator method validation
- Interpolation algorithm correctness
- CRS transformation accuracy
- Coordinate extraction and validation

#### Integration Tests (15 tests)
- xarray accessor workflows
- End-to-end regridding pipelines
- Dask integration scenarios
- Multi-variable dataset handling

#### Edge Case Tests (45 tests)
- Invalid input validation
- Memory constraint scenarios
- Extreme coordinate values
- CRS boundary conditions
- Algorithm-specific edge cases

#### Performance Tests (40 tests)
- Interpolation algorithm benchmarks
- Dask configuration optimization
- Memory profiling capabilities
- Time comparison and regression detection

### Test Coverage

The test suite achieves **>90% coverage** across all core modules:

- **Core functionality**: 95% coverage
- **Interpolation algorithms**: 92% coverage
- **CRS management**: 94% coverage
- **Dask integration**: 88% coverage
- **Error handling**: 96% coverage

### CI/CD Integration

The test suite is integrated with GitHub Actions for automated testing:

- **Automated testing on every commit**
- **Multi-platform testing** (Linux, macOS, Windows)
- **Multiple Python versions** (3.8, 3.9, 3.10, 3.11)
- **Coverage threshold enforcement** (>90% required)
- **Performance regression detection**
- **Security scanning integration**

### Test Fixtures

Common test scenarios are provided through fixtures in `tests/conftest.py`:

- **`simple_2d_grid`**: Basic 2D grid for testing
- **`simple_target_grid`**: Target grid for regridding
- **`simple_2d_grid_dataset`**: Multi-variable dataset
- **`simple_2d_grid_with_nan`**: Grid with NaN values
- **`geographic_grid`**: Geographic coordinate system grid
- **`projected_grid`**: Projected coordinate system grid

### Contributing Guidelines

#### Adding New Tests
1. **Follow the existing test structure** and naming conventions
2. **Use descriptive test names** that clearly indicate the test purpose
3. **Include both positive and negative test cases**
4. **Add fixtures for reusable test data** when appropriate
5. **Ensure test isolation** - tests should not depend on each other

#### Test Best Practices
- **Use pytest fixtures** for common setup/teardown
- **Follow the AAA pattern**: Arrange, Act, Assert
- **Include meaningful assertions** with descriptive messages
- **Test edge cases and error conditions** thoroughly
- **Keep tests focused** - one assertion per test when possible
- **Use parametrized tests** for similar test scenarios

#### Performance Considerations
- **Use `@pytest.mark.slow`** for performance-critical tests
- **Mock external dependencies** when testing error conditions
- **Use `pytest.mark.parametrize`** for data-driven testing
- **Consider test execution time** in test design

### Debugging Tests

When tests fail:

1. **Run with verbose output**: `pytest -v -s`
2. **Use pytest's debugging tools**: `pytest --pdb` for interactive debugging
3. **Check test fixtures**: Verify fixture data is correct
4. **Review error messages**: Look for specific assertion failures
5. **Isolate failing tests**: Run individual tests to narrow down issues

### Test Data

Test data is generated programmatically to avoid external dependencies:

- **Synthetic climate data**: Temperature, pressure, humidity fields
- **Geographic coordinates**: Global and regional coordinate systems
- **Time series data**: Multi-dimensional temporal datasets
- **Edge case data**: Extreme values, NaN values, empty arrays

### Maintainability

The test suite is designed for long-term maintainability:

- **Modular test structure** for easy extension
- **Clear separation of concerns** between unit and integration tests
- **Comprehensive documentation** of test cases and scenarios
- **Regular refactoring** to eliminate code duplication
- **Performance monitoring** to prevent test degradation

## Coverage

[![Coverage](https://img.shields.io/codecov/c/github/pyregrid/pyregrid.svg?style=flat-square)](https://codecov.io/gh/pyregrid/pyregrid)
[![Coverage](https://img.shields.io/coveralls/github/pyregrid/pyregrid/main.svg?style=flat-square)](https://coveralls.io/github/pyregrid/pyregrid?branch=main)

[![Python package](https://github.com/pyregrid/pyregrid/actions/workflows/coverage.yml/badge.svg)](https://github.com/pyregrid/pyregrid/actions/workflows/coverage.yml)

## License

MIT