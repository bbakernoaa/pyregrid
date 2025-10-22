# Installation

This guide covers how to install PyRegrid in various environments.

## Prerequisites

PyRegrid requires Python 3.8 or higher. It is built on top of several scientific Python libraries, including:

- NumPy
- Xarray
- Dask
- Cartopy
- Scipy

## Installing with pip

The easiest way to install PyRegrid is using pip:

```bash
pip install pyregrid
```

## Installing with Conda

PyRegrid can also be installed using Conda. It is recommended to create a new environment for PyRegrid to avoid dependency conflicts.

```bash
conda create -n pyregrid_env python=3.9
conda activate pyregrid_env
conda install -c conda-forge numpy xarray dask cartopy scipy
pip install pyregrid
```

## Development Installation

To install PyRegrid for development, clone the repository and install in editable mode:

```bash
git clone https://github.com/pyregrid/pyregrid
cd pyregrid
pip install -e .
```

For development with all optional dependencies:

```bash
pip install -e ".[dev]"
```

## Verifying Installation

To verify that PyRegrid is installed correctly, run:

```python
import pyregrid
print(pyregrid.__version__)
```

## Troubleshooting

If you encounter issues during installation:

1. Ensure you have Python 3.8 or higher
2. Try installing in a fresh virtual environment
3. Check that you have the necessary system dependencies for the underlying libraries
4. Refer to the [GitHub Issues](https://github.com/pyregrid/pyregrid/issues) for known problems