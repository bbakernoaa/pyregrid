"""
Test fixtures for PyRegrid library.

This module contains shared test fixtures for creating common data scenarios
used throughout the test suite.
"""

import pytest
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime


@pytest.fixture
def simple_2d_grid():
    """Create a simple 2D grid for testing."""
    # Create a simple 2D grid with lat/lon coordinates
    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-90, 90, 5)
    
    # Create data array with some sample values
    data = np.random.rand(5, 10)
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        dims=['lat', 'lon'],
        coords={
            'lon': lons,
            'lat': lats
        },
        name='temperature'
    )
    
    return da


@pytest.fixture
def simple_2d_grid_dataset():
    """Create a simple 2D grid dataset with multiple variables."""
    # Create a simple 2D grid with lat/lon coordinates
    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-90, 90, 5)
    
    # Create data arrays with some sample values
    temp_data = np.random.rand(5, 10)
    pressure_data = np.random.rand(5, 10) * 1000
    
    # Create xarray Dataset
    ds = xr.Dataset({
        'temperature': (['lat', 'lon'], temp_data),
        'pressure': (['lat', 'lon'], pressure_data)
    }, coords={
        'lon': lons,
        'lat': lats
    })
    
    return ds


@pytest.fixture
def simple_point_data():
    """Create simple point data for testing."""
    # Create a simple DataFrame with point data
    data = {
        'longitude': [-120, -110, -100, -90, -80],
        'latitude': [30, 40, 50, 60, 70],
        'temperature': [20, 25, 30, 35, 40]
    }
    
    df = pd.DataFrame(data)
    
    return df


@pytest.fixture
def simple_point_dict():
    """Create simple point data as dictionary for testing."""
    return {
        'longitude': [-120, -110, -100, -90, -80],
        'latitude': [30, 40, 50, 60, 70],
        'temperature': [20, 25, 30, 35, 40]
    }


@pytest.fixture
def simple_target_grid():
    """Create a simple target grid for regridding."""
    lons = np.linspace(-170, 170, 8)
    lats = np.linspace(-80, 80, 4)
    
    # Create xarray DataArray
    da = xr.DataArray(
        np.zeros((4, 8)),
        dims=['lat', 'lon'],
        coords={
            'lon': lons,
            'lat': lats
        }
    )
    
    return da


@pytest.fixture
def simple_3d_grid():
    """Create a simple 3D grid with time dimension."""
    times = pd.date_range('2020-01-01', periods=3)
    lons = np.linspace(-180, 180, 5)
    lats = np.linspace(-90, 90, 3)
    
    # Create data array with some sample values
    data = np.random.rand(3, 3, 5)
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        dims=['time', 'lat', 'lon'],
        coords={
            'time': times,
            'lon': lons,
            'lat': lats
        },
        name='temperature'
    )
    
    return da


@pytest.fixture
def simple_2d_grid_with_nan():
    """Create a 2D grid with NaN values for testing edge cases."""
    lons = np.linspace(-180, 180, 10)
    lats = np.linspace(-90, 90, 5)
    
    # Create data array with some NaN values
    data = np.random.rand(5, 10)
    data[1, 3] = np.nan  # Add a NaN value
    data[4, 7] = np.nan  # Add another NaN value
    
    # Create xarray DataArray
    da = xr.DataArray(
        data,
        dims=['lat', 'lon'],
        coords={
            'lon': lons,
            'lat': lats
        },
        name='temperature'
    )
    
    return da