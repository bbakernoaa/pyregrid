"""
Dask integration module for PyRegrid.

This module provides Dask-based implementations for scalable regridding operations.
"""
from .dask_regridder import DaskRegridder
from .chunking import ChunkingStrategy
from .memory_management import MemoryManager
from .parallel_processing import ParallelProcessor

__all__ = ['DaskRegridder', 'ChunkingStrategy', 'MemoryManager', 'ParallelProcessor']