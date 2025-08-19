"""
High Performance Computing (HPC) module for hybrid CPU/GPU array operations.
"""

from .hpc import globals, is_use_njit
from .hybrid_array import HybridArray

__all__ = [
    'globals',
    'is_use_njit',
    'HybridArray',
]
