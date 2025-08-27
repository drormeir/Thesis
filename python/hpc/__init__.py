"""
High Performance Computing (HPC) module for hybrid CPU/GPU array operations.
"""

from .hpc import globals, is_use_njit, is_use_gpu, raise_cuda_not_available, raise_njit_not_available
from .hybrid_array import HybridArray

__all__ = [
    'globals',
    'is_use_njit',
    'is_use_gpu',
    'raise_cuda_not_available',
    'raise_njit_not_available',
    'HybridArray',
]
