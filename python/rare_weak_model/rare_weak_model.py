import math
import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, simple_data_size_to_grid_block_2D, HybridArray
from python.rare_weak_model.python_native import random_p_values_matrix_py, random_p_values_series_py, random_modified_p_values_matrix_py
from python.rare_weak_model.numba_cpu import random_p_values_matrix_cpu_njit, random_p_values_series_cpu_njit, random_modified_p_values_matrix_cpu_njit
from python.rare_weak_model.numba_gpu import random_p_values_matrix_gpu, random_p_values_series_gpu, random_modified_p_values_matrix_gpu
from python.random_integers.random_integers import random_num_steps

def rare_weak_model(\
        data: HybridArray,\
        r: np.float64,\
        beta: np.float64,\
        num_steps: int|np.uint32|None=None,\
        use_njit: bool|None = None) -> None:
    assert 0.0 <= beta <= 1.0
    data.astype(np.float64)
    num_monte, N = data.shape()
    n1 = int(N**beta+0.5)
    mu = math.sqrt(2*r*math.log(N))
    random_modified_p_values_matrix(\
        data = data.crop(0,num_monte,0,n1),\
        mu = mu,\
        offset_row0=0,\
        offset_col0=0,\
        num_steps=num_steps,\
        use_njit=use_njit)
    random_p_values_matrix(\
        data = data.crop(0,num_monte,n1,N),\
        offset_row0=0,\
        offset_col0=n1,\
        num_steps=num_steps,
        use_njit=use_njit)

def random_modified_p_values_matrix(\
        data: HybridArray,\
        mu: float|np.float64,\
        offset_row0: int|np.uint32,\
        offset_col0: int|np.uint32,\
        num_steps: int|np.uint32|None=None,\
        use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = random_num_steps(num_steps)
    mu = np.float64(mu)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = simple_data_size_to_grid_block_2D(data.shape())
        random_modified_p_values_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, mu, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_modified_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=data.data)
        else:
            random_modified_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=data.data)


def random_p_values_matrix(data: HybridArray,\
                           offset_row0: int|np.uint32,\
                           offset_col0: int|np.uint32,\
                           num_steps: int|np.uint32|None = None,\
                           use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = random_num_steps(num_steps)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = simple_data_size_to_grid_block_2D(data.shape())
        random_p_values_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)
        else:
            random_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)



def random_p_values_series(data: HybridArray, seed: int|np.uint64, use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    seed = np.uint64(seed)
    assert data.ndim() == 1
    if data.is_gpu():
        # GPU mode
        random_p_values_series_gpu[1, 1](seed, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_series_cpu_njit(seed=seed, out=data.data)
        else:
            random_p_values_series_py(seed=seed, out=data.data)
