import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, simple_data_size_to_grid_block_2D, HybridArray
from python.rare_weak_model.python_native import random_p_values_matrix_py, random_p_values_series_py, random_modified_p_values_matrix_py, modify_p_values_matrix_py, sort_and_count_labels_rows_py
from python.rare_weak_model.numba_cpu import random_p_values_matrix_cpu_njit, random_p_values_series_cpu_njit, random_modified_p_values_matrix_cpu_njit, modify_p_values_matrix_cpu_njit, sort_and_count_labels_rows_cpu_njit
from python.rare_weak_model.numba_gpu import random_p_values_matrix_gpu, random_p_values_series_gpu, random_modified_p_values_matrix_gpu, modify_p_values_matrix_gpu, sort_and_count_labels_rows_gpu
from python.random_integers.random_integers import random_num_steps

def rare_weak_model(\
        data: HybridArray,\
        counts: HybridArray,\
        mu: float|np.float64,\
        n1: int|np.uint32,\
        num_steps: int|np.uint32|None=None,\
        use_njit: bool|None = None,
        sort_labels: bool = True) -> None:
    data.astype(np.float64)
    num_monte, N = data.shape()
    assert 0 <= n1 <= N
    random_p_values_matrix(\
        data = data,\
        offset_row0=0,\
        offset_col0=0,\
        num_steps=num_steps,
        use_njit=use_njit)
    if n1 > 0:
        modify_p_values_matrix(\
            data = data.crop(0,num_monte,0,n1),\
            mu = mu,\
            use_njit=use_njit)
    data.uncrop()
    if sort_labels:
        sort_and_count_labels_rows(data=data, counts=counts, n1=n1, use_njit=use_njit)

def sort_and_count_labels_rows(\
        data: HybridArray,\
        counts: HybridArray,\
        n1: int|np.uint32,\
        use_njit: bool|None = None) -> None:
    counts.realloc(shape=data.shape(), dtype=np.uint32, use_gpu=data.is_gpu())
    n1 = np.uint32(n1)
    if data.is_gpu():
        # GPU mode
        sort_and_count_labels_rows_gpu(data=data.gpu_data(), n1=n1, counts=counts.gpu_data())
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        assert isinstance(counts.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            sort_and_count_labels_rows_cpu_njit(data=data.data, n1=n1, counts=counts.data)
        else:
            sort_and_count_labels_rows_py(data=data.data, n1=n1, counts=counts.data)
    
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
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_modified_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=data.numpy())
        else:
            random_modified_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=data.numpy())

def modify_p_values_matrix(\
        data: HybridArray,\
        mu: float|np.float64,\
        use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    mu = np.float64(mu)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = simple_data_size_to_grid_block_2D(data.shape())
        modify_p_values_matrix_gpu[grid_shape, block_shape](data.data, mu) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            modify_p_values_matrix_cpu_njit(out=data.numpy(), mu=mu)
        else:
            modify_p_values_matrix_py(out=data.numpy(), mu=mu)


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
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())
        else:
            random_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())



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
