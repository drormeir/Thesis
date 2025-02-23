import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.rare_weak_model.python_native import random_p_values_matrix_py, random_p_values_series_py, random_modified_p_values_matrix_py, modify_p_values_matrix_py, sort_and_count_labels_rows_py
from python.rare_weak_model.numba_cpu import random_p_values_matrix_cpu_njit, random_p_values_series_cpu_njit, random_modified_p_values_matrix_cpu_njit, modify_p_values_matrix_cpu_njit, sort_and_count_labels_rows_cpu_njit
from python.rare_weak_model.numba_gpu import random_p_values_matrix_gpu, random_p_values_series_gpu, random_modified_p_values_matrix_gpu, modify_p_values_matrix_gpu, sort_and_count_labels_rows_gpu
from python.random_integers.random_integers import random_num_steps
from python.array_math_utils.array_math_utils import sort_rows_inplace

def rare_weak_null_hypothesis(\
        sorted_p_values_output: HybridArray,\
        ind_model: int|np.uint32 = 0,\
        **kwargs) -> None:
    random_p_values_matrix(\
        p_values_output = sorted_p_values_output,\
        offset_row0= np.uint32(ind_model) * sorted_p_values_output.nrows(),\
        offset_col0=0,\
        **kwargs)
    sort_rows_inplace(array=sorted_p_values_output, **kwargs)
    
def rare_weak_model(\
        sorted_p_values_output: HybridArray,\
        cumulative_counts_output: HybridArray|None,\
        n1: int|np.uint32,\
        mu: float|np.float64|np.float32,\
        ind_model: int|np.uint32 = 0,\
        **kwargs) -> None:
    random_p_values_matrix(\
        p_values_output = sorted_p_values_output,\
        offset_row0= np.uint32(ind_model) * sorted_p_values_output.nrows(),\
        offset_col0=0,\
        **kwargs)
    modify_p_values_submatrix(p_values_inoutput = sorted_p_values_output,\
                              mu=mu, n1=n1, **kwargs)
    if cumulative_counts_output is None:
        sort_rows_inplace(array=sorted_p_values_output, **kwargs)
    else:
        sort_and_count_labels_rows(sorted_p_values_inoutput=sorted_p_values_output,\
                                    cumulative_counts_output=cumulative_counts_output,\
                                    n1=n1, **kwargs)

def sort_and_count_labels_rows(\
        sorted_p_values_inoutput: HybridArray,\
        cumulative_counts_output: HybridArray,\
        n1: int|np.uint32,\
        use_njit: bool|None = None) -> None:
    if n1 < 1:
        return
    cumulative_counts_output.realloc(like=sorted_p_values_inoutput, dtype=np.uint32)
    n1 = np.uint32(n1)
    if sorted_p_values_inoutput.is_gpu():
        # GPU mode
        sort_and_count_labels_rows_gpu(data=sorted_p_values_inoutput.gpu_data(), n1=n1,\
                                       counts=cumulative_counts_output.gpu_data())
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            sort_and_count_labels_rows_cpu_njit(data=sorted_p_values_inoutput.numpy(), n1=n1,\
                                                counts=cumulative_counts_output.numpy())
        else:
            sort_and_count_labels_rows_py(data=sorted_p_values_inoutput.numpy(), n1=n1,\
                                          counts=cumulative_counts_output.numpy())
    
def random_modified_p_values_matrix(\
        p_values_output: HybridArray,\
        mu: float|np.float64,\
        offset_row0: int|np.uint32,\
        offset_col0: int|np.uint32,\
        num_steps: int|np.uint32|None=None,\
        use_njit: bool|None = None) -> None:
    p_values_output.astype(np.float64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = random_num_steps(num_steps)
    mu = np.float64(mu)
    if p_values_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = p_values_output.gpu_grid_block2D_square_shapes()
        random_modified_p_values_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, mu, p_values_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_modified_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=p_values_output.numpy())
        else:
            random_modified_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, mu=mu, out=p_values_output.numpy())

def modify_p_values_submatrix(\
        p_values_inoutput: HybridArray,\
        mu: float|np.float64|np.float32,\
        n1: int|np.uint32,\
        use_njit: bool|None = None) -> None:
    rows, N = p_values_inoutput.shape()
    assert 0 <= n1 <= N
    if n1 < 1:
        return
    p_values_inoutput.astype(np.float64)
    data = p_values_inoutput.crop(0,rows,0,n1)
    mu = np.float64(mu)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = data.gpu_grid_block2D_square_shapes()
        modify_p_values_matrix_gpu[grid_shape, block_shape](data.gpu_data(), mu) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            modify_p_values_matrix_cpu_njit(out=data.numpy(), mu=mu)
        else:
            modify_p_values_matrix_py(out=data.numpy(), mu=mu)
    p_values_inoutput.uncrop()

def random_p_values_matrix(p_values_output: HybridArray,\
                           offset_row0: int|np.uint32,\
                           offset_col0: int|np.uint32,\
                           num_steps: int|np.uint32|None = None,\
                           use_njit: bool|None = None) -> None:
    p_values_output.astype(np.float64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = random_num_steps(num_steps)
    if p_values_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = p_values_output.gpu_grid_block2D_square_shapes()
        random_p_values_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, p_values_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=p_values_output.numpy())
        else:
            random_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=p_values_output.numpy())



def random_p_values_series(p_values_output: HybridArray, seed: int|np.uint64, use_njit: bool|None = None) -> None:
    p_values_output.astype(np.float64)
    seed = np.uint64(seed)
    assert p_values_output.ndim() == 1
    if p_values_output.is_gpu():
        # GPU mode
        random_p_values_series_gpu[1, 1](seed, p_values_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_series_cpu_njit(seed=seed, out=p_values_output.numpy())
        else:
            random_p_values_series_py(seed=seed, out=p_values_output.numpy())
