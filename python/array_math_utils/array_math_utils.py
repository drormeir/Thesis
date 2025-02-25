import numpy as np
from python.hpc import use_njit, HybridArray
from python.array_math_utils.numba_gpu import array_transpose_gpu, sort_rows_inplace_gpu, average_row_gpu, average_column_gpu, cumulative_argmin_gpu, cumulative_min_inplace_gpu, cumulative_dominant_argmin_gpu, cumulative_dominant_min_inplace_gpu, max_along_rows_gpu
from python.array_math_utils.numba_cpu import array_transpose_cpu_njit, average_row_cpu_njit, average_column_cpu_njit, sort_rows_inplace_cpu_njit, cumulative_argmin_cpu_njit, cumulative_min_inplace_cpu_njit, cumulative_dominant_argmin_cpu_njit, cumulative_dominant_min_inplace_cpu_njit, max_along_rows_cpu_njit
from python.array_math_utils.python_native import cumulative_argmin_py, cumulative_min_inplace_py, cumulative_dominant_argmin_py, cumulative_dominant_min_inplace_py

def array_transpose_inplace(array: HybridArray, **kwargs) -> None:
    work = HybridArray()
    array_transpose(array=array, out=work, **kwargs)
    array.swap(work)
    work.close()
    
def array_transpose(array: HybridArray, out: HybridArray, **kwargs) -> None:
    if array.is_empty():
        return # clear output array???
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    out.realloc(like=array, shape=array.shape()[::-1])    
    if array.is_gpu():
        # GPU mode
        array_transpose_gpu(array=array.gpu_data(), out=out.gpu_data())
    else:
        # CPU mode
        if use_njit(**kwargs):
            array_transpose_cpu_njit(array=array.numpy(), out=out.numpy())
        else:
            out.numpy()[:] = array.numpy().T

def average_row(array: HybridArray, out_row: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    out_row.realloc(like=array, shape=(1,array.ncols()))
    if array.is_gpu():
        # GPU mode
        average_row_gpu(array=array.gpu_data(), out_row=out_row.gpu_data())
    else:
        # CPU mode
        if use_njit(**kwargs):
            average_row_cpu_njit(array=array.numpy(), out_row=out_row.numpy())
        else:
            np.mean(a=array.numpy(), axis=0, keepdims=True, out=out_row.numpy())

def average_column(array: HybridArray, out_column: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    out_column.realloc(like=array, shape=(array.nrows(),1))
    if array.is_gpu():
        # GPU mode
        average_column_gpu(array=array.gpu_data(), out_column=out_column.gpu_data())
    else:
        # CPU mode
        if use_njit(**kwargs):
            average_column_cpu_njit(array=array.numpy(), out_column=out_column.numpy())
        else:
            np.mean(a=array.numpy(), axis=1, keepdims=True, out=out_column.numpy())

def sort_rows_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        sort_rows_inplace_gpu(array.gpu_data())
    else:
        # CPU mode
        if use_njit(**kwargs):
            sort_rows_inplace_cpu_njit(array.numpy())
        else:
            array.numpy().sort(axis=1)

def cumulative_argmin(array: HybridArray, argmin: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    argmin.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        cumulative_argmin_gpu[grid_shape, block_shape](array.gpu_data(), argmin.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            cumulative_argmin_cpu_njit(array=array.numpy(), argmin=argmin.numpy())
        else:
            cumulative_argmin_py(array=array.numpy(), argmin=argmin.numpy())


def cumulative_min_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        cumulative_min_inplace_gpu[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            cumulative_min_inplace_cpu_njit(array=array.numpy())
        else:
            cumulative_min_inplace_py(array=array.numpy())

def cumulative_dominant_argmin(array: HybridArray, argmin: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'    
    argmin.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        cumulative_dominant_argmin_gpu[grid_shape, block_shape](array.gpu_data(), argmin.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            cumulative_dominant_argmin_cpu_njit(array=array.numpy(), argmin=argmin.numpy())
        else:
            cumulative_dominant_argmin_py(array=array.numpy(), argmin=argmin.numpy())


def cumulative_dominant_min_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        cumulative_dominant_min_inplace_gpu[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            cumulative_dominant_min_inplace_cpu_njit(array=array.numpy())
        else:
            cumulative_dominant_min_inplace_py(array=array.numpy())

def max_along_rows(array: HybridArray, argmax: HybridArray, maxval: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    shape = (1,array.nrows())
    argmax.realloc(like=array, shape=shape, dtype=np.uint32)
    maxval.realloc(like=array, shape=shape)
    if array.is_gpu():
        # GPU mode
        max_along_rows_gpu(array=array.gpu_data(), argmax=argmax.gpu_data(), maxval=maxval.gpu_data())
    else:
        # CPU mode
        array_numpy = array.numpy()
        argmax_numpy = argmax.numpy().reshape(-1)
        maxval_numpy = maxval.numpy().reshape(-1)
        if use_njit(**kwargs):
            max_along_rows_cpu_njit(array=array_numpy, argmax=argmax_numpy, maxval=maxval_numpy)
        else:
            array_numpy.argmax(axis=1, out=argmax_numpy)
            array_numpy.max(axis=1, out=maxval_numpy)
