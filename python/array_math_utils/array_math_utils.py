import numpy as np
from python.hpc import is_use_njit, HybridArray
from python.array_math_utils import python_native, numba_cpu, numba_gpu

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
        numba_gpu.array_transpose(array=array.gpu_data(), out=out.gpu_data())
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.array_transpose(array=array.numpy(), out=out.numpy())
        else:
            python_native.array_transpose(array=array.numpy(), out=out.numpy())

def average_row(array: HybridArray, out_row: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    out_row.realloc(like=array, shape=(1,array.ncols()))
    if array.is_gpu():
        # GPU mode
        numba_gpu.average_row(array=array.gpu_data(), out_row=out_row.gpu_data())
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.average_row(array=array.numpy(), out_row=out_row.numpy())
        else:
            python_native.average_row(array=array.numpy(), out_row=out_row.numpy())

def average_column(array: HybridArray, out_column: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    out_column.realloc(like=array, shape=(array.nrows(),1))
    if array.is_gpu():
        # GPU mode
        numba_gpu.average_column(array=array.gpu_data(), out_column=out_column.gpu_data())
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.average_column(array=array.numpy(), out_column=out_column.numpy())
        else:
            python_native.average_column(array=array.numpy(), out_column=out_column.numpy())

def sort_rows_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        numba_gpu.sort_rows_inplace(array.gpu_data())
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.sort_rows_inplace(array.numpy())
        else:
            python_native.sort_rows_inplace(array.numpy())


def cumulative_argmin(array: HybridArray, argmin: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    argmin.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_argmin[grid_shape, block_shape](array.gpu_data(), argmin.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_argmin(array=array.numpy(), argmin=argmin.numpy())
        else:
            python_native.cumulative_argmin(array=array.numpy(), argmin=argmin.numpy())


def cumulative_argmax(array: HybridArray, argmax: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    argmax.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_argmax[grid_shape, block_shape](array.gpu_data(), argmax.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_argmax(array=array.numpy(), argmax=argmax.numpy())
        else:
            python_native.cumulative_argmax(array=array.numpy(), argmax=argmax.numpy())


def cumulative_min_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_min_inplace[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_min_inplace(array=array.numpy())
        else:
            python_native.cumulative_min_inplace(array=array.numpy())


def cumulative_max_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_max_inplace[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_max_inplace(array=array.numpy())
        else:
            python_native.cumulative_max_inplace(array=array.numpy())


def cumulative_dominant_argmin(array: HybridArray, argmin: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'    
    argmin.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_dominant_argmin[grid_shape, block_shape](array.gpu_data(), argmin.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_dominant_argmin(array=array.numpy(), argmin=argmin.numpy())
        else:
            python_native.cumulative_dominant_argmin(array=array.numpy(), argmin=argmin.numpy())


def cumulative_dominant_argmax(array: HybridArray, argmax: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'    
    argmax.realloc(like=array, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_dominant_argmax[grid_shape, block_shape](array.gpu_data(), argmax.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_dominant_argmax(array=array.numpy(), argmax=argmax.numpy())
        else:
            python_native.cumulative_dominant_argmax(array=array.numpy(), argmax=argmax.numpy())


def cumulative_dominant_min_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_dominant_min_inplace[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_dominant_min_inplace(array=array.numpy())
        else:
            python_native.cumulative_dominant_min_inplace(array=array.numpy())


def cumulative_dominant_max_inplace(array: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    if array.is_gpu():
        # GPU mode
        grid_shape, block_shape = array.gpu_grid_block1D_rows_shapes()
        numba_gpu.cumulative_dominant_max_inplace[grid_shape, block_shape](array.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.cumulative_dominant_max_inplace(array=array.numpy())
        else:
            python_native.cumulative_dominant_max_inplace(array=array.numpy())


def max_column_along_rows(array: HybridArray, maxval: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    shape = (1,array.nrows())
    maxval.realloc(like=array, shape=shape)
    if array.is_gpu():
        # GPU mode
        numba_gpu.max_column_along_rows(array=array.gpu_data(), maxval=maxval.gpu_data())
    else:
        # CPU mode
        array_numpy = array.numpy()
        maxval_numpy = maxval.numpy().reshape(-1)
        if is_use_njit(**kwargs):
            numba_cpu.max_column_along_rows(array=array_numpy, maxval=maxval_numpy)
        else:
            python_native.max_column_along_rows(array=array_numpy, maxval=maxval_numpy)


def argmax_column_along_rows(array: HybridArray, argmax: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    shape = (1,array.nrows())
    argmax.realloc(like=array, shape=shape, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        numba_gpu.argmax_column_along_rows(array=array.gpu_data(), argmax=argmax.gpu_data())
    else:
        # CPU mode
        array_numpy = array.numpy()
        argmax_numpy = argmax.numpy().reshape(-1)
        if is_use_njit(**kwargs):
            numba_cpu.argmax_column_along_rows(array=array_numpy, argmax=argmax_numpy)
        else:
            python_native.argmax_column_along_rows(array=array_numpy, argmax=argmax_numpy)


def min_column_along_rows(array: HybridArray, minval: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    shape = (1,array.nrows())
    minval.realloc(like=array, shape=shape)
    if array.is_gpu():
        # GPU mode
        numba_gpu.min_column_along_rows(array=array.gpu_data(), minval=minval.gpu_data())
    else:
        # CPU mode
        array_numpy = array.numpy()
        minval_numpy = minval.numpy().reshape(-1)
        if is_use_njit(**kwargs):
            numba_cpu.min_column_along_rows(array=array_numpy, minval=minval_numpy)
        else:
            python_native.min_column_along_rows(array=array_numpy, minval=minval_numpy)


def argmin_column_along_rows(array: HybridArray, argmin: HybridArray, **kwargs) -> None:
    assert array.dtype() == np.float64, f'{array.dtype()=}'
    shape = (1,array.nrows())
    argmin.realloc(like=array, shape=shape, dtype=np.uint32)
    if array.is_gpu():
        # GPU mode
        numba_gpu.argmin_column_along_rows(array=array.gpu_data(), argmin=argmin.gpu_data())
    else:
        # CPU mode
        array_numpy = array.numpy()
        argmin_numpy = argmin.numpy().reshape(-1)
        if is_use_njit(**kwargs):
            numba_cpu.argmin_column_along_rows(array=array_numpy, argmin=argmin_numpy)
        else:
            python_native.argmin_column_along_rows(array=array_numpy, argmin=argmin_numpy)
