import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.array_math_utils.numba_gpu import array_transpose_gpu, sort_rows_inplace_gpu, average_rows_gpu
from python.array_math_utils.numba_cpu import array_transpose_cpu_njit, average_rows_cpu_njit, sort_rows_inplace_cpu_njit

def array_transpose_inplace(array: HybridArray, use_njit: bool|None = None) -> None:
    work = HybridArray()
    array_transpose(array=array, out=work, use_njit=use_njit)
    array.swap(work)
    work.close()
    
def array_transpose(array: HybridArray, out: HybridArray, use_njit: bool|None = None) -> None:
    if array.is_empty():
        return # clear output array???
    out.realloc(like=array, shape=array.shape()[::-1])    
    if array.is_gpu():
        # GPU mode
        array_transpose_gpu(array=array.gpu_data(), out=out.gpu_data())
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            array_transpose_cpu_njit(array=array.numpy(), out=out.numpy())
        else:
            out.numpy()[:] = array.numpy().T

def average_rows(array: HybridArray, out_row: HybridArray, use_njit: bool|None = None) -> None:
    if array.is_gpu():
        # GPU mode
        average_rows_gpu(array=array.gpu_data(), out_row=out_row.gpu_data())
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            average_rows_cpu_njit(array=array.numpy(), out_row=out_row.numpy())
        else:
            np.mean(a=array.numpy(), axis=0, keepdims=True, out=out_row.numpy())

def sort_rows_inplace(array: HybridArray, use_njit: bool|None = None) -> None:
    if array.is_gpu():
        sort_rows_inplace_gpu(array.gpu_data())
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            sort_rows_inplace_cpu_njit(array.numpy())
        else:
            array.numpy().sort(axis=1)
