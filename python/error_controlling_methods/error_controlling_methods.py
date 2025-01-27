import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, simple_data_size_to_grid_block_1D, HybridArray
from python.error_controlling_methods.numba_gpu import benjamini_hochberg_gpu, bonferroni_gpu, topk_gpu
from python.error_controlling_methods.numba_cpu import benjamini_hochberg_cpu_njit, bonferroni_cpu_njit, topk_cpu_njit
from python.error_controlling_methods.python_native import benjamini_hochberg_py, bonferroni_py, topk_py

def topk(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               use_njit: bool|None = None) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = simple_data_size_to_grid_block_1D(num_discoveries_output.nrows())
        topk_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            topk_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                          num_discoveries_output=num_discoveries_output.numpy())
        else:
            topk_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                    num_discoveries_output=num_discoveries_output.numpy())
            

def bonferroni(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               use_njit: bool|None = None) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
                # GPU mode
        grid_shape, block_shape =simple_data_size_to_grid_block_1D(num_discoveries_output.nrows())
        bonferroni_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            bonferroni_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            bonferroni_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())
            
def benjamini_hochberg(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               use_njit: bool|None = None) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
                # GPU mode
        grid_shape, block_shape = simple_data_size_to_grid_block_1D(num_discoveries_output.nrows())
        benjamini_hochberg_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            benjamini_hochberg_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            benjamini_hochberg_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())

