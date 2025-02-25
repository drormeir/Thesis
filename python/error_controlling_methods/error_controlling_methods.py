import numpy as np
from python.hpc import use_njit, HybridArray
from python.error_controlling_methods.numba_gpu import benjamini_hochberg_gpu, bonferroni_gpu, topk_gpu
from python.error_controlling_methods.numba_cpu import benjamini_hochberg_cpu_njit, bonferroni_cpu_njit, topk_cpu_njit
from python.error_controlling_methods.python_native import benjamini_hochberg_py, bonferroni_py, topk_py

def topk(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        topk_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            topk_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                          num_discoveries_output=num_discoveries_output.numpy())
        else:
            topk_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                    num_discoveries_output=num_discoveries_output.numpy())
            

def bonferroni(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        bonferroni_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            bonferroni_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            bonferroni_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())
            
def benjamini_hochberg(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
                # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        benjamini_hochberg_gpu[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            benjamini_hochberg_cpu_njit(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            benjamini_hochberg_py(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())

