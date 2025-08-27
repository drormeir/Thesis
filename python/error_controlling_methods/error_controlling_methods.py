import numpy as np
from python.hpc import is_use_njit, HybridArray
from python.error_controlling_methods import numba_gpu, numba_cpu, python_native

def top_k(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        numba_gpu.top_k[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.top_k(sorted_p_values_input=sorted_p_values_input.numpy(),\
                          num_discoveries_output=num_discoveries_output.numpy())
        else:
            python_native.top_k(sorted_p_values_input=sorted_p_values_input.numpy(),\
                    num_discoveries_output=num_discoveries_output.numpy())
            

def bonferroni(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        numba_gpu.bonferroni[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.bonferroni(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            python_native.bonferroni(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())
            
def benjamini_hochberg(sorted_p_values_input: HybridArray,\
               num_discoveries_output: HybridArray,\
               **kwargs) -> None:
    num_discoveries_output.realloc(like=sorted_p_values_input, dtype=np.uint32)
    if num_discoveries_output.is_gpu():
                # GPU mode
        grid_shape, block_shape = num_discoveries_output.gpu_grid_block1D_rows_shapes()
        numba_gpu.benjamini_hochberg[grid_shape, block_shape](sorted_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.benjamini_hochberg(sorted_p_values_input=sorted_p_values_input.numpy(),\
                                num_discoveries_output=num_discoveries_output.numpy())
        else:
            python_native.benjamini_hochberg(sorted_p_values_input=sorted_p_values_input.numpy(),\
                         num_discoveries_output=num_discoveries_output.numpy())

