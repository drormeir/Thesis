import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.adaptive_methods.numba_gpu import higher_criticism_stable_gpu, higher_criticism_unstable_gpu, berk_jones_gpu, discover_argmin_gpu, discover_dominant_gpu
from python.adaptive_methods.numba_cpu import higher_criticism_stable_cpu_njit, higher_criticism_unstable_cpu_njit, berk_jones_cpu_njit, discover_argmin_cpu_njit, discover_dominant_cpu_njit
from python.adaptive_methods.python_native import higher_criticism_stable_py, higher_criticism_unstable_py, berk_jones_py, discover_argmin_py, discover_dominant_py


def higher_criticism_stable(\
        sorted_p_values_input_output: HybridArray,\
        num_discoveries_output: HybridArray,\
        discover_method: str = 'argmin',\
        use_njit: bool|None = None) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = sorted_p_values_input_output.gpu_grid_block_shapes()
        higher_criticism_stable_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            higher_criticism_stable_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            higher_criticism_stable_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
    discover_by_method(transformed_p_values_input=sorted_p_values_input_output,\
                        num_discoveries_output=num_discoveries_output,\
                        discover_method=discover_method,\
                        use_njit=use_njit)

def higher_criticism_unstable(\
        sorted_p_values_input_output: HybridArray,\
        num_discoveries_output: HybridArray,\
        discover_method: str = 'argmin',\
        use_njit: bool|None = None) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = sorted_p_values_input_output.gpu_grid_block_shapes()
        higher_criticism_unstable_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            higher_criticism_unstable_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            higher_criticism_unstable_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
    discover_by_method(transformed_p_values_input=sorted_p_values_input_output,\
                        num_discoveries_output=num_discoveries_output,\
                        discover_method=discover_method,\
                        use_njit=use_njit)


def berk_jones(\
        sorted_p_values_input_output: HybridArray,\
        num_discoveries_output: HybridArray,\
        discover_method: str = 'argmin',\
        use_njit: bool|None = None,\
        grid_block_shape_debug: int|None = None) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape =\
            sorted_p_values_input_output.gpu_grid_block_shapes(\
                debug=grid_block_shape_debug,\
                registers_per_thread=256)
        berk_jones_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            berk_jones_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            berk_jones_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
    discover_by_method(transformed_p_values_input=sorted_p_values_input_output,\
                        num_discoveries_output=num_discoveries_output,\
                        discover_method=discover_method,\
                        use_njit=use_njit)

def discover_by_method(\
        transformed_p_values_input: HybridArray,\
        num_discoveries_output: HybridArray,\
        discover_method: str = 'argmin',\
        use_njit: bool|None = None) -> None:
    if discover_method == 'argmin':
        discover_argmin(transformed_p_values_input=transformed_p_values_input,\
                        num_discoveries_output=num_discoveries_output,\
                        use_njit=use_njit)
    else:
        discover_dominant(transformed_p_values_input=transformed_p_values_input,\
                        num_discoveries_output=num_discoveries_output,\
                        use_njit=use_njit)

def discover_argmin(\
        transformed_p_values_input: HybridArray,\
        num_discoveries_output: HybridArray,\
        use_njit: bool|None = None) -> None:
    num_discoveries_output.realloc(like=transformed_p_values_input, dtype=np.uint32)
    if transformed_p_values_input.is_gpu():
        # GPU mode
        grid_shape, block_shape = transformed_p_values_input.rows_gpu_grid_block_shapes()
        discover_argmin_gpu[grid_shape, block_shape](transformed_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            discover_argmin_cpu_njit(transformed_p_values_input=transformed_p_values_input.numpy(),\
                                     num_discoveries_output=num_discoveries_output.numpy())
        else:
            discover_argmin_py(\
                transformed_p_values_input=transformed_p_values_input.numpy(),\
                num_discoveries_output=num_discoveries_output.numpy())


def discover_dominant(\
        transformed_p_values_input: HybridArray,\
        num_discoveries_output: HybridArray,\
        use_njit: bool|None = None) -> None:
    num_discoveries_output.realloc(like=transformed_p_values_input, dtype=np.uint32)
    if transformed_p_values_input.is_gpu():
        # GPU mode
        grid_shape, block_shape = transformed_p_values_input.rows_gpu_grid_block_shapes()
        discover_dominant_gpu[grid_shape, block_shape](transformed_p_values_input.gpu_data(), num_discoveries_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            discover_dominant_cpu_njit(transformed_p_values_input=transformed_p_values_input.numpy(),\
                                     num_discoveries_output=num_discoveries_output.numpy())
        else:
            discover_dominant_py(\
                transformed_p_values_input=transformed_p_values_input.numpy(),\
                num_discoveries_output=num_discoveries_output.numpy())

