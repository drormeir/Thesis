import numpy as np
from python.hpc import use_njit, HybridArray
from python.adaptive_methods.numba_gpu import higher_criticism_stable_gpu, higher_criticism_unstable_gpu, berk_jones_gpu
from python.adaptive_methods.numba_cpu import higher_criticism_stable_cpu_njit, higher_criticism_unstable_cpu_njit, berk_jones_cpu_njit
from python.adaptive_methods.python_native import higher_criticism_stable_py, higher_criticism_unstable_py, berk_jones_py
from python.array_math_utils.array_math_utils import cumulative_argmin, cumulative_min_inplace, cumulative_dominant_argmin, cumulative_dominant_min_inplace


def apply_transform_discovery_method(\
        sorted_p_values_input_output: HybridArray,\
        num_discoveries_output: HybridArray|None,\
        **kwargs) -> None:
    apply_transform_method(\
        sorted_p_values_input_output=sorted_p_values_input_output,\
        **kwargs)
    if num_discoveries_output is None:
        apply_discovery_method_on_transformation(\
            transformed_p_values_input=sorted_p_values_input_output,\
            **kwargs)
    else:
        discover_by_method(\
            transformed_p_values_input=sorted_p_values_input_output,\
            num_discoveries_output=num_discoveries_output,\
            **kwargs)
        
def apply_transform_method(\
        sorted_p_values_input_output: HybridArray,\
        transform_method: str,\
        **kwargs) -> None:
    if transform_method == 'higher_criticism_stable':
        higher_criticism_stable(sorted_p_values_input_output,**kwargs)
    elif transform_method == 'higher_criticism_unstable':
        higher_criticism_unstable(sorted_p_values_input_output,**kwargs)
    elif transform_method == 'berk_jones':
        berk_jones(sorted_p_values_input_output,**kwargs)
    else:
        assert False, f'{transform_method=}'
        
def higher_criticism_stable(sorted_p_values_input_output: HybridArray, **kwargs) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = sorted_p_values_input_output.gpu_grid_block2D_square_shapes()
        higher_criticism_stable_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            higher_criticism_stable_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            higher_criticism_stable_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())

def higher_criticism_unstable(sorted_p_values_input_output: HybridArray, **kwargs) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = sorted_p_values_input_output.gpu_grid_block2D_square_shapes()
        higher_criticism_unstable_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            higher_criticism_unstable_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            higher_criticism_unstable_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())


def berk_jones(\
        sorted_p_values_input_output: HybridArray,\
        **kwargs) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape =\
            sorted_p_values_input_output.gpu_grid_block2D_square_shapes(\
                registers_per_thread=256)
        berk_jones_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            berk_jones_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            berk_jones_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())

def apply_discovery_method_on_transformation(\
        transformed_p_values_input: HybridArray,\
        discovery_method: str = 'argmin',\
        **kwargs) -> None:
    if discovery_method == 'argmin':
        cumulative_min_inplace(array=transformed_p_values_input, **kwargs)
    else:
        cumulative_dominant_min_inplace(array=transformed_p_values_input, **kwargs)

def discover_by_method(\
        transformed_p_values_input: HybridArray,\
        num_discoveries_output: HybridArray,\
        discovery_method: str,\
        **kwargs) -> None:
    if discovery_method == 'argmin':
        cumulative_argmin(array=transformed_p_values_input,\
                        argmin=num_discoveries_output,\
                        **kwargs)
    else:
        cumulative_dominant_argmin(array=transformed_p_values_input,\
                        argmin=num_discoveries_output,\
                        **kwargs)

