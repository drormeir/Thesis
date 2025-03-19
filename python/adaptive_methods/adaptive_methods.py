import numpy as np
from tqdm import tqdm
from python.hpc import use_njit, HybridArray
from python.adaptive_methods.numba_gpu import higher_criticism_gpu, higher_criticism_unstable_gpu, berk_jones_gpu, calc_lgamma_gpu, berk_jones_gpu_max_iter, berk_jones_legacy_gpu_max_iter, berk_jones_gpu_execute
from python.adaptive_methods.numba_cpu import higher_criticism_cpu_njit, higher_criticism_unstable_cpu_njit, berk_jones_cpu_njit, calc_lgamma_cpu_njit
from python.adaptive_methods.python_native import higher_criticism_py, higher_criticism_unstable_py, berk_jones_py, calc_lgamma_py
from python.array_math_utils.array_math_utils import cumulative_argmin, cumulative_argmax, cumulative_min_inplace, cumulative_max_inplace, cumulative_dominant_argmin, cumulative_dominant_argmax, cumulative_dominant_min_inplace, cumulative_dominant_max_inplace
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis

available_transform_methods = ['higher_criticism', 'higher_criticism_unstable','berk_jones','identity']

def test_speed_transforms(\
        N: int,\
        num_monte: int,\
        num_executions: int,\
        transform_method: str,\
        use_gpu: bool|None=None,\
        lgamma_cache: HybridArray|None=None,\
        **kwargs) -> None:
    assert transform_method in available_transform_methods
    desc = f'Test Speed Transforms {transform_method=}'
    local_lgamma = lgamma_cache is None
    if local_lgamma:
        lgamma_cache = HybridArray()
    with HybridArray().realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu) as noise:
        for ind_execution in tqdm(range(num_executions), desc=desc, unit="step"):
            rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_execution, **kwargs)
            apply_transform_method(sorted_p_values_input_output=noise,\
                                   transform_method=transform_method,\
                                   lgamma_cache = lgamma_cache,\
                                   **kwargs)
            pass
        pass
    if local_lgamma:
        lgamma_cache.close()


def test_speed_berk_jones(\
        N: int,\
        num_monte: int,\
        num_executions: int,\
        use_gpu: bool|None=None,\
        lgamma_cache: HybridArray|None=None,\
        **kwargs) -> None:
    desc = f'Test Speed Berk Jones {use_gpu=} use_njit={use_njit(**kwargs)}'
    local_lgamma = lgamma_cache is None
    if local_lgamma:
        lgamma_cache = HybridArray()
    with HybridArray().realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu) as noise:
        for ind_execution in tqdm(range(num_executions), desc=desc, unit="step"):
            rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_execution, **kwargs)
            berk_jones(sorted_p_values_input_output=noise,lgamma_cache=lgamma_cache,**kwargs)
            pass
        pass
    if local_lgamma:
        lgamma_cache.close()


def berk_jones_max_iter(\
        max_iter_output: HybridArray,
        max_iter_legacy_output: HybridArray,
        N: int,\
        num_monte: int,\
        lgamma_cache: HybridArray|None=None,\
        **kwargs) -> None:
    local_lgamma = lgamma_cache is None
    if local_lgamma:
        lgamma_cache = HybridArray()
    calc_lgamma(lgamma_cache, N, use_gpu=True)
    max_iter_output.realloc(shape=(num_monte,N), dtype=np.uint32, use_gpu=True)
    max_iter_legacy_output.realloc_like(max_iter_output)
    grid_shape, block_shape = max_iter_output.gpu_grid_block2D_square_shapes(registers_per_thread=128)
    with HybridArray().realloc(like=max_iter_output, dtype=np.float64) as noise:
        rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=0, **kwargs)
        berk_jones_gpu_max_iter[grid_shape, block_shape](noise.gpu_data(), lgamma_cache.gpu_data(), max_iter_output.gpu_data()) # type: ignore
        rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=0, **kwargs)
        berk_jones_legacy_gpu_max_iter[grid_shape, block_shape](noise.gpu_data(), lgamma_cache.gpu_data(), max_iter_legacy_output.gpu_data()) # type: ignore
    if local_lgamma:
        lgamma_cache.close()


def apply_transform_discovery_method(\
        sorted_p_values_input_output: HybridArray,\
        num_discoveries_output: HybridArray|None,\
        transform_method: str,\
        discover_dominant: bool|None = None,\
        discover_min: bool|None = None,\
        **kwargs) -> None:
    assert transform_method in available_transform_methods, f'{transform_method=} not in {available_transform_methods=}'
    apply_transform_method(\
        sorted_p_values_input_output=sorted_p_values_input_output,\
        transform_method=transform_method,\
        **kwargs)
    if transform_method == 'identity':
        assert discover_dominant is None or not discover_dominant
        assert discover_min is None or not discover_min
        discover_dominant = False
        discover_min = False
    else:
        if discover_dominant is None:
            discover_dominant = False
        if discover_min is None:
            discover_min = True
    assert discover_dominant is not None
    assert discover_min is not None
    if num_discoveries_output is None:
        discover_by_method_inplace(\
            transformed_p_values_inoutput=sorted_p_values_input_output,\
            discover_dominant=discover_dominant, discover_min=discover_min,\
            **kwargs)
    else:
        discover_by_method_arg(\
            transformed_p_values_input=sorted_p_values_input_output,\
            num_discoveries_output=num_discoveries_output,\
            discover_dominant=discover_dominant, discover_min=discover_min,\
            **kwargs)
        

def apply_transform_method(\
        sorted_p_values_input_output: HybridArray,\
        transform_method: str,\
        **kwargs) -> None:
    assert transform_method in available_transform_methods, f'{transform_method=} not in {available_transform_methods=}'
    if transform_method == 'higher_criticism':
        higher_criticism(sorted_p_values_input_output,**kwargs)
    elif transform_method == 'higher_criticism_unstable':
        higher_criticism_unstable(sorted_p_values_input_output,**kwargs)
    elif transform_method == 'berk_jones':
        berk_jones(sorted_p_values_input_output,**kwargs)
    elif transform_method == 'identity':
        return
    else:
        assert False, f'{transform_method=}'
        

def higher_criticism(sorted_p_values_input_output: HybridArray, **kwargs) -> None:
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        grid_shape, block_shape = sorted_p_values_input_output.gpu_grid_block2D_square_shapes()
        higher_criticism_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            higher_criticism_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
        else:
            higher_criticism_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())


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
        lgamma_cache: HybridArray|None = None,\
        **kwargs) -> None:
    local_lgamma = lgamma_cache is None
    if local_lgamma:
        lgamma_cache = HybridArray()
    calc_lgamma(lgamma_cache, sorted_p_values_input_output.ncols(), use_gpu=sorted_p_values_input_output.is_gpu())
    if sorted_p_values_input_output.is_gpu():
        # GPU mode
        berk_jones_gpu(sorted_p_values_input_output, lgamma_cache, **kwargs)
    else:
        # CPU mode
        if use_njit(**kwargs):
            berk_jones_cpu_njit(sorted_p_values_input_output=sorted_p_values_input_output.numpy(),\
                                lgamma_cache=lgamma_cache.numpy())
        else:
            berk_jones_py(sorted_p_values_input_output=sorted_p_values_input_output.numpy())
    if local_lgamma:
        lgamma_cache.close()


def calc_lgamma(lgamma_cache: HybridArray, N: int|np.uint32, use_gpu: bool, **kwargs) -> None:
    # because I want to calc from lgamma(1) to lgamma(N+1) inclusive and put them in the same indexes
    N += 2
    if N <= lgamma_cache.size():
        return
    lgamma_cache.realloc(shape=(N,), dtype=np.float64, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        calc_lgamma_gpu[1, 1](lgamma_cache.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            calc_lgamma_cpu_njit(lgamma_cache.numpy())
        else:
            calc_lgamma_py(lgamma_cache.numpy())



def discover_by_method_inplace(\
        transformed_p_values_inoutput: HybridArray,\
        discover_dominant: bool,\
        discover_min: bool,\
        **kwargs) -> None:
    if discover_dominant:
        if discover_min:
            cumulative_dominant_min_inplace(array=transformed_p_values_inoutput, **kwargs)
        else:
            cumulative_dominant_max_inplace(array=transformed_p_values_inoutput, **kwargs)
    else:
        if discover_min:
            cumulative_min_inplace(array=transformed_p_values_inoutput, **kwargs)
        else:
            cumulative_max_inplace(array=transformed_p_values_inoutput, **kwargs)


def discover_by_method_arg(\
        transformed_p_values_input: HybridArray,\
        num_discoveries_output: HybridArray,\
        discover_dominant: bool,\
        discover_min: bool,\
        **kwargs) -> None:
    if discover_dominant:
        if discover_min:
            cumulative_dominant_argmin(array=transformed_p_values_input,\
                                       argmin=num_discoveries_output,\
                                       **kwargs)
        else:
            cumulative_dominant_argmax(array=transformed_p_values_input,\
                                       argmax=num_discoveries_output,\
                                       **kwargs)
    else:
        if discover_min:
            cumulative_argmin(array=transformed_p_values_input,\
                            argmin=num_discoveries_output,\
                            **kwargs)
        else:
            cumulative_argmax(array=transformed_p_values_input,\
                            argmin=num_discoveries_output,\
                            **kwargs)


def str_transform_method(\
        transform_method: str,\
        discover_dominant: bool|None = None,\
        discover_min: bool|None = None,\
        **kwargs) -> str:
    assert transform_method in available_transform_methods, f'{transform_method=} not in {available_transform_methods=}'
    if transform_method == 'identity':
        assert discover_dominant is None or not discover_dominant
        assert discover_min is None or not discover_min
        return f'Identity (p-value at alpha)'
    if discover_dominant is None:
        discover_dominant = False
    if discover_min is None:
        discover_min = True
    str_discover_min_max = 'Min' if discover_min else 'Max'
    if discover_dominant:
        return transform_method + ': Dominant ' + str_discover_min_max
    else:
        return transform_method + ': ' + str_discover_min_max + ' p-value'
