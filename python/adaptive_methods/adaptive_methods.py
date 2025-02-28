import numpy as np
from tqdm import tqdm
from python.hpc import use_njit, HybridArray
from python.adaptive_methods.numba_gpu import higher_criticism_gpu, higher_criticism_unstable_gpu, berk_jones_gpu, calc_lgamma_gpu, berk_jones_gpu_max_iter, berk_jones_legacy_gpu_max_iter
from python.adaptive_methods.numba_cpu import higher_criticism_cpu_njit, higher_criticism_unstable_cpu_njit, berk_jones_cpu_njit, calc_lgamma_cpu_njit
from python.adaptive_methods.python_native import higher_criticism_py, higher_criticism_unstable_py, berk_jones_py, calc_lgamma_py
from python.array_math_utils.array_math_utils import cumulative_argmin, cumulative_min_inplace, cumulative_dominant_argmin, cumulative_dominant_min_inplace
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis


def test_speed_transforms(\
        N: int,\
        num_monte: int,\
        num_executions: int,\
        use_gpu: bool|None=None,\
        transform_method: str ='identity',
        lgamma_cache: HybridArray|None=None,\
        **kwargs) -> None:
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
    is_njit = use_njit(**kwargs)
    desc = f'Test Speed Berk Jones {use_gpu=} use_njit={is_njit}'
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
        debug = kwargs.get('debug_berk_jones_gpu_block_size', None)
        grid_shape, block_shape =\
            sorted_p_values_input_output.gpu_grid_block2D_columns_shapes(registers_per_thread=100, debug=debug)
        berk_jones_gpu[grid_shape, block_shape](sorted_p_values_input_output.gpu_data(), lgamma_cache.gpu_data()) # type: ignore
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

