import numpy as np
from python.hpc import use_njit, HybridArray
from python.random_integers.numba_gpu import random_integers_matrix_gpu, splitmix64_matrix_gpu, random_integer_base_states_matrix_gpu, random_integers_series_gpu, random_integers_2_p_values_gpu
from python.random_integers.numba_cpu import random_integers_matrix_cpu_njit, splitmix64_from_states_cpu_njit, random_integer_base_states_from_seeds_cpu_njit, random_integers_series_cpu_njit, random_integers_2_p_values_cpu_njit
from python.random_integers.python_native import random_integers_matrix_py, splitmix64_from_states_py, random_integer_base_states_from_seeds_py, random_integers_series_py

def random_num_steps(**kwargs) -> np.uint32:
    num_steps = kwargs.get('num_steps', None)
    return np.uint32(2) if num_steps is None else np.uint32(num_steps)


def random_integers_matrix(data: HybridArray,\
                           offset_row0: int|np.uint32,\
                           offset_col0: int|np.uint32,\
                           **kwargs) -> None:
    data.astype(np.uint64)
    assert data.dtype() == np.uint64, f'{data.dtype()=}'
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = random_num_steps(**kwargs)
    if data.is_gpu():
        # GPU mode
        assert data.gpu_data().dtype == np.uint64, f'{data.dtype()=} {data.gpu_data().dtype=}'
        grid_shape, block_shape = data.gpu_grid_block2D_square_shapes()
        random_integers_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, data.gpu_data()) # type: ignore
    else:
        # CPU mode
        assert data.numpy().dtype == np.uint64, f'{data.dtype()=} {data.numpy().dtype=}'
        if use_njit(**kwargs):
            random_integers_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())
        else:
            random_integers_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())


def splitmix64_matrix(states: np.ndarray,\
               out_states: HybridArray,\
               out_z: HybridArray,\
               use_gpu: bool,\
               **kwargs) -> None:
    assert states.dtype == np.uint64
    states_array = HybridArray().clone_from_numpy(states, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_states.realloc_like(states_array)
        out_z.realloc_like(states_array)
        grid_shape, block_shape = states_array.gpu_grid_block2D_square_shapes()
        splitmix64_matrix_gpu[grid_shape, block_shape](states_array.data, out_states.data, out_z.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(states_array.data, np.ndarray)
        if use_njit(**kwargs):
            out_z_data, out_states_data = splitmix64_from_states_cpu_njit(states=states_array.data)
        else:
            out_z_data, out_states_data = splitmix64_from_states_py(states=states_array.data)
        out_states.clone_from_numpy(out_states_data)
        out_z.clone_from_numpy(out_z_data)
    states_array.close()


def random_integers_base_states_matrix(\
        seeds: np.ndarray,\
        out_s0: HybridArray,\
        out_s1: HybridArray,\
        use_gpu: bool,\
        **kwargs) -> None:
    assert seeds.dtype == np.uint64
    seeds_array = HybridArray().clone_from_numpy(seeds, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_s0.realloc_like(seeds_array)
        out_s1.realloc_like(seeds_array)
        grid_shape, block_shape = seeds_array.gpu_grid_block2D_square_shapes()
        random_integer_base_states_matrix_gpu[grid_shape, block_shape](seeds_array.data, out_s0.data, out_s1.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(seeds_array.data, np.ndarray)
        if use_njit(**kwargs):
            out_s0_data, out_s1_data = random_integer_base_states_from_seeds_cpu_njit(seeds=seeds_array.data)
        else:
            out_s0_data, out_s1_data = random_integer_base_states_from_seeds_py(seeds=seeds_array.data)
        out_s0.clone_from_numpy(out_s0_data)
        out_s1.clone_from_numpy(out_s1_data)
    seeds_array.close()


def random_integers_series(data: HybridArray, seed: int|np.uint64, **kwargs) -> None:
    data.astype(np.uint64)
    seed = np.uint64(seed)
    if data.is_gpu():
        # GPU mode
        random_integers_series_gpu[1, 1](seed, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if use_njit(**kwargs):
            random_integers_series_cpu_njit(seed=seed, out=data.data)
        else:
            random_integers_series_py(seed=seed, out=data.data)


def random_integers_2_p_values(integers: HybridArray, p_values: HybridArray, **kwargs) -> None:
    assert integers.dtype() == np.uint64, f'{integers.dtype()=}'
    p_values.realloc(like=integers, dtype=np.float64)
    if integers.is_gpu():
        # GPU mode
        grid_shape, block_shape = integers.gpu_grid_block2D_square_shapes()
        random_integers_2_p_values_gpu[grid_shape, block_shape](integers.gpu_data(), p_values.gpu_data()) # type: ignore
    else:
        # CPU mode
        if use_njit(**kwargs):
            random_integers_2_p_values_cpu_njit(integers.numpy(), p_values.numpy())
        else:
            p_values.numpy()[:] = (integers.numpy() + np.float64(0.5))/np.float64(2.0**64)
