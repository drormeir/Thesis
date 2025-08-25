import numpy as np
from python.hpc import is_use_njit, HybridArray
from python.random_integers import python_native, numba_cpu, numba_gpu



def get_num_steps(**kwargs) -> np.uint32:
    num_steps = kwargs.get('num_steps', None)
    return np.uint32(2) if num_steps is None else np.uint32(num_steps)


def matrix(data: HybridArray,\
            offset_row0: int|np.uint32,\
            offset_col0: int|np.uint32,\
            **kwargs) -> None:
    data.astype(np.uint64)
    assert data.dtype() == np.uint64, f'{data.dtype()=}'
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = get_num_steps(**kwargs)
    if data.is_gpu():
        # GPU mode
        assert data.gpu_data().dtype == np.uint64, f'{data.dtype()=} {data.gpu_data().dtype=}'
        grid_shape, block_shape = data.gpu_grid_block2D_square_shapes()
        numba_gpu.matrix[grid_shape, block_shape](num_steps, offset_row0, offset_col0, data.gpu_data()) # type: ignore
    else:
        # CPU mode
        assert data.numpy().dtype == np.uint64, f'{data.dtype()=} {data.numpy().dtype=}'
        if is_use_njit(**kwargs):
            numba_cpu.matrix(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())
        else:
            python_native.matrix(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.numpy())


def matrix_splitmix64(states: np.ndarray,\
               out_states: HybridArray,\
            out_z: HybridArray,
            use_gpu: bool,
            **kwargs) -> None:
    assert states.dtype == np.uint64
    states_array = HybridArray().clone_from_numpy(states, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_states.realloc_like(states_array)
        out_z.realloc_like(states_array)
        grid_shape, block_shape = states_array.gpu_grid_block2D_square_shapes()
        numba_gpu.matrix_splitmix64[grid_shape, block_shape](states_array.data, out_states.data, out_z.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(states_array.data, np.ndarray)
        if is_use_njit(**kwargs):
            out_z_data, out_states_data = numba_cpu.matrix_splitmix64(states=states_array.data)
        else:
            out_z_data, out_states_data = python_native.matrix_splitmix64(states=states_array.data)
        out_states.clone_from_numpy(out_states_data)
        out_z.clone_from_numpy(out_z_data)
    states_array.close()


def matrix_base_states(
        seeds: np.ndarray,\
            out_s0: HybridArray,
            out_s1: HybridArray,
            use_gpu: bool,
        **kwargs) -> None:
    assert seeds.dtype == np.uint64
    seeds_array = HybridArray().clone_from_numpy(seeds, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_s0.realloc_like(seeds_array)
        out_s1.realloc_like(seeds_array)
        grid_shape, block_shape = seeds_array.gpu_grid_block2D_square_shapes()
        numba_gpu.matrix_base_states[grid_shape, block_shape](seeds_array.data, out_s0.data, out_s1.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(seeds_array.data, np.ndarray)
        if is_use_njit(**kwargs):
            out_s0_data, out_s1_data = numba_cpu.matrix_base_states(seeds=seeds_array.data)
        else:
            out_s0_data, out_s1_data = python_native.matrix_base_states(seeds=seeds_array.data)
        out_s0.clone_from_numpy(out_s0_data)
        out_s1.clone_from_numpy(out_s1_data)
    seeds_array.close()


def series(data: HybridArray, seed: int|np.uint64, **kwargs) -> None:
    data.astype(np.uint64)
    seed = np.uint64(seed)
    if data.is_gpu():
        # GPU mode
        numba_gpu.series[1, 1](seed, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if is_use_njit(**kwargs):
            numba_cpu.series(seed=seed, out=data.data)
        else:
            python_native.series(seed=seed, out=data.data)


def matrix_2_p_values(integers: HybridArray, p_values: HybridArray, **kwargs) -> None:
    assert integers.dtype() == np.uint64, f'{integers.dtype()=}'
    p_values.realloc(like=integers, dtype=np.float64)
    if integers.is_gpu():
        # GPU mode
        grid_shape, block_shape = integers.gpu_grid_block2D_square_shapes()
        numba_gpu.matrix_2_p_values[grid_shape, block_shape](integers.gpu_data(), p_values.gpu_data()) # type: ignore
    else:
        # CPU mode
        if is_use_njit(**kwargs):
            numba_cpu.matrix_2_p_values(integers.numpy(), p_values.numpy())
        else:
            python_native.matrix_2_p_values(integers.numpy(), p_values.numpy())
