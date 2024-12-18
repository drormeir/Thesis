from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, simple_data_size_to_grid_block, HybridArray


def random_p_values_matrix(data: HybridArray,\
                           offset_row0: int|np.uint32,\
                           offset_col0: int|np.uint32,\
                           num_steps: int|np.uint32 = 1,\
                           use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = np.uint32(num_steps)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = calculate_grid_and_block_2d(data)
        random_p_values_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)
        else:
            random_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)

def random_integers_matrix(data: HybridArray, offset_row0: int|np.uint32, offset_col0: int|np.uint32, num_steps: int|np.uint32 = 1, use_njit: bool|None = None) -> None:
    data.astype(np.uint64)
    offset_row0 = np.uint32(offset_row0)
    offset_col0 = np.uint32(offset_col0)
    num_steps = np.uint32(num_steps)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = calculate_grid_and_block_2d(data)
        random_integers_matrix_gpu[grid_shape, block_shape](num_steps, offset_row0, offset_col0, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_integers_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)
        else:
            random_integers_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=data.data)


def calculate_grid_and_block_2d(data: HybridArray) -> tuple[tuple, tuple]:
    data_shape = data.shape()
    if len(data_shape) != 2:
        raise ValueError("This function only supports 2D data.")

    grid_shape_x, block_shape_x = simple_data_size_to_grid_block(data_shape[1], suggested_block_size=globals.max_threads_per_block)
    grid_shape_y, block_shape_y = simple_data_size_to_grid_block(data_shape[0], suggested_block_size=1)

    block_shape = (block_shape_y, block_shape_x)
    grid_shape = (grid_shape_y, grid_shape_x)

    return grid_shape, block_shape    

def splitmix64_matrix(states: np.ndarray,\
               out_states: HybridArray,\
               out_z: HybridArray,\
               use_gpu: bool,\
               use_njit: bool|None = None) -> None:
    assert states.dtype == np.uint64
    states_array = HybridArray().clone_from_numpy(states, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_states.realloc_like(states_array)
        out_z.realloc_like(states_array)
        grid_shape, block_shape = calculate_grid_and_block_2d(states_array)
        splitmix64_matrix_gpu[grid_shape, block_shape](states_array.data, out_states.data, out_z.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(states_array.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
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
        use_njit: bool|None = None) -> None:
    assert seeds.dtype == np.uint64
    seeds_array = HybridArray().clone_from_numpy(seeds, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        out_s0.realloc_like(seeds_array)
        out_s1.realloc_like(seeds_array)
        grid_shape, block_shape = calculate_grid_and_block_2d(seeds_array)
        random_integer_base_states_matrix_gpu[grid_shape, block_shape](seeds_array.data, out_s0.data, out_s1.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(seeds_array.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            out_s0_data, out_s1_data = random_integer_base_states_from_seeds_cpu_njit(seeds=seeds_array.data)
        else:
            out_s0_data, out_s1_data = random_integer_base_states_from_seeds_py(seeds=seeds_array.data)
        out_s0.clone_from_numpy(out_s0_data)
        out_s1.clone_from_numpy(out_s1_data)
    seeds_array.close()

###########################################################################################################

def random_p_values_series(data: HybridArray, seed: int|np.uint64, use_njit: bool|None = None) -> None:
    data.astype(np.float64)
    seed = np.uint64(seed)
    assert data.ndim() == 1
    if data.is_gpu():
        # GPU mode
        assert isinstance(data.data, DeviceNDArray)
        assert data.data.ndim == 1, f"Expected 1D array, got {data.data.ndim}D"
        random_p_values_series_gpu[1, 1](seed, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_series_cpu_njit(seed=seed, out=data.data)
        else:
            random_p_values_series_py(seed=seed, out=data.data)


def random_integers_series(data: HybridArray, seed: int|np.uint64, use_njit: bool|None = None) -> None:
    data.astype(np.uint64)
    seed = np.uint64(seed)
    if data.is_gpu():
        # GPU mode
        assert isinstance(data.data, DeviceNDArray)
        random_integers_series_gpu[1, 1](seed, data.data) # type: ignore
    else:
        # CPU mode
        assert isinstance(data.data, np.ndarray)
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            random_integers_series_cpu_njit(seed=seed, out=data.data)
        else:
            random_integers_series_py(seed=seed, out=data.data)


##########################################################################

def random_p_values_matrix_py(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
    random_integers_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
    out += 0.5
    out /= np.float64(2.0**64)

def random_integers_matrix_py(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
    col_seeds = np.arange(offset_col0, offset_col0 + out.shape[1], dtype=np.uint64).reshape(1,-1)
    row_seeds = np.arange(offset_row0, offset_row0 + out.shape[0], dtype=np.uint64).reshape(-1,1)
    seeds = (row_seeds << np.uint64(32)) + col_seeds
    if num_steps == np.uint32(-1):
        out[:] = seeds
        return
    s0, s1 = random_integer_base_states_from_seeds_py(seeds=seeds)
    if num_steps == np.uint32(-2):
        out[:] = s0
        return
    if num_steps == np.uint32(-3):
        out[:] = s1
        return
    for _ in range(num_steps):
        s0, s1 = random_integer_states_transition_from_states_py(s0=s0, s1=s1)
    random_integer_result_from_states_py(s0=s0, s1=s1, result=out)


def random_p_values_series_py(seed: np.uint64, out: np.ndarray) -> None:
    norm_factor = 1.0 / np.float64(2.0**64)
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        rand_int = random_integer_result_py(s0=s0, s1=s1)
        out[i] = (rand_int + 0.5) * norm_factor

def random_integers_series_py(seed: np.uint64, out: np.ndarray) -> None:
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        out[i] = random_integer_result_py(s0=s0, s1=s1)

def random_integer_base_states_from_seeds_py(seeds: np.ndarray)-> tuple[np.ndarray,np.ndarray]:
    splitmix_states     = seeds
    s0, splitmix_states = splitmix64_from_states_py(splitmix_states)
    s1, splitmix_states = splitmix64_from_states_py(splitmix_states)
    return s0, s1

def random_integer_base_states_py(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
    splitmix_state     = seed
    s0, splitmix_state = splitmix64_py(splitmix_state)
    s1, splitmix_state = splitmix64_py(splitmix_state)
    return s0, s1

def random_integer_states_transition_from_states_py(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    s1 ^= s0
    s0 = rotl64_array_py(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64_array_py(s1, np.uint64(28))
    return s0, s1

def random_integer_states_transition_py(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
    s1 ^= s0
    s0 = rotl64_py(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64_py(s1, np.uint64(28))
    return s0, s1

def random_integer_result_from_states_py(s0: np.ndarray, s1: np.ndarray, result: np.ndarray) -> None:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result[:] = rotl64_array_py(s0 + s1, np.uint64(17)) + s0


def random_integer_result_py(s0: np.uint64, s1: np.uint64) -> np.uint64:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result64 = rotl64_py(s0 + s1, np.uint64(17)) + s0
    return result64

def splitmix64_from_states_py(states: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        states += np.uint64(0x9E3779B97F4A7C15)
        z = states
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, states

def splitmix64_py(state: np.uint64) -> tuple[np.uint64,np.uint64]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state

def rotl64_array_py(x: np.ndarray, k: np.uint64) -> np.ndarray:
    return (x << k) | (x >> (np.uint64(64) - k))

def rotl64_py(x: np.uint64, k: np.uint64) -> np.uint64:
    return (x << k) | (x >> (np.uint64(64) - k))

###########################################################################################################

if not globals.cpu_njit_num_threads:
    # Mock API
    def random_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_integers_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_series_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_integers_series_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numba

    @numba.njit(parallel=False)
    def random_p_values_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
        out_uint64 = np.empty_like(out, dtype=np.uint64)
        random_integers_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out_uint64)
        out = (out_uint64+0.5) / np.float64(2.0**64)

    @numba.njit(parallel=False)
    def random_integers_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
        col_seeds = np.arange(offset_col0, offset_col0 + out.shape[1], dtype=np.uint64).reshape(1,-1)
        row_seeds = np.arange(offset_row0, offset_row0 + out.shape[0], dtype=np.uint64).reshape(-1,1)
        seeds = (row_seeds << np.uint64(32)) + col_seeds
        s0, s1 = random_integer_base_states_from_seeds_cpu_njit(seeds=seeds)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_from_states_cpu_njit(s0=s0, s1=s1)
        random_integer_result_from_states_cpu_njit(s0=s0, s1=s1, result=out)

    @numba.njit(parallel=False)
    def random_integer_cpu_njit(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0=s0, s1=s1)
        result64 = random_integer_result_cpu_njit(s0=s0, s1=s1)
        return result64

    @numba.njit(parallel=False)
    def random_p_values_series_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        norm_factor = 1.0 / np.float64(2.0**64)
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            rand_int = random_integer_result_cpu_njit(s0, s1)
            out[i] = (rand_int + 0.5) * norm_factor

    @numba.njit(parallel=False)
    def random_integers_series_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            out[i] = random_integer_result_cpu_njit(s0, s1)

    @numba.njit(parallel=False)
    def random_integer_base_states_from_seeds_cpu_njit(seeds: np.ndarray)-> tuple[np.ndarray,np.ndarray]:
        splitmix_states     = seeds
        s0, splitmix_states = splitmix64_from_states_cpu_njit(splitmix_states)
        s1, splitmix_states = splitmix64_from_states_cpu_njit(splitmix_states)
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_base_states_cpu_njit(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
        splitmix_state     = seed
        s0, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        s1, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_states_transition_from_states_cpu_njit(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        s1 ^= s0
        s0 = rotl64_array_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_array_cpu_njit(s1, np.uint64(28))
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_states_transition_cpu_njit(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
        s1 ^= s0
        s0 = rotl64_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_cpu_njit(s1, np.uint64(28))
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_result_from_states_cpu_njit(s0: np.ndarray, s1: np.ndarray, result: np.ndarray) -> None:
        result[:] = rotl64_array_cpu_njit(s0 + s1, np.uint64(17)) + s0

    @numba.njit(parallel=False)
    def random_integer_result_cpu_njit(s0: np.uint64, s1: np.uint64) -> np.uint64:
        result64 = rotl64_cpu_njit(s0 + s1, np.uint64(17)) + s0
        return result64


    @numba.njit(parallel=False)
    def splitmix64_from_states_cpu_njit(states: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        states += np.uint64(0x9E3779B97F4A7C15)
        z = states
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, states
    
    @numba.njit(parallel=False)
    def splitmix64_cpu_njit(state: np.uint64) -> tuple[np.uint64,np.uint64]:
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state
    
    @numba.njit(parallel=False)
    def rotl64_array_cpu_njit(x: np.ndarray, k: np.uint64) -> np.ndarray:
        return (x << k) | (x >> (np.uint64(64) - k))

    @numba.njit(parallel=False)
    def rotl64_cpu_njit(x: np.uint64, k: np.uint64) -> np.uint64:
        return (x << k) | (x >> (np.uint64(64) - k))

###########################################################################################################

if not globals.cuda_available:
    # Mock API
    def random_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integers_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_series_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integers_series_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()

else:
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    @numba.cuda.jit(device=False)
    def random_p_values_matrix_gpu(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: DeviceNDArray) -> None:
        norm_factor = 1.0 / np.float64(2.0**64)
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = (np.uint64(offset_row0 + ind_row) << np.uint64(32)) + offset_col0
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                rand_int = random_integer_gpu(seed_row + np.uint64(ind_col), num_steps)
                out_row[ind_col] = (rand_int + 0.5) * norm_factor

    @numba.cuda.jit(device=False)
    def random_integers_matrix_gpu(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = (np.uint64(offset_row0 + ind_row) << np.uint64(32)) + offset_col0
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                out_row[ind_col] = random_integer_gpu(seed_row + np.uint64(ind_col), num_steps)
    
    @numba.cuda.jit(device=False)
    def splitmix64_matrix_gpu(states: DeviceNDArray, new_states: DeviceNDArray, z: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, z.shape[0], row_stride):
            states_row = states[ind_row]
            new_states_row = new_states[ind_row]
            z_row = z[ind_row]
            for ind_col in range(ind_col0, z.shape[1], col_stride):
                z_row[ind_col], new_states_row[ind_col] = splitmix64_gpu(states_row[ind_col])

    @numba.cuda.jit(device=False)
    def random_integer_base_states_matrix_gpu(seeds: DeviceNDArray, s0: DeviceNDArray, s1: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, seeds.shape[0], row_stride):
            seeds_row = seeds[ind_row]
            s0_row = s0[ind_row]
            s1_row = s1[ind_row]
            for ind_col in range(ind_col0, seeds.shape[1], col_stride):
                s0_row[ind_col], s1_row[ind_col] = random_integer_base_states_gpu(seeds_row[ind_col])

    @numba.cuda.jit(device=False)
    def random_p_values_series_gpu(seed: np.uint64, out: DeviceNDArray) -> None:
        norm_factor = 1.0 / np.float64(2.0**64)
        s0, s1 = random_integer_base_states_gpu(seed)
        num_steps = out.size
        ind_start = numba.cuda.grid(1) # type: ignore
        ind_stride = numba.cuda.gridsize(1) # type: ignore
        for i in range(ind_start, num_steps, ind_stride):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
            rand_int = random_integer_result_gpu(s0, s1)
            out[i] = (rand_int + 0.5) * norm_factor
        
    @numba.cuda.jit(device=False)
    def random_integers_series_gpu(seed: np.uint64, out: DeviceNDArray):
        s0, s1 = random_integer_base_states_gpu(seed)
        num_steps = out.size
        ind_start = numba.cuda.grid(1) # type: ignore
        ind_stride = numba.cuda.gridsize(1) # type: ignore
        for i in range(ind_start, num_steps, ind_stride):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
            out[i] = random_integer_result_gpu(s0, s1)

                
    @numba.cuda.jit(device=True)
    def random_integer_gpu(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        if num_steps == np.uint32(-1): # debug mode
            return seed
        s0, s1 = random_integer_base_states_gpu(seed)
        if num_steps == np.uint32(-2): # debug mode
            return s0
        if num_steps == np.uint32(-3): # debug mode
            return s1
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
        result64 = random_integer_result_gpu(s0, s1)
        return result64
    
    @numba.cuda.jit(device=True)
    def random_integer_base_states_gpu(seed: np.uint64) -> tuple[np.uint64,np.uint64]:
        splitmix_state     = seed
        s0, splitmix_state = splitmix64_gpu(splitmix_state)
        s1, splitmix_state = splitmix64_gpu(splitmix_state)
        return s0, s1

    @numba.cuda.jit(device=True)
    def random_integer_states_transition_gpu(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
        s1 ^= s0
        s0 = rotl64_gpu(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_gpu(s1, np.uint64(28))
        return s0, s1

    @numba.cuda.jit(device=True)
    def random_integer_result_gpu(s0: np.uint64, s1: np.uint64) -> np.uint64:
        result64 = rotl64_gpu(s0 + s1, np.uint64(17)) + s0
        return result64

    @numba.cuda.jit(device=True)
    def splitmix64_gpu(state: np.uint64) -> tuple[np.uint64, np.uint64]:
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state

    @numba.cuda.jit(device=True)
    def rotl64_gpu(x: np.uint64, k: np.uint64) -> np.uint64:
        return (x << k) | (x >> (np.uint64(64) - k))



