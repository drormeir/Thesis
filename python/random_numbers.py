from python.cuda import cuda_available, cpu_njit_num_threads, raise_cuda_not_available, raise_njit_not_available, HybridArray, max_threads_per_block
import numpy as np


def random_p_values_matrix(data: HybridArray, seed_row0: int|np.uint64, seed_col0: int|np.uint64, use_njit: bool|None = None) -> HybridArray:
    seed_row0 = np.uint64(seed_row0)
    seed_col0 = np.uint64(seed_col0)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = calculate_grid_and_block_2d(data.shape())
        random_p_values_matrix_gpu[*grid_shape, *block_shape](seed0, data.data) # type: ignore
    else:
        # CPU mode
        if cpu_njit_num_threads and (use_njit is None or use_njit):
            random_p_values_matrix_cpu_njit(seed_row0=seed_row0, seed_col0=seed_col0, out=data.data)
        else:
            random_p_values_matrix_py(seed_row0=seed_row0, seed_col0=seed_col0, out=data.data)
    return data


def random_integers_matrix(data: HybridArray, seed_row0: int|np.uint64, seed_col0: int|np.uint64, num_steps: int|np.uint32 = 1, use_njit: bool|None = None) -> HybridArray:
    seed_row0 = np.uint64(seed_row0)
    seed_col0 = np.uint64(seed_col0)
    num_steps = np.uint32(num_steps)
    if data.is_gpu():
        # GPU mode
        grid_shape, block_shape = calculate_grid_and_block_2d(data.shape())
        random_integers_matrix_gpu[*grid_shape, *block_shape](seed0, data.data) # type: ignore
    else:
        # CPU mode
        if cpu_njit_num_threads and (use_njit is None or use_njit):
            random_integers_matrix_cpu_njit(seed_row0=seed_row0, seed_col0=seed_col0, num_steps=num_steps, out=data.data)
        else:
            random_integers_matrix_py(seed_row0=seed_row0, seed_col0=seed_col0, num_steps=num_steps, out=data.data)
    return data


def calculate_grid_and_block_2d(data_shape):
    """
    Calculate grid and block shapes for CUDA kernel execution (2D data only).

    Parameters:
        data_shape (tuple): Shape of the 2D data as (rows, cols).

    Returns:
        tuple: (grid_shape, block_shape)
    """
    if not cuda_available:
        raise_cuda_not_available()    

    if len(data_shape) != 2:
        raise ValueError("This function only supports 2D data.")

    block_shape = (1, max_threads_per_block)

    grid_shape = (
        (data_shape[0] + block_shape[0] - 1) // block_shape[0],  # Rows
        (data_shape[1] + block_shape[1] - 1) // block_shape[1]   # Columns
    )

    return grid_shape, block_shape    


###########################################################################################################

def random_p_values_matrix_py(seed_row0: np.uint64, seed_col0: np.uint64, out: np.ndarray) -> None:
    norm_factor = 1.0 / np.float64(2**64)
    for ind_row in np.arange(out.shape[0], dtype=np.uint64):
        for ind_col in np.arange(out.shape[1], dtype=np.uint64):
            seed = (ind_row+seed_row0) << np.uint64(32) | (ind_col+seed_col0)
            rand_int = random_integer_py(seed=seed, num_steps=np.uint32(1))
            out[ind_row][ind_col] = (rand_int + 0.5) * norm_factor

def random_integers_matrix_py(seed_row0: np.uint64, seed_col0: np.uint64, num_steps: np.uint32, out: np.ndarray) -> None:
    """
    Generates random numbers using a xoroshiro128++.
    
    Parameters:
        seed_row0 (int): base seed for first row
        seed_col0 (int): base seed for first column
        out (numpy array): Output array to store random values.
    """    
    for ind_row in np.arange(out.shape[0], dtype=np.uint64):
        for ind_col in np.arange(out.shape[1], dtype=np.uint64):
            seed = (ind_row+seed_row0) << np.uint64(32) | (ind_col+seed_col0)
            out[ind_row][ind_col] = random_integer_py(seed=seed, num_steps=num_steps)

def random_integer_py(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
    s0, s1 = random_integer_base_states_py(seed=seed)
    for _ in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
    result64 = random_integer_result_py(s0=s0, s1=s1)
    return result64

def random_p_values_vector_py(seed: np.uint64, out: np.ndarray) -> None:
    norm_factor = 1.0 / np.float64(2**64)
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        rand_int = random_integer_result_py(s0=s0, s1=s1)
        out[i] = (rand_int + 0.5) * norm_factor

def random_integers_vector_py(seed: np.uint64, out: np.ndarray) -> None:
    """
    Generates random numbers using a xoroshiro128++.
    
    Parameters:
        seed (int): base seed
        out (numpy array): Output array to store random values.
    """
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        out[i] = random_integer_result_py(s0=s0, s1=s1)

def random_integer_base_states_py(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
    splitmix_state     = seed
    s0, splitmix_state = splitmix64_py(splitmix_state)
    s1, splitmix_state = splitmix64_py(splitmix_state)
    return s0, s1

def random_integer_states_transition_py(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
    s1 ^= s0
    s0 = rotl64_py(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64_py(s1, np.uint64(28))
    return s0, s1

def random_integer_result_py(s0: np.uint64, s1: np.uint64) -> np.uint64:
    result64 = rotl64_py(s0 + s1, np.uint64(17)) + s0
    return result64

def splitmix64_py(state: np.uint64) -> tuple[np.uint64,np.uint64]:
    state += np.uint64(0x9E3779B97F4A7C15)
    z = state
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return z, state

def rotl64_py(x: np.uint64, k: np.uint64) -> np.uint64:
    return (x << k) | (x >> (np.uint64(64) - k))

###########################################################################################################

if not cpu_njit_num_threads:
    def random_p_values_matrix_cpu_njit(seed_row0, seed_col0, out) -> None: # type: ignore
        raise_njit_not_available()
    def random_integers_matrix_cpu_njit(seed_row0, seed_col0, num_steps, out) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_vector_cpu_njit(seed, out) -> None: # type: ignore
        raise_njit_not_available()
    def random_integers_vector_cpu_njit(seed, out) -> None: # type: ignore
        raise_njit_not_available()
    def random_integer_cpu_njit(seed, num_steps) -> np.uint64: # type: ignore
        raise_njit_not_available()
    def random_integer_base_states_cpu_njit(seed)-> tuple:  # type: ignore       
        raise_njit_not_available()
    def random_integer_states_transition_cpu_njit(s0, s1) -> tuple:  # type: ignore       
        raise_njit_not_available()
    def random_integer_result_cpu_njit(s0, s1) -> np.uint64:  # type: ignore       
        raise_njit_not_available()
    def splitmix64_cpu_njit(state) -> tuple:  # type: ignore       
        raise_njit_not_available()
    def rotl64_cpu_njit(x, k) -> np.uint64:  # type: ignore       
        raise_njit_not_available()
else:
    import numba

    @numba.njit(parallel=True)
    def random_p_values_matrix_cpu_njit(seed_row0: np.uint64, seed_col0: np.uint64, out: np.ndarray) -> None:
        norm_factor = 1.0 / np.float64(2**64)
        num_rows, num_cols = out.shape
        num_cells = num_rows * num_cols
        for ind_cell in numba.prange(num_cells):
            ind_row = ind_cell // out.shape[0]
            ind_col = ind_cell % out.shape[0]
            seed = np.uint64(ind_row+seed_row0) << np.uint64(32) | np.uint64(ind_col+seed_col0)
            rand_int = random_integer_cpu_njit(seed=seed, num_steps=np.uint32(1))
            out[ind_row][ind_col] = (rand_int + 0.5) * norm_factor

    @numba.njit(parallel=True)
    def random_integers_matrix_cpu_njit(seed_row0: np.uint64, seed_col0: np.uint64, num_steps: np.uint32, out: np.ndarray) -> None:
        """
        Generates random numbers using a xoroshiro128++.
        
        Parameters:
            seed0 (int): base seed
            out (numpy array): Output array to store random values.
        """
        num_rows, num_cols = out.shape
        num_cells = num_rows * num_cols
        for ind_cell in numba.prange(num_cells):
            ind_row = ind_cell // out.shape[0]
            ind_col = ind_cell % out.shape[0]
            seed = np.uint64(ind_row+seed_row0) << np.uint64(32) | np.uint64(ind_col+seed_col0)
            out[ind_row][ind_col] = random_integer_cpu_njit(seed=seed, num_steps=num_steps)


    @numba.njit(parallel=False)
    def random_integer_cpu_njit(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0=s0, s1=s1)
        result64 = random_integer_result_cpu_njit(s0=s0, s1=s1)
        return result64

    @numba.njit(parallel=False)
    def random_p_values_vector_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        norm_factor = 1.0 / np.float64(2**64)
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            rand_int = random_integer_result_cpu_njit(s0, s1)
            out[i] = (rand_int + 0.5) * norm_factor

    @numba.njit(parallel=False)
    def random_integers_vector_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        """
        Generates random numbers using a xoroshiro128++.
        
        Parameters:
            seed0 (int): base seed
            out (numpy array): Output array to store random values.
        """
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            out[i] = random_integer_result_cpu_njit(s0, s1)

    @numba.njit(parallel=False)
    def random_integer_base_states_cpu_njit(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
        splitmix_state     = seed
        s0, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        s1, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_states_transition_cpu_njit(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
        s1 ^= s0
        s0 = rotl64_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_cpu_njit(s1, np.uint64(28))
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_result_cpu_njit(s0: np.uint64, s1: np.uint64) -> np.uint64:
        result64 = rotl64_cpu_njit(s0 + s1, np.uint64(17)) + s0
        return result64


    @numba.njit(parallel=False)
    def splitmix64_cpu_njit(state: np.uint64) -> tuple[np.uint64,np.uint64]:
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state
    
    @numba.njit(parallel=False)
    def rotl64_cpu_njit(x: np.uint64, k: np.uint64) -> np.uint64:
        return (x << k) | (x >> (np.uint64(64) - k))

###########################################################################################################

if not cuda_available:
    def random_p_values_matrix_gpu(seed_row0, seed_col0, out) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integers_matrix_gpu(seed_row0, seed_col0, num_steps, out) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_vector_gpu(seed, out) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integers_vector_gpu(seed, out) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integer_gpu(seed, num_steps) -> np.uint64: # type: ignore
        raise_cuda_not_available()
    def random_integer_base_states_gpu(seed)-> tuple:  # type: ignore       
        raise_cuda_not_available()
    def random_integer_states_transition_gpu(s0, s1) -> tuple:  # type: ignore       
        raise_cuda_not_available()
    def random_integer_result_gpu(s0, s1) -> np.uint64:  # type: ignore
        raise_cuda_not_available()
    def splitmix64_gpu(state) -> tuple[np.uint64, np.uint64]:  # type: ignore
        raise_cuda_not_available()        
    def rotl64_gpu(x, k) -> np.uint64: # type: ignore
        raise_cuda_not_available()

else:
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    @numba.cuda.jit(device=False)
    def random_p_values_matrix_gpu(seed_row0: np.uint64, seed_col0: np.uint64, out: DeviceNDArray) -> None:
        norm_factor = 1.0 / np.float64(2**64)
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = np.uint64(ind_row+seed_row0) << np.uint64(32)
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                seed = seed_row | np.uint64(ind_col+seed_col0)
                rand_int = random_integer_gpu(seed=seed, num_steps=np.uint32(1))
                out_row[ind_col] = (rand_int + 0.5) * norm_factor

    @numba.cuda.jit(device=False)
    def random_integers_matrix_gpu(seed_row0: np.uint64, seed_col0: np.uint64, num_steps: np.uint32, out: DeviceNDArray):
        """
        CUDA kernel to set random integers values in a np.uint64 matrix.

        Parameters:
            seed0 (int): base random seed
            out (device array): Output matrix to store random values.
        """
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = np.uint64(ind_row+seed_row0) << np.uint64(32)
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                seed = seed_row | np.uint64(ind_col+seed_col0)
                out_row[ind_col] = random_integer_gpu(seed=seed, num_steps=num_steps)
    
    @numba.cuda.jit(device=False)
    def random_p_values_vector_gpu(seed: np.uint64, out: DeviceNDArray) -> None:
        norm_factor = 1.0 / np.float64(2**64)
        s0, s1 = random_integer_base_states_gpu(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
            rand_int = random_integer_result_gpu(s0, s1)
            out[i] = (rand_int + 0.5) * norm_factor

    @numba.cuda.jit(device=False)
    def random_integers_vector_gpu(seed: np.uint64, out: DeviceNDArray):
        """
        CUDA kernel to set uniform random values using xoroshiro128++.

        Parameters:
            seed (int): random seed.
            ind_row (int): index of the row to fill inside the output matrix.
            out (device array): Output matrix to store random values.
        """
        s0, s1 = random_integer_base_states_gpu(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
            out[i] = random_integer_result_gpu(s0, s1)

    @numba.cuda.jit(device=True)
    def random_integer_gpu(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        s0, s1 = random_integer_base_states_gpu(seed=seed)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_gpu(s0=s0, s1=s1)
        result64 = random_integer_result_gpu(s0=s0, s1=s1)
        return result64

    @numba.cuda.jit(device=True)
    def random_integer_base_states_gpu(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
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



