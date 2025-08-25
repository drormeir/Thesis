import numpy as np
from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def matrix(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def series(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def integer_from_seed(**kwargs) -> np.uint64: # type: ignore
        raise_cuda_not_available()
    def matrix_splitmix64(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def matrix_base_states(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def integer_base_states(**kwargs) -> tuple[np.uint64,np.uint64]: # type: ignore
        raise_cuda_not_available()
    def integer_states_transition(**kwargs) -> tuple[np.uint64,np.uint64]: # type: ignore
        raise_cuda_not_available()
    def integer_result(**kwargs) -> np.uint64: # type: ignore
        raise_cuda_not_available()
    def splitmix64(**kwargs) -> tuple[np.uint64,np.uint64]: # type: ignore
        raise_cuda_not_available()
    def rotl64(**kwargs) -> np.uint64: # type: ignore
        raise_cuda_not_available()
    def scramble_seed(**kwargs) -> np.uint64: # type: ignore
        raise_cuda_not_available()
else:
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    @numba.cuda.jit(device=False)
    def matrix(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = (np.uint64(offset_row0 + ind_row) << np.uint64(32)) + offset_col0
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                out_row[ind_col] = integer_from_seed(seed_row + np.uint64(ind_col), num_steps)
    
    @numba.cuda.jit(device=False)
    def series(seed: np.uint64, out: DeviceNDArray):
        s0, s1 = integer_base_states(seed)
        num_steps = out.size
        ind_start = numba.cuda.grid(1) # type: ignore
        ind_stride = numba.cuda.gridsize(1) # type: ignore
        for i in range(ind_start, num_steps, ind_stride):
            s0, s1 = integer_states_transition(s0, s1)
            out[i] = integer_result(s0, s1)
    
    @numba.cuda.jit(device=True)
    def integer_from_seed(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        s0, s1 = integer_base_states(seed)
        for _ in range(num_steps):
            s0, s1 = integer_states_transition(s0, s1)
        result64 = integer_result(s0, s1)
        return result64

    @numba.cuda.jit(device=False)
    def matrix_splitmix64(states: DeviceNDArray, new_states: DeviceNDArray, z: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, z.shape[0], row_stride):
            states_row = states[ind_row]
            new_states_row = new_states[ind_row]
            z_row = z[ind_row]
            for ind_col in range(ind_col0, z.shape[1], col_stride):
                z_row[ind_col], new_states_row[ind_col] = splitmix64(states_row[ind_col])

    @numba.cuda.jit(device=False)
    def matrix_base_states(seeds: DeviceNDArray, s0: DeviceNDArray, s1: DeviceNDArray):
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, seeds.shape[0], row_stride):
            seeds_row = seeds[ind_row]
            s0_row = s0[ind_row]
            s1_row = s1[ind_row]
            for ind_col in range(ind_col0, seeds.shape[1], col_stride):
                s0_row[ind_col], s1_row[ind_col] = integer_base_states(seeds_row[ind_col])

    @numba.cuda.jit(device=True)
    def integer_base_states(seed: np.uint64) -> tuple[np.uint64,np.uint64]:
        seed = scramble_seed(seed)
        splitmix_state     = seed
        s0, splitmix_state = splitmix64(splitmix_state)
        s1, splitmix_state = splitmix64(splitmix_state)
        return s0, s1

    @numba.cuda.jit(device=True)
    def integer_states_transition(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
        s1 ^= s0
        s0 = rotl64(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64(s1, np.uint64(28))
        return s0, s1

    @numba.cuda.jit(device=True)
    def integer_result(s0: np.uint64, s1: np.uint64) -> np.uint64:
        result64 = rotl64(s0 + s1, np.uint64(17)) + s0
        return result64

    @numba.cuda.jit(device=True)
    def splitmix64(state: np.uint64) -> tuple[np.uint64, np.uint64]:
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state

    @numba.cuda.jit(device=True)
    def rotl64(x: np.uint64, k: np.uint64) -> np.uint64:
        return (x << k) | (x >> (np.uint64(64) - k))

    @numba.cuda.jit(device=True)
    def scramble_seed(seed: np.uint64) -> np.uint64:
        # adding one column and one row to avoid zero seed
        seed += (np.uint64(1) << np.uint64(32)) + np.uint64(1)
        # Apply a series of XOR and multiplication steps to mix the bits.
        seed ^= (seed >> np.uint64(33))
        seed *= np.uint64(0xff51afd7ed558ccd) # Large constant from MurmurHash3
        seed ^= (seed >> np.uint64(33))
        seed *= np.uint64(0xc4ceb9fe1a85ec53)  # Another mixing constant
        seed ^= (seed >> np.uint64(33))
        return seed

