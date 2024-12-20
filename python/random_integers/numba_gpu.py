from python.hpc import globals, raise_cuda_not_available

if not globals.cuda_available:
    # Mock API
    def random_integers_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_integers_series_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def splitmix64_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

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



