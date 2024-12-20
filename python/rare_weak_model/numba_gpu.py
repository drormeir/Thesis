from python.hpc import globals, raise_cuda_not_available

if not globals.cuda_available:
    # Mock API
    def random_modified_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_series_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()

else:
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from python.random_integers.numba_gpu import random_integer_gpu, random_integer_base_states_gpu, random_integer_states_transition_gpu, random_integer_result_gpu
    from scipy.stats import norm

    @numba.cuda.jit(device=False)
    def random_modified_p_values_matrix_gpu(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: DeviceNDArray) -> None:
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
                p_value = (rand_int + 0.5) * norm_factor
                # sf(x) = 0.5 * erfc(x / sqrt(2))
                out_row[ind_col] = norm.sf(norm.isf(p_value) + mu)

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
