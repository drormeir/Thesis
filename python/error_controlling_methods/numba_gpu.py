from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def topk_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def bonferroni_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def benjamini_hochberg_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import math
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from python.random_integers.numba_gpu import random_integer_gpu, random_integer_base_states_gpu, random_integer_states_transition_gpu, random_integer_result_gpu
    import cupy

    @numba.cuda.jit(device=False)
    def topk_gpu(sorted_p_values_input: DeviceNDArray,\
               num_discoveries_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        nrows, N = num_discoveries_output.shape
        for row in range(row0, nrows, row_stride):
            out_row = num_discoveries_output[row]
            for ind_col in range(N):
                out_row[ind_col] = ind_col+1

    @numba.cuda.jit(device=False)
    def bonferroni_gpu(sorted_p_values_input: DeviceNDArray,\
               num_discoveries_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        num_monte, N = num_discoveries_output.shape
        for row in range(row0, num_monte, row_stride):
            inp_row = sorted_p_values_input[row]
            out_row = num_discoveries_output[row]
            num_discover = np.uint32(0)
            for ind_col in range(N):
                bonferroni_threshold = (ind_col+1)/N
                while num_discover < N and inp_row[num_discover] <= bonferroni_threshold:
                    num_discover += 1
                out_row[ind_col] = num_discover

    @numba.cuda.jit(device=False)
    def benjamini_hochberg_gpu(sorted_p_values_input: DeviceNDArray,\
               num_discoveries_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        num_monte, N = num_discoveries_output.shape
        for row in range(row0, num_monte, row_stride):
            inp_row = sorted_p_values_input[row]
            out_row = num_discoveries_output[row]
            num_discover = np.uint32(N)
            for col in range(N-1,-1,-1):
                bonferroni_threshold = (col+1)/N
                while num_discover > 0 and inp_row[num_discover-1] > bonferroni_threshold*(num_discover/N):
                    num_discover -= 1
                out_row[col] = num_discover
