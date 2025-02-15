from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def detect_signal_auc_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from python.hpc import HybridArray
    from python.array_math_utils.numba_gpu import average_row_gpu

    def detect_signal_auc_gpu(\
            noise_input: HybridArray,\
            signal_input_work: HybridArray,\
            auc_out_row: HybridArray) -> None:
        grid_shape, block_shape = signal_input_work.gpu_grid_block2D_columns_shapes(registers_per_thread=128)
        detect_signal_auc_gpu_kernel[grid_shape, block_shape](noise_input.gpu_data(), signal_input_work.gpu_data()) # type: ignore
        average_row_gpu(array=signal_input_work.gpu_data(), out_row=auc_out_row.gpu_data())
        grid_shape, block_shape = auc_out_row.gpu_grid_block1D_cols_shapes()
        maximize_auc_gpu_kernel[grid_shape, block_shape](auc_out_row.gpu_data()) # type: ignore

    @numba.cuda.jit(device=False)
    def detect_signal_auc_gpu_kernel(\
            noise_input: DeviceNDArray,\
            signal_input_work: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = signal_input_work.shape
        noise_size = np.uint32(noise_input.shape[1])
        for ind_col in range(ind_col0, N, col_stride):
            noise_row = noise_input[ind_col]
            for ind_row in range(ind_row0, num_monte, row_stride):
                val = signal_input_work[ind_row][ind_col]
                ind_below = np.uint32(0)
                ind_above = noise_size-1
                while ind_below < ind_above-1:
                    ind_middle = ind_below + ((ind_above-ind_below) >> 1)
                    if noise_row[ind_middle] <= val:
                        ind_below = ind_middle
                    else:
                        ind_above = ind_middle
                if noise_row[ind_below] >= val:
                    count_below = ind_below
                else:
                    count_below = ind_above + np.uint32(noise_row[ind_above] <= val)
                signal_input_work[ind_row][ind_col] = count_below/noise_size


    @numba.cuda.jit(device=False)
    def maximize_auc_gpu_kernel(auc_out_row: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_col0 = numba.cuda.grid(1) # type: ignore
        # Calculate the stride
        col_stride = numba.cuda.gridsize(1) # type: ignore
        _, N = auc_out_row.shape
        auc_row = auc_out_row[0]
        for ind_col in range(ind_col0, N, col_stride):
            auc = auc_row[ind_col]
            auc_row[ind_col] = max(auc,np.float64(1)-auc)
