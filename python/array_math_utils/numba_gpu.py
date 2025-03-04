from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def array_transpose_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def average_row_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def average_column_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def sort_rows_inplace_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_argmin_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_min_inplace_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_argmin_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_min_inplace_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def max_column_along_rows_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    import cupy

    def array_transpose_gpu(array: DeviceNDArray, out: DeviceNDArray) -> None:
        cupy.asarray(out)[:] = cupy.asarray(array).T # cupy.ascontiguousarray(cupy_T)

    def average_row_gpu(array: DeviceNDArray, out_row: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        avg_cupy = cupy.asarray(out_row)
        cupy.mean(array_cupy, axis=0, keepdims=True, out=avg_cupy)

    def average_column_gpu(array: DeviceNDArray, out_column: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        avg_cupy = cupy.asarray(out_column)
        cupy.mean(array_cupy, axis=1, keepdims=True, out=avg_cupy)

    def sort_rows_inplace_gpu(array: DeviceNDArray) -> None:
        cupy.asarray(array).sort(axis=1)

    def max_column_along_rows_gpu(array: DeviceNDArray, argmax: DeviceNDArray, maxval: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        maxval_cupy = cupy.asarray(maxval).reshape(-1)
        argmax_cupy = cupy.asarray(argmax).reshape(-1)
        cupy.max(array_cupy, axis=1, out=maxval_cupy)
        cupy.argmax(array_cupy, axis=1, out=argmax_cupy, dtype=np.uint32)

    @numba.cuda.jit(device=False)
    def cumulative_argmin_gpu(array: DeviceNDArray, argmin: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            input_row = array[ind_row]
            output_row = argmin[ind_row]
            current_min = input_row[0]
            current_idx = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val < current_min:
                    current_idx = np.uint32(j)
                    current_min = curr_val
                output_row[j] = current_idx


    @numba.cuda.jit(device=False)
    def cumulative_min_inplace_gpu(array: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            row = array[ind_row]
            current_min = row[0]
            for j in range(1, cols):
                curr_val = row[j]
                if curr_val < current_min:
                    current_min = curr_val
                row[j] = current_min


    @numba.cuda.jit(device=False)
    def cumulative_dominant_argmin_gpu(array: DeviceNDArray, argmin: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            input_row = array[ind_row]
            output_row = argmin[ind_row]
            current_min = input_row[0]
            current_ind_min = np.uint32(0)
            current_ind_dominant = np.uint32(0)
            max_dominant_length = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val < current_min:
                    curr_val = current_min
                    current_ind_min = np.uint32(j)
                curr_dominant_length = np.uint32(j) - current_ind_min
                if curr_dominant_length >= max_dominant_length:
                    current_ind_dominant = current_ind_min
                    max_dominant_length = curr_dominant_length
                output_row[j] = current_ind_dominant


    @numba.cuda.jit(device=False)
    def cumulative_dominant_min_inplace_gpu(array: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        num_monte, N = array.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = array[ind_row]
            current_ind_min = np.uint32(0)
            current_min = current_dominant = row[0]
            max_dominant_length = np.uint32(0)
            for j in range(1, N):
                curr_val = row[j]
                if curr_val < current_min:
                    current_ind_min = np.uint32(j)
                    current_min = curr_val
                curr_dominant_length = np.uint32(j) - current_ind_min
                if curr_dominant_length >= max_dominant_length:
                    current_dominant = current_min
                    max_dominant_length = curr_dominant_length
                row[j] = current_dominant

