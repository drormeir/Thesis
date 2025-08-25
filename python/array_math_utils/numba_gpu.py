from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def array_transpose(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def average_row(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def average_column(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def sort_rows_inplace(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_argmin(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_argmax(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_min_inplace(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_max_inplace(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_argmin(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_argmax(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_min_inplace(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def cumulative_dominant_max_inplace(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def max_column_along_rows(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def argmax_column_along_rows(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def min_column_along_rows(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def argmin_column_along_rows(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    import cupy

    def array_transpose(array: DeviceNDArray, out: DeviceNDArray) -> None:
        cupy.asarray(out)[:] = cupy.asarray(array).T # cupy.ascontiguousarray(cupy_T)

    def average_row(array: DeviceNDArray, out_row: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        avg_cupy = cupy.asarray(out_row)
        cupy.mean(array_cupy, axis=0, keepdims=True, out=avg_cupy)

    def average_column(array: DeviceNDArray, out_column: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        avg_cupy = cupy.asarray(out_column)
        cupy.mean(array_cupy, axis=1, keepdims=True, out=avg_cupy)


    def sort_rows_inplace(array: DeviceNDArray) -> None:
        cupy.asarray(array).sort(axis=1)


    def max_column_along_rows(array: DeviceNDArray, maxval: DeviceNDArray) -> None:
        assert array.dtype == maxval.dtype
        array_cupy = cupy.asarray(array)
        maxval_cupy = cupy.asarray(maxval).reshape(-1)
        cupy.max(array_cupy, axis=1, out=maxval_cupy)


    def argmax_column_along_rows(array: DeviceNDArray, argmax: DeviceNDArray) -> None:
        assert argmax.dtype == np.uint32
        array_cupy = cupy.asarray(array)
        argmax_cupy = cupy.asarray(argmax).reshape(-1)
        cupy.argmax(array_cupy, axis=1, out=argmax_cupy)


    def min_column_along_rows(array: DeviceNDArray, minval: DeviceNDArray) -> None:
        assert array.dtype == minval.dtype
        array_cupy = cupy.asarray(array)
        minval_cupy = cupy.asarray(minval).reshape(-1)
        cupy.min(array_cupy, axis=1, out=minval_cupy)


    def argmin_column_along_rows(array: DeviceNDArray, argmin: DeviceNDArray) -> None:
        assert argmin.dtype == np.uint32
        array_cupy = cupy.asarray(array)
        argmin_cupy = cupy.asarray(argmin).reshape(-1)
        cupy.argmin(array_cupy, axis=1, out=argmin_cupy)

    @numba.cuda.jit(device=False)
    def cumulative_argmin(array: DeviceNDArray, argmin: DeviceNDArray) -> None:
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
    def cumulative_argmax(array: DeviceNDArray, argmax: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            input_row = array[ind_row]
            output_row = argmax[ind_row]
            current_max = input_row[0]
            current_idx = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val > current_max:
                    current_idx = np.uint32(j)
                    current_max = curr_val
                output_row[j] = current_idx


    @numba.cuda.jit(device=False)
    def cumulative_min_inplace(array: DeviceNDArray) -> None:
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
    def cumulative_max_inplace(array: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            row = array[ind_row]
            current_max = row[0]
            for j in range(1, cols):
                curr_val = row[j]
                if curr_val < current_max:
                    current_max = curr_val
                row[j] = current_max


    @numba.cuda.jit(device=False)
    def cumulative_dominant_argmin(array: DeviceNDArray, argmin: DeviceNDArray) -> None:
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
    def cumulative_dominant_argmax(array: DeviceNDArray, argmax: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        rows, cols = array.shape
        for ind_row in range(ind_row0, rows, row_stride):
            input_row = array[ind_row]
            output_row = argmax[ind_row]
            current_max = input_row[0]
            current_ind_max = np.uint32(0)
            current_ind_dominant = np.uint32(0)
            max_dominant_length = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val > current_max:
                    curr_val = current_max
                    current_ind_max = np.uint32(j)
                curr_dominant_length = np.uint32(j) - current_ind_max
                if curr_dominant_length >= max_dominant_length:
                    current_ind_dominant = current_ind_max
                    max_dominant_length = curr_dominant_length
                output_row[j] = current_ind_dominant


    @numba.cuda.jit(device=False)
    def cumulative_dominant_min_inplace(array: DeviceNDArray) -> None:
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


    @numba.cuda.jit(device=False)
    def cumulative_dominant_max_inplace(array: DeviceNDArray) -> None:
        # Get the 1D indices of the current thread within the grid
        ind_row0 = numba.cuda.grid(1) # type: ignore
        # Calculate the strides
        row_stride = numba.cuda.gridsize(1) # type: ignore
        num_monte, N = array.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = array[ind_row]
            current_ind_max = np.uint32(0)
            current_max = current_dominant = row[0]
            max_dominant_length = np.uint32(0)
            for j in range(1, N):
                curr_val = row[j]
                if curr_val > current_max:
                    current_ind_max = np.uint32(j)
                    current_max = curr_val
                curr_dominant_length = np.uint32(j) - current_ind_max
                if curr_dominant_length >= max_dominant_length:
                    current_dominant = current_max
                    max_dominant_length = curr_dominant_length
                row[j] = current_dominant

