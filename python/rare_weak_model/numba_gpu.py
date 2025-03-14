from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def random_modified_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def random_p_values_series_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def arg_sort_rows_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def modify_p_values_matrix_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def sort_and_count_labels_rows_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import math
    import numpy as np
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from python.random_integers.numba_gpu import random_integer_gpu, random_integer_base_states_gpu, random_integer_states_transition_gpu, random_integer_result_gpu
    import cupy

    def sort_and_count_labels_rows_gpu(data: DeviceNDArray, n1: np.uint32, counts: DeviceNDArray) -> None:
        cupy_arr = cupy.asarray(data)
        idx_sorted = cupy.argsort(cupy_arr, axis=1)
        cupy_arr_sorted = cupy.take_along_axis(cupy_arr, idx_sorted, axis=1)
        data[:] = numba.cuda.as_cuda_array(cupy_arr_sorted)
        cupy.cumsum(idx_sorted<n1, axis=1, dtype=cupy.uint32, out=cupy.asarray(counts))

    @numba.cuda.jit(device=False)
    def random_modified_p_values_matrix_gpu(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: DeviceNDArray) -> None:
        norm_factor = np.float64(1.0) / np.float64(2.0**64)
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = (np.uint64(offset_row0 + ind_row) << np.uint64(32)) + offset_col0
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                rand_int = random_integer_gpu(seed_row + np.uint64(ind_col), num_steps)
                p_value = (rand_int + np.float64(0.5)) * norm_factor
                isf = standard_normal_isf_newton_gpu(p_value)
                out_row[ind_col] = standard_normal_sf_gpu(isf + mu)

    @numba.cuda.jit(device=False)
    def modify_p_values_matrix_gpu(out: DeviceNDArray, mu: np.float64) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                isf = standard_normal_isf_newton_gpu(out_row[ind_col])
                out_row[ind_col] = standard_normal_sf_gpu(isf + mu)

    @numba.cuda.jit(device=False)
    def random_p_values_matrix_gpu(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: DeviceNDArray) -> None:
        norm_factor = np.float64(1.0) / np.float64(2.0**64)
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        for ind_row in range(ind_row0, out.shape[0], row_stride):
            out_row = out[ind_row]
            seed_row = (np.uint64(offset_row0 + ind_row) << np.uint64(32)) + offset_col0
            for ind_col in range(ind_col0, out.shape[1], col_stride):
                rand_int = random_integer_gpu(seed_row + np.uint64(ind_col), num_steps)
                out_row[ind_col] = (rand_int + np.float64(0.5)) * norm_factor

    @numba.cuda.jit(device=False)
    def random_p_values_series_gpu(seed: np.uint64, out: DeviceNDArray) -> None:
        norm_factor = np.float64(1.0) / np.float64(2.0**64)
        s0, s1 = random_integer_base_states_gpu(seed)
        num_steps = out.size
        ind_start = numba.cuda.grid(1) # type: ignore
        ind_stride = numba.cuda.gridsize(1) # type: ignore
        for i in range(ind_start, num_steps, ind_stride):
            s0, s1 = random_integer_states_transition_gpu(s0, s1)
            rand_int = random_integer_result_gpu(s0, s1)
            out[i] = (rand_int + np.float64(0.5)) * norm_factor


    @numba.cuda.jit(device=True)
    def standard_normal_isf_newton_gpu(p: np.float64) -> np.float64:
        """
        Compute the ISF (inverse survival function) for the standard normal
        by solving SF(z) = p via Newton–Raphson starting from z0 = rational approximation

        Args:
            p       : Probability in (0,1).
            tol     : Convergence tolerance.

        Returns:
            The value of x such that SF(z) ≈ p.
        """
        # Initial guess for z 
        z = standard_normal_isf_rational_approximation_gpu(p)
        for _ in range(5):  # usually no more than 3 iterations
            # f(z)   = SF(z) - p
            # f'(z)  = SF'(z)
            f_val = standard_normal_sf_gpu(z) - p
            f_prime = standard_normal_sf_derivative_gpu(z)
            dz = - f_val / f_prime
            z += dz
            '''
            if abs(dz) < tol:
                break # Converged
            '''
        return z


    @numba.cuda.jit(device=True)
    def standard_normal_isf_rational_approximation_gpu(p: np.float64) -> np.float64:
        """
        Classic Abramowitz & Stegun approximation (formula 26.2.23).
        """
        # Coefficients, can not use lists in numba.cuda
        c0, c1, c2 = np.float64(2.515517), np.float64(0.802853), np.float64(0.010328)
        d0, d1, d2 = np.float64(1.432788), np.float64(0.189269), np.float64(0.001308)
        one = np.float64(1.0)
        if p > 0.5:
            q = one-p
            f = -one
        else:
            q = p
            f = one
        t = math.sqrt(np.float64(-2.0) * math.log(q))

        numerator = (c2*t + c1)*t + c0
        denominator = ((d2*t + d1)*t + d0)*t + one
        return f*(t - numerator / denominator)


    @numba.cuda.jit(device=True)
    def standard_normal_sf_gpu(z: np.float64) -> np.float64:
        """
        Standard normal survival function, SF(z) = 1 - Phi(z),
        implemented using math.erfc from the standard library.

        SF(z) = 0.5 * erfc(z / sqrt(2)).
        """
        return np.float64(0.5) * math.erfc(z / math.sqrt(np.float64(2.0)))

    @numba.cuda.jit(device=True)
    def standard_normal_sf_derivative_gpu(z: np.float64) -> np.float64:
        """
        Derivative of the standard normal survival function SF(z).
        This is -phi(z), where phi(z) is the standard normal PDF.
        """
        # phi(z) = 1/sqrt(2π) * exp(-z^2/2)
        pdf_z = math.exp(np.float64(-0.5) * z*z) / math.sqrt(np.float64(2.0) * np.float64(math.pi))
        return -np.float64(pdf_z)

