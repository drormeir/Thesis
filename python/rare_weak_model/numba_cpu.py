from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def random_modified_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_series_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def modify_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def sort_and_count_labels_rows_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    import math
    import numba
    from python.random_integers.numba_cpu import random_integers_matrix_cpu_njit, random_integer_base_states_cpu_njit, random_integer_states_transition_cpu_njit, random_integer_result_cpu_njit

    @numba.njit(parallel=True)
    def sort_and_count_labels_rows_cpu_njit(data: np.ndarray, n1: np.uint32, counts: np.ndarray) -> None:
        nrows, ncols = data.shape
        for i in numba.prange(nrows):
            idx = np.argsort(data[i])
            data[i][:] = data[i][idx]
            cum_sum: np.uint32 = np.uint32(0)
            counts_i = counts[i]
            for j in range(ncols):
                cum_sum += np.uint32(idx[j] < n1)
                counts_i[j] = cum_sum

    @numba.njit(parallel=False)
    def random_modified_p_values_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: np.ndarray) -> None:
        random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
        modify_p_values_matrix_cpu_njit(out=out, mu=mu)

    @numba.njit(parallel=True)
    def modify_p_values_matrix_cpu_njit(out: np.ndarray, mu: np.float64) -> None:
        rows, cols = out.shape
        for ind in numba.prange(rows*cols):
            row = ind // cols
            col = ind % cols
            isf = standard_normal_isf_newton_cpu_njit(out[row][col])
            out[row][col] = standard_normal_sf_cpu_njit(isf + mu)

    @numba.njit(parallel=True)
    def random_p_values_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
        out_uint64 = np.empty_like(out, dtype=np.uint64)
        random_integers_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out_uint64)
        out[:] = (out_uint64+np.float64(0.5)) / np.float64(2.0**64)

    @numba.njit(parallel=False)
    def random_p_values_series_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        norm_factor = np.float64(1.0) / np.float64(2.0**64)
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            rand_int = random_integer_result_cpu_njit(s0, s1)
            out[i] = (rand_int + np.float64(0.5)) * norm_factor

    @numba.njit(parallel=False)
    def standard_normal_isf_newton_cpu_njit(p: np.float64) -> np.float64:
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
        z = standard_normal_isf_rational_approximation_cpu_njit(p)
        for _ in range(5):  # usually no more than 3 iterations
            # f(z)   = SF(z) - p
            # f'(z)  = SF'(z)
            f_val = standard_normal_sf_cpu_njit(z) - p
            f_prime = standard_normal_sf_derivative_cpu_njit(z)
            dz = - f_val / f_prime
            z += dz
            '''
            if abs(dz) < tol:
                break # Converged
            '''
        return z

    @numba.njit(parallel=False)
    def standard_normal_isf_rational_approximation_cpu_njit(p: np.float64) -> np.float64:
        """
        Classic Abramowitz & Stegun approximation (formula 26.2.23).
        """
        # Coefficients
        c = [np.float64(2.515517), np.float64(0.802853), np.float64(0.010328)]
        d = [np.float64(1.432788), np.float64(0.189269), np.float64(0.001308)]
        one = np.float64(1.0)
        if p > np.float64(0.5):
            q = one - p
            f = -one
        else:
            q = p
            f = one
        t = math.sqrt(-np.float64(2.0) * math.log(q))
        numerator = (c[2]*t + c[1])*t + c[0]
        denominator = ((d[2]*t + d[1])*t + d[0])*t + one
        return f*(t - numerator / denominator)


    @numba.njit(parallel=False)
    def standard_normal_sf_cpu_njit(z: np.float64) -> np.float64:
        """
        Standard normal survival function, SF(z) = 1 - Phi(z),
        implemented using math.erfc from the standard library.

        SF(z) = 0.5 * erfc(z / sqrt(2)).
        """
        return np.float64(0.5) * math.erfc(z / math.sqrt(np.float64(2.0)))

    @numba.njit(parallel=False)
    def standard_normal_sf_derivative_cpu_njit(z: np.float64) -> np.float64:
        """
        Derivative of the standard normal survival function SF(z).
        This is -phi(z), where phi(z) is the standard normal PDF.
        """
        # phi(z) = 1/sqrt(2π) * exp(-z^2/2)
        pdf_z = math.exp(np.float64(-0.5) * z*z) / math.sqrt(np.float64(2.0) * np.float64(math.pi))
        return -np.float64(pdf_z)
