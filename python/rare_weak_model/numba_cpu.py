from python.hpc import globals, raise_njit_not_available

if not globals.cpu_njit_num_threads:
    # Mock API
    def random_modified_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_p_values_series_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    from scipy.stats import norm
    import math
    import numba
    from python.random_integers.numba_cpu import random_integers_matrix_cpu_njit, random_integer_base_states_cpu_njit, random_integer_states_transition_cpu_njit, random_integer_result_cpu_njit


    @numba.njit(parallel=True)
    def random_modified_p_values_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: np.ndarray) -> None:
        random_p_values_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
        # sf(x) = 0.5 * erfc(x / sqrt(2))
        out[:] = norm.sf(norm.isf(out) + mu)

    @numba.njit(parallel=True)
    def random_p_values_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
        out_uint64 = np.empty_like(out, dtype=np.uint64)
        random_integers_matrix_cpu_njit(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out_uint64)
        out[:] = (out_uint64+0.5) / np.float64(2.0**64)

    @numba.njit(parallel=False)
    def random_p_values_series_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        norm_factor = 1.0 / np.float64(2.0**64)
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            rand_int = random_integer_result_cpu_njit(s0, s1)
            out[i] = (rand_int + 0.5) * norm_factor
