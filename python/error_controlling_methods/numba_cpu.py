from python.hpc import globals, raise_njit_not_available

if not globals.cpu_njit_num_threads:
    # Mock API
    def topk_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def bonferroni_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def benjamini_hochberg_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    import numba

    @numba.njit(parallel=True)
    def topk_cpu_njit(sorted_p_values_input: np.ndarray,\
                num_discoveries_output: np.ndarray) -> None:
        num_monte, N = sorted_p_values_input.shape
        num_monte = np.uint32(num_monte)
        N = np.uint32(N)
        one = np.uint32(1)
        num_discoveries = np.arange(one,N+one,dtype=np.uint32).reshape(1,-1)
        for row in numba.prange(num_monte):
            num_discoveries_output[row,:] = num_discoveries
        
    @numba.njit(parallel=True)
    def bonferroni_cpu_njit(sorted_p_values_input: np.ndarray,\
                num_discoveries_output: np.ndarray) -> None:
        num_monte, N = num_discoveries_output.shape
        for row in numba.prange(num_monte):
            inp_row = sorted_p_values_input[row]
            out_row = num_discoveries_output[row]
            num_discover = np.uint32(0)
            for ind_col in range(N):
                bonferroni_threshold = (ind_col+1)/N
                while num_discover < N and inp_row[num_discover] <= bonferroni_threshold:
                    num_discover += 1
                out_row[ind_col] = num_discover

    @numba.njit(parallel=True)
    def benjamini_hochberg_cpu_njit(sorted_p_values_input: np.ndarray,\
                num_discoveries_output: np.ndarray) -> None:
        num_monte, N = num_discoveries_output.shape
        for row in numba.prange(num_monte):
            inp_row = sorted_p_values_input[row]
            out_row = num_discoveries_output[row]
            num_discover = np.uint32(N)
            for col in range(N-1,-1,-1):
                bonferroni_threshold = (col+1)/N
                while num_discover > 0 and inp_row[num_discover-1] > bonferroni_threshold*(num_discover/N):
                    num_discover -= 1
                out_row[col] = num_discover
