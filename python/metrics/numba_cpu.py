from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def detect_signal_auc_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    import numba
    from python.hpc import HybridArray

    @numba.njit(parallel=True)
    def detect_signal_auc_cpu_njit(\
            noise_input: np.ndarray,\
            signal_input: np.ndarray,\
            auc_out_row: np.ndarray) -> None:
        one = np.float64(1)
        noise_size = np.uint32(noise_input.shape[1])
        counts2auc = (one/noise_size)*(one/noise_size)
        num_monte, N = signal_input.shape
        for ind_col in numba.prange(N):
            noise_row = noise_input[ind_col]
            count_below = np.uint32(0)
            for ind_row in range(num_monte):
                val = signal_input[ind_row][ind_col]
                ind_below = np.uint32(0)
                ind_above = noise_size-1
                while ind_below < ind_above-1:
                    ind_middle = ind_below + ((ind_above-ind_below) >> 1)
                    if noise_row[ind_middle] <= val:
                        ind_below = ind_middle
                    else:
                        ind_above = ind_middle
                if noise_row[ind_below] >= val:
                    count_below += ind_below
                else:
                    count_below += ind_above + np.uint32(noise_row[ind_above] <= val)
            auc = count_below*counts2auc
            auc_out_row[0][ind_col] = max(auc,one-auc)
