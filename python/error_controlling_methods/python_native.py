import numpy as np

def topk_py(sorted_p_values_input: np.ndarray,\
            num_discoveries_output: np.ndarray) -> None:
    num_monte, N = sorted_p_values_input.shape
    num_discoveries_output[:] = np.repeat(np.arange(1,N+1).reshape(1,-1), num_monte, axis=0)

def bonferroni_py(sorted_p_values_input: np.ndarray,\
               num_discoveries_output: np.ndarray) -> None:
    num_monte, N = num_discoveries_output.shape
    for row in range(num_monte):
        inp_row = sorted_p_values_input[row]
        out_row = num_discoveries_output[row]
        num_discover = np.uint32(0)
        for col in range(N):
            bonferroni_threshold = (col+1)/N
            while num_discover < N and inp_row[num_discover] <= bonferroni_threshold:
                num_discover += 1
            out_row[col] = num_discover

def benjamini_hochberg_py(sorted_p_values_input: np.ndarray,\
               num_discoveries_output: np.ndarray) -> None:
    num_monte, N = num_discoveries_output.shape
    for row in range(num_monte):
        inp_row = sorted_p_values_input[row]
        out_row = num_discoveries_output[row]
        num_discover = np.uint32(N)
        for col in range(N-1,-1,-1):
            bonferroni_threshold = (col+1)/N
            while num_discover > 0 and inp_row[num_discover-1] > bonferroni_threshold*(num_discover/N):
                num_discover -= 1
            out_row[col] = num_discover
