import numpy as np
from scipy.stats import norm
from python.random_integers.python_native import random_integers_matrix_py, random_integer_base_states_py, random_integer_states_transition_py, random_integer_result_py

def sort_and_count_labels_rows_py(data: np.ndarray, n1: np.uint32, counts: np.ndarray) -> None:
    idx_sorted = np.argsort(data, axis=1)
    data[:] = np.take_along_axis(data, idx_sorted, axis=1)
    np.cumsum(idx_sorted<n1, axis=1, dtype=np.uint32, out=counts)

def random_modified_p_values_matrix_py(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: np.ndarray) -> None:
    random_p_values_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
    modify_p_values_matrix_py(out=out, mu=mu)

def random_p_values_matrix_py(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
    random_integers_matrix_py(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
    out += np.float64(0.5)
    out /= np.float64(2.0**64)


def random_p_values_series_py(seed: np.uint64, out: np.ndarray) -> None:
    norm_factor = np.float64(1.0) / np.float64(2.0**64)
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        rand_int = random_integer_result_py(s0=s0, s1=s1)
        out[i] = (rand_int + np.float64(0.5)) * norm_factor

def modify_p_values_matrix_py(out: np.ndarray, mu: np.float64) -> None:
    out[:] = norm.sf(norm.isf(out) + mu)
    