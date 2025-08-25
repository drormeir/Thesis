import numpy as np
from scipy.stats import norm
from python.random_integers import python_native as random_integers

def sort_and_count_labels_rows(data: np.ndarray, n1: np.uint32, counts: np.ndarray) -> None:
    idx_sorted = np.argsort(data, axis=1)
    data[:] = np.take_along_axis(data, idx_sorted, axis=1)
    np.cumsum(idx_sorted<n1, axis=1, dtype=np.uint32, out=counts)


def random_modified_p_values(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, mu: np.float64, out: np.ndarray) -> None:
    random_p_values_matrix(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=out)
    modify_p_values(out=out, mu=mu)


def random_p_values_matrix(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
    work_array = np.empty_like(out, dtype=np.uint64)
    random_integers.matrix(num_steps=num_steps, offset_row0=offset_row0, offset_col0=offset_col0, out=work_array)
    out[:] = (work_array + np.float64(0.5)) / np.float64(2**64)


def random_p_values_series(seed: np.uint64, out: np.ndarray) -> None:
    work_array = np.empty_like(out, dtype=np.uint64)
    random_integers.series(seed=seed, out=work_array)
    out[:] = (work_array + np.float64(0.5)) / np.float64(2**64)


def modify_p_values(out: np.ndarray, mu: np.float64) -> None:
    out[:] = norm.sf(norm.isf(out) + mu)
    