import numpy as np

def cumulative_argmin_py(array: np.ndarray, argmin: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
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


def cumulative_argmax_py(array: np.ndarray, argmax: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
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


def cumulative_min_inplace_py(array: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
        row = array[ind_row]
        current_min = row[0]
        for j in range(1, cols):
            curr_val = row[j]
            if curr_val < current_min:
                current_min = curr_val
            row[j] = current_min


def cumulative_max_inplace_py(array: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
        row = array[ind_row]
        current_max = row[0]
        for j in range(1, cols):
            curr_val = row[j]
            if curr_val > current_max:
                current_max = curr_val
            row[j] = current_max


def cumulative_dominant_argmin_py(array: np.ndarray, argmin: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
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
                current_ind_min = np.uint32(j)
                current_min = curr_val
            curr_dominant_length = np.uint32(j) - current_ind_min
            if curr_dominant_length >= max_dominant_length:
                current_ind_dominant = current_ind_min
                max_dominant_length = curr_dominant_length
            output_row[j] = current_ind_dominant


def cumulative_dominant_argmax_py(array: np.ndarray, argmax: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
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
                current_ind_max = np.uint32(j)
                current_max = curr_val
            curr_dominant_length = np.uint32(j) - current_ind_max
            if curr_dominant_length >= max_dominant_length:
                current_ind_dominant = current_ind_max
                max_dominant_length = curr_dominant_length
            output_row[j] = current_ind_dominant


def cumulative_dominant_min_inplace_py(array: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
        row = array[ind_row]
        current_ind_min = np.uint32(0)
        current_min = current_dominant = row[0]
        max_dominant_length = np.uint32(0)
        for j in range(1, cols):
            curr_val = row[j]
            if curr_val < current_min:
                current_ind_min = np.uint32(j)
                current_min = curr_val
            curr_dominant_length = np.uint32(j) - current_ind_min
            if curr_dominant_length >= max_dominant_length:
                current_dominant = current_min
                max_dominant_length = curr_dominant_length
            row[j] = current_dominant


def cumulative_dominant_max_inplace_py(array: np.ndarray) -> None:
    rows, cols = array.shape
    for ind_row in range(rows):
        row = array[ind_row]
        current_ind_max = np.uint32(0)
        current_max = current_dominant = row[0]
        max_dominant_length = np.uint32(0)
        for j in range(1, cols):
            curr_val = row[j]
            if curr_val > current_max:
                current_ind_max = np.uint32(j)
                current_max = curr_val
            curr_dominant_length = np.uint32(j) - current_ind_max
            if curr_dominant_length >= max_dominant_length:
                current_dominant = current_max
                max_dominant_length = curr_dominant_length
            row[j] = current_dominant


def max_column_along_rows_py(array: np.ndarray, maxval: np.ndarray) -> None:
    array.max(axis=1, out=maxval)


def argmax_column_along_rows_py(array: np.ndarray, argmax: np.ndarray) -> None:
    array.argmax(axis=1, out=argmax)


def min_column_along_rows_py(array: np.ndarray, minval: np.ndarray) -> None:
    array.min(axis=1, out=minval)


def argmin_column_along_rows_py(array: np.ndarray, argmin: np.ndarray) -> None:
    array.argmin(axis=1, out=argmin)

def average_row_py(array: np.ndarray, out_row: np.ndarray) -> None:
    np.mean(array, axis=0, keepdims=True, out=out_row)


def average_column_py(array: np.ndarray, out_column: np.ndarray) -> None:
    np.mean(array, axis=1, keepdims=True, out=out_column)

def array_transpose_py(array: np.ndarray, out: np.ndarray) -> None:
    out[:] = array.T

def sort_rows_inplace_py(array: np.ndarray) -> None:
    array.sort(axis=1)
