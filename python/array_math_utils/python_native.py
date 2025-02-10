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
