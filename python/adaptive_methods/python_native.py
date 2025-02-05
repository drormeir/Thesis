import math
import numpy as np
from scipy.stats import beta

def higher_criticism_stable_py(\
        sorted_p_values_input_output: np.ndarray) -> None:
    _, N = sorted_p_values_input_output.shape
    p_base_line = np.arange(1,N+1, dtype=np.float64).reshape(1,-1)/(N+1)
    stdev = np.sqrt(p_base_line)*np.sqrt(1-p_base_line)
    sqrtN = math.sqrt(np.float64(N))
    sorted_p_values_input_output[:] = sqrtN*((sorted_p_values_input_output - p_base_line)/stdev)
    
def higher_criticism_unstable_py(\
        sorted_p_values_input_output: np.ndarray) -> None:
    _, N = sorted_p_values_input_output.shape
    p_base_line = np.arange(1,N+1, dtype=np.float64).reshape(1,-1)/(N+1)
    stdev = np.sqrt(sorted_p_values_input_output)*np.sqrt(1-sorted_p_values_input_output)
    sqrtN = math.sqrt(np.float64(N))
    sorted_p_values_input_output[:] = sqrtN*((sorted_p_values_input_output - p_base_line)/stdev)

def berk_jones_py(\
    sorted_p_values_input_output: np.ndarray) -> None:
    _, N = sorted_p_values_input_output.shape
    for col in range(N):
        a = col+1
        b = N-col
        x = sorted_p_values_input_output[:,col]
        sorted_p_values_input_output[:,col] = beta.cdf(x,a,b)

def discover_argmin_py(\
        transformed_p_values_input: np.ndarray,\
        num_discoveries_output: np.ndarray) -> None:
    num_monte, N = transformed_p_values_input.shape
    for ind_row in range(num_monte):
        input_row = transformed_p_values_input[ind_row]
        output_row = num_discoveries_output[ind_row]
        current_idx = 0
        output_row[0] = 1
        for j in range(1, N):
            if input_row[j] < input_row[current_idx]:
                current_idx = j
            output_row[j] = current_idx+1


def discover_dominant_py(\
        transformed_p_values_input: np.ndarray,\
        num_discoveries_output: np.ndarray) -> None:
    num_monte, N = transformed_p_values_input.shape
    for ind_row in range(num_monte):
        input_row = transformed_p_values_input[ind_row]
        output_row = num_discoveries_output[ind_row]
        current_ind_min = 0
        current_ind_dominant = 0
        max_dominant_length = 0
        output_row[0] = 1
        for j in range(1, N):
            if input_row[j] < input_row[current_ind_min]:
                current_ind_min = j
            else:
                curr_dominant_length = j - current_ind_min
                if curr_dominant_length >= max_dominant_length:
                    current_ind_dominant = current_ind_min
                    max_dominant_length = curr_dominant_length
            output_row[j] = current_ind_dominant+1

