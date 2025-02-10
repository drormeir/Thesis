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

