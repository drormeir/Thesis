from .random_integers import random_num_steps,\
    random_integers_matrix, splitmix64_matrix,\
    random_integers_base_states_matrix, random_integers_series, random_integers_2_p_values

class gpu:
    from .numba_gpu import (
        random_integers_matrix_gpu as random_integers_matrix,\
        splitmix64_matrix_gpu as splitmix64_maxtrix,\
        random_integers_base_states_matrix_gpu as random_integers_base_states_matrix,\
        random_integers_series_gpu as random_integers_series,\
        random_integers_2_p_values_gpu as random_integers_2_p_values
    )

class cpu:
    from .numba_cpu import (
        random_integers_matrix_cpu_njit as random_integers_matrix,\
        splitmix64_matrix_cpu_njit as splitmix64_matrix,\
        random_integers_base_states_matrix_cpu_njit as random_integers_base_states_matrix,\
        random_integers_series_cpu_njit as random_integers_series,\
        random_integers_2_p_values_cpu_njit as random_integers_2_p_values
    )

class python:
    from .python_native import (
        random_integers_matrix_py as random_integers_matrix,\
        splitmix64_matrix_py as splitmix64_matrix,\
        random_integers_base_states_matrix_py as random_integers_base_states_matrix,\
        random_integers_series_py as random_integers_series,\
        random_integers_2_p_values_py as random_integers_2_p_values
    )   




__all__ = [
    'random_integers_matrix',
    'random_integers_2_p_values',
    'random_integers_series',
    'random_integers_base_states_matrix',
    'splitmix64_matrix',
    'random_num_steps'
]
