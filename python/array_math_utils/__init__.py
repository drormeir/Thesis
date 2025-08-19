from .array_math_utils import\
    array_transpose_inplace,\
    array_transpose,\
    average_row,\
    average_column,\
    sort_rows_inplace,\
    cumulative_argmin,\
    cumulative_argmax,\
    cumulative_min_inplace,\
    cumulative_max_inplace,\
    cumulative_dominant_argmin,\
    cumulative_dominant_argmax,\
    cumulative_dominant_min_inplace,\
    cumulative_dominant_max_inplace,\
    max_column_along_rows,\
    argmax_column_along_rows,\
    min_column_along_rows,\
    argmin_column_along_rows

class cpu:
    from .numba_cpu import (
        array_transpose_cpu_njit as array_transpose,
        average_row_cpu_njit as average_row,
        average_column_cpu_njit as average_column,
        sort_rows_inplace_cpu_njit as sort_rows_inplace,
        cumulative_argmin_cpu_njit as cumulative_argmin,
        cumulative_argmax_cpu_njit as cumulative_argmax,
        cumulative_min_inplace_cpu_njit as cumulative_min_inplace,
        cumulative_max_inplace_cpu_njit as cumulative_max_inplace,
        cumulative_dominant_argmin_cpu_njit as cumulative_dominant_argmin,
        cumulative_dominant_argmax_cpu_njit as cumulative_dominant_argmax,
        cumulative_dominant_min_inplace_cpu_njit as cumulative_dominant_min_inplace,
        cumulative_dominant_max_inplace_cpu_njit as cumulative_dominant_max_inplace,
        max_column_along_rows_cpu_njit as max_column_along_rows,
        argmax_column_along_rows_cpu_njit as argmax_column_along_rows,
        min_column_along_rows_cpu_njit as min_column_along_rows,
        argmin_column_along_rows_cpu_njit as argmin_column_along_rows
    )

class gpu:
    from .numba_gpu import (
        array_transpose_gpu as array_transpose,
        average_row_gpu as average_row,
        average_column_gpu as average_column,
        sort_rows_inplace_gpu as sort_rows_inplace,
        cumulative_argmin_gpu as cumulative_argmin,
        cumulative_argmax_gpu as cumulative_argmax,
        cumulative_min_inplace_gpu as cumulative_min_inplace,
        cumulative_max_inplace_gpu as cumulative_max_inplace,
        cumulative_dominant_argmin_gpu as cumulative_dominant_argmin,
        cumulative_dominant_argmax_gpu as cumulative_dominant_argmax,
        cumulative_dominant_min_inplace_gpu as cumulative_dominant_min_inplace,
        cumulative_dominant_max_inplace_gpu as cumulative_dominant_max_inplace,
        max_column_along_rows_gpu as max_column_along_rows,
        argmax_column_along_rows_gpu as argmax_column_along_rows,
        min_column_along_rows_gpu as min_column_along_rows,
        argmin_column_along_rows_gpu as argmin_column_along_rows
    )

class python:
    from .python_native import (
        array_transpose_py as array_transpose,
        average_row_py as average_row,
        average_column_py as average_column,
        sort_rows_inplace_py as sort_rows_inplace,
        cumulative_argmin_py as cumulative_argmin,
        cumulative_argmax_py as cumulative_argmax,
        cumulative_min_inplace_py as cumulative_min_inplace,
        cumulative_max_inplace_py as cumulative_max_inplace,
        cumulative_dominant_argmin_py as cumulative_dominant_argmin,
        cumulative_dominant_argmax_py as cumulative_dominant_argmax,
        cumulative_dominant_min_inplace_py as cumulative_dominant_min_inplace,
        cumulative_dominant_max_inplace_py as cumulative_dominant_max_inplace,
        max_column_along_rows_py as max_column_along_rows,
        argmax_column_along_rows_py as argmax_column_along_rows,
        min_column_along_rows_py as min_column_along_rows,
        argmin_column_along_rows_py as argmin_column_along_rows
    )

__all__ = [
    'array_transpose_inplace',
    'array_transpose',
    'average_row',
    'average_column',
    'sort_rows_inplace',
    'cumulative_argmin',
    'cumulative_argmax',
    'cumulative_min_inplace',
    'cumulative_max_inplace',
    'cumulative_dominant_argmin',
    'cumulative_dominant_argmax',
    'cumulative_dominant_min_inplace',
    'cumulative_dominant_max_inplace',
    'max_column_along_rows',
    'argmax_column_along_rows',
    'min_column_along_rows',
    'argmin_column_along_rows'
]
