import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.metrics.numba_gpu import detect_signal_auc_gpu
from python.metrics.numba_cpu import detect_signal_auc_cpu_njit
from python.metrics.python_native import detect_signal_auc_py
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis, rare_weak_model
from python.adaptive_methods.adaptive_methods import apply_transform_method
from python.array_math_utils.array_math_utils import array_transpose_inplace, sort_rows_inplace

def create_noise_4_auc(noise: HybridArray, shape: tuple, transform_method: str,\
                       use_gpu: bool|None = None, use_njit: bool|None = None,\
                       num_steps: int|None=None, ind_model: int=0) -> None:
    noise.realloc(shape=shape, dtype=np.float64, use_gpu=use_gpu)
    rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_model,\
                              num_steps=num_steps, use_njit=use_njit)
    apply_transform_method(sorted_p_values_input_output=noise,
                           transform_method=transform_method,\
                           use_njit=use_njit)
    array_transpose_inplace(noise, use_njit=use_njit)
    sort_rows_inplace(noise, use_njit=use_njit)

def create_signal_4_auc(signal: HybridArray, counts: HybridArray,\
                        shape: tuple, transform_method: str, ind_model: int,\
                        epsilon: np.float64|np.float32|float, mu: np.float64|np.float32|float,\
                        use_gpu: bool|None = None, use_njit: bool|None = None,\
                        num_steps: int|None = None) -> None:
    signal.realloc(shape=shape, dtype=np.float64, use_gpu=use_gpu)
    n1 = max(np.uint32(1),np.uint32(epsilon*shape[1]))
    rare_weak_model(sorted_p_values_output=signal, cumulative_counts_output=counts,\
                    ind_model=ind_model, mu=mu, n1=n1, num_steps=num_steps, use_njit=use_njit)
    apply_transform_method(sorted_p_values_input_output=signal,
                            transform_method=transform_method,\
                            use_njit=use_njit)

def detect_signal_auc(\
        noise_input: HybridArray,\
        signal_input_work: HybridArray,\
        auc_out_row: HybridArray,\
        use_njit: bool|None = None) -> None:
    assert signal_input_work.is_gpu() == noise_input.is_gpu()
    assert signal_input_work.is_gpu() == auc_out_row.is_gpu()
    noise_shape = noise_input.shape()
    signal_shape = signal_input_work.shape()
    auc_shape = auc_out_row.shape()
    assert noise_shape[0] == signal_shape[1], f'{noise_shape=} {signal_shape=}'
    assert auc_shape == (1,signal_shape[1])
    if signal_input_work.is_gpu():
        # GPU mode
        detect_signal_auc_gpu(noise_input, signal_input_work, auc_out_row)
    else:
        # CPU mode
        if globals.cpu_njit_num_threads and (use_njit is None or use_njit):
            detect_signal_auc_cpu_njit(\
                noise_input.numpy(),\
                signal_input_work.numpy(), auc_out_row.numpy())
        else:
            detect_signal_auc_py(noise_input, signal_input_work, auc_out_row)
