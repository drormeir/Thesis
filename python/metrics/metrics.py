import numpy as np
from python.hpc import globals, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.metrics.numba_gpu import detect_signal_auc_gpu
from python.metrics.numba_cpu import detect_signal_auc_cpu_njit
from python.metrics.python_native import detect_signal_auc_py
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis, rare_weak_model
from python.adaptive_methods.adaptive_methods import apply_transform_discovery_method
from python.array_math_utils.array_math_utils import array_transpose_inplace, sort_rows_inplace, max_along_rows
from tqdm import tqdm

def analyze_multi_auc(\
        shape: tuple,\
        transform_method: str,\
        discovery_method: str,\
        n1s: list|np.ndarray, mus: list|np.ndarray,\
        alpha_selection_method: str|float|None=None,\
        use_gpu: bool|None = None, use_njit: bool|None = None,\
        num_steps: int|None=None) -> np.ndarray:
    if isinstance(n1s, np.ndarray):
        n1s = n1s.reshape(-1).tolist()
    if isinstance(mus, np.ndarray):
        mus = mus.reshape(-1).tolist()
    assert n1s and mus
    assert len(n1s) == len(mus)
    num_executions = len(n1s)
    N = shape[1]
    auc_results = HybridArray().realloc(shape=(num_executions,N), dtype=np.float64, use_gpu=use_gpu)

    with (HybridArray() as noise,\
            HybridArray() as signal,\
            tqdm(total=num_executions+1, desc="Processing", unit="step") as pbar):
        pbar.set_postfix({"Current Step": 0})  # Set dynamic message
        create_noise_4_auc(noise=noise, shape=shape,\
                        transform_method=transform_method,\
                        discovery_method=discovery_method,\
                        use_gpu=use_gpu, use_njit=use_njit, num_steps=num_steps, ind_model=num_executions)
        pbar.update(1)
        for ind_model,n1,mu in zip(range(num_executions), n1s, mus):
            pbar.set_postfix({"Current Step": ind_model+1})  # Set dynamic message
            create_signal_4_auc(\
                signal=signal, shape=shape,\
                transform_method=transform_method,\
                discovery_method=discovery_method,\
                ind_model=ind_model, n1=n1, mu=mu,\
                use_gpu=use_gpu, use_njit=use_njit,\
                num_steps=num_steps)
            auc_results.select_row(ind_model)
            detect_signal_auc(noise_input=noise, signal_input_work=signal,\
                            auc_out_row=auc_results, use_njit=use_njit)
            pbar.update(1)

    if alpha_selection_method is None:
        ret = auc_results.numpy()
    elif isinstance(auc_results,str) and auc_results == 'max':
        with (HybridArray() as argmax, HybridArray() as maxval):
            max_along_rows(auc_results,argmax=argmax,maxval=maxval)
            argmax_result = ((argmax.numpy()+1)/N).astype(np.float64)
            maxval_result = maxval.numpy()
            ret = np.hstack([argmax_result, maxval_result])
    elif isinstance(alpha_selection_method, (float, np.floating)):
        ind_col = min(np.uint32(N*max(alpha_selection_method,0.0) + 0.5),np.uint32(N-1))
        ret = auc_results.select_col(ind_col).numpy()
    else:
        assert False, f'{alpha_selection_method=}'
    return ret

def create_noise_4_auc(noise: HybridArray, shape: tuple,\
                       transform_method: str,\
                       discovery_method: str,\
                       use_gpu: bool|None = None,\
                       use_njit: bool|None = None,\
                       num_steps: int|None=None,\
                       ind_model: int=0) -> None:
    noise.realloc(shape=shape, dtype=np.float64, use_gpu=use_gpu)
    rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_model,\
                              num_steps=num_steps, use_njit=use_njit)
    apply_transform_discovery_method(
        sorted_p_values_input_output=noise,
        num_discoveries_output=None,\
        transform_method=transform_method,\
        discovery_method=discovery_method,\
        use_njit=use_njit)
    array_transpose_inplace(noise, use_njit=use_njit)
    sort_rows_inplace(noise, use_njit=use_njit)


def create_signal_4_auc(signal: HybridArray, shape: tuple,\
                        transform_method: str,\
                        discovery_method: str,\
                        ind_model: int,\
                        n1: np.uint32|int, mu: np.float64|np.float32|float,\
                        use_gpu: bool|None = None, use_njit: bool|None = None,\
                        num_steps: int|None = None) -> None:
    signal.realloc(shape=shape, dtype=np.float64, use_gpu=use_gpu)
    rare_weak_model(sorted_p_values_output=signal,\
                    cumulative_counts_output=None,\
                    ind_model=ind_model, mu=mu, n1=n1, num_steps=num_steps, use_njit=use_njit)
    apply_transform_discovery_method(\
        sorted_p_values_input_output=signal,\
        num_discoveries_output=None,\
        transform_method=transform_method,\
        discovery_method=discovery_method,\
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
    assert auc_shape == (1,signal_shape[1]), f'{auc_shape=}'
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
