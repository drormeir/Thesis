import itertools
import numpy as np
from tqdm import tqdm
from python.hpc import use_njit, HybridArray
from python.metrics.numba_gpu import detect_signal_auc_gpu
from python.metrics.numba_cpu import detect_signal_auc_cpu_njit
from python.metrics.python_native import detect_signal_auc_py
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis, rare_weak_model
from python.adaptive_methods.adaptive_methods import apply_transform_discovery_method
from python.array_math_utils.array_math_utils import array_transpose_inplace, sort_rows_inplace, max_along_rows, array_transpose

def analyze_auc_r_beta_ranges(\
        N: int,\
        r_range: np.ndarray|list,\
        beta_range: np.ndarray|list,\
        alpha_selection_method: str|float,\
        **kwargs) -> np.ndarray:
    r_range = np.asarray(r_range).reshape(-1)
    beta_range = np.asarray(beta_range).reshape(-1)
    out_shape = (r_range.size,beta_range.size)
    mus = np.sqrt(2*r_range*np.log(N))
    epsilons = np.power(N,-beta_range)
    n1s = np.clip((epsilons*N+0.5).astype(np.uint32),np.uint32(1),N)
    mus, n1s = zip(*itertools.product(mus,n1s))
    ret = analyze_multi_auc(N=N, alpha_selection_method=alpha_selection_method,\
                            n1s=n1s, mus=mus, **kwargs)
    if isinstance(alpha_selection_method, str):
        ret = ret.reshape((2,)+out_shape)
    else:
        ret = ret.reshape(out_shape)
    return ret

def analyze_multi_auc(\
        N: int,\
        n1s: tuple|list|np.ndarray, mus: tuple|list|np.ndarray,\
        alpha_selection_method: str|float|None=None,\
        **kwargs) -> np.ndarray:
    if isinstance(n1s, np.ndarray):
        n1s = n1s.reshape(-1).tolist()
    if isinstance(mus, np.ndarray):
        mus = mus.reshape(-1).tolist()
    assert n1s and mus
    assert len(n1s) == len(mus)
    num_executions = len(n1s)
    use_gpu = kwargs.get('use_gpu', None)
    auc_results = HybridArray().realloc(shape=(num_executions,N), dtype=np.float64, use_gpu=use_gpu)
    with (HybridArray() as noise,\
            HybridArray() as signal,\
            tqdm(total=num_executions+1, desc="Processing", unit="step") as pbar):
        pbar.set_postfix({"Current Step": 0})  # Set dynamic message
        create_noise_4_auc(noise=noise, N=N, ind_model=num_executions, **kwargs)
        pbar.update(1)
        for ind_model,n1,mu in zip(range(num_executions), n1s, mus):
            pbar.set_postfix({"Current Step": ind_model+1})  # Set dynamic message
            create_signal_4_auc(\
                signal=signal, N=N,\
                ind_model=ind_model, n1=n1, mu=mu,\
                **kwargs)
            detect_signal_auc(noise_input=noise, signal_input_work=signal,\
                            auc_out_row=auc_results.select_row(ind_model),\
                                **kwargs)
            pbar.update(1)
    auc_results.uncrop()
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

def test_speed_neto_detect_signal_auc(\
        N: int,\
        num_monte: int,\
        n1: int|np.uint32,\
        mu: float|np.float64|np.float32,\
        num_executions: int,\
        transform_method: str ='identity',
        detect_signal: bool = True,\
        create_signal: bool = True,\
        **kwargs) -> None:
    use_gpu = kwargs.get('use_gpu', None)
    with (
        HybridArray() as auc_results,\
        HybridArray() as noise,\
        HybridArray().realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu) as signal):
        create_noise_4_auc(noise=noise, N=N, ind_model=num_executions, num_monte=num_monte,\
                           transform_method=transform_method, **kwargs)
        if not create_signal:
            array_transpose(array=noise, out=signal, **kwargs)
        for ind_execution in tqdm(range(num_executions), desc="Test Speed Detect Signal AUC", unit="step"):
            if create_signal:
                create_signal_4_auc(\
                    signal=signal, N=N, num_monte=num_monte,\
                    ind_model=ind_execution, n1=n1, mu=mu,\
                    transform_method=transform_method,\
                    **kwargs)
            if detect_signal:
                detect_signal_auc(noise_input=noise, signal_input_work=signal,\
                                auc_out_row=auc_results,\
                                    **kwargs)
            print(f'Done iter {ind_execution+1} out of {num_executions}')
    print('Done!')


def create_noise_4_auc(noise: HybridArray,\
                       num_monte: int, N: int,\
                       ind_model: int=0,
                       **kwargs) -> None:
    use_gpu = kwargs.get('use_gpu', None)
    noise.realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu)
    rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_model,\
                              **kwargs)
    apply_transform_discovery_method(
        sorted_p_values_input_output=noise,
        num_discoveries_output=None,\
        **kwargs)
    array_transpose_inplace(noise, **kwargs)
    sort_rows_inplace(noise, **kwargs)


def create_signal_4_auc(signal: HybridArray,\
                       num_monte: int, N: int,\
                        ind_model: int,\
                        n1: np.uint32|int,\
                        mu: np.float64|np.float32|float,\
                        **kwargs) -> None:
    use_gpu = kwargs.get('use_gpu', None)
    signal.realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu)
    rare_weak_model(sorted_p_values_output=signal,\
                    cumulative_counts_output=None,\
                    ind_model=ind_model, mu=mu, n1=n1,\
                    **kwargs)
    apply_transform_discovery_method(\
        sorted_p_values_input_output=signal,\
        num_discoveries_output=None,\
        **kwargs)


def detect_signal_auc(\
        noise_input: HybridArray,\
        signal_input_work: HybridArray,\
        auc_out_row: HybridArray,\
        **kwargs) -> None:
    signal_shape = signal_input_work.shape()
    auc_out_row.realloc(like=signal_input_work, shape=(1,signal_shape[1]))
    assert signal_input_work.is_gpu() == noise_input.is_gpu()
    assert signal_input_work.is_gpu() == auc_out_row.is_gpu()
    noise_shape = noise_input.shape()
    auc_shape = auc_out_row.shape()
    assert noise_shape[0] == signal_shape[1], f'{noise_shape=} {signal_shape=}'
    assert auc_shape == (1,signal_shape[1]), f'{auc_shape=}'
    if signal_input_work.is_gpu():
        # GPU mode
        detect_signal_auc_gpu(noise_input, signal_input_work, auc_out_row)
    else:
        # CPU mode
        if use_njit(**kwargs):
            detect_signal_auc_cpu_njit(\
                noise_input.numpy(),\
                signal_input_work.numpy(), auc_out_row.numpy())
        else:
            detect_signal_auc_py(noise_input, signal_input_work, auc_out_row)
