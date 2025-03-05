import itertools
import numpy as np
from tqdm import tqdm
from python.hpc import use_njit, HybridArray
from python.metrics.numba_gpu import detect_signal_auc_gpu
from python.metrics.numba_cpu import detect_signal_auc_cpu_njit
from python.metrics.python_native import detect_signal_auc_py
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis, rare_weak_model
from python.adaptive_methods.adaptive_methods import apply_transform_discovery_method
from python.array_math_utils.array_math_utils import array_transpose_inplace, sort_rows_inplace, max_column_along_rows, array_transpose

def analyze_auc_r_beta_ranges(\
        N: int,\
        r_range: np.ndarray|list,\
        beta_range: np.ndarray|list,\
        alpha_selection_methods: list|str|float|None,\
        **kwargs) -> np.ndarray:
    r_range = np.asarray(r_range).reshape(-1)
    beta_range = np.asarray(beta_range).reshape(-1)
    mus = np.sqrt(2*r_range*np.log(N))
    epsilons = np.power(N,-beta_range)
    n1s = np.clip((epsilons*N+0.5).astype(np.uint32),np.uint32(1),N)
    assert n1s.dtype == np.uint32
    mu_n1_tuples = list(itertools.product(mus,n1s))
    ret = analyze_auc_multi_tuples_mu_n1(N=N, alpha_selection_methods=alpha_selection_methods,\
                            mu_n1_tuples=mu_n1_tuples, **kwargs)
    n1_mu_shape = (mus.size,n1s.size)
    if alpha_selection_methods is None:
        return ret.reshape(n1_mu_shape+(N,))    
    return ret.reshape((ret.shape[0],)+n1_mu_shape).squeeze()


def analyze_auc_multi_tuples_mu_n1(\
        N: int,\
        mu_n1_tuples: tuple[float,int|np.uint32]|list[tuple[float,int|np.uint32]],\
        transform_method: str,\
        alpha_selection_methods: list|str|float|None=None,\
        **kwargs) -> np.ndarray:
    if not isinstance(mu_n1_tuples, list):
        mu_n1_tuples = [mu_n1_tuples]
    num_executions = len(mu_n1_tuples)
    use_gpu = kwargs.get('use_gpu', None)
    auc_results = HybridArray().realloc(shape=(num_executions,N), dtype=np.float64, use_gpu=use_gpu)
    with (HybridArray() as noise,\
            HybridArray() as signal,\
            tqdm(total=num_executions+1, desc=f'Processing {transform_method}', unit='step') as pbar):
        pbar.set_postfix({"Current Step": 0})  # Set dynamic message
        create_noise_4_auc(noise=noise, N=N, ind_model=num_executions, transform_method=transform_method, **kwargs)
        pbar.update(1)
        for ind_model,(mu,n1) in enumerate(mu_n1_tuples):
            assert isinstance(n1,int) or np.issubdtype(n1,np.integer), f'analyze_auc_multi_tuples_n1_mu{mu_n1_tuples[0]=} {type(n1)=}'
            pbar.set_postfix({"Current Step": ind_model+1})  # Set dynamic message
            create_signal_4_auc(\
                signal=signal, N=N,\
                ind_model=ind_model, n1=n1, mu=mu,\
                transform_method=transform_method,\
                **kwargs)
            detect_signal_auc(noise_input=noise, signal_input_work=signal,\
                            auc_out_row=auc_results.select_row(ind_model),\
                                **kwargs)
            pbar.update(1)
    auc_results.uncrop()
    if alpha_selection_methods is None:
        assert auc_results.shape() == (num_executions,N)
        return auc_results.numpy()
    if not isinstance(alpha_selection_methods,list):
        alpha_selection_methods = [alpha_selection_methods]
    ret_list: list = []
    save_max_label = ''
    save_max_values = None
    for alpha_method in alpha_selection_methods:
        if isinstance(alpha_method,str):
            if alpha_method == save_max_label:
                ret_list.append(save_max_values)
                continue
            if alpha_method in ['max', 'argmax']: 
                with (HybridArray() as argmax, HybridArray() as maxval):
                    max_column_along_rows(auc_results,argmax=argmax,maxval=maxval)
                    maxval_numpy = maxval.numpy()
                    argmax_numpy = ((argmax.numpy()+1)/N).astype(np.float64)
                    if alpha_method == 'max':
                        ret_list.append(maxval_numpy)
                        save_max_label = 'argmax'
                        save_max_values = argmax_numpy
                    else:
                        ret_list.append(argmax_numpy)
                        save_max_label = 'max'
                        save_max_values = maxval_numpy
            elif alpha_method == 'first':
                ret_list.append(auc_results.select_col(0).numpy())
                auc_results.uncrop()
            else:
                assert False, f'{alpha_selection_methods=}'
        elif isinstance(alpha_method, (float, np.floating)):
            ind_col = min(np.uint32(N*max(alpha_method,0.0) + 0.5),np.uint32(N-1))
            ret_list.append(auc_results.select_col(ind_col).numpy())
            auc_results.uncrop()
        else:
            assert False, f'{alpha_selection_methods=}'
    if len(ret_list) == 1:
        ret_numpy = ret_list[0]
    else:
        ret_numpy = np.hstack(ret_list)
    return ret_numpy.T


def test_speed_neto_detect_signal_auc(\
        N: int,\
        num_monte: int,\
        n1: int|np.uint32,\
        mu: float|np.float64|np.float32,\
        num_executions: int,\
        transform_method: str ='identity',
        create_signal: bool = True,\
        detect_signal: bool = True,\
        **kwargs) -> None:
    use_gpu = kwargs.get('use_gpu', None)
    desc = f'Test Speed Detect Signal AUC {transform_method=} {create_signal=} {detect_signal=}'
    with (
        HybridArray().realloc(shape=(1,N), dtype=np.float64, use_gpu=use_gpu) as auc_results,\
        HybridArray() as noise,\
        HybridArray().realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu) as signal):
        create_noise_4_auc(noise=noise, N=N, ind_model=num_executions, num_monte=num_monte,\
                           transform_method=transform_method, **kwargs)
        if not create_signal:
            array_transpose(array=noise, out=signal, **kwargs)
        for ind_execution in tqdm(range(num_executions), desc=desc, unit="step"):
            if create_signal:
                create_signal_4_auc(\
                    signal=signal, N=N, num_monte=num_monte,\
                    ind_model=ind_execution, n1=n1, mu=mu,\
                    transform_method=transform_method,\
                    **kwargs)
            if detect_signal:
                detect_signal_auc(noise_input=noise, signal_input_work=signal, auc_out_row=auc_results, **kwargs)
            pass
        pass

def create_noise_4_auc(noise: HybridArray,\
                       num_monte: int, N: int,\
                       transform_method: str,\
                       ind_model: int=0,
                       **kwargs) -> None:
    use_gpu = kwargs.get('use_gpu', None)
    noise.realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu)
    rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=ind_model,\
                              **kwargs)
    apply_transform_discovery_method(
        sorted_p_values_input_output=noise,
        num_discoveries_output=None,\
        transform_method=transform_method,\
        **kwargs)
    array_transpose_inplace(noise, **kwargs)
    sort_rows_inplace(noise, **kwargs)


def create_signal_4_auc(signal: HybridArray,\
                        num_monte: int, N: int,\
                        transform_method: str,\
                        ind_model: int,\
                        n1: np.uint32|int,\
                        mu: np.float64|np.float32|float,\
                        **kwargs) -> None:
    assert isinstance(n1,int) or np.issubdtype(n1,np.integer), f'create_signal_4_auc({n1=})'
    use_gpu = kwargs.get('use_gpu', None)
    signal.realloc(shape=(num_monte,N), dtype=np.float64, use_gpu=use_gpu)
    rare_weak_model(sorted_p_values_output=signal,\
                    cumulative_counts_output=None,\
                    ind_model=ind_model, mu=mu, n1=n1,\
                    **kwargs)
    apply_transform_discovery_method(\
        sorted_p_values_input_output=signal,\
        num_discoveries_output=None,\
        transform_method=transform_method,\
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
