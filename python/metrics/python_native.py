import numpy as np
from sklearn.metrics import roc_auc_score
from python.hpc import HybridArray

def detect_signal_auc_py(\
            noise_input: HybridArray|np.ndarray,\
            signal_input: HybridArray|np.ndarray,\
            auc_out_row: HybridArray|np.ndarray) -> None:
    if isinstance(noise_input,HybridArray):
        noise_input = noise_input.numpy()
    if isinstance(signal_input,HybridArray):
        signal_input = signal_input.numpy()
    if isinstance(auc_out_row,HybridArray):
        auc_out_row = auc_out_row.numpy()
    noise_size = np.uint32(noise_input.shape[1])
    num_monte, N = signal_input.shape
    y_true = np.concatenate([np.zeros(shape=(noise_size,),dtype=np.uint32),\
                             np.ones(shape=(num_monte,),dtype=np.uint32)])
    for ind_col in range(N):
        y_score = np.concatenate([\
            noise_input[ind_col],\
            signal_input[:,ind_col]])
        auc = roc_auc_score(y_true=y_true, y_score=y_score)
        auc = np.float64(auc)
        auc_out_row[0][ind_col] = max(auc,1-auc)
