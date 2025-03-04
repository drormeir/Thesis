import numpy as np
from python.metrics.metrics import analyze_auc_r_beta_ranges
from python.analysis import visualization

def single_heatmap_auc_vs_r_beta_range(\
        r_range: np.ndarray|list, beta_range: np.ndarray|list,\
        N:int, num_monte: int,\
        transform_method: str,\
        alpha_selection_method:float|str,\
        **kwargs) -> None:
    print(f'Running on single_heatmap_auc_vs_r_beta_range {kwargs}')
    auc =analyze_auc_r_beta_ranges(\
            r_range=r_range, beta_range=beta_range,\
            N=N, num_monte=num_monte,\
            transform_method=transform_method,\
            alpha_selection_methods=alpha_selection_method,\
            use_gpu=True, **kwargs)
    visualization.heatmap_r_beta_range(\
            r_range=r_range,\
            beta_range=beta_range,\
            data=auc,\
            value_name='AUC', data_min=0.5, data_max=None,\
            title=f'p_value transform: {transform_method}\n{N=} {num_monte=} {alpha_selection_method=}')
