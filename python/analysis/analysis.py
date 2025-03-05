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


def multi_heatmap_auc_vs_r_beta_range(\
        r_range: np.ndarray|list, beta_range: np.ndarray|list,\
        N:int, num_monte: int,\
        recipe: list[str|tuple[str,str,str]|tuple[str,str,float]],
        **kwargs) -> None:
    print(f'Running on single_heatmap_auc_vs_r_beta_range {kwargs}')
    aucs = []
    titles = []
    for recipe_method in recipe:
        if isinstance(recipe_method,str):
            assert recipe_method=='identity'
            auc =analyze_auc_r_beta_ranges(\
                r_range=r_range, beta_range=beta_range,\
                N=N, num_monte=num_monte,\
                transform_method=recipe_method,\
                alpha_selection_methods=[0.0],\
                use_gpu=True, **kwargs)
            visualization.heatmap_r_beta_range(\
                r_range=r_range,\
                beta_range=beta_range,\
                data=auc,\
                value_name='AUC', data_min=0.5, data_max=None,\
                title=f'p_value transform: Original. Statisti=Lowest p_value\n{N=} {num_monte=}')
            titles.append('Lowest p_value')
        else:
            transform_method, discovery_method, alpha_method = recipe_method
            auc =analyze_auc_r_beta_ranges(\
                r_range=r_range, beta_range=beta_range,\
                N=N, num_monte=num_monte,\
                transform_method=transform_method,\
                discovery_method=discovery_method,\
                alpha_selection_methods=alpha_method,\
                use_gpu=True, **kwargs)
            title = transform_method
            if discovery_method:
                title += f' {discovery_method=}'
            title += f' (alpha={alpha_method})'
            visualization.heatmap_r_beta_range(\
                r_range=r_range,\
                beta_range=beta_range,\
                data=auc,\
                value_name='AUC', data_min=0.5, data_max=None,\
                title=f'{title}\n{N=} {num_monte=}')
            titles.append(title)
        aucs.append(auc)
    aucs = np.array(aucs)
    argmax = aucs.argmax(axis=0).astype(np.uint32)
    visualization.imagemap_r_beta_range(r_range=r_range, beta_range=beta_range,\
                                        data=argmax, labels=titles,\
                                        title='Best statisti to detect signal using AUC')