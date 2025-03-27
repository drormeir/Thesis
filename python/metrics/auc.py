import numpy as np
from tqdm import tqdm
from python.hpc import use_njit, HybridArray
from python.metrics.numba_gpu import detect_signal_auc_gpu
from python.metrics.numba_cpu import detect_signal_auc_cpu_njit
from python.metrics.python_native import detect_signal_auc_py
from python.rare_weak_model.rare_weak_model import rare_weak_null_hypothesis, rare_weak_model
from python.adaptive_methods.adaptive_methods import apply_transform_discovery_method, str_transform_method
from python.array_math_utils.array_math_utils import array_transpose_inplace, sort_rows_inplace, array_transpose
from python.metrics.r_beta_space import R_Beta_Space


class AUC_analysis(R_Beta_Space):
    def __init__(self, N: int,\
                 num_monte: int,\
                 r_range: np.ndarray|list,\
                 beta_range: np.ndarray|list,\
                 **kwargs) -> None:
        super(AUC_analysis, self).__init__(\
            N=N, num_monte=num_monte,\
            r_range=r_range, beta_range=beta_range,\
            base_metric_for_risk_factor=None,\
            **kwargs)


    def analyze(self, alpha_selection_methods, **kwargs) -> dict:
        use_gpu = kwargs.get('use_gpu', None)
        num_executions = self.r_beta_size
        auc_result = self.alloc_r_beta_full_results(use_gpu=use_gpu)
        str_method = str_transform_method(**kwargs)
        with (tqdm(total=num_executions+1, desc=f'Processing {str_method}', unit='step') as pbar,\
              HybridArray().realloc(shape=self.single_rare_weak_shape, dtype=np.float64, use_gpu=use_gpu) as noise,\
              HybridArray().realloc(shape=self.single_rare_weak_shape, dtype=np.float64, use_gpu=use_gpu) as signal,\
        ):
            pbar.set_postfix({"Current Step": 0})  # Set dynamic message
            # Create null hypothesis noise
            noise.realloc(like=signal)
            rare_weak_null_hypothesis(sorted_p_values_output=noise, ind_model=num_executions, **kwargs)
            apply_transform_discovery_method(
                sorted_p_values_input_output=noise,
                num_discoveries_output=None,\
                **kwargs)
            array_transpose_inplace(noise, **kwargs)
            sort_rows_inplace(noise, **kwargs)
            pbar.update(1)
            for ind_model in range(num_executions):
                pbar.set_postfix({"Current Step": ind_model+1})  # Set dynamic message
                # Create signal
                mu, n1 = self.mu_n1_tuples[ind_model]
                rare_weak_model(sorted_p_values_output=signal,\
                                cumulative_counts_output=None,\
                                ind_model=ind_model, mu=mu, n1=n1,\
                                **kwargs)
                apply_transform_discovery_method(\
                    sorted_p_values_input_output=signal,\
                    num_discoveries_output=None,\
                    **kwargs)
                # Detect signal
                auc_row = auc_result.select_row(ind_model)
                if use_gpu:
                    # GPU mode
                    detect_signal_auc_gpu(noise_input=noise, signal_input_work=signal, auc_out_row=auc_row)
                else:
                    # CPU mode
                    if use_njit(**kwargs):
                        detect_signal_auc_cpu_njit(noise.numpy(), signal.numpy(), auc_row.numpy())
                    else:
                        detect_signal_auc_py(noise.numpy(), signal.numpy(), auc_row.numpy())
                auc_result.uncrop()
                pbar.update(1)
        return self.select_by_alpha(r_beta_full_results=auc_result, alpha_selection_methods=alpha_selection_methods)


    def single_heatmap(self,\
            alpha_selection_method:float|str,\
            **kwargs) -> None:
        print(f'Running on single_heatmap_auc_vs_r_beta_range {kwargs}')
        kwargs['use_gpu'] = True
        auc_dict =self.analyze(alpha_selection_methods=alpha_selection_method, **kwargs)
        alphas, aucs = self.collect_values(auc_dict)
        num_monte, N = self.single_rare_weak_shape
        str_method = str_transform_method(**kwargs)
        title=f'p_value transform: {str_method}\n{N=} {num_monte=} {alphas[0]}'
        self.heatmap(data=aucs[0], title=title, data_min=0.5, data_max=1.0, value_name='AUC')


    def multi_heatmap(self, recipe: list[str|tuple], alpha_methods: list, seperate_max_best: bool=True, **kwargs) -> None:
        assert len(recipe) == len(alpha_methods)
        all_methods_aucs = []
        all_methods_titles = []
        max_methods_aucs = []
        max_methods_titles = []
        num_monte, N = self.single_rare_weak_shape
        for recipe_method, alpha_method in zip(recipe,alpha_methods):
            if isinstance(recipe_method,str):
                recipe_method = (recipe_method,None,None)
            transform_method, discover_dominant, discover_min = recipe_method
            auc_dict = self.analyze(alpha_selection_methods=alpha_method,\
                                    transform_method=transform_method,\
                                    discover_dominant=discover_dominant,\
                                    discover_min=discover_min,\
                                    use_gpu=True, **kwargs)
            alphas, aucs = self.collect_values(auc_dict)
            str_method = str_transform_method(transform_method=transform_method,\
                                              discover_dominant=discover_dominant,\
                                              discover_min=discover_min)
            for alpha, auc in zip(alphas,aucs):
                title=f'p_value transform: {str_method}\n{N=} {num_monte=} {alpha}'
                if 'arg' in alpha:
                    self.heatmap(data=auc, title=title, data_min=0.0, data_max=1.0, value_name='Optimal alpha')
                    continue
                if 'risk' in alpha:
                    self.heatmap(data=auc, title=title, data_min=0.0, data_max=0.2, value_name='AUC Risk Diff')
                    continue
                self.heatmap(data=auc, title=title, data_min=0.5, data_max=1.0, value_name='AUC')
                best_title = str_method + f' ({alpha})'
                if 'max' in alpha and seperate_max_best:
                    max_methods_aucs.append(auc)
                    max_methods_titles.append(best_title)
                else:
                    all_methods_aucs.append(auc)
                    all_methods_titles.append(best_title)
        for aucs, titles in zip([all_methods_aucs, max_methods_aucs], [all_methods_titles, max_methods_titles]):
            assert len(aucs) == len(titles)
            if len(aucs) < 2:
                continue
            ind_best_method, diff = self.select_best_param(aucs, True)
            self.imagemap(data=ind_best_method, labels=titles, title='Best statisti to detect signal using AUC', **kwargs)
            self.heatmap(data=diff, title='Second best analysis of methods', data_min=0.0, data_max=0.1, value_name='Diff to 2nd best AUC')
