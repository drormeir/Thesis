from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Synthetic_Data_Generators import Multi_Class_Normal_Population as Data_Generator
from Synthetic_Data_Generators import Two_Lists_Tuple, Data_Generator_Noise, Data_Generator_Base, signal_2_noise_roc
from Higher_Criticism import *
from typing import Any, Sequence

def AUC_best_objectives_from_random_values(
        hc_models: list,\
        signal_generators: Sequence[Data_Generator_Base],\
        monte_carlo: int = 10000, chunk_size: int = 100) -> np.ndarray:
    N = signal_generators[0].N
    data_generators: list[Data_Generator_Base] = [Data_Generator_Noise(N)]
    data_generators += signal_generators
    num_signal_generators = len(signal_generators)
    num_models = len(hc_models)
    signal_values = np.empty(shape=(num_signal_generators+1,num_models,monte_carlo))
    for seed0 in tqdm(list(range(0,monte_carlo,chunk_size))):
        seed1 = min(seed0+chunk_size,monte_carlo)
        seeds = list(range(seed0,seed1))
        random_values = Data_Generator_Base.generate_random_values(N=N, seeds=seeds)
        for ind_generator, data_generator in enumerate(data_generators):
            data_generator.generate_from_random_values(random_values=random_values)
            for ind_model, hc_model in enumerate(hc_models):
                hc_model.run_sorted_p(data_generator.p_values)
                signal_values[ind_generator, ind_model, seed0:seed1] = hc_model.best_objective
    result = np.empty(shape=(num_models, num_signal_generators))
    noise_values, signal_values = signal_values[0], signal_values[1:]
    auc_params = Two_Lists_Tuple(list(enumerate(hc_models)), list(range(num_signal_generators)))
    for (ind_model, hc_model), ind_signal in tqdm(auc_params):
        auc, _, _ = signal_2_noise_roc(\
            signal_values=signal_values[ind_signal][ind_model],\
            noise_values=noise_values[ind_model])
        result[ind_model, ind_signal] = auc
    return result


def AUC_asymptotic_analysis_multi_size(\
        N_range: list[int], beta_range: list[float], r_range: list[float], hc_models: list,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = [(ind_beta,beta,ind_r,r) for ind_beta, beta in enumerate(beta_range) for ind_r, r in enumerate(r_range)]
    model_result_shape = (len(N_range),len(r_range),len(beta_range))
    collect_results = {model.full_name: np.empty(shape=model_result_shape, dtype=np.float32) for model in hc_models}

    for ind_N, N in enumerate(N_range):
        print(f'Working on sample size: {N}')
        signal_generators = [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for ind_beta, beta, ind_r, r in params_list]
        auc_results = AUC_best_objectives_from_random_values(\
            hc_models=hc_models, signal_generators=signal_generators,\
            monte_carlo=monte_carlo, chunk_size=chunk_size)
            
        for ind_model, hc_model in enumerate(hc_models):
            for ind_param, param in enumerate(params_list):
                ind_beta, beta, ind_r, r = param
                collect_results[hc_model.full_name][ind_N, ind_r, ind_beta] = auc_results[ind_model][ind_param]

    for ind_beta, beta, ind_r, r in params_list:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        max_auc = 0
        for key in collect_results:
            auc = collect_results[key][:, ind_r, ind_beta].reshape(-1)
            # dictionary type hinting to avoid warnings
            line_params: dict[str, int | str] = {}
            key_lower = key.lower()
            if 'hc' in key_lower:
                line_params['linestyle'] = 'dashed' if 'power' in key_lower else 'solid'
            else:
                line_params['linestyle'] = 'dotted'
                line_params['linewidth'] = 3
            ax.plot(N_range, auc, label=key, **line_params)
            max_auc = max(max_auc, auc.max())
        if max_auc >= 0.9:
            ax.set_ylim(top=1.0)
        ax.set_xlim(left=0)
        ax.set_title(f'AUC values as function of number of samples using {monte_carlo} monte carlo runs.\n' + f'r={r:.2f} beta={beta:.2f}')
        ax.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
        plt.show()


def AUC_full_analysis_single_size(\
        N: int, beta_range: list[float], r_range: list[float],\
        gamma_range: list[float],\
        stables_flags: int = 3, global_max_flags: int = 3,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    assert 1 <= stables_flags <= 3
    assert 1 <= global_max_flags <= 3
    params_list = Two_Lists_Tuple(r_range, beta_range)
    signal_generators = [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for r, beta in params_list]
    # hc_models = [Bonferroni(alpha=-1), Benjamini_Hochberg(alpha=-1),]
    hc_models = []
    num_major_models = 0
    if (stables_flags & 1) and (global_max_flags & 1):
        num_major_models += 1
        hc_models += [Higher_Criticism(stable=True, global_max=True, gamma=gamma) for gamma in gamma_range]
    if (stables_flags & 2) and (global_max_flags & 1):
        num_major_models += 1
        hc_models += [Higher_Criticism(stable=False, global_max=True, gamma=gamma) for gamma in gamma_range]
    if (stables_flags & 1) and (global_max_flags & 2):
        num_major_models += 1
        hc_models += [Higher_Criticism(stable=True, global_max=False, gamma=gamma) for gamma in gamma_range]
    if (stables_flags & 2) and (global_max_flags & 2):
        num_major_models += 1
        hc_models += [Higher_Criticism(stable=False, global_max=False, gamma=gamma) for gamma in gamma_range]
    auc_results = AUC_best_objectives_from_random_values(\
        hc_models=hc_models, signal_generators=signal_generators,\
        monte_carlo=monte_carlo, chunk_size=chunk_size)
    num_gamma = len(gamma_range)
    num_beta = len(beta_range)
    num_r = len(r_range)
    im_colorbar = ScalarMappable()
    im_colorbar.set_clim(vmin=0.5, vmax=1.0)
    im_colorbar.set_cmap('rainbow')
    x_ticks = np.linspace(0.5, num_beta-0.5, num=num_beta)
    x_tick_labels = [f'{beta:.2f}' for beta in beta_range]
    y_ticks = np.linspace(0.5, num_r-0.5, num=num_r)
    y_tick_labels = [f'{r:.2f}' for r in r_range]
    fig, axs = plt.subplots(figsize=(num_major_models*4, 5*num_gamma),
                            nrows=num_gamma, ncols=num_major_models, sharex=False, sharey=True)
    ind_param_ind_model = list(enumerate(auc_results.argmax(axis=0)))
    ind_params_best_per_model = [[]] * len(hc_models)
    for ind_model, ind_params_best in enumerate(ind_params_best_per_model):
        for ind_param_best in [ind_param for ind_param, ind_m in ind_param_ind_model if ind_m == ind_model]:
            ind_r_best, ind_beta_best = ind_param_best // num_beta, ind_param_best % num_beta
            ind_params_best.append((ind_r_best,ind_beta_best))
    for ind_model, hc_model in enumerate(hc_models):
        auc_model = auc_results[ind_model].reshape(num_r, num_beta)
        row, col = ind_model % num_gamma, ind_model // num_gamma
        ax = axs[row,col]
        im = ax.pcolor(auc_model, cmap=im_colorbar.get_cmap(), clim=im_colorbar.get_clim())
        ax.set_title(hc_model.name)
        ax.set_xlabel('Beta')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        if col == 0:
            ax.set_ylabel(hc_model.str_gamma + '\n\nr')
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
        is_best_r_beta = np.zeros_like(auc_model, dtype=bool)
        for ind_r_best, ind_beta_best in ind_params_best_per_model[ind_model]:
            is_best_r_beta[ind_r_best, ind_beta_best] = True
        for ind_r in range(num_r):
            for ind_beta in range(num_beta):
                color = "w" if is_best_r_beta[ind_r, ind_beta] else "black"
                str_value = f'{auc_model[ind_r, ind_beta]:.2f}'
                ax.text(ind_beta+0.5, ind_r+0.5, str_value, ha="center", va="center", color=color)
    fig.suptitle(f'AUC according to selected value of objective function for signal detection in sample size: {N}', y=1)
    fig.tight_layout()
    # set color bars only after tight layout
    row_height = 1/num_gamma
    x0 = 1
    width = 0.03
    height = 0.82*row_height
    for row in range(num_gamma):
        dy_above_bottom = 0.03 + 0.1*row/num_gamma
        bottom_y = (num_gamma-1-row+dy_above_bottom)*row_height
        cax = fig.add_axes((x0, bottom_y, width, height))
        fig.colorbar(im_colorbar, cax=cax)
    plt.show()


class Monte_carlo_Confusion_Matrices:
    def __init__(self,
        hc_models: list,\
        signal_generators: Sequence[Data_Generator_Base],\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
        N = signal_generators[0].N
        num_signal_generators = len(signal_generators)
        num_models = len(hc_models)
        self.confusion_matrices = np.empty(shape=(num_models,num_signal_generators,monte_carlo,4))
        for seed0 in tqdm(list(range(0,monte_carlo,chunk_size))):
            seed1 = min(seed0+chunk_size,monte_carlo)
            seeds = list(range(seed0,seed1))
            random_values = Data_Generator_Base.generate_random_values(N=N, seeds=seeds)
            for ind_generator, signal_generator in enumerate(signal_generators):
                signal_generator.generate_from_random_values(random_values=random_values)
                for ind_model, hc_model in enumerate(hc_models):
                    hc_model.run_sorted_p(signal_generator.p_values)
                    self.confusion_matrices[ind_model, ind_generator, seed0:seed1] = signal_generator.calc_confusion(hc_model.num_rejected)
        
    def apply_func(self, func) -> np.ndarray:
        num_models, num_signal_generators = self.confusion_matrices.shape[:2]
        ret = np.empty(shape=(num_models, num_signal_generators))
        for ind_model, conf_matrices_per_model in enumerate(self.confusion_matrices):
            for ind_generator, monte_carlo_conf_matrices in enumerate(conf_matrices_per_model):
                true_positive = monte_carlo_conf_matrices[:,0]
                false_positive = monte_carlo_conf_matrices[:,1]
                true_negative = monte_carlo_conf_matrices[:,2]
                false_negative = monte_carlo_conf_matrices[:,3]
                ret[ind_model, ind_generator] = func(true_positive, false_positive, true_negative, false_negative)
        return ret

    def apply_mean_false_disovery_rate(self) -> np.ndarray:
        def mean_fdr(true_positive, false_positive, true_negative, false_negative) -> float:
            num_rejected = false_positive + true_positive
            with np.errstate(divide='ignore', invalid='ignore'):
                fdr_all = false_positive/num_rejected
            fdr_all[~ np.isfinite(fdr_all)] = 1
            return np.mean(fdr_all)
        return self.apply_func(mean_fdr)

    def apply_mean_misdetection_rate(self) -> np.ndarray:
        def mean_mdr(true_positive, false_positive, true_negative, false_negative) -> float:
            original_signal_count = false_negative + true_positive
            with np.errstate(divide='ignore', invalid='ignore'):
                mdr_all = false_negative/original_signal_count
            mdr_all[~ np.isfinite(mdr_all)] = 1
            return np.mean(mdr_all)
        return self.apply_func(mean_mdr)
    
    def apply_sqrt_mean_misclassification_rate(self) -> np.ndarray:
        def mean_mdr(true_positive, false_positive, true_negative, false_negative) -> float:
            mcr_all = true_positive == 0
            return mcr_all.mean()**0.5
        return self.apply_func(mean_mdr)

