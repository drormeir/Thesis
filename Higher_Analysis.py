from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Synthetic_Data_Generators import Multi_Class_Normal_Population as Data_Generator
from Synthetic_Data_Generators import Two_Lists_Tuple, Data_Generator_Noise, Data_Generator_Base, signal_2_noise_roc
from Higher_Criticism import *
from typing import Any, Sequence

def AUC_best_objectives_from_random_values(
        hc_models: Sequence[Base_Rejection_Method],\
        signal_generators: Sequence[Data_Generator_Base],\
        better: bool = False,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> np.ndarray:
    N = signal_generators[0].N
    data_generators: list[Data_Generator_Base] = [Data_Generator_Noise(N)]
    data_generators += signal_generators
    num_signal_generators = len(signal_generators)
    num_data_generators = num_signal_generators + 1
    num_gamma_per_model = [model.num_gamma for model in hc_models]
    ind_gamma_model_base = np.cumsum([0] + num_gamma_per_model)
    signal_values = np.empty(shape=(num_data_generators,ind_gamma_model_base[-1],monte_carlo))
    signal_rejected = np.empty_like(signal_values, dtype=np.int32)
    with tqdm(total=monte_carlo*num_data_generators) as progress_bar:
        for seed0 in tqdm(list(range(0,monte_carlo,chunk_size))):
            seed1 = min(seed0+chunk_size,monte_carlo)
            seeds = list(range(seed0,seed1))
            random_values = Data_Generator_Base.generate_random_values(N=N, seeds=seeds)
            for ind_generator, data_generator in enumerate(data_generators):
                data_generator.generate_from_random_values(random_values=random_values)
                for ind_base_model, hc_model in enumerate(hc_models):
                    hc_model.run_sorted_p(data_generator.p_values)
                    for ind_gamma in range(num_gamma_per_model[ind_base_model]):
                        ind_model_gamma = ind_gamma_model_base[ind_base_model] + ind_gamma
                        signal_values[ind_generator, ind_model_gamma, seed0:seed1] = hc_model.best_objective[:,ind_gamma]
                        signal_rejected[ind_generator, ind_model_gamma, seed0:seed1] = hc_model.num_rejected[:,ind_gamma]
                progress_bar.update(len(seeds))
    noise_values, signal_values = signal_values[0], signal_values[1:]
    noise_rejected, signal_rejected = signal_rejected[0], signal_rejected[1:]
    result = np.empty(shape=(ind_gamma_model_base[-1], num_signal_generators))
    auc_params = Two_Lists_Tuple(list(enumerate(hc_models)), list(range(num_signal_generators)))
    def rejected_signal_dict(num_rejected, signal_values) -> dict[int,list[np.float32]]:
        res = {}
        for rej,val in zip(num_rejected,signal_values):
            if rej in res:
                res[rej].append(val)
            else:
                res[rej] = [val]
        return res
    with tqdm(total=np.sum(num_gamma_per_model)*num_signal_generators) as progress_bar:
        for (ind_base_model, hc_model), ind_signal in tqdm(auc_params):
            values_direction = 1 if hc_model.is_method_max else -1
            for ind_gamma in range(num_gamma_per_model[ind_base_model]):
                ind_model_gamma = ind_gamma_model_base[ind_base_model] + ind_gamma
                signal_values_gamma = signal_values[ind_signal][ind_model_gamma]
                noise_values_gamma = noise_values[ind_model_gamma]
                if better:
                    signal_rejected_gamma = signal_rejected[ind_signal][ind_model_gamma]
                    noise_rejected_gamma = noise_rejected[ind_model_gamma]
                    noise_dict = rejected_signal_dict(noise_rejected_gamma,values_direction*noise_values_gamma)
                    signal_dict = rejected_signal_dict(signal_rejected_gamma,values_direction*signal_values_gamma)
                    noise_rejected_gamma = []
                    noise_unique_rejections = np.sort(list(noise_dict.keys()))
                    for noise_vals in noise_dict.values():
                        num_noise = len(noise_vals)
                        noise_rejected_gamma.extend(np.arange(num_noise+1)/num_noise)
                    signal_values_gamma = []
                    for signal_rej, signal_vals in signal_dict.items():
                        if signal_rej > noise_rejected_gamma[-1]:
                            signal_values_gamma.extend([2]*len(signal_vals))
                            continue
                        ind_noise_rej = np.searchsorted(noise_unique_rejections, signal_rej)
                        noise_vals = np.sort(noise_dict[noise_unique_rejections[ind_noise_rej]])
                        num_noise = noise_vals.size
                        signal_vals = np.searchsorted(noise_vals,signal_vals)
                        beyond = signal_vals == num_noise
                        signal_vals = signal_vals.astype(np.float32)/num_noise
                        signal_vals[beyond] = 2
                        signal_values_gamma.extend(signal_vals.tolist())
                    auc, _, _ = signal_2_noise_roc(signal_values=signal_values_gamma, noise_values=noise_values_gamma, direction=1)
                else:
                    auc, _, _ = signal_2_noise_roc(signal_values=signal_values_gamma, noise_values=noise_values_gamma, direction=values_direction)
                result[ind_model_gamma, ind_signal] = auc
                progress_bar.update(1)
    return result


def AUC_asymptotic_analysis_multi_size(\
        N_range: list[int], beta_range: list[float], r_range: list[float], hc_models: list,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = [(ind_beta,beta,ind_r,r) for ind_beta, beta in enumerate(beta_range) for ind_r, r in enumerate(r_range)]
    model_result_shape = (len(N_range),len(r_range),len(beta_range))
    num_gamma_per_model = [model.num_gamma for model in hc_models]
    ind_gamma_model_base = np.cumsum([0] + num_gamma_per_model)

    collect_results = {}
    line_params_per_model = {}
    for model in hc_models:
        for ind_gamma, gamma in enumerate(model.gamma):
            full_name = model.get_full_name_with_gamma(ind_gamma=ind_gamma)
            # dictionary type hinting to avoid warnings
            line_params: dict[str, int | str] = {}
            if isinstance(model, (Higher_Criticism, Import_HC)):
                if isinstance(gamma,float) and gamma > 0:
                    line_params['linestyle'] = 'solid'
                else:
                    line_params['linestyle'] = 'dashed'
            else:
                line_params['linestyle'] = 'dotted'
                line_params['linewidth'] = 3
            line_params_per_model[full_name] = line_params
            collect_results[full_name] = np.empty(shape=model_result_shape, dtype=np.float32)

    for ind_N, N in enumerate(N_range):
        print(f'Working on sample size: {N}')
        signal_generators = [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for ind_beta, beta, ind_r, r in params_list]
        auc_results = AUC_best_objectives_from_random_values(\
            hc_models=hc_models, signal_generators=signal_generators,\
            monte_carlo=monte_carlo, chunk_size=chunk_size)
            
        for ind_base_model, model in enumerate(hc_models):
            for ind_gamma in range(num_gamma_per_model[ind_base_model]):
                full_name = model.get_full_name_with_gamma(ind_gamma=ind_gamma)
                ind_gamma_model = ind_gamma_model_base[ind_base_model] + ind_gamma
                for ind_param, param in enumerate(params_list):
                    ind_beta, beta, ind_r, r = param
                    collect_results[full_name][ind_N, ind_r, ind_beta] = auc_results[ind_gamma_model][ind_param]

    for ind_beta, beta, ind_r, r in params_list:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        max_auc = 0
        for key in collect_results:
            auc = collect_results[key][:, ind_r, ind_beta].reshape(-1)
            line_params = line_params_per_model[key]
            ax.plot(N_range, auc, label=key, **line_params)
            max_auc = max(max_auc, auc.max())
        if max_auc >= 0.9:
            ax.set_ylim(top=1.0)
        ax.set_xlim(left=0)
        ax.set_title(f'AUC values as function of number of samples using {monte_carlo} monte carlo runs.\n' + f'r={r:.2f} beta={beta:.2f}')
        ax.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
        plt.show()


class ValuesTextColors:
    def __init__(self, min_color: str, max_color: str, num_gamma: int, num_major_models: int, num_r: int, num_beta: int, results: np.ndarray) -> None:
        self.min_color = min_color
        self.max_color = max_color
        # for _ in... --> different instances of list
        self.ind_params_max_per_model = [[] for _ in range(num_gamma*num_major_models)]
        for ind_param_max, ind_model_gamma in enumerate(results.argmax(axis=0)):
            ind_r_max, ind_beta_max = ind_param_max // num_beta, ind_param_max % num_beta
            self.ind_params_max_per_model[ind_model_gamma].append((ind_r_max, ind_beta_max))
        self.ind_params_min_per_model = [[] for _ in range(num_gamma*num_major_models)]
        for ind_param_min, ind_model_gamma in enumerate(results.argmin(axis=0)):
            ind_r_min, ind_beta_min = ind_param_min // num_beta, ind_param_min % num_beta
            self.ind_params_min_per_model[ind_model_gamma].append((ind_r_min, ind_beta_min))
        self.is_extreme_r_beta = np.empty(shape=(num_r,num_beta), dtype=np.int32)
    
    def build_extreme_values(self, ind_model_gamma: int) -> None:
        self.is_extreme_r_beta.fill(0)
        for ind_r_max, ind_beta_max in self.ind_params_max_per_model[ind_model_gamma]:
            self.is_extreme_r_beta[ind_r_max, ind_beta_max] = 1
        for ind_r_min, ind_beta_min in self.ind_params_min_per_model[ind_model_gamma]:
            self.is_extreme_r_beta[ind_r_min, ind_beta_min] = -1

    def get_color(self, ind_r: int, ind_beta: int) -> str:
        type_r_beta = self.is_extreme_r_beta[ind_r, ind_beta]
        if type_r_beta > 0:
            return self.max_color
        if type_r_beta < 0:
            return self.min_color
        return 'gray'

def AUC_full_analysis_single_size(\
        N: int, beta_range: list[float], r_range: list[float],\
        major_models: Sequence[Base_Rejection_Method],\
        better: bool = False,
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = Data_Generator.params_list_from_N_r_beta(N=N, r=r_range, beta=beta_range)
    signal_generators = [Data_Generator(**params) for params in params_list]
    num_major_models = len(major_models)
    auc_results = AUC_best_objectives_from_random_values(\
        hc_models=major_models, signal_generators=signal_generators,\
        better=better,\
        monte_carlo=monte_carlo, chunk_size=chunk_size)
    num_r = len(r_range)
    num_beta = len(beta_range)
    num_gamma = major_models[0].num_gamma
    assert auc_results.shape == (num_major_models*num_gamma,num_r*num_beta)
    im_colorbar = ScalarMappable()
    im_colorbar.set_clim(vmin=0.5, vmax=1.0)
    im_colorbar.set_cmap('rainbow')
    x_ticks = np.linspace(0.5, num_beta-0.5, num=num_beta)
    x_tick_labels = [f'{beta:.2f}' for beta in beta_range]
    y_ticks = np.linspace(0.5, num_r-0.5, num=num_r)
    y_tick_labels = [f'{r:.2f}' for r in r_range]
    fig, axs = plt.subplots(figsize=(num_major_models*4, 5*num_gamma),
                            nrows=num_gamma, ncols=num_major_models, sharex=False, sharey=True)
    values_text_colors = ValuesTextColors(min_color='black', max_color='white',\
                                          num_gamma=num_gamma, num_major_models=num_major_models, num_r=num_r, num_beta=num_beta,\
                                          results=auc_results)
    for ind_base_model, base_model in enumerate(major_models):
        for ind_gamma in range(num_gamma):
            ind_model_gamma = ind_base_model*num_gamma+ind_gamma
            auc_model = auc_results[ind_model_gamma].reshape(num_r, num_beta)
            if num_major_models == 1:
                ax = axs[ind_gamma]
            elif num_gamma == 1:
                ax = axs[ind_base_model]
            else:
                ax = axs[ind_gamma,ind_base_model]
            im = ax.pcolor(auc_model, cmap=im_colorbar.get_cmap(), clim=im_colorbar.get_clim())
            ax.set_title(base_model.full_name_param)
            ax.set_xlabel('Beta')
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            if ind_base_model == 0:
                ax.set_ylabel(base_model.str_gamma_list[ind_gamma] + '\n\nr')
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels)
            values_text_colors.build_extreme_values(ind_model_gamma=ind_model_gamma)
            for ind_r in range(num_r):
                for ind_beta in range(num_beta):
                    color = values_text_colors.get_color(ind_r=ind_r, ind_beta=ind_beta)
                    str_value = f'{auc_model[ind_r, ind_beta]:.4f}'
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

def display_simulation_results(
        suptitle: str, simulation_results: np.ndarray, major_models: list[Base_Rejection_Method],\
        r_range: list[float], beta_range: list[float], vmin: float, vmax: float,\
        min_color: str = 'white', max_color: str = 'black'):
    num_r = len(r_range)
    num_beta = len(beta_range)
    num_major_models = len(major_models)
    num_gamma = major_models[0].num_gamma

    assert simulation_results.shape == (num_major_models*num_gamma,num_r*num_beta)
    im_colorbar = ScalarMappable()
    im_colorbar.set_clim(vmin=vmin, vmax=vmax)
    im_colorbar.set_cmap('rainbow')
    x_ticks = np.linspace(0.5, num_beta-0.5, num=num_beta)
    x_tick_labels = [f'{beta:.2f}' for beta in beta_range]
    y_ticks = np.linspace(0.5, num_r-0.5, num=num_r)
    y_tick_labels = [f'{r:.2f}' for r in r_range]
    fig, axs = plt.subplots(figsize=(num_major_models*4, 5*num_gamma),
                            nrows=num_gamma, ncols=num_major_models, sharex=False, sharey=True)
    values_text_colors = ValuesTextColors(min_color=min_color, max_color=max_color,\
                                          num_gamma=num_gamma, num_major_models=num_major_models, num_r=num_r, num_beta=num_beta,\
                                          results=simulation_results)
    for ind_base_model, base_model in enumerate(major_models):
        for ind_gamma in range(num_gamma):
            ind_model_gamma = ind_base_model*num_gamma+ind_gamma
            auc_model = simulation_results[ind_model_gamma].reshape(num_r, num_beta)
            ax = axs[ind_gamma,ind_base_model]
            im = ax.pcolor(auc_model, cmap=im_colorbar.get_cmap(), clim=im_colorbar.get_clim())
            ax.set_title(base_model.full_name_param)
            ax.set_xlabel('Beta')
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels)
            if ind_base_model == 0:
                ax.set_ylabel(base_model.str_gamma_list[ind_gamma] + '\n\nr')
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_tick_labels)
            values_text_colors.build_extreme_values(ind_model_gamma=ind_model_gamma)
            for ind_r in range(num_r):
                for ind_beta in range(num_beta):
                    color = values_text_colors.get_color(ind_r, ind_beta)
                    str_value = f'{auc_model[ind_r, ind_beta]:.2f}'
                    ax.text(ind_beta+0.5, ind_r+0.5, str_value, ha="center", va="center", color=color)
    fig.suptitle(suptitle, y=1)
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
        num_gamma_per_model = [model.num_gamma for model in hc_models]
        ind_gamma_model_base = np.cumsum([0] + num_gamma_per_model)
        self.confusion_matrices = np.empty(shape=(ind_gamma_model_base[-1],num_signal_generators,monte_carlo,4))
        with tqdm(total=monte_carlo*num_signal_generators) as progress_bar:
            for seed0 in list(range(0,monte_carlo,chunk_size)):
                seed1 = min(seed0+chunk_size,monte_carlo)
                seeds = list(range(seed0,seed1))
                random_values = Data_Generator_Base.generate_random_values(N=N, seeds=seeds)
                for ind_generator, signal_generator in enumerate(signal_generators):
                    signal_generator.generate_from_random_values(random_values=random_values)
                    for ind_base_model, hc_model in enumerate(hc_models):
                        hc_model.run_sorted_p(signal_generator.p_values)
                        for ind_gamma in range(num_gamma_per_model[ind_base_model]):
                            ind_model_gamma = ind_gamma_model_base[ind_base_model] + ind_gamma
                            confusion_per_seeds = signal_generator.calc_confusion(hc_model.num_rejected[:,ind_gamma])
                            self.confusion_matrices[ind_model_gamma, ind_generator, seed0:seed1] = confusion_per_seeds
                            pass
                    progress_bar.update(len(seeds))
        
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
    
    def apply_sqrt_mean_entire_misclassification_rate(self) -> np.ndarray:
        def mean_mdr(true_positive, false_positive, true_negative, false_negative) -> float:
            mcr_all = true_positive == 0
            return mcr_all.mean()**0.5
        return self.apply_func(mean_mdr)

    def apply_mean_f1_score(self) -> np.ndarray:
        def mean_f1_score(true_positive, false_positive, true_negative, false_negative) -> float:
            f1_score = true_positive/(true_positive+0.5*(false_positive+false_negative))
            return f1_score.mean()
        return self.apply_func(mean_f1_score)

