import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Synthetic_Data_Generators import Multi_Class_Normal_Population as Data_Generator
from Synthetic_Data_Generators import Two_Lists_Tuple, Data_Generator_Noise, Data_Generator_Base, signal_2_noise_roc
from Higher_Criticism import Higher_Criticism
from typing import Sequence

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


def asymptotic_analysis_multi_size(\
        N_range: list[int], beta_range: list[float], r_range: list[float], hc_models: list,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = [(ind_beta,beta,ind_r,r) for ind_beta, beta in enumerate(beta_range) for ind_r, r in enumerate(r_range)]
    model_result_shape = (len(N_range),len(r_range),len(beta_range))
    collect_results = {str(model): np.empty(shape=model_result_shape, dtype=np.float32) for model in hc_models}

    for ind_N, N in enumerate(N_range):
        print(f'Working on sample size: {N}')
        signal_generators = [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for ind_beta, beta, ind_r, r in params_list]
        auc_results = AUC_best_objectives_from_random_values(\
            hc_models=hc_models, signal_generators=signal_generators,\
            monte_carlo=monte_carlo, chunk_size=chunk_size)
            
        for ind_model, hc_model in enumerate(hc_models):
            for ind_param, param in enumerate(params_list):
                ind_beta, beta, ind_r, r = param
                collect_results[str(hc_model)][ind_N, ind_r, ind_beta] = auc_results[ind_model][ind_param]

    for ind_beta, beta, ind_r, r in params_list:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        max_auc = 0
        for key in collect_results:
            auc = collect_results[key][:, ind_r, ind_beta].reshape(-1)
            # dictionary type hinting to avoid warnings
            line_params: dict[str, int | str] = {'linestyle': 'dashed' if 'power' in key else 'solid'}
            if 'HC' not in key:
                line_params['linewidth'] = 3
            ax.plot(N_range, auc, label=key, **line_params)
            max_auc = max(max_auc, auc.max())
        if max_auc >= 0.9:
            ax.set_ylim(top=1.0)
        ax.set_xlim(left=0)
        ax.set_title(f'AUC values as function of number of samples using {monte_carlo} monte carlo runs.\n' + f'r={r:.2f} beta={beta:.2f}')
        ax.legend(loc='center right', bbox_to_anchor=(1.7, 0.5))
        plt.show()


def full_analysis_single_size(\
        N: int, beta_range: list[float], r_range: list[float],\
        gamma_range: list[float], monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = Two_Lists_Tuple(r_range, beta_range)
    signal_generators = [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for r, beta in params_list]
    hc_models = []
    hc_models += [Higher_Criticism(work_mode='hc', global_max=True, gamma=gamma) for gamma in gamma_range]
    hc_models += [Higher_Criticism(work_mode='unstable', global_max=True, gamma=gamma) for gamma in gamma_range]
    hc_models += [Higher_Criticism(work_mode='hc', global_max=False, gamma=gamma) for gamma in gamma_range]
    hc_models += [Higher_Criticism(work_mode='unstable', global_max=False, gamma=gamma) for gamma in gamma_range]
    num_major_models = 4
    num_gamma = len(gamma_range)
    num_beta = len(beta_range)
    num_r = len(r_range)
    auc_results = AUC_best_objectives_from_random_values(\
        hc_models=hc_models, signal_generators=signal_generators,\
        monte_carlo=monte_carlo, chunk_size=chunk_size)

    x_ticks = np.linspace(0.5, num_beta-0.5, num=num_beta)
    x_tick_labels = [f'{beta:.2f}' for beta in beta_range]
    y_ticks = np.linspace(0.5, num_r-0.5, num=num_r)
    y_tick_labels = [f'{r:.2f}' for r in r_range]
    # https://matplotlib.org/stable/gallery/subplots_axes_and_figures/figure_title.html
    fig, axs = plt.subplots(figsize=(num_major_models*4, 5*num_gamma), nrows=num_gamma, ncols=num_major_models, sharex=True, sharey=True)
    ind_model_best_per_param = auc_results.argmax(axis=0)
    for ind_model, hc_model in enumerate(hc_models):
        auc_model = auc_results[ind_model].reshape(num_r, num_beta)
        row, col = ind_model % num_gamma, ind_model // num_gamma
        ax = axs[row,col]
        im = ax.pcolor(auc_model, cmap='rainbow', vmin=0.5, vmax=1.0)
        ax.set_title(hc_model.str_sub_model_type())
        #ax.set_xlabel("Beta")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels)
        if col == 0:
            ax.set_ylabel(hc_model.str_gamma() + '\n\nr')
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels)
        if col == num_major_models - 1:
            fig.colorbar(im, ax=ax)
        for ind_param_best in [ind_param for ind_param, ind_m in enumerate(ind_model_best_per_param) if ind_m == ind_model]:
            ind_r_best, ind_beta_best = ind_param_best // num_beta, ind_param_best % num_beta
            ax.text(ind_beta_best+0.5, ind_r_best+0.5, "X", ha="center", va="center", color="w")
    fig.suptitle(f'AUC according to selected value of objective function for signal detection in sample size: {N}', y=0.9)
    fig.supxlabel('Beta', y=0.1)
    #fig.tight_layout()
    plt.show()
