import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Synthetic_Data_Generators import Multi_Class_Normal_Population as Data_Generator
from Synthetic_Data_Generators import Two_Lists_Tuple, Data_Generator_Noise, Data_Generator_Base, signal_2_noise_roc
from Higher_Criticism import Higher_Criticism

def asymptotic_analysis_multi_size(\
        N_range: list[int], beta_range: list[float], r_range: list[float], hc_models: list,\
        monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = [(ind_beta,beta,ind_r,r) for ind_beta, beta in enumerate(beta_range) for ind_r, r in enumerate(r_range)]
    num_models = len(hc_models)
    model_result_shape = (len(N_range),len(r_range),len(beta_range))
    collect_results = {str(model): np.empty(shape=model_result_shape, dtype=np.float32) for model in hc_models}
    num_generators = len(params_list) + 1
    signal_values = np.empty(shape=(num_generators,num_models,monte_carlo))

    for ind_N, N in enumerate(N_range):
        print(f'Working on sample size: {N}')
        signal_generators = [Data_Generator_Noise(N)]
        signal_generators += [Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta)) for ind_beta, beta, ind_r, r in params_list]
        for seed0 in tqdm(list(range(0,monte_carlo,chunk_size))):
            seed1 = min(seed0+chunk_size,monte_carlo)
            seeds = list(range(seed0,seed1))
            random_values = Data_Generator_Base.generate_random_values(N=N, seeds=seeds)
            for ind_generator, signal_generator in enumerate(signal_generators):
                signal_values[ind_generator,:, seed0:seed1] =\
                    Higher_Criticism.best_objectives_from_random_values(\
                        hc_models=hc_models, data_generator=signal_generator, random_values=random_values)
            
        for ind_model, hc_model in enumerate(hc_models):
            noise_values = signal_values[0][ind_model]
            for ind_param, param in enumerate(params_list):
                ind_beta, beta, ind_r, r = param
                auc, _, _ = signal_2_noise_roc(signal_values=signal_values[ind_param+1][ind_model], noise_values=noise_values)
                collect_results[str(hc_model)][ind_N, ind_r, ind_beta] = auc

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


def asymptotic_analysis_single_size(N: int, beta_range: list[float], r_range: list[float],\
        hc_models: list[tuple], monte_carlo: int = 10000, chunk_size: int = 100) -> None:
    params_list = Two_Lists_Tuple(list(enumerate(beta_range)), list(enumerate(r_range)))
    params_list = [((-1,0),(-1,0))] + params_list
    num_models = len(hc_models)
    model_result_shape = (len(N_range),len(r_range),len(beta_range))
    collect_results = {str(model): np.empty(shape=model_result_shape, dtype=np.float32) for model in hc_models}
    noise_values = np.empty(shape=0)  # pivot value to avoid missing definition warnings
    
    for hc_model_type, hc_model_params, model_param_key, model_param_values in hc_models:
        print(f'Working on sample size: {N}')
        for (ind_beta, beta), (ind_r, r) in tqdm(params_list):
            if ind_beta < 0:
                noise_generator = Data_Generator_Base(N)
                noise_values = Higher_Criticism.monte_carlo_best_objectives(\
                    hc_models=hc_models, data_generator=noise_generator, monte_carlo=monte_carlo, disable_tqdm=True,\
                    chunk_size=chunk_size)
                assert noise_values.shape == (num_models, monte_carlo)
                continue
            signal_generator = Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta))
            signal_values = Higher_Criticism.monte_carlo_best_objectives(
                hc_models=hc_models, data_generator=signal_generator, monte_carlo=monte_carlo, disable_tqdm=True,\
                chunk_size=chunk_size)
            
            for ind_model, hc_model in enumerate(hc_models):
                auc, _, _ = signal_2_noise_roc(signal_values=signal_values[ind_model], noise_values=noise_values[ind_model])
                collect_results[str(hc_model)][ind_N, ind_r, ind_beta] = auc

    for (ind_beta, beta), (ind_r, r) in params_list:
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
