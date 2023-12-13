import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from Synthetic_Data_Generators import Multi_Class_Normal_Population as Data_Generator
from Synthetic_Data_Generators import Two_Lists_Tuple, Data_Generator_Base
from Higher_Criticism import Higher_Criticism

def asymptotic_analysis(N_range: list[int], beta_range: list[float], r_range: list[float], hc_models: list, monte_carlo: int = 1000) -> None:
    params_list = Two_Lists_Tuple(list(enumerate(beta_range)), list(enumerate(r_range)))
    collect_results = {}
    many_params = len(params_list) > 1
    for ind_N, N in enumerate(N_range):
        print(f'Working on sample size: {N}')
        noise_generator = Data_Generator_Base(N)
        noise_values = Higher_Criticism.monte_carlo_best_objectives(hc_models=hc_models, data_generator=noise_generator, monte_carlo=monte_carlo, disable_tqdm=many_params)
        for (ind_beta, beta), (ind_r, r) in tqdm(params_list, disable= not many_params):
            signal_generator = Data_Generator(**Data_Generator.params_from_N_r_beta(N=N, r=r, beta=beta))
            hc_monte_carlo = Higher_Criticism.monte_carlo_statistics_HC(hc_models=hc_models, noise_values=noise_values, data_generator=signal_generator, disable_tqdm=many_params)
            for key, auc in hc_monte_carlo.items():
                if key not in collect_results:
                    collect_results[key] = np.empty(shape=(len(N_range),len(r_range),len(beta_range)), dtype=np.float32)
                collect_results[key][ind_N, ind_r, ind_beta] = auc

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
