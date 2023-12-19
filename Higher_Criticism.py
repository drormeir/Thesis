from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import kstest
from Synthetic_Data_Generators import Data_Generator_Base, signal_2_noise_roc

class Higher_Criticism:
    def __init__(self, work_mode: str = 'hc', gamma: float = 0.3, global_max: bool = True):
        self.work_mode = work_mode.lower()
        assert self.work_mode in ['hc', 'unstable', 'import', 'bonferroni', 'bh', 'ks1', 'ks2']
        self.gamma = gamma
        self.num_rejected = 0
        self.p_threshold = 0
        self.best_objective = 0
        self.global_max = global_max

    @staticmethod
    def monte_carlo_statistics_HC(hc_models: list, noise_values: np.ndarray, data_generator: Data_Generator_Base,
                                  disable_tqdm: bool, chunk_size: int) -> dict:
        monte_carlo = noise_values.shape[1]
        signal_values = Higher_Criticism.monte_carlo_best_objectives(hc_models=hc_models, data_generator=data_generator,\
                                                                     monte_carlo=monte_carlo, disable_tqdm=disable_tqdm,\
                                                                     chunk_size=chunk_size)
        result = {}
        for ind_model, hc_model in enumerate(hc_models):
            auc, _, _ = signal_2_noise_roc(signal_values=signal_values[ind_model], noise_values=noise_values[ind_model])
            result[str(hc_model)] = auc
        return result
    
    def __str__(self) -> str:
        if self.work_mode == 'bonferroni':
            return 'Bonferroni'
        if self.work_mode == 'bh':
            return 'Benjamini Hochberg'
        if 'ks' in self.work_mode:
            ret = 'Kolmogorov-Smirnov '
            if '1' in self.work_mode:
                ret += 'one'
            else:
                ret += 'two'
            return ret
        if 'hc' == self.work_mode:
            ret = 'HC'
        elif 'unstable' == self.work_mode:
            ret = 'HC_unstable'
        else:
            ret = 'import_HC'
        ret += '_'
        if self.gamma <= 0:
            ret += f'gammapower_{-self.gamma:.2f}'
        else:
            ret += f'gamma_{self.gamma:.2f}'
        ret += f'_{"global" if self.global_max else "local"}_max'
        return ret
    
    def monte_carlo_statistics(self, monte_carlo: int, data_generator: Data_Generator_Base, disable_tqdm: bool = False) -> dict:
        nums_rejected = []
        best_objectives = []
        first_p_value = []
        lowest_angle = []
        N = data_generator.N
        angle_x_axis = np.arange(1,N+1)
        for i in tqdm(range(monte_carlo), disable=disable_tqdm):
            data_generator.generate(seeds=[i])
            p_values = np.sort(data_generator.p_values)
            self.run_sorted_p(p_values)
            # collecting data from objective function
            best_objectives.append(self.best_objective)
            nums_rejected.append(self.num_rejected)
            first_p_value.append(p_values[0])
            lowest_angle.append((p_values/angle_x_axis).min())
        return {'nums_rejected' : nums_rejected,
                'best_objectives':best_objectives,
                'first_p_value': first_p_value,
                'lowest_angle':lowest_angle}

    @staticmethod
    def monte_carlo_best_objectives(hc_models: list, data_generator: Data_Generator_Base, monte_carlo: int,
                                    disable_tqdm: bool, chunk_size: int) -> np.ndarray:
        ret = np.empty(shape=(len(hc_models),monte_carlo), dtype=np.float32)
        for j in tqdm(range(0,monte_carlo,chunk_size), disable=disable_tqdm):
            seeds = list(range(j,min(j+chunk_size,monte_carlo)))
            data_generator.generate(seeds=seeds)
            p_values = np.sort(data_generator.p_values, axis=1)
            for i, hc_model in enumerate(hc_models):
                hc_model.run_sorted_p(p_values)
                ret[i, np.asarray(seeds, dtype=np.int32)] = hc_model.best_objective
        return ret

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        num_p_vectors, N = p_values_sorted.shape
        gamma = N**(-self.gamma-1.0) if self.gamma <= 0 else self.gamma
        self.p_threshold = np.zeros(shape=num_p_vectors, dtype=float)
        self.best_objective = np.zeros_like(self.p_threshold)
        self.num_rejected = np.zeros(shape=num_p_vectors, dtype=np.int32)
        self.objectives = np.empty_like(p_values_sorted)
        if self.work_mode == 'import':
            gamma = min(gamma,(N-1)/N)  # avoid last element division by zero
            for ind_p_vector, p_vector in enumerate(p_values_sorted):
                mtest = MultiTest(p_vector, stbl=True)
                _, self.p_threshold[ind_p_vector] = mtest.hc(gamma=gamma)
                self.objectives[ind_p_vector] = mtest._zz
            self.num_rejected = np.sum(p_values_sorted <= self.p_threshold, axis=1)
        elif self.work_mode in ['hc', 'unstable']:
            i_N = np.arange(1,N+1) / N
            nominator = math.sqrt(N)*(i_N - p_values_sorted)
            if self.work_mode == 'hc':
                denominator = np.sqrt(i_N*(1-i_N))
            else:
                denominator = np.sqrt(p_values_sorted*(1-p_values_sorted))
            max_len = int(N*gamma)
            # trimming zeros in denominator
            zero_denom = denominator <= 1e-8 * np.abs(nominator)
            if zero_denom.any():
                max_len = min(zero_denom.argmax(), max_len)  # first zero
            nominator = nominator[:,:max_len]
            denominator = denominator[:max_len] if denominator.ndim == 1 else denominator[:,:max_len]            
            self.objectives = nominator / denominator
            if self.global_max:
                ind_best = np.argmax(self.objectives, axis=1)
            else:
                ind_best = np.full(fill_value=N - 1, shape=num_p_vectors, dtype=np.int32)
                ind_lowest_objective = np.copy(ind_best)
                best_beyond = np.zeros_like(ind_best)
                for ind_objective in self.objectives.argsort(axis=1).T[::-1]:
                    beyond_objective = ind_lowest_objective - ind_objective
                    better = beyond_objective >= best_beyond
                    ind_best[better] = ind_objective[better]
                    best_beyond[better] = beyond_objective[better]
                    ind_lowest_objective = np.minimum(ind_lowest_objective, ind_objective)
            self.num_rejected = ind_best + 1
        elif self.work_mode == 'bonferroni':
            self.objectives = 1 - p_values_sorted
            self.num_rejected = np.ones(shape=num_p_vectors, dtype=np.int32)
        elif self.work_mode == 'bh':  # Benjamini Hochberg
            self.objectives = - p_values_sorted / np.arange(1,N+1)
            self.num_rejected = np.argmax(self.objectives,axis=1) + 1
        elif 'ks' in self.work_mode:  # Kolmogorov-Smirnov test
            objective = np.asarray([p_values_sorted - np.arange(N)/N, p_values_sorted - np.arange(1,N+1)/N])
            if '2' in self.work_mode:
                objective = np.abs(objective)
            self.objectives = objective.max(axis=0)
            self.num_rejected = np.argmax(self.objectives, axis=1) + 1
        for ind_seed, num_rejected in enumerate(self.num_rejected):
            if num_rejected < 1:
                self.p_threshold[ind_seed] = 0
                self.best_objective[ind_seed] = self.objectives[0]
            else:
                self.p_threshold[ind_seed] = p_values_sorted[ind_seed,num_rejected-1]
                self.best_objective[ind_seed] = self.objectives[ind_seed,num_rejected-1]


    def plot_objectives(self, title: str = 'Higher Criticism objectives values'):
        hc_objectives = self.objectives
        HC_best_ID = self.num_rejected
        plt.figure(figsize=(20,10))
        plt.title(label=title)
        plt.plot(np.arange(1,1+len(hc_objectives)), hc_objectives, label='HC objectives', c='blue')
        plt.vlines(x=HC_best_ID, ymin=hc_objectives.min(), ymax=hc_objectives.max(),
                   label=f'Optimal HC rejects {self.num_rejected} best objective value={self.best_objective:.2f}',
                   linestyles='dashed', colors='red')
        plt.xlabel(xlabel='Sorted Samples by P-value')
        plt.ylabel(ylabel='HC values')
        plt.xlim(xmin=0, xmax=len(hc_objectives))
        plt.legend()
        plt.show()

