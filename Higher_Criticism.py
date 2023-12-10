from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from Synthetic_Data_Generators import Data_Generator_Base, signal_2_noise_roc

class Higher_Criticism:
    def __init__(self, work_mode: str = 'hc', gamma: float = 0.3, global_max: bool = True):
        self.work_mode = work_mode.lower()
        assert self.work_mode in ['hc', 'import', 'bonferroni', 'bh']
        self.gamma = gamma
        self.num_rejected = 0
        self.p_threshold = 0
        self.best_objective = 0
        self.global_max = global_max

    @staticmethod
    def monte_carlo_statistics_HC(hc_models: list, noise_values: np.ndarray, data_generator: Data_Generator_Base, disable_tqdm: bool) -> dict:
        monte_carlo = noise_values.shape[1]
        signal_values = Higher_Criticism.monte_carlo_best_objectives(hc_models=hc_models, data_generator=data_generator,\
                                                                     monte_carlo=monte_carlo, disable_tqdm=disable_tqdm)
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
        ret = 'HC' if 'hc' == self.work_mode else 'import_HC'
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
            data_generator.generate(seed=i)
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
    def monte_carlo_best_objectives(hc_models: list, data_generator: Data_Generator_Base, monte_carlo: int, disable_tqdm: bool) -> np.ndarray:
        ret = np.empty(shape=(len(hc_models),monte_carlo), dtype=np.float32)
        for j in tqdm(range(monte_carlo), disable=disable_tqdm):
            data_generator.generate(seed=j)
            p_values = np.sort(data_generator.p_values)
            for i, hc_model in enumerate(hc_models):
                hc_model.run_sorted_p(p_values)
                ret[i,j] = hc_model.best_objective
        return ret

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        N = int(p_values_sorted.size)
        gamma = N**(-self.gamma-1.0) if self.gamma <= 0 else self.gamma
        if self.work_mode == 'import':
            mtest = MultiTest(p_values_sorted)
            _, self.p_threshold = mtest.hc(gamma=gamma)  # avoid last element division by zero
            self.objectives = mtest._zz
            self.num_rejected = np.sum(p_values_sorted <= self.p_threshold)
        elif self.work_mode == 'hc':
            i_N = np.arange(1,N+1) / N
            nominator = math.sqrt(N)*(i_N - p_values_sorted)
            denominator = np.sqrt(i_N*(1-i_N))
            max_len = int(N*gamma)
            # trimming zeros in denominator
            zero_denom = denominator <= 1e-8 * np.abs(nominator)
            if zero_denom.any():
                max_len = min(zero_denom.argmax(), max_len)  # first zero
            nominator = nominator[:max_len]
            denominator = denominator[:max_len]            
            self.objectives = nominator / denominator
            if self.global_max:
                ind_best = np.argmax(self.objectives)
            else:
                ind_best = N - 1
                ind_lowest_objective = N - 1
                best_beyond = 0
                for ind_objective in self.objectives.argsort()[::-1]:
                    beyond_objective = ind_lowest_objective - ind_objective
                    if beyond_objective >= best_beyond:
                        ind_best = ind_objective
                        best_beyond = beyond_objective
                    ind_lowest_objective = min(ind_lowest_objective, ind_objective)
            self.num_rejected = ind_best + 1
        elif self.work_mode == 'bonferroni':
            self.objectives = 1 - p_values_sorted
            self.num_rejected = 1
        else:  # Benjamini Hochberg
            self.objectives = - p_values_sorted / np.arange(1,N+1)
            self.num_rejected = np.argmax(self.objectives) + 1
        if self.num_rejected < 1:
            self.p_threshold = 0
            self.best_objective = self.objectives[0]
        else:
            self.p_threshold = p_values_sorted[self.num_rejected-1]
            self.best_objective = self.objectives[self.num_rejected-1]


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

