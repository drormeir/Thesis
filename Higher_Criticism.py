from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import kstest
from Synthetic_Data_Generators import Data_Generator_Base

class Higher_Criticism:
    def __init__(self, work_mode: str = 'hc', alpha: float = -1, gamma: float = 0.3, global_max: bool = True):
        self.work_mode = work_mode.lower()
        assert self.work_mode in ['hc', 'unstable', 'import', 'bonferroni', 'bh', 'ks1', 'ks2']
        self.gamma = gamma
        self.num_rejected = 0
        self.p_threshold = 0
        self.best_objective = 0
        self.global_max = global_max
        self.i_N = np.empty(shape=0)
        self.denominator = np.empty(shape=0)
        self.alpha = alpha

    def __str__(self) -> str:
        if self.work_mode == 'bonferroni':
            return 'Bonferroni' + self.str_alpha(space=' ')
        if self.work_mode == 'bh':
            return 'Benjamini Hochberg' + self.str_alpha(space=' ')
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
        ret += '_' + self.str_gamma()
        ret += f'_{"global" if self.global_max else "local"}_max'
        return ret
    
    def str_alpha(self, space: str ='') -> str:
        return '' if self.alpha < 0 else space + f'alpha={self.alpha:.2}'
    
    def str_gamma(self) -> str:
        return f'gammapower={-self.gamma:.2f}' if self.gamma <= 0 else f'gamma={self.gamma:.2f}'

    def str_sub_model_type(self) -> str:
        stable = 'Unstable' if self.work_mode == 'unstable' else 'Stable'
        max_type = f'{"Global" if self.global_max else "Local"}_max'
        return stable + '_' + max_type
    
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
            for i, hc_model in enumerate(hc_models):
                hc_model.run_sorted_p(data_generator.p_values)
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
            if self.i_N.size != N:
                self.i_N = np.arange(1,N+1).reshape(1,-1) / N
                self.denominator = np.sqrt(self.i_N*(1-self.i_N))
            nominator = math.sqrt(N)*(self.i_N - p_values_sorted)
            if self.work_mode == 'unstable':
                self.denominator = np.sqrt(p_values_sorted*(1-p_values_sorted))
            with np.errstate(divide='ignore', invalid='ignore'):
                self.objectives = nominator / self.denominator
            isfinite = np.isfinite(self.objectives)
            # trimming zeros in denominator
            max_N = int(N*gamma)
            max_len = np.full(shape=num_p_vectors,fill_value=max_N,dtype=np.int32)
            for ind_p_vector, ind_zero_denom in enumerate(isfinite.argmin(axis=1)):
                if isfinite[ind_p_vector,ind_zero_denom]:
                    continue
                len_row = min(max_N, ind_zero_denom)
                max_len[ind_p_vector] = len_row
                self.objectives[ind_p_vector,len_row:] = 0
            if self.global_max:
                ind_best = np.minimum(np.argmax(self.objectives, axis=1), max_len)
            else:
                ind_best = max_len
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
            if self.alpha < 0:
                self.num_rejected = np.ones(shape=num_p_vectors, dtype=np.int32)
            else:
                self.num_rejected = np.sum(p_values_sorted <= self.alpha, axis=1)
        elif self.work_mode == 'bh':  # Benjamini Hochberg
            self.objectives = - p_values_sorted / np.arange(1,N+1)
            if self.alpha < 0:
                self.num_rejected = np.argmax(self.objectives,axis=1) + 1
            else:
                below_bh_line = p_values_sorted <= np.arange(1,N+1)*self.alpha
                ind_last_rejected = N - 1 - below_bh_line[:,::-1].argmax(axis=1)
                for ind_row, ind_last_reject in enumerate(ind_last_rejected):
                    if below_bh_line[ind_row, ind_last_reject]:
                        self.num_rejected[ind_row] = ind_last_reject + 1
                    else:
                        self.num_rejected[ind_row] = 0

        elif 'ks' in self.work_mode:  # Kolmogorov-Smirnov test
            objective = np.asarray([p_values_sorted - np.arange(N)/N, p_values_sorted - np.arange(1,N+1)/N])
            if '2' in self.work_mode:
                objective = np.abs(objective)
            self.objectives = objective.max(axis=0)
            self.num_rejected = np.argmax(self.objectives, axis=1) + 1
        for ind_seed, num_rejected in enumerate(self.num_rejected):
            if num_rejected < 1:
                self.p_threshold[ind_seed] = 0
                self.best_objective[ind_seed] = self.objectives[ind_seed, 0]
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

