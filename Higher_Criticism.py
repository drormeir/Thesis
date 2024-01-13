from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import kstest
from Synthetic_Data_Generators import Data_Generator_Base

class Base_Rejection_Method:
    def __init__(self, name: str, str_param: str, plot_color: str) -> None:
        self.name = name
        self.str_param = str_param
        self.plot_color = plot_color
        self.num_rejected = np.empty(shape=0, dtype=np.int32)
        self.best_objective = np.empty(shape=0, dtype=float)
        self.p_threshold = np.empty(shape=0, dtype=float)
        self.i_N = np.empty(shape=0, dtype=float)

    def full_name(self):
        return self.name + ' ' + self.str_param if self.str_param else self.name

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        num_p_vectors, N = p_values_sorted.shape
        self.p_threshold = np.zeros(shape=num_p_vectors, dtype=float)
        self.best_objective = np.zeros_like(self.p_threshold)
        self.num_rejected = np.zeros(shape=num_p_vectors, dtype=np.int32)
        self.objectives = np.empty_like(p_values_sorted)
        if self.i_N.size != N:
            self.i_N = np.arange(1,N+1).reshape(1,-1) / N

        self.work_sorted_p(p_values_sorted)

        for ind_seed, num_rejected in enumerate(self.num_rejected):
            if num_rejected < 1:
                self.p_threshold[ind_seed] = 0
                self.best_objective[ind_seed] = self.objectives[ind_seed, 0]
            else:
                self.p_threshold[ind_seed] = p_values_sorted[ind_seed,num_rejected-1]
                self.best_objective[ind_seed] = self.objectives[ind_seed,num_rejected-1]

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        assert 0

class Bonferroni(Base_Rejection_Method):
    def __init__(self, alpha: float = 1, plot_color: str = ''):
        super().__init__(name='Bonferroni', str_param='' if alpha <= 0 else f'alpha={alpha:.2f}', plot_color=plot_color)
        self.alpha = alpha

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        _, N = p_values_sorted.shape
        self.objectives = 1 - p_values_sorted
        if self.alpha <= 0:
            self.num_rejected.fill(1)
        else:
            self.num_rejected = np.sum(p_values_sorted <= self.alpha/N, axis=1)


class Benjamini_Hochberg(Base_Rejection_Method):
    def __init__(self, alpha: float = 1, plot_color: str = ''):
        super().__init__(name='Benjamini-Hochberg', str_param='' if alpha <= 0 else f'alpha={alpha:.2f}', plot_color=plot_color)
        self.alpha = alpha

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        _, N = p_values_sorted.shape
        self.objectives = - p_values_sorted / self.i_N
        if self.alpha <= 0:
            self.num_rejected = np.argmax(self.objectives,axis=1) + 1
            return
        below_bh_line = p_values_sorted <= self.i_N*self.alpha
        ind_last_rejected = N - 1 - below_bh_line[:,::-1].argmax(axis=1)
        for ind_row, ind_last_reject in enumerate(ind_last_rejected):
            if below_bh_line[ind_row, ind_last_reject]:
                self.num_rejected[ind_row] = ind_last_reject + 1
            else:
                self.num_rejected[ind_row] = 0


class Kolmogorov_Smirnov(Base_Rejection_Method):  # Kolmogorov-Smirnov test
    def __init__(self, mode: int = 1, plot_color: str = '') -> None:
        super().__init__(name='Kolmogorov-Smirnov ' + ('one' if mode == 1 else 'two'), str_param='', plot_color=plot_color)
        self.mode = mode
        self.i_N0 = np.empty(shape=0)

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        _, N = p_values_sorted.shape
        if self.i_N0.size != N:
            self.i_N0 = np.arange(N).reshape(1,-1)/N
        # two rows of objective function
        objective = np.asarray([p_values_sorted - self.i_N0, p_values_sorted - self.i_N])
        if self.mode == 2:
            objective = np.abs(objective)
        self.objectives = objective.max(axis=0)
        self.num_rejected = np.argmax(self.objectives, axis=1) + 1


class Import_HC(Base_Rejection_Method):
    def __init__(self, stable=True, gamma: float = 0.3, plot_color: str = '') -> None:
        super().__init__(name='Import HC ' + ('Stable' if stable else 'Unstable'), str_param=f'gamma={gamma:.2f}', plot_color=plot_color)
        self.gamma = gamma
        self.stable = stable

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        _, N = p_values_sorted.shape
        for ind_p_vector, p_vector in enumerate(p_values_sorted):
            mtest = MultiTest(p_vector, stbl=self.stable)
            _, self.p_threshold[ind_p_vector] = mtest.hc(gamma=self.gamma)
            self.objectives[ind_p_vector] = mtest._zz
        self.num_rejected = np.sum(p_values_sorted <= self.p_threshold, axis=1)


class Higher_Criticism(Base_Rejection_Method):
    def __init__(self, stable=True, global_max: bool = True, gamma: float = 0.3, plot_color: str = ''):
        super().__init__(name=f'Higher Criticism{"" if stable else " Unstable"}{"" if global_max else " Local Max"}',\
                         str_param=f'{"gamma" if gamma > 0 else "gammaPower"}={abs(gamma):.2f}',\
                         plot_color=plot_color)
        self.gamma = gamma
        self.stable = stable
        self.global_max = global_max
        self.denominator = np.empty(shape=0)
    
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

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        num_p_vectors, N = p_values_sorted.shape
        nominator = math.sqrt(N)*(self.i_N - p_values_sorted)
        if self.stable:
            if self.denominator.shape != self.i_N.shape:
                self.denominator = np.sqrt(self.i_N*(1-self.i_N))
        else:
            self.denominator = np.sqrt(p_values_sorted*(1-p_values_sorted))
        with np.errstate(divide='ignore', invalid='ignore'):
            self.objectives = nominator / self.denominator
        isfinite = np.isfinite(self.objectives)
        # trimming zeros in denominator
        if self.gamma > 0:
            gamma = self.gamma
        else:
            gamma = N**(-self.gamma-1.0)
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

