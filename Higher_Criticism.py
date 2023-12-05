from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from Synthetic_Data_Generators import Data_Generator_Base
from tqdm import tqdm

class Higher_Criticism:
    def __init__(self, use_import: bool, gamma: float = 1.0, correct_symetric_baseline_p_values: bool = True):
        self.use_import = use_import
        self.start_baseline_index_p_values = 0.5 if correct_symetric_baseline_p_values else 1.0
        self.gamma = gamma
        self.num_rejected = 0
        self.p_threshold = 0
        self.best_objective = 0

    @staticmethod
    def monte_carlo_statistics_HC(monte_carlo: int, data_generator: Data_Generator_Base, disable_tqdm: bool = False) -> dict:
        results = [[],[],[]]
        hc_models = [Higher_Criticism(use_import=True), Higher_Criticism(use_import=False,correct_symetric_baseline_p_values=False),
                     Higher_Criticism(use_import=False, correct_symetric_baseline_p_values=True)]
        N = data_generator.N
        for i in tqdm(range(monte_carlo), disable=disable_tqdm):
            data_generator.generate(seed=i)
            p_values = np.sort(data_generator.p_values)
            for r,hc in zip(results,hc_models):
                hc.run_sorted_p(p_values)
                r.append(hc.best_objective)
        return {'import_HC' : results[0],
                'HC_1_0':results[1],
                'HC_0_5': results[2]}
    
    def monte_carlo_statistics(self, monte_carlo: int, data_generator: Data_Generator_Base, disable_tqdm: bool = False) -> dict:
        nums_rejected = []
        best_objectives = []
        first_p_value = []
        lowest_angle = []
        N = data_generator.N
        angle_x_axis = self.p_values_index(N)
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

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        N = int(p_values_sorted.size)
        gamma = min(self.gamma, (N-1)/N)  # avoid last element devision by zero
        if self.use_import:
            mtest = MultiTest(p_values_sorted)
            _, self.p_threshold = mtest.hc(gamma=gamma)
            self.objectives = mtest._zz
            self.num_rejected = np.sum(p_values_sorted <= self.p_threshold)
        else:
            i_N = self.p_values_index(N) / N
            nominator = math.sqrt(N)*(i_N - p_values_sorted)
            denominator = np.sqrt(i_N*(1-i_N))
            # trimming zeros in denominator
            zero_denom = denominator <= 1e-8 * np.abs(nominator)
            if zero_denom.any():
                ind_zero = zero_denom.argmax()  # first zero
                nominator = nominator[:ind_zero]
                denominator = denominator[:ind_zero]            
            self.objectives = nominator / denominator
            self.objectives = self.objectives[:int(N*self.gamma)]
            sorted_objectives = self.objectives.argsort()[::-1]
            ind_lowest_objective = N - 1
            ind_best = N - 1
            best_beyond = 0
            for ind_objective in sorted_objectives:
                beyond_objective = ind_lowest_objective - ind_objective
                if beyond_objective >= best_beyond:
                    ind_best = ind_objective
                    best_beyond = beyond_objective
                ind_lowest_objective = min(ind_lowest_objective, ind_objective)
            self.num_rejected = ind_best + 1
        if self.num_rejected < 1:
            self.p_threshold = 0
            self.best_objective = self.objectives[0]
        else:
            self.p_threshold = p_values_sorted[self.num_rejected-1]
            self.best_objective = self.objectives[self.num_rejected-1]

    def p_values_index(self, N: int) -> np.ndarray:
        return np.arange(start=self.start_baseline_index_p_values,stop=N+0.1,step=1.0, dtype=np.float32)

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

