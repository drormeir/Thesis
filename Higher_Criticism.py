from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math

class Higher_Criticism:
    def __init__(self, use_import: bool, correct_symetric_baseline_p_values: bool = True, gamma: float = 0.3):
        self.use_import = use_import
        self.correct_symetric_baseline_p_values = correct_symetric_baseline_p_values
        self.gamma = gamma

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        if self.use_import:
            mtest = MultiTest(p_values_sorted)
            _, self.p_threshold = mtest.hc(gamma=self.gamma)
            self.objectives = mtest._zz
            self.num_rejected = np.sum(p_values_sorted <= self.p_threshold)
            self.best_objective = self.objectives[self.num_rejected-1]
            return
        N = int(p_values_sorted.size)
        start = 0.5 if self.correct_symetric_baseline_p_values else 1.0
        i_N = np.arange(start=start,stop=N+0.1,step=1, dtype=np.float32) / N
        nominator = math.sqrt(N)*(i_N - p_values_sorted)
        denominator = np.sqrt(i_N*(1-i_N))
        zero_denom = denominator <= 1e-8 * np.abs(nominator)
        if zero_denom.any():
            ind_zero = zero_denom.argmax()  # first zero
            nominator = nominator[:ind_zero]
            denominator = denominator[:ind_zero]
        self.objectives = nominator / denominator
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
        self.p_threshold = p_values_sorted[ind_best]
        self.num_rejected = ind_best + 1
        self.best_objective = self.objectives[self.num_rejected-1]

    def plot_objectives(self, title: str = 'Higher Criticism objectives values'):
        hc_objectives = self.objectives
        HC_best_ID = self.num_rejected
        plt.figure(figsize=(20,10))
        plt.title(label=title)
        plt.plot(np.arange(1,1+len(hc_objectives)), hc_objectives, label='HC objectives', c='blue')
        plt.vlines(x=HC_best_ID, ymin=hc_objectives.min(), ymax=hc_objectives.max(),
                   label=f'Optimal HC rejects {HC_best_ID} best objective value={self.best_objective:.2f}',
                   linestyles='dashed', colors='red')
        plt.xlabel(xlabel='Sorted Samples by P-value')
        plt.ylabel(ylabel='HC values')
        plt.xlim(xmin=0, xmax=len(hc_objectives))
        plt.legend()
        plt.show()

