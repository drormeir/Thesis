from multitest import MultiTest
import matplotlib.pyplot as plt
import numpy as np
import math
from tqdm import tqdm
from scipy.stats import beta
from Synthetic_Data_Generators import Data_Generator_Base

class Base_Rejection_Method:
    Selection_Method_ArgMin = 'argmin'
    Selection_Method_ArgMax = 'argmax'
    Selection_Method_LastNegative = 'lastnegative'
    Selection_Method_LocalMin = 'localmin'
    Selection_Methods = [Selection_Method_ArgMin, Selection_Method_LastNegative, Selection_Method_LocalMin, Selection_Method_ArgMax]

    def __init__(self, name: str, gamma: float | list[float|str] | np.ndarray,\
                 selection_method: str,\
                 save_objectives: bool,\
                 str_param: str = '') -> None:
        if isinstance(gamma,float):
            self.gamma = [gamma]
        elif isinstance(gamma,np.ndarray):
            self.gamma = gamma.tolist()
        elif isinstance(gamma,list):
            self.gamma = gamma
        else:
            self.gamma = []
        for g in self.gamma:
            if isinstance(g,float):
                assert -1.0 < g <= 1.0
            else:
                assert isinstance(g,str)
                assert g == 'logsqrt'

        assert selection_method in Base_Rejection_Method.Selection_Methods
        self.name = name
        self.str_param = str_param
        self.selection_method = selection_method
        self.num_gamma = len(self.gamma)
        self.N_gamma = np.zeros(shape=(self.num_gamma,), dtype=np.int32)
        self.effective_gamma = np.zeros(shape=(self.num_gamma,), dtype=np.float32)
        self.str_gamma_list = []
        for g in self.gamma:
            if isinstance(g,str):
                self.str_gamma_list.append(g)
            elif g > 0:
                self.str_gamma_list.append(f'gamma={g:.2f}')
            else:
                self.str_gamma_list.append(f'gammaPower={-g:.2f}')
        self.num_rejected = np.empty(shape=0, dtype=np.int32)
        self.best_objective = np.empty(shape=0, dtype=float)
        self.p_threshold = np.empty(shape=0, dtype=float)
        self.i_N = np.empty(shape=0, dtype=float)
        self.full_name_param = name # + f'({self.selection_method})'
        if self.str_param:
            self.full_name_param += ' ' + self.str_param
        self.full_name = self.full_name_param
        self.is_method_min = self.selection_method != Base_Rejection_Method.Selection_Method_ArgMax
        self.is_method_max = ~self.is_method_min
        self.objectives = np.empty(shape=0) if save_objectives else None

    def get_full_name_with_gamma(self, ind_gamma: int) -> str:
        return self.full_name_param + self.str_gamma_list[ind_gamma]
    
    def calc_i_N(self, N: int):
        if self.i_N.size == N:
            return
        self.i_N = np.arange(1,N+1).reshape(1,-1) / N
        N_gamma = np.empty_like(self.effective_gamma)
        for ind_gamma, gamma in enumerate(self.gamma):
            if isinstance(gamma,str):
                N_gamma[ind_gamma] = min(math.log(N)*math.sqrt(N),N)
            elif gamma <= 0:
                N_gamma[ind_gamma] = N**-gamma
            else:
                N_gamma[ind_gamma] = N*gamma
        self.N_gamma[:] = np.maximum(1,N_gamma+0.5).astype(self.N_gamma.dtype)
        self.effective_gamma = np.minimum((self.N_gamma+1e-3)/N,1.0)
        self.gamma_ip = self.N_gamma - 1
        self.max_N_gamma = self.N_gamma.max()
        self.max_effective_gamma = self.effective_gamma.max()
        self.argsort_N_gamma = self.N_gamma.argsort()

    def run_unsorted_p(self, p_values_unsorted: np.ndarray) -> None:
        self.run_sorted_p(np.sort(p_values_unsorted))

    def run_sorted_p(self, p_values_sorted: np.ndarray) -> None:
        num_p_vectors, N = p_values_sorted.shape
        self.p_threshold = np.empty(shape=(num_p_vectors,self.num_gamma), dtype=float)
        self.best_objective = np.empty_like(self.p_threshold)
        self.calc_i_N(N)
        objectives = self.work_sorted_p(p_values_sorted)
        assert objectives.shape[0] == p_values_sorted.shape[0]
        assert objectives.shape[1] <= p_values_sorted.shape[1]
        if self.objectives is not None:
            self.objectives = objectives
        max_N = self.max_N_gamma
        objectives = objectives[:,:max_N]
        if self.selection_method == Base_Rejection_Method.Selection_Method_LastNegative:
            selected_indexes = np.empty_like(self.best_objective, dtype=np.int32)
            below_threshold_reversed = (objectives <= 0)[:,::-1]
            first_ind_gamma = self.argsort_N_gamma[0]
            first_N_gamma = self.N_gamma[first_ind_gamma]
            selected_indexes[:,first_ind_gamma] = first_N_gamma-1-below_threshold_reversed[:,-first_N_gamma:].argmax(axis=1)
            ind_p_negative_values = [ind_p_vector for ind_p_vector, curr_ip in enumerate(selected_indexes[:,first_ind_gamma]) if objectives[ind_p_vector,curr_ip] > 0]
            selected_indexes[ind_p_negative_values,first_ind_gamma] = 0
            for ind_ind_gamma, curr_ind_gamma in enumerate(self.argsort_N_gamma[1:], start=1):
                prev_ind_gamma = self.argsort_N_gamma[ind_ind_gamma-1]
                prev_N_gamma = self.N_gamma[prev_ind_gamma]
                curr_N_gamma = self.N_gamma[curr_ind_gamma]
                if curr_N_gamma == prev_N_gamma:
                    selected_indexes[:,curr_ind_gamma] = selected_indexes[:,prev_ind_gamma]
                    continue
                selected_indexes[:,curr_ind_gamma] = curr_N_gamma-1-below_threshold_reversed[:,-curr_N_gamma:-prev_N_gamma].argmax(axis=1)
                ind_p_negative_values = [ind_p_vector for ind_p_vector, curr_ip in enumerate(selected_indexes[:,curr_ind_gamma]) if objectives[ind_p_vector,curr_ip] > 0]
                selected_indexes[ind_p_negative_values, curr_ind_gamma] = selected_indexes[ind_p_negative_values, prev_ind_gamma]
        elif self.selection_method == Base_Rejection_Method.Selection_Method_ArgMin:
            selected_indexes = np.empty_like(self.best_objective, dtype=np.int32)
            first_ind_gamma = self.argsort_N_gamma[0]
            first_N_gamma = self.N_gamma[first_ind_gamma]
            selected_indexes[:,first_ind_gamma] = objectives[:,:first_N_gamma].argmin(axis=1)
            for ind_ind_gamma, curr_ind_gamma in enumerate(self.argsort_N_gamma[1:], start=1):
                prev_ind_gamma = self.argsort_N_gamma[ind_ind_gamma-1]
                prev_N_gamma = self.N_gamma[prev_ind_gamma]
                curr_N_gamma = self.N_gamma[curr_ind_gamma]
                if curr_N_gamma == prev_N_gamma:
                    selected_indexes[:,curr_ind_gamma] = selected_indexes[:,prev_ind_gamma]
                    continue
                selected_indexes[:,curr_ind_gamma] = objectives[:,prev_N_gamma:curr_N_gamma].argmin(axis=1) + prev_N_gamma
                for ind_p_vector in range(num_p_vectors):
                    curr_ip = selected_indexes[ind_p_vector,curr_ind_gamma]
                    prev_ip = selected_indexes[ind_p_vector,prev_ind_gamma]
                    if objectives[ind_p_vector,curr_ip] >= objectives[ind_p_vector,prev_ip]:
                        # new argmin does not improve --> keep old argmin
                        selected_indexes[ind_p_vector,curr_ind_gamma] = prev_ip
        elif self.selection_method == Base_Rejection_Method.Selection_Method_ArgMax:
            selected_indexes = np.empty_like(self.best_objective, dtype=np.int32)
            first_ind_gamma = self.argsort_N_gamma[0]
            first_N_gamma = self.N_gamma[first_ind_gamma]
            selected_indexes[:,first_ind_gamma] = objectives[:,:first_N_gamma].argmax(axis=1)
            for ind_ind_gamma, curr_ind_gamma in enumerate(self.argsort_N_gamma[1:], start=1):
                prev_ind_gamma = self.argsort_N_gamma[ind_ind_gamma-1]
                prev_N_gamma = self.N_gamma[prev_ind_gamma]
                curr_N_gamma = self.N_gamma[curr_ind_gamma]
                if curr_N_gamma == prev_N_gamma:
                    selected_indexes[:,curr_ind_gamma] = selected_indexes[:,prev_ind_gamma]
                    continue
                selected_indexes[:,curr_ind_gamma] = objectives[:,prev_N_gamma:curr_N_gamma].argmax(axis=1) + prev_N_gamma
                for ind_p_vector in range(num_p_vectors):
                    curr_ip = selected_indexes[ind_p_vector,curr_ind_gamma]
                    prev_ip = selected_indexes[ind_p_vector,prev_ind_gamma]
                    if objectives[ind_p_vector,curr_ip] <= objectives[ind_p_vector,prev_ip]:
                        # new argmax does not improve --> keep old argmax
                        selected_indexes[ind_p_vector,curr_ind_gamma] = prev_ip
        else:  # local min
            selected_indexes = np.repeat(self.N_gamma.reshape(1,-1), repeats=num_p_vectors, axis=0)
            ind_lowest_objective = np.copy(selected_indexes)
            best_beyond = np.zeros_like(selected_indexes)
            for ind_objective in objectives.argsort(axis=1).T:
                ind_objective = np.repeat(ind_objective.reshape(-1,1), repeats=self.num_gamma, axis=1)
                beyond_objective = ind_lowest_objective - ind_objective
                better_mask = beyond_objective >= best_beyond
                selected_indexes[better_mask] = ind_objective[better_mask]
                best_beyond[better_mask] = beyond_objective[better_mask]
                ind_lowest_objective = np.minimum(ind_lowest_objective, ind_objective)
        self.best_objective = np.array([p_vector_objectives[p_vector_select] for p_vector_select, p_vector_objectives in zip(selected_indexes,objectives)])
        self.p_threshold = np.array([p_vector[p_vector_select] for p_vector_select, p_vector in zip(selected_indexes,p_values_sorted)])
        self.num_rejected = selected_indexes + 1
        '''
        invalid_selections = self.best_objective > 0
        self.p_threshold[invalid_selections] = 0
        self.num_rejected[invalid_selections] = 0
        '''

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        assert 0
        return np.empty_like(p_values_sorted)
    
class Bonferroni(Base_Rejection_Method):
    def __init__(self,\
                 alpha: float = 1,\
                 gamma: list[float|str] | np.ndarray = [1.],\
                 selection_method: str = Base_Rejection_Method.Selection_Method_LastNegative,\
                 save_objectives: bool = False):
        super().__init__(name='Bonferroni',\
                         str_param=f'alpha={alpha:.2f}',\
                         gamma=gamma,\
                         selection_method=selection_method,\
                         save_objectives=save_objectives)
        self.alpha = alpha

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        N = self.i_N.size
        return p_values_sorted - self.alpha/N

class Benjamini_Hochberg(Base_Rejection_Method):
    def __init__(self, alpha: float,\
                 gamma: list[float|str] | np.ndarray = [1.],\
                 selection_method: str = Base_Rejection_Method.Selection_Method_LastNegative,\
                 save_objectives: bool = False):
        super().__init__(name='Benjamini-Hochberg', gamma=gamma,
                         str_param=f'alpha={alpha:.2f}', selection_method=selection_method, save_objectives=save_objectives)
        self.alpha = alpha

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        return p_values_sorted - self.i_N[:,:p_values_sorted.shape[1]]*self.alpha



class Lowest_Angle(Base_Rejection_Method):
    def __init__(self, gamma: list[float|str] | np.ndarray = [1.],\
                 selection_method: str = Base_Rejection_Method.Selection_Method_ArgMin,\
                 save_objectives: bool = False):
        super().__init__(name='Lowest-Angle', gamma=gamma,
                         selection_method=selection_method, save_objectives=save_objectives)

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        return p_values_sorted / self.i_N[:,:p_values_sorted.shape[1]] - 1


class Kolmogorov_Smirnov(Base_Rejection_Method):  # Kolmogorov-Smirnov test
    def __init__(self, mode: int = 1, gamma: list[float|str] | np.ndarray = [1.],\
                 selection_method: str = Base_Rejection_Method.Selection_Method_ArgMin,\
                 save_objectives: bool = False) -> None:
        super().__init__(name='Kolmogorov-Smirnov ' + ('one' if mode == 1 else 'two'),\
                         gamma=gamma, selection_method=selection_method, save_objectives=save_objectives)
        self.mode = mode
        self.i_N0 = np.empty(shape=0)
    
    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        N = self.i_N.size
        max_N = p_values_sorted.shape[1]
        if self.mode == 1:
            # searching only for low p values
            objective = p_values_sorted - self.i_N[:,:max_N]
            if not self.is_method_min:
                objective *= -1
        else:
            if self.i_N0.size != N:
                self.i_N0 = np.arange(N).reshape(1,-1)/N
            # two rows of objective function
            objective = np.asarray([p_values_sorted - self.i_N[:,:max_N], p_values_sorted - self.i_N0[:,:max_N]])
            objective = np.abs(objective).max(axis=0)
            if self.is_method_min:
                objective *= -1
        objective = math.sqrt(N)*objective  # normalization
        return objective


class Import_HC(Base_Rejection_Method):
    def __init__(self, gamma: list[float|str] | np.ndarray = [0.3], stable: bool = True,\
                 save_objectives: bool = False) -> None:
        super().__init__(name='Import HC ' + ('Stable' if stable else 'Unstable'),
                         gamma=gamma, selection_method=Base_Rejection_Method.Selection_Method_ArgMax, save_objectives=save_objectives)
        self.stable = stable
    
    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        gamma = self.max_effective_gamma
        objectives = []
        for p_vector in p_values_sorted:
            mtest = MultiTest(p_vector, stbl=self.stable)
            mtest.hc(gamma=gamma)
            objectives.append(mtest._zz.reshape(1,-1))
        objectives = np.array(objectives)
        if self.is_method_min:
            objectives *= -1
        return objectives


class Higher_Criticism(Base_Rejection_Method):
    def __init__(self, gamma: float | list[float|str] | np.ndarray, selection_method: str = Base_Rejection_Method.Selection_Method_ArgMax,\
                 stable: bool = True, save_objectives: bool = False):
        super().__init__(name=f'Higher Criticism{"" if stable else " Unstable"}',\
                         gamma=gamma, selection_method=selection_method, save_objectives=save_objectives)
        self.stable = stable
        self.denominator = np.empty(shape=0)
        
    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        N = self.i_N.size
        if self.stable:
            if self.denominator.shape != self.i_N.shape:
                self.denominator = np.sqrt(self.i_N*(1-self.i_N))
            active_N = min(p_values_sorted.shape[1],N-1)  # avoid division by zero
            denominator = self.denominator[:,:active_N]
        else:
            denominator = np.sqrt(p_values_sorted*(1-p_values_sorted))
        active_N = denominator.shape[1]
        nominator = math.sqrt(N)*(self.i_N[:,:active_N] - p_values_sorted[:,:active_N])
        objectives = nominator / denominator
        if not self.is_method_max:
            objectives *= -1
        return objectives

    def monte_carlo_statistics(self, monte_carlo: int, data_generator: Data_Generator_Base,\
                               chunk_size: int = 100, disable_tqdm: bool = False) -> dict:
        nums_rejected = []
        best_objectives = []
        first_p_value = []
        lowest_angle = []
        for i in tqdm(range(0,monte_carlo,chunk_size), disable=disable_tqdm):
            data_generator.generate(seeds=list(range(i,min(i+chunk_size,monte_carlo))))
            self.run_sorted_p(data_generator.p_values)
            # collecting data from objective function
            best_objectives += list(self.best_objective.reshape(-1))
            nums_rejected += list(self.num_rejected.reshape(-1))
            first_p_value += list(data_generator.p_values[:,0].reshape(-1))
            lowest_angle += list(np.min(data_generator.p_values/self.i_N, axis=1))
        assert max(nums_rejected) > 0
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

    def plot_objectives(self, title: str = 'Higher Criticism objectives values'):
        assert isinstance(self.objectives, np.ndarray)
        hc_objectives = self.objectives[0]
        num_rejected = self.num_rejected[0]
        best_objective = self.best_objective[0][0]
        plt.figure(figsize=(20,10))
        plt.title(label=title)
        plt.plot(np.arange(1,1+len(hc_objectives)), hc_objectives, label='HC objectives', c='blue')
        plt.vlines(x=num_rejected, ymin=hc_objectives.min(), ymax=hc_objectives.max(),
                   label=f'Optimal HC rejects {num_rejected} best objective value={best_objective:.2f}',
                   linestyles='dashed', colors='red')
        plt.xlabel(xlabel='Sorted Samples by P-value')
        plt.ylabel(ylabel='HC values')
        plt.xlim(xmin=0, xmax=len(hc_objectives))
        plt.legend()
        plt.show()


class Berk_Jones(Base_Rejection_Method):
    def __init__(self, selection_method: str = Base_Rejection_Method.Selection_Method_ArgMin,\
                 gamma: list[float|str] | np.ndarray = [1.0],\
                 save_objectives: bool = False) -> None:
        super().__init__(name='Berk-Jones', gamma=gamma,\
                         selection_method=selection_method, save_objectives=save_objectives)
        self.ii = np.empty(shape=0)

    def work_sorted_p(self, p_values_sorted: np.ndarray) -> np.ndarray:
        N = self.i_N.size
        if self.ii.shape != self.i_N.shape:
            self.ii = np.arange(1,N+1, dtype=np.int32).reshape(self.i_N.shape)
        ii = self.ii[:,:p_values_sorted.shape[1]]
        # make rare p values have negative objective
        objectives = beta.cdf(p_values_sorted, ii, N - ii + 1) - 0.5
        if not self.is_method_min:
            objectives *= -1
        return objectives
