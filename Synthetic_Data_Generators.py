import numpy as np
from scipy.stats import norm
import itertools
import math
from sklearn import metrics
import functools
import multiprocessing


class Data_Generator_Base:
    def __init__(self, N: int) -> None:
        self.N = N
        self.p_values = np.empty(shape=0)
        self.argsort = np.empty(shape=0)

    def generate(self, seeds: list[int]) -> None:
        random_values = Data_Generator_Base.generate_random_values(self.N, seeds=seeds)
        self.generate_from_random_values(random_values)
    
    def generate_from_random_values(self, random_values: np.ndarray) -> None:
        p_values = self.generate_p_values_from_random_values(random_values)
        self.argsort = p_values.argsort(axis=1)
        self.p_values = np.sort(p_values, axis=1)

    def generate_p_values_from_random_values(self, random_values: np.ndarray) -> np.ndarray:
        return np.empty(shape=0)

    def calc_confusion(self, num_rejected: np.ndarray) -> np.ndarray:
        return np.empty(shape=0)
    
    @staticmethod
    def generate_noise_p_values_parallel_row(ind_seed: int, seed: int, N: int, p_values: np.ndarray) -> None:
        p_values[ind_seed] = np.random.default_rng(seed).random(size=N)

    @staticmethod
    def generate_noise_p_values_parallel(N: int, seeds: list[int]) -> np.ndarray:
        n_seeds = len(seeds)
        p_values = np.empty(shape=(n_seeds, N))
        args_list = [(ind,s,N,p_values) for ind,s in enumerate(seeds)]
        with multiprocessing.Pool() as pool:
            pool.starmap(Data_Generator_Base.generate_noise_p_values_parallel_row, args_list)
        return p_values

    @staticmethod
    def generate_random_values(N: int, seeds: list[int]) -> np.ndarray:
        p_values = np.empty(shape=(len(seeds), N))
        for ind_seed, seed in enumerate(seeds):
            p_values[ind_seed] = np.random.default_rng(seed).random(size=N)
        return p_values

class Data_Generator_Noise(Data_Generator_Base):
    def __init__(self, N: int) -> None:
        super().__init__(N)
    
    def generate_p_values_from_random_values(self, random_values: np.ndarray) -> np.ndarray:
        return random_values

class Multi_Class_Normal_Population(Data_Generator_Base):
    def __init__(self, sizes: list[int], mus: list[float], sigmas: list[float], sides: int = 1) -> None:
        super().__init__(N = sum(sizes))
        self.sizes = sizes
        self.mus = mus
        self.sigmas = sigmas
        self.original_labels = np.hstack([[label]*size for label,size in enumerate(self.sizes)]).astype(int)
        self.mu_vector = np.array([self.mus[label] for label in self.original_labels], dtype=np.float32)
        self.std_vector = np.array([self.sigmas[label] for label in self.original_labels], dtype=np.float32)
        self.mu_vector0 = (self.mu_vector - self.mus[0])/self.sigmas[0]
        self.std_vector0 = self.std_vector/self.sigmas[0]
        if sides == 1:
            N0 = self.sizes[0]
            self.mu_vector0 = self.mu_vector0[N0:]
            self.std_vector0 = self.std_vector0[N0:]
        self.sides = sides

    @staticmethod
    def params_pure_noise(N: int) -> dict:
        return {'sizes': [N], 'mus': [0.], 'sigmas': [1.]}
    
    @staticmethod
    def params_from_N_mu_fraction(N: int, mu: float, fraction: float, sides: int = 1) -> dict:
        n = min(int(0.5+N*fraction),N)
        if fraction >= 1e-9:
            n = max(n,1)
        if fraction <= 1-1e-9:
            n = min(n, N-1)
        return {'sizes': [N-n, n], 'mus': [0., mu], 'sigmas': [1.,1.], 'sides': sides}

    @staticmethod
    def params_from_N_r_beta(N: int, r: float, beta: float) -> dict:
        mu = math.sqrt(2*r*math.log(N))
        fraction = math.pow(N,-beta)
        return Multi_Class_Normal_Population.params_from_N_mu_fraction(N=N,mu=mu,fraction=fraction)
    
    @staticmethod
    def params_list_from_N_r_beta(N: int|list[int], r: float|list[float], beta: float|list[float]) -> list[dict]:
        if not isinstance(N,list):
            N = [N]
        if not isinstance(r,list):
            r = [r]
        if not isinstance(beta,list):
            beta = [beta]
        return [Multi_Class_Normal_Population.params_from_N_r_beta(*params) for params in itertools.product(N,r,beta)]

    @staticmethod
    def params_dicts_from_lists_N_mu_fraction(N_range: list[int], mu_range: list[float], fraction_range: list[float]) -> list[dict]:
        params_tuples = list(itertools.product(N_range,mu_range,fraction_range))
        return [Multi_Class_Normal_Population.params_from_N_mu_fraction(*params,sides=2) for params in params_tuples]

    def generate_p_values_from_random_values(self, random_values: np.ndarray) -> np.ndarray:
        if self.sides == 1:
            N0 = self.sizes[0]
            z = norm.ppf(random_values[:,N0:]) * self.std_vector0 + self.mu_vector0
            p_values = np.hstack([random_values[:,:N0], norm.sf(z)])
        else:
            z = norm.ppf(random_values) * self.std_vector0 + self.mu_vector0
            p_values = norm.sf(np.abs(z)) * 2
        return p_values

    def calc_confusion(self, num_rejected: np.ndarray) -> np.ndarray:
        N = np.sum(self.sizes)
        result = np.empty(shape=(num_rejected.size,4), dtype=np.int32)
        for ind_row, num_reject in enumerate(num_rejected.reshape(-1)):
            labels = self.original_labels[self.argsort[ind_row]]
            true_positive = np.sum(labels[:num_reject] == 1)
            false_positive = num_reject - true_positive
            true_negative = np.sum(labels[num_reject:] == 0)
            false_negative = N - num_reject - true_negative
            result[ind_row][:] = [true_positive, false_positive, true_negative, false_negative]
        return result

class Multi_Class_Normal_Population_Uniform(Multi_Class_Normal_Population):
    def generate(self, seed: int) -> None:
        p_values = np.hstack([np.linspace(start=0.5/size, stop=1-0.5/size, num=size) for size in self.sizes if size])
        z = norm.ppf(p_values) * self.std_vector0 + self.mu_vector0
        if self.sides == 1:
            p_values = norm.sf(z)
        else:
            p_values = norm.sf(np.abs(z)) * 2
        self.p_values = np.sort(p_values, axis=1)


def Two_Lists_Tuple(list1: list, list2: list) -> list[tuple]:
    return list(itertools.product(list1,list2))

def signal_2_noise_roc(signal_values, noise_values) -> tuple[float,np.ndarray,np.ndarray]:
    sort_v_factor = -1 if np.mean(signal_values) > np.mean(noise_values) else 1
    roc_values_tuples = [(0,v*sort_v_factor) for v in noise_values] + [(1,v*sort_v_factor) for v in signal_values]
    def cmp_signal_values(t1, t2):
        v1, v2 = t1[1], t2[1]
        diff_v12 = v1 - v2
        if abs(diff_v12) <= 1e-6*min(abs(v1),abs(v2)):
            return t1[0] - t2[0]
        return 1 if v1 > v2 else -1
    sorted_tuples = sorted(roc_values_tuples, key = functools.cmp_to_key(cmp_signal_values))
    sorted_labels = [roc_tuple[0] for roc_tuple in sorted_tuples]
    sorted_values = [roc_tuple[1] for roc_tuple in reversed(sorted_tuples)]
    fpr, tpr, _ = metrics.roc_curve(y_true=sorted_labels, y_score= sorted_values)
    roc_auc = metrics.auc(fpr,tpr)
    return float(roc_auc), fpr, tpr
