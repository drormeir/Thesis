import numpy as np
from scipy.stats import norm
import itertools
import math
from sklearn import metrics
import functools

class Data_Generator_Base:
    def __init__(self, N: int) -> None:
        self.N = N
        self.p_values = np.empty(shape=0)

    def generate(self, seeds: list[int]) -> None:
        self.p_values = np.empty(shape=(len(seeds),self.N))
        for ind_seed, seed in enumerate(seeds):
            np.random.seed(seed=seed)
            self.p_values[ind_seed] = np.random.uniform(size=self.N)

    @staticmethod
    def params_pure_noise(N: int) -> dict:
        return {'sizes': [N], 'mus': [0.], 'sigmas': [1.]}
    
    @staticmethod
    def params_from_N_mu_fraction(N: int, mu: float, fraction: float) -> dict:
        n = min(int(0.5+N*fraction),N)
        if fraction >= 1e-9:
            n = max(n,1)
        if fraction <= 1-1e-9:
            n = min(n, N-1)
        return {'sizes': [N-n, n], 'mus': [0., mu], 'sigmas': [1.,1.]}

    @staticmethod
    def params_from_N_r_beta(N: int, r: float, beta: float) -> dict:
        mu = math.sqrt(2*r*math.log(N))
        fraction = math.pow(N,-beta)
        return Data_Generator_Base.params_from_N_mu_fraction(N=N,mu=mu,fraction=fraction)


class Multi_Class_Normal_Population(Data_Generator_Base):
    def __init__(self, sizes: list[int], mus: list[float], sigmas: list[float], sides: int = 1) -> None:
        super().__init__(N = sum(sizes))
        self.sizes = sizes
        self.mus = mus
        self.sigmas = sigmas
        self.labels = None
        self.values = None
        self.original_labels = np.hstack([[label]*size for label,size in enumerate(self.sizes)]).astype(int)
        self.mu_vector = np.array([self.mus[label] for label in self.original_labels], dtype=np.float32)
        self.std_vector = np.array([self.sigmas[label] for label in self.original_labels], dtype=np.float32)
        self.sides = sides

    def generate(self, seeds: list[int]) -> None:
        n_array = np.arange(self.N)
        z = np.empty(shape=(len(seeds), self.N))
        ind_permute = np.empty_like(z, dtype=n_array.dtype)
        for ind_seed, seed in enumerate(seeds):
            np.random.seed(seed=seed)
            z[ind_seed] = np.hstack([np.random.standard_normal(size=size).astype(np.float32) for size in self.sizes])
            ind_permute[ind_seed] = np.random.permutation(n_array)
        self.generate_from_z(z, ind_permute=ind_permute)

    def generate_from_z(self, z: np.ndarray, ind_permute: np.ndarray = np.empty(shape=0)) -> None:
        values = z * self.std_vector + self.mu_vector
        self.labels = np.empty_like(ind_permute)
        self.values = np.empty_like(values)
        for ind_row, row_permute in enumerate(ind_permute):
            self.labels[ind_row] = self.original_labels[row_permute]
            self.values[ind_row] = values[ind_row,row_permute]
        ppf = (self.values - self.mus[0])/self.sigmas[0]
        if self.sides == 1:
            self.p_values = norm.sf(ppf)
        else:
            self.p_values = norm.sf(np.abs(ppf)) * 2


class Multi_Class_Normal_Population_Uniform(Multi_Class_Normal_Population):
    def generate(self, seed: int) -> None:
        np.random.seed(seed=seed)
        z = np.hstack([norm.ppf(np.linspace(start=0.5/size, stop=1-0.5/size, num=size)) for size in self.sizes if size])
        self.generate_from_z(z)


def Two_Lists_Tuple(list1: list, list2: list) -> list[tuple]:
    return list(itertools.product(list1,list2))

def signal_2_noise_roc(signal_values, noise_values) -> tuple[float,np.ndarray,np.ndarray]:
    sort_v_factor = -1 if np.mean(signal_values) > np.mean(noise_values) else 1
    roc_values_tuples = []
    for v in noise_values:
        roc_values_tuples.append((0,v*sort_v_factor))
    for v in signal_values:
        roc_values_tuples.append((1,v*sort_v_factor))
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
