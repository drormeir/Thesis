import numpy as np
from scipy.stats import norm
import itertools


class Data_Generator_Base:
    def __init__(self, N: int) -> None:
        self.N = N
        self.p_values = np.empty(shape=0)

    def generate(self, seed: int) -> None:
        pass


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

    def generate(self, seed: int) -> None:
        np.random.seed(seed=seed)
        z = np.hstack([np.random.standard_normal(size=size).astype(np.float32) for size in self.sizes])
        self.generate_from_z(z)

    def generate_from_z(self, z: np.ndarray) -> None:
        ind_premute = np.random.permutation(np.arange(self.N))
        values = z * self.std_vector + self.mu_vector
        self.labels = self.original_labels[ind_premute]
        self.values = values[ind_premute]
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
