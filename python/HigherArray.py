import numpy as np
from python.cuda import cuda_available, cpu_njit_available, get_optimal_thread_block_size
from python.random_numbers import uniform_random_matrix

class NumbaArray:
    def __init__(self, is_host: bool = True) -> None:
        self.is_host = is_host if cuda_available else True
        self.data = None

    def set_uniform_random_values(self, N:int, seed0: int, seed1: int) -> None:
        """
        Sets uniform random values in the array.

        Parameters:
        N (int): Number of random values per seed.
        seed0 (int): Starting seed for RNG.
        seed1 (int): Ending seed for RNG.
        """
        self.data = uniform_random_matrix(N=N, seed0=seed0, seed1=seed1, use_cuda=self.is_host)

    def to_numpy(self) -> np.ndarray:
        """
        Converts the NumbaArray to a NumPy array.

        Returns:
            np.ndarray: The data as a NumPy array.
        """
        if self.is_host:
            return self.data
        return self.data.copy_to_host()        
