from python.cuda import cuda_available, cpu_njit_available, raise_cuda_not_available, raise_njit_not_available, HybridArray
from python.random_numbers import random_p_values_row_py
from python.random_numbers import random_integers_row_gpu
from python.random_numbers import random_integers_row_cpu_njit
import numpy as np
from scipy.stats import norm

def rare_weak_no_labels_row_py(ind_row_out: np.uint32, n1: np.uint32, mu: np.float32, ind_row_work: np.uint32, seed: np.uint64, work_random_integers: np.ndarray, out: np.ndarray) -> None:
    random_p_values_row_py(ind_row_out=ind_row_out, ind_row_work=ind_row_work, seed=seed, work_random_integers=work_random_integers, out=out)
    out[ind_row_out][:n1] = norm.sf(norm.isf(out[ind_row_out][:n1]) + mu)
    out.sort()

def rare_weak_matrix_no_labels_py(seed0: np.uint64, n1: np.uint32, mu: np.float32, work_random_integers: np.ndarray, out: np.ndarray) -> None:
    """
    Generates random numbers using a xoroshiro128++.
    
    Parameters:
        seed0 (int): base seed
        out (numpy array): Output array to store random values.
    """
    for ind_seed in np.arange(0,out.shape[0], work_random_integers.shape[0], dtype=np.uint32):
        random_p_values_row_py(ind_row_out=ind_row_out, ind_row_work=ind_row_work, seed=seed, work_random_integers=work_random_integers, out=out)

        rare_weak_row_py(ind_row_out=ind_seed, n1=n1, mu=mu, ind_row_work=0, seed=seed0 + ind_seed, work_random_integers=work_row, out=out)


if cpu_njit_available:
    import numba

    @numba.njit(parallel=True)
    def uniform_random_matrix_cpu_njit(seed0: int, out: np.ndarray) -> None:
        """
        Generates random numbers using a xoroshiro128++.
        
        Parameters:
            seed0 (int): base seed
            out (numpy array): Output array to store random values.
        """
        def splitmix64_cpu_njit(state):
            state = np.uint64(state + 0x9E3779B97F4A7C15)
            z = state
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z = z ^ (z >> np.uint64(31))
            return z, state

        def rotl64_cpu_njit(x, k):
            x = np.uint64(x)
            return (x << np.uint64(k)) | (x >> np.uint64(64 - k))
        
        num_seeds, N = out.shape
        for ind_seed in numba.prange(num_seeds):
            splitmix_state = np.uint64(seed0 + ind_seed)
            s0, splitmix_state = splitmix64_cpu_njit(splitmix_state)
            s1, splitmix_state = splitmix64_cpu_njit(splitmix_state)
            # create N random numbers
            for i in range(N):
                result = rotl64_cpu_njit(s0 + s1, 17) + s0
                out[ind_seed, i] = np.uint32(result >> 32) # higher bit are better randomized
                # state transition
                s1 ^= s0
                s0 = rotl64_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
                s1 = rotl64_cpu_njit(s1, np.uint64(28))
else:
    def uniform_random_matrix_cpu_njit(seed0: int, out: np.ndarray) -> None:
        raise_njit_not_available()

if cuda_available:
    import numba
    import numba.types
    import numba.cuda

    @numba.cuda.jit(device=True)
    def splitmix64_gpu(state):
        state += numba.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> numba.uint64(30))) * numba.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> numba.uint64(27))) * numba.uint64(0x94D049BB133111EB)
        z = z ^ (z >> numba.uint64(31))
        return z, state

    @numba.cuda.jit(numba.types.uint64(numba.types.uint64, numba.types.uint64), device=True)
    def rotl64_gpu(x, k):
        return (x << k) | (x >> (64 - k))

    @numba.cuda.jit(device=False)
    def uniform_random_matrix_gpu(seed0, out):
        """
        CUDA kernel to set uniform random values using xoroshiro128++.

        Parameters:
            seed0 (int): base seed
            out (device array): Output array to store random values.
        """
        ind_start = numba.cuda.grid(1)  # type: ignore
        ind_stride = numba.cuda.gridsize(1) # type: ignore
        num_seeds, N = out.shape
        for ind_seed in range(ind_start, num_seeds, ind_stride):
            splitmix_state = numba.uint64(seed0 + ind_seed)
            s0, splitmix_state = splitmix64_gpu(splitmix_state)
            s1, splitmix_state = splitmix64_gpu(splitmix_state)
            # create N random numbers
            for i in range(N):
                result = rotl64_gpu(s0 + s1, 17) + s0
                out[ind_seed, i] = numba.uint32(result >> 32) # higher bit are better randomized
                # state transition
                s1 ^= s0
                s0 = rotl64_gpu(s0, numba.uint64(49)) ^ s1 ^ (s1 << numba.uint64(21))
                s1 = rotl64_gpu(s1, numba.uint64(28))
else:
    def uniform_random_matrix_gpu(seed0, out):
        raise_cuda_not_available()



def uniform_random_matrix(data: HybridArray, N:int, seed0: int, seed1: int, use_gpu: bool|None = None, use_njit: bool|None = None) -> HybridArray:
    assert 0 < N
    assert 0 <= seed0 < seed1
    if use_gpu is None:
        use_gpu = cuda_available
    elif use_gpu and not cuda_available:
        raise_cuda_not_available()
    data.realloc(shape=(seed1 - seed0, N), dtype=np.uint32, use_gpu=use_gpu)
    if use_gpu:
        # GPU mode
        blocks_grid, per_block_threads_grid = data.optimal_block_size_for_rows()
        uniform_random_matrix_gpu[blocks_grid, per_block_threads_grid](seed0, data.data) # type: ignore - Suppress Pylance Object of type "(N: Unknown, rng_states: Unknown, out: Unknown) -> None" is not subscriptable
    else:
        # CPU mode
        if cpu_njit_available and (use_njit is None or use_njit):
            uniform_random_matrix_cpu_njit(seed0=seed0, out=data.data)
        else:
            uniform_random_matrix_py(seed0=seed0, out=data.data)
    return data
