from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def random_integers_matrix_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def random_integers_series_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    import numba

    @numba.njit(parallel=True)
    def random_integers_matrix_cpu_njit(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
        col_seeds = np.arange(offset_col0, offset_col0 + out.shape[1], dtype=np.uint64).reshape(1,-1)
        row_seeds = np.arange(offset_row0, offset_row0 + out.shape[0], dtype=np.uint64).reshape(-1,1)
        seeds = (row_seeds << np.uint64(32)) + col_seeds
        s0, s1 = random_integer_base_states_from_seeds_cpu_njit(seeds=seeds)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_from_states_cpu_njit(s0=s0, s1=s1)
        random_integer_result_from_states_cpu_njit(s0=s0, s1=s1, result=out)

    @numba.njit(parallel=False)
    def random_integer_cpu_njit(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        for _ in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0=s0, s1=s1)
        result64 = random_integer_result_cpu_njit(s0=s0, s1=s1)
        return result64

    @numba.njit(parallel=False)
    def random_integers_series_cpu_njit(seed: np.uint64, out: np.ndarray) -> None:
        s0, s1 = random_integer_base_states_cpu_njit(seed=seed)
        num_steps = out.size
        for i in range(num_steps):
            s0, s1 = random_integer_states_transition_cpu_njit(s0, s1)
            out[i] = random_integer_result_cpu_njit(s0, s1)

    @numba.njit(parallel=False)
    def random_integer_base_states_from_seeds_cpu_njit(seeds: np.ndarray)-> tuple[np.ndarray,np.ndarray]:
        splitmix_states     = seeds
        s0, splitmix_states = splitmix64_from_states_cpu_njit(splitmix_states)
        s1, splitmix_states = splitmix64_from_states_cpu_njit(splitmix_states)
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_base_states_cpu_njit(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
        splitmix_state     = seed
        s0, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        s1, splitmix_state = splitmix64_cpu_njit(splitmix_state)
        return s0, s1

    @numba.njit(parallel=True)
    def random_integer_states_transition_from_states_cpu_njit(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        s1 ^= s0
        s0 = rotl64_array_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_array_cpu_njit(s1, np.uint64(28))
        return s0, s1

    @numba.njit(parallel=False)
    def random_integer_states_transition_cpu_njit(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
        s1 ^= s0
        s0 = rotl64_cpu_njit(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
        s1 = rotl64_cpu_njit(s1, np.uint64(28))
        return s0, s1

    @numba.njit(parallel=True)
    def random_integer_result_from_states_cpu_njit(s0: np.ndarray, s1: np.ndarray, result: np.ndarray) -> None:
        result[:] = rotl64_array_cpu_njit(s0 + s1, np.uint64(17)) + s0

    @numba.njit(parallel=False)
    def random_integer_result_cpu_njit(s0: np.uint64, s1: np.uint64) -> np.uint64:
        result64 = rotl64_cpu_njit(s0 + s1, np.uint64(17)) + s0
        return result64


    @numba.njit(parallel=True)
    def splitmix64_from_states_cpu_njit(states: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
        states += np.uint64(0x9E3779B97F4A7C15)
        z = states
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, states
    
    @numba.njit(parallel=False)
    def splitmix64_cpu_njit(state: np.uint64) -> tuple[np.uint64,np.uint64]:
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state
    
    @numba.njit(parallel=True)
    def rotl64_array_cpu_njit(x: np.ndarray, k: np.uint64) -> np.ndarray:
        return (x << k) | (x >> (np.uint64(64) - k))

    @numba.njit(parallel=False)
    def rotl64_cpu_njit(x: np.uint64, k: np.uint64) -> np.uint64:
        return (x << k) | (x >> (np.uint64(64) - k))
