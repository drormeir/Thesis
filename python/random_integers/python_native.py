import numpy as np

def random_integers_matrix_py(num_steps: np.uint32, offset_row0: np.uint32, offset_col0: np.uint32, out: np.ndarray) -> None:
    col_seeds = np.arange(offset_col0, offset_col0 + out.shape[1], dtype=np.uint64).reshape(1,-1)
    row_seeds = np.arange(offset_row0, offset_row0 + out.shape[0], dtype=np.uint64).reshape(-1,1)
    seeds = (row_seeds << np.uint64(32)) + col_seeds
    s0, s1 = random_integer_base_states_from_seeds_py(seeds=seeds)
    for _ in range(num_steps):
        s0, s1 = random_integer_states_transition_from_states_py(s0=s0, s1=s1)
    random_integer_result_from_states_py(s0=s0, s1=s1, result=out)


def random_p_values_series_py(seed: np.uint64, out: np.ndarray) -> None:
    norm_factor = 1.0 / np.float64(2.0**64)
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        rand_int = random_integer_result_py(s0=s0, s1=s1)
        out[i] = (rand_int + 0.5) * norm_factor

def random_integers_series_py(seed: np.uint64, out: np.ndarray) -> None:
    s0, s1 = random_integer_base_states_py(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = random_integer_states_transition_py(s0=s0, s1=s1)
        out[i] = random_integer_result_py(s0=s0, s1=s1)

def random_integer_base_states_from_seeds_py(seeds: np.ndarray)-> tuple[np.ndarray,np.ndarray]:
    splitmix_states     = seeds
    s0, splitmix_states = splitmix64_from_states_py(splitmix_states)
    s1, splitmix_states = splitmix64_from_states_py(splitmix_states)
    return s0, s1

def random_integer_base_states_py(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
    splitmix_state     = seed
    s0, splitmix_state = splitmix64_py(splitmix_state)
    s1, splitmix_state = splitmix64_py(splitmix_state)
    return s0, s1

def random_integer_states_transition_from_states_py(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    s1 ^= s0
    s0 = rotl64_array_py(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64_array_py(s1, np.uint64(28))
    return s0, s1

def random_integer_states_transition_py(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
    s1 ^= s0
    s0 = rotl64_py(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64_py(s1, np.uint64(28))
    return s0, s1

def random_integer_result_from_states_py(s0: np.ndarray, s1: np.ndarray, result: np.ndarray) -> None:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result[:] = rotl64_array_py(s0 + s1, np.uint64(17)) + s0


def random_integer_result_py(s0: np.uint64, s1: np.uint64) -> np.uint64:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result64 = rotl64_py(s0 + s1, np.uint64(17)) + s0
    return result64

def splitmix64_from_states_py(states: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        states += np.uint64(0x9E3779B97F4A7C15)
        z = states
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, states

def splitmix64_py(state: np.uint64) -> tuple[np.uint64,np.uint64]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state

def rotl64_array_py(x: np.ndarray, k: np.uint64) -> np.ndarray:
    return (x << k) | (x >> (np.uint64(64) - k))

def rotl64_py(x: np.uint64, k: np.uint64) -> np.uint64:
    return (x << k) | (x >> (np.uint64(64) - k))
