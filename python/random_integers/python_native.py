import numpy as np

def matrix(num_steps: np.uint32,\
            offset_row0: np.uint32, offset_col0: np.uint32,\
            out: np.ndarray) -> None:
    assert out.dtype == np.uint64, f'{out.dtype=}'
    row_seeds = np.arange(offset_row0, offset_row0 + out.shape[0], dtype=np.uint64).reshape(-1,1)
    col_seeds = np.arange(offset_col0, offset_col0 + out.shape[1], dtype=np.uint64).reshape(1,-1)
    # Combine row and col into a single 64-bit integer
    with np.errstate(over='ignore'):
        out[:] = (row_seeds << np.uint64(32)) + col_seeds

    s0, s1 = matrix_base_states(out)
    for _ in range(num_steps):
        s0, s1 = matrix_states_transition(s0=s0, s1=s1)
    matrix_result(s0=s0, s1=s1, result=out)

def series(seed: np.uint64, out: np.ndarray) -> None:
    s0, s1 = integer_base_states(seed=seed)
    num_steps = out.size
    for i in range(num_steps):
        s0, s1 = integer_states_transition(s0=s0, s1=s1)
        out[i] = integer_result(s0=s0, s1=s1)

def integer_from_seed(seed: np.uint64, num_steps: np.uint32) -> np.uint64:
    s0, s1 = integer_base_states(seed=seed)
    for _ in range(num_steps):
        s0, s1 = integer_states_transition(s0=s0, s1=s1)
    result64 = integer_result(s0=s0, s1=s1)
    return result64

def matrix_base_states(seeds: np.ndarray)-> tuple[np.ndarray,np.ndarray]:
    matrix_scramble_seeds(seeds)
    splitmix_states     = seeds
    s0, splitmix_states = matrix_splitmix64(splitmix_states)
    s1, splitmix_states = matrix_splitmix64(splitmix_states)
    return s0, s1

def integer_base_states(seed: np.uint64)-> tuple[np.uint64,np.uint64]:
    seed = scramble_seed(seed=seed)
    splitmix_state     = seed
    s0, splitmix_state = splitmix64(splitmix_state)
    s1, splitmix_state = splitmix64(splitmix_state)
    return s0, s1

def matrix_states_transition(s0: np.ndarray, s1: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    s1 ^= s0
    s0 = matrix_rotl64(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = matrix_rotl64(s1, np.uint64(28))
    return s0, s1

def integer_states_transition(s0: np.uint64, s1: np.uint64) -> tuple[np.uint64,np.uint64]:
    s1 ^= s0
    s0 = rotl64(s0, np.uint64(49)) ^ s1 ^ (s1 << np.uint64(21))
    s1 = rotl64(s1, np.uint64(28))
    return s0, s1

def matrix_result(s0: np.ndarray, s1: np.ndarray, result: np.ndarray) -> None:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result[:] = matrix_rotl64(s0 + s1, np.uint64(17)) + s0


def integer_result(s0: np.uint64, s1: np.uint64) -> np.uint64:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        result64 = rotl64(s0 + s1, np.uint64(17)) + s0
    return result64

def matrix_splitmix64(states: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        states += np.uint64(0x9E3779B97F4A7C15)
        z = states
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, states

def splitmix64(state: np.uint64) -> tuple[np.uint64,np.uint64]:
    with np.errstate(over='ignore'):  # Suppress overflow warnings
        state += np.uint64(0x9E3779B97F4A7C15)
        z = state
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
        return z, state

def matrix_rotl64(x: np.ndarray, k: np.uint64) -> np.ndarray:
    return (x << k) | (x >> (np.uint64(64) - k))

def rotl64(x: np.uint64, k: np.uint64) -> np.uint64:
    return (x << k) | (x >> (np.uint64(64) - k))

def matrix_scramble_seeds(seeds: np.ndarray) -> None:
    # Combine row and col into a single 64-bit integer
    with np.errstate(over='ignore'):
        # adding one column and one row to avoid zero seed
        seeds += (np.uint64(1) << np.uint64(32)) + np.uint64(1)
        # Apply a series of XOR and multiplication steps to mix the bits.
        seeds ^= (seeds >> np.uint64(33))
        seeds *= np.uint64(0xff51afd7ed558ccd) # Large constant from MurmurHash3
        seeds ^= (seeds >> np.uint64(33))
        seeds *= np.uint64(0xc4ceb9fe1a85ec53)  # Another mixing constant
        seeds ^= (seeds >> np.uint64(33))

def scramble_seed(seed: np.uint64) -> np.uint64:
    with np.errstate(over='ignore'):
        # adding one column and one row to avoid zero seed
        seed += (np.uint64(1) << np.uint64(32)) + np.uint64(1)
        # Apply a series of XOR and multiplication steps to mix the bits.
        seed ^= (seed >> np.uint64(33))
        seed *= np.uint64(0xff51afd7ed558ccd) # Large constant from MurmurHash3
        seed ^= (seed >> np.uint64(33))
        seed *= np.uint64(0xc4ceb9fe1a85ec53)  # Another mixing constant
        seed ^= (seed >> np.uint64(33))
    return seed
