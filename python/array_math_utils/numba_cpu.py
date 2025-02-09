from python.hpc import globals
cpu_njit_num_threads = globals.cpu_njit_num_threads # numba njit compatible

if not cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def array_transpose_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def average_rows_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def sort_rows_inplace_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import math
    import numpy as np
    import numba

    @numba.njit(parallel=True)
    def array_transpose_cpu_njit(array: np.ndarray, out: np.ndarray) -> None:
        chunks_ranges = split2chunks(out.shape[0])
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            out[begin:end] = array[:,begin:end].T
    
    @numba.njit(parallel=True)
    def average_rows_cpu_njit(array: np.ndarray, out_row: np.ndarray) -> None:
        chunks_ranges = split2chunks(array.shape[1])
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            np.mean(array[:,begin:end], axis=0, keepdims=True, out=out_row[0,begin:end])

    @numba.njit(parallel=True)
    def sort_rows_inplace_cpu_njit(array: np.ndarray) -> None:
        for ind_row in numba.prange(array.shape[0]):
            array[ind_row,:] = np.sort(array[ind_row])

    @numba.njit(parallel=False)
    def split2chunks(size: int) -> np.ndarray:
        num_chunks = min(size, cpu_njit_num_threads)
        chunk_base_size = size // num_chunks
        chunk_residue = size % num_chunks
        ranges = np.empty((num_chunks, 2), dtype=np.uint32)
        for ind_chunk in range(num_chunks):
            current_residue = int(ind_chunk < chunk_residue)
            current_size = chunk_base_size + current_residue
            chunk_offset = ind_chunk*(current_residue + chunk_base_size)+chunk_residue*(1-current_residue)
            ranges[ind_chunk, 0] = chunk_offset
            ranges[ind_chunk, 1] = chunk_offset + current_size
        return ranges
