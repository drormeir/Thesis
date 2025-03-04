from python.hpc import globals
cpu_njit_num_threads = globals.cpu_njit_num_threads # numba njit compatible

if not cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def array_transpose_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def average_row_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def sort_rows_inplace_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def cumulative_argmin_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def cumulative_min_inplace_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def cumulative_dominant_argmin_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def cumulative_dominant_min_inplace_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def max_column_along_rows_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import numpy as np
    import numba

    @numba.njit(parallel=True)
    def array_transpose_cpu_njit(array: np.ndarray, out: np.ndarray) -> None:
        chunks_ranges = split2chunks(out.shape[0])
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            out[begin:end] = array[:,begin:end].T
    
    @numba.njit(parallel=True)
    def average_row_cpu_njit(array: np.ndarray, out_row: np.ndarray) -> None:
        chunks_ranges = split2chunks(array.shape[1])
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            np.mean(array[:,begin:end], axis=0, keepdims=True, out=out_row[0,begin:end])

    @numba.njit(parallel=True)
    def average_column_cpu_njit(array: np.ndarray, out_column: np.ndarray) -> None:
        chunks_ranges = split2chunks(array.shape[0])
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            np.mean(array[begin:end,:], axis=1, keepdims=True, out=out_column[begin:end,0])

    @numba.njit(parallel=True)
    def sort_rows_inplace_cpu_njit(array: np.ndarray) -> None:
        for ind_row in numba.prange(array.shape[0]):
            array[ind_row,:] = np.sort(array[ind_row])

    @numba.njit(parallel=True)
    def cumulative_argmin_cpu_njit(array: np.ndarray, argmin: np.ndarray) -> None:
        rows, cols = array.shape
        for ind_row in numba.prange(rows):
            input_row = array[ind_row]
            output_row = argmin[ind_row]
            current_min = input_row[0]
            current_idx = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val < current_min:
                    current_idx = np.uint32(j)
                    current_min = curr_val
                output_row[j] = current_idx

    @numba.njit(parallel=True)
    def cumulative_min_inplace_cpu_njit(array: np.ndarray) -> None:
        rows, cols = array.shape
        for ind_row in numba.prange(rows):
            row = array[ind_row]
            current_min = row[0]
            for j in range(1, cols):
                curr_val = row[j]
                if curr_val < current_min:
                    current_min = curr_val
                row[j] = current_min

    @numba.njit(parallel=True)
    def cumulative_dominant_argmin_cpu_njit(array: np.ndarray, argmin: np.ndarray) -> None:
        rows, cols = array.shape
        for ind_row in numba.prange(rows):
            input_row = array[ind_row]
            output_row = argmin[ind_row]
            current_min = input_row[0]
            current_ind_min = np.uint32(0)
            current_ind_dominant = np.uint32(0)
            max_dominant_length = np.uint32(0)
            output_row[0] = np.uint32(0)
            for j in range(1, cols):
                curr_val = input_row[j]
                if curr_val < current_min:
                    current_ind_min = np.uint32(j)
                    current_min = curr_val
                curr_dominant_length = np.uint32(j) - current_ind_min
                if curr_dominant_length >= max_dominant_length:
                    current_ind_dominant = current_ind_min
                    max_dominant_length = curr_dominant_length
                output_row[j] = current_ind_dominant

    @numba.njit(parallel=True)
    def cumulative_dominant_min_inplace_cpu_njit(array: np.ndarray) -> None:
        rows, cols = array.shape
        for ind_row in numba.prange(rows):
            row = array[ind_row]
            current_min = current_dominant = row[0]
            current_ind_min = np.uint32(0)
            max_dominant_length = np.uint32(0)
            for j in range(1, cols):
                curr_val = row[j]
                if curr_val < current_min:
                    current_ind_min = np.uint32(j)
                    current_min = curr_val
                curr_dominant_length = np.uint32(j) - current_ind_min
                if curr_dominant_length >= max_dominant_length:
                    current_dominant = current_min
                    max_dominant_length = curr_dominant_length
                row[j] = current_dominant

    @numba.njit(parallel=True)
    def max_column_along_rows_cpu_njit(array: np.ndarray, argmax: np.ndarray, maxval: np.ndarray) -> None:
        rows = array.shape[0]
        chunks_ranges = split2chunks(rows)
        for ind_chunck in numba.prange(chunks_ranges.shape[0]):
            begin, end = chunks_ranges[ind_chunck]
            array[begin:end].argmax(axis=1, out=argmax[begin:end])
            array[begin:end].max(axis=1, out=maxval[begin:end])

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


