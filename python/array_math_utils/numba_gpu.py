from python.hpc import globals

if not globals.cuda_available:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def array_transpose_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def average_rows_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def sort_rows_inplace_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import math
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    import cupy

    def array_transpose_gpu(array: DeviceNDArray, out: DeviceNDArray) -> None:
        cupy.asarray(out)[:] = cupy.asarray(array).T # cupy.ascontiguousarray(cupy_T)

    def average_rows_gpu(array: DeviceNDArray, out_row: DeviceNDArray) -> None:
        array_cupy = cupy.asarray(array)
        avg_cupy = cupy.asarray(out_row)
        cupy.mean(array_cupy, axis=0, keepdims=True, out=avg_cupy)

    def sort_rows_inplace_gpu(array: DeviceNDArray) -> None:
        cupy.asarray(array).sort(axis=1)
