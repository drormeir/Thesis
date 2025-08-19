import numba
import numba.cuda
from numba.cuda.cudadrv.devicearray import DeviceNDArray
import numpy as np
import sys

if len(sys.argv) != 3:
    print("ERROR: Usage: python test_cuda_kernel.py <grid_size> <block_dim>", file=sys.stderr)
    sys.exit(1)

grid_size = int(sys.argv[1])
block_dim = int(sys.argv[2])

@numba.cuda.jit
def dummy_kernel(out: DeviceNDArray):
    idx = numba.cuda.grid(1) # type: ignore
    if idx < out.size:
        out[idx] = idx

test_data = np.empty(grid_size * block_dim, dtype=np.uint32)
d_test_data = numba.cuda.to_device(test_data)
dummy_kernel[grid_size, block_dim](d_test_data) # type: ignore
d_test_data.copy_to_host()

sys.exit(0)
