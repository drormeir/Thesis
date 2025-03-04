import warnings
import gc
import numpy as np
from types import SimpleNamespace

globals = SimpleNamespace(
    numba_import_success = False,
    cpu_njit_num_threads = np.uint32(0),
    cuda_available = False,
    max_threads_per_block = np.uint32(1),
    min_grid_size = np.uint32(1),
    warp_size = np.uint32(32),
    grid_block_shape_debug = np.uint32(0),
    max_registers_per_block = np.uint32(1),
)

try:
    import numba
    print("Numba version:", numba.__version__)
    globals.numba_import_success = True
except ImportError as e:
    print(f'Could not import Numba.\n{e}')


def init_njit() -> np.uint32:
    if not globals.numba_import_success:
        return np.uint32(0)
    
    try:
        # Attempt to define and call a simple njit function
        @numba.njit
        def test_func(x):
            return x + 1
        test_func(1)
        ret = np.uint32(numba.get_num_threads())
    except (NameError, ImportError, TypeError):
        ret = np.uint32(0)

    if ret:
        print("numba.njit is available.")
    else:
        print("numba.njit not available on this system.")
    return ret


def raise_njit_not_available():
    if globals.cpu_njit_num_threads > 0:
        raise AssertionError("Invalid state encountered in raise_njit_not_available()")
    else:
        raise AssertionError("numba.njit not available on this system.")


###########################################################################################


try:
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from numba.core.errors import NumbaPerformanceWarning
    from numba.cuda.cudadrv.error import CudaSupportError
    cuda_import_success = True
except ImportError as e:
    print(f'Could not import CUDA.\n{e}')
     # Define mock class to satisfy type checks
    class DeviceNDArray:
        def __init__(self, shape: tuple, strides, dtype, **kwargs) -> None:
            self.size = 0
            self.shape: tuple = shape
            self.strides = strides
            self.dtype=dtype
            raise_cuda_not_available()

        def __setitem__(self, key, value):
            raise_cuda_not_available()

        def __getitem__(self, key) -> 'DeviceNDArray':
            raise_cuda_not_available()
            return self

        def __repr__(self) -> str:
            raise_cuda_not_available()
            return f"MockDeviceNDArray(shape={self.shape}, dtype={self.dtype})"
        
        def reshape(self, shape: tuple|int) -> 'DeviceNDArray':
            raise_cuda_not_available()
            self.shape = shape if isinstance(shape,tuple) else (shape,)
            return self
        
        def copy_to_host(self) -> np.ndarray:
            raise_cuda_not_available()
            return np.empty(shape=(0,))
        

def init_cuda() -> bool:
    if not cuda_import_success:
        return False
    
    try:
        # numba.cuda.is_supported_version() and 
        if not numba.cuda.is_available():
            print('CUDA libraries are installed but this system does not supports CUDA operations.')
            return False
        
        minimal_grid_size = find_minimal_grid_size()
        
        # Test compatibility with np.uint64
        @numba.cuda.jit
        def test_uint64(seed: np.uint64, out: DeviceNDArray):
            idx = numba.cuda.grid(1) # type: ignore
            if idx < out.size:
                out[idx] = seed + idx

        seed_value = np.uint64(42)
        test_grid_size = minimal_grid_size
        test_block_size = 1
        test_size = test_grid_size * test_block_size
        test_data = np.empty(test_size, dtype=np.uint64) 
        d_test_data = numba.cuda.to_device(test_data)
        test_uint64[test_grid_size, test_block_size](seed_value, d_test_data) # type: ignore
        retrieved_data = d_test_data.copy_to_host()
        expected_data = seed_value + np.arange(test_size, dtype=np.uint64)
        if not np.array_equal(retrieved_data, expected_data):
            print(f'CUDA version is not compatible with np.uint64.\n'
                  f'Retrieved: {retrieved_data}\nExpected: {expected_data}')
            return False
        
    except Exception as e:
        print(f'CUDA initialization failed with error: {e}')
        return False
    print('CUDA is available and will be used for GPU operations.')
    return True

    
def print_cuda_device_attributes():
    print('Printing CUDA active device attributes:\n'+'='*50)
    context                       = numba.cuda.current_context()
    context_mem_info              = context.get_memory_info()
    device                        = context.device
    # The number of Streaming Multiprocessors (SMs) affects how many blocks can run concurrently.
    multi_processor_count         = device.MULTIPROCESSOR_COUNT
    min_grid_size                 = find_minimal_grid_size()
    max_grid_dimensions_XYZ       = (device.MAX_GRID_DIM_X,device.MAX_GRID_DIM_Y,device.MAX_GRID_DIM_Z)
    max_block_dimensions_XYZ      = (device.MAX_BLOCK_DIM_X,device.MAX_BLOCK_DIM_Y,device.MAX_BLOCK_DIM_Z)
    globals.max_threads_per_block = device.MAX_THREADS_PER_BLOCK
    globals.warp_size             = device.WARP_SIZE
    max_shared_memory_per_block   = device.MAX_SHARED_MEMORY_PER_BLOCK
    globals.max_registers_per_block = device.MAX_REGISTERS_PER_BLOCK
    memory_bus_width_bits         = device.GLOBAL_MEMORY_BUS_WIDTH
    total_constant_memory         = device.TOTAL_CONSTANT_MEMORY
    memory_clock_rate_MHz         = device.MEMORY_CLOCK_RATE/1000 # from KHz to MHz
    print(f'    Name:                               {device.name.decode("utf-8")}')
    print(f'    Free Memory:                        {context_mem_info.free//1024} [KB]',)
    print(f'    Total Memory:                       {context_mem_info.total//1024} [KB]',)
    print(f'    Compute capability:                 {device.compute_capability[0]}.{device.compute_capability[1]}')
    print(f'    Clock rate:                         {device.CLOCK_RATE/1000:.2f} [MHz]')
    print(f'    Memory clock rate:                  {memory_clock_rate_MHz:.2f} [MHz]')
    print(f'    Memory bus width:                   {memory_bus_width_bits} bits')
    '''
    Multiply by 2: Because GDDR5 memory uses double data rate (DDR).
    Divide by 8: To convert bits to bytes.
    Divide by 1000: To convert MB/s to GB/s.
    '''
    print(f'    Memory band width (theoretical)     {2*memory_clock_rate_MHz * (memory_bus_width_bits/8) / 1000:.2f} [GByte/Sec]')
    print(f'    Number of multiprocessors:          {multi_processor_count}')
    print(f'    Minimal grid size:                  {min_grid_size}')
    print(f'    Maximum grid size:                  {max_grid_dimensions_XYZ}')
    print(f'    Maximum block dimensions:           {max_block_dimensions_XYZ}')
    print(f'    Maximum threads per block:          {globals.max_threads_per_block}')
    print(f'    Warp size:                          {globals.warp_size}')
    print(f'    Maximum shared memory per block:    {max_shared_memory_per_block} [bytes]')
    print(f'    Maximum registers per block:        {globals.max_registers_per_block}')
    print(f'    Total constant memory:              {total_constant_memory} [bytes]')
    print(f'    Asynchronous engine count:          {device.ASYNC_ENGINE_COUNT}')
    print(f'    L2 cache size:                      {device.L2_CACHE_SIZE} [bytes]')
    print(f'    ECC support enabled:                {bool(device.ECC_ENABLED)}')
    # Unavailable GPU attributes
    #print(f'    Maximum threads per multiprocessor: {device.MAX_THREADS_PER_MULTIPROCESSOR}')

'''
The number of CUDA cores per SM varies depending on the GPU architecture:

Kepler (Compute Capability 3.x): 192 CUDA cores per SM
Maxwell (Compute Capability 5.x): 128 CUDA cores per SM
Pascal (Compute Capability 6.x): 64 CUDA cores per SM
Turing (Compute Capability 7.5): 64 CUDA cores per SM
Ampere (Compute Capability 8.x): 128 CUDA cores per SM

Compute Capability	Architecture	CUDA Cores per SM
5.x	Maxwell	128
6.x	Pascal	128
7.0	Volta	64
7.5	Turing	64
8.x	Ampere	64 (FP32 cores)
'''

def find_minimal_grid_size() -> int:
    """
    Dynamically finds the minimal grid size required to avoid NumbaPerformanceWarning.
    
    Returns:
        int: The minimal grid size.
    """
    # Check if the static variable exists
    if hasattr(find_minimal_grid_size, 'previous_result'):
        return find_minimal_grid_size.previous_result  # Return cached value
    

    @numba.cuda.jit
    def dummy_kernel():
        pass

    grid_size = 1  # Start with 1 block

    while True:
        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always", category=NumbaPerformanceWarning)

            try:
                dummy_kernel[grid_size, 1]()  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Kernel launch failed at grid size {grid_size}: {e}")

            # Check if any warnings were captured
            if not any(w.category == NumbaPerformanceWarning for w in captured_warnings):
                break

        grid_size += 1

        # Sanity check to avoid infinite loops
        if grid_size > 1024 * 1024:  # Arbitrary large grid size limit
            raise RuntimeError("Exceeded reasonable grid size without suppressing warnings.")

    # Cache the result using a function attribute
    find_minimal_grid_size.previous_result = grid_size

    return grid_size

def raise_cuda_not_available():
    if globals.cuda_available:
        raise CudaSupportError("Invalid state encountered in raise_cuda_not_available()")
    else:
        raise AssertionError("CUDA is not available on this system.")
    
def cuda_garbage_collect() -> None:
    if not globals.cuda_available:
        raise_cuda_not_available()
    gc.collect(0) # generation 0 should be sufficient to release GPU memory
    numba.cuda.current_context().memory_manager.deallocations.clear()



class HybridArray:
    def __init__(self) -> None:
        self._clear_state()

    def __del__(self) -> None:
        self.close()

    def __enter__(self) -> 'HybridArray':
        return self  # Return the object if needed

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
        
    def close(self, garbage_collect: bool = True) -> None:
        self._clear_state()
        if garbage_collect:
            cuda_garbage_collect()

    def _clear_state(self) -> None:
        # dereferencing self.data (view) BEFORE any self.original_data
        self.data: np.ndarray | DeviceNDArray | None = None
        self.original_numpy_data: np.ndarray | None = None
        self.original_numba_data: DeviceNDArray | None = None

    def realloc_like(self, other: 'HybridArray') -> 'HybridArray':
        return self.realloc(shape=other.shape(), dtype=other.dtype(), use_gpu=other.is_gpu())
    
    def realloc(self,\
                like: 'HybridArray | None' = None,\
                shape: tuple | None = None,\
                dtype: type | None = None,\
                use_gpu: bool | None = None) -> 'HybridArray':
        # determine shape
        if shape is None:
            shape = self.shape() if like is None else like.shape()
        assert shape and len(shape)
        new_size = np.uint64(np.prod(shape))
        assert new_size > 0
        # determine dtype
        self_dtype = self.dtype()
        if dtype is None:
            dtype = self_dtype if like is None else like.dtype()
            assert dtype is not None
        # determine is_gpu
        curr_is_gpu = self.is_gpu()
        if use_gpu is None:
            use_gpu = curr_is_gpu if like is None else like.is_gpu()
        if use_gpu and not globals.cuda_available:
            raise_cuda_not_available()
        original_size = self.original_size()
        if use_gpu == curr_is_gpu and original_size >= new_size and self_dtype == dtype:
            # reusing existing data
            return self.reshape(shape=shape)
        # allocating new data
        self.close()
        if use_gpu:
            self.original_numba_data = numba.cuda.device_array(shape=shape, dtype=dtype)
        else:
            self.original_numpy_data = np.empty(shape=shape, dtype=dtype)
        return self.uncrop()
    
    def to_cpu(self) -> 'HybridArray':
        if self.original_numba_data is None:
            return self
        original_numpy_data = self.original_numba_data.copy_to_host()
        data_shape = self.shape()
        self.close()
        self.original_numpy_data = original_numpy_data
        return self.reshape(data_shape, inplace=True)
    
    def to_gpu(self) -> 'HybridArray':
        if not globals.cuda_available:
            raise_cuda_not_available()
        if self.original_numpy_data is None:
            return self
        original_numba_data = numba.cuda.to_device(self.original_numpy_data)
        data_shape = self.shape()
        self.close()
        self.original_numba_data = original_numba_data
        return self.reshape(data_shape, inplace=True)
    
    def clone_from_numpy(self, data: np.ndarray, use_gpu: bool|None = None) -> 'HybridArray':
        if use_gpu is None:
            use_gpu = self.is_gpu()
        self.close()
        if use_gpu:
            self.original_numba_data = numba.cuda.to_device(data)
        else:
            self.original_numpy_data = np.copy(data)
        return self.uncrop()

    def clone_to_numpy(self) -> np.ndarray:
        if self.original_numpy_data is not None:
            assert isinstance(self.data, np.ndarray)
            return np.copy(self.data)
        if self.original_numba_data is not None:
            assert isinstance(self.data, DeviceNDArray)
            return self.data.copy_to_host()
        return np.empty(shape=(0,))
    
    def numpy(self) -> np.ndarray:
        if self.original_numpy_data is not None:
            assert isinstance(self.data, np.ndarray)
            return self.data
        if self.original_numba_data is not None:
            assert isinstance(self.data, DeviceNDArray)
            return self.data.copy_to_host()
        return np.empty(shape=(0,))
    
    def astype(self, dtype: type, inplace: bool = True, suppress_warning: bool = True) -> 'HybridArray':
        assert inplace
        if self.dtype() == dtype:
            return self
        data_shape = self.shape()
        self.data = None  # erase the reference to the old data
        with warnings.catch_warnings():
            if suppress_warning:
                warnings.simplefilter("ignore", RuntimeWarning)
            if self.original_numpy_data is not None:
                self.original_numpy_data = self.original_numpy_data.astype(dtype=dtype)
            elif self.original_numba_data is not None:
                original_numpy = self.original_numba_data.copy_to_host().astype(dtype=dtype)
                self.close()
                self.original_numba_data = numba.cuda.to_device(original_numpy)
        return self.reshape(data_shape, inplace=inplace)
    
    def reshape(self, shape, inplace: bool = True) -> 'HybridArray':
        assert inplace
        if shape == self.shape():
            return self
        size = np.uint64(np.prod(shape))
        if self.original_numpy_data is not None:
            self.data = self.original_numpy_data.reshape(-1)[:size].reshape(shape)
        elif self.original_numba_data is not None:
            self.data = self.original_numba_data.reshape(-1)[:size].reshape(shape)
        return self
    
    def crop(self, row0: int|np.uint32, row1: int|np.uint32, col0: int|np.uint32, col1: int|np.uint32) -> 'HybridArray':
        assert self.ndim() == 2
        assert isinstance(row0,int) or np.issubdtype(row0,np.integer), f'HybridArray.crop({row0=})'
        assert isinstance(row1,int) or np.issubdtype(row1,np.integer), f'HybridArray.crop({row1=})'
        assert isinstance(col0,int) or np.issubdtype(col0,np.integer), f'HybridArray.crop({col0=})'
        assert isinstance(col1,int) or np.issubdtype(col1,np.integer), f'HybridArray.crop({col1=})'
        if self.original_numpy_data is not None:
            self.data = self.original_numpy_data[row0:row1,col0:col1]
        elif self.original_numba_data is not None:
            self.data = self.original_numba_data[row0:row1,col0:col1]
        return self
    
    def uncrop(self) -> 'HybridArray':
        if self.original_numpy_data is not None:
            self.data = self.original_numpy_data
        elif self.original_numba_data is not None:
            self.data = self.original_numba_data
        return self

    def shape_size(self) -> tuple[tuple,np.uint64]:
        shape = self.shape()
        return shape, np.uint64(np.prod(shape))
    
    def size(self) -> np.uint64:
        shape = self.shape()
        if not shape or len(shape) < 1:
            return np.uint64(0)
        return np.uint64(np.prod(shape))
    
    def ndim(self) -> int:
        shape = self.shape()
        return len(shape)
    
    def shape(self) -> tuple:
        if self.data is None:
            return ()
        return self.data.shape
    
    def nrows(self) -> np.uint32:
        assert self.data is not None
        return np.uint32(self.data.shape[0])
    
    def ncols(self) -> np.uint32:
        assert self.data is not None
        return np.uint32(self.data.shape[1])
    
    def is_gpu(self) -> bool:
        # In case nothing is allocated yet, this function returns default: False
        return self.original_numba_data is not None
    
    def is_cpu(self) -> bool:
        # In case nothing is allocated yet, this function returns default: True
        return self.original_numba_data is None
    
    def dtype(self) -> type|None:
        if self.original_numpy_data is not None:
            return self.original_numpy_data.dtype
        if self.original_numba_data is not None:
            return self.original_numba_data.dtype
        return None
    
    def is_empty(self) -> bool:
        return self.original_numba_data is None and self.original_numpy_data is None
    
    def original_size(self) -> np.uint64:
        original_shape = self.original_shape()
        return np.uint64(np.prod(original_shape))
    
    def original_shape(self) -> tuple:
        if self.original_numpy_data is not None:
            return self.original_numpy_data.shape
        if self.original_numba_data is not None:
            return self.original_numba_data.shape
        return (0,)

    def gpu_data(self) -> DeviceNDArray:
        assert isinstance(self.data, DeviceNDArray)
        return self.data
        
    def select_row(self, i: int|np.uint32) -> 'HybridArray':
        if self.original_numpy_data is not None:
            self.data = self.original_numpy_data[i:i+1, :]
        elif self.original_numba_data is not None:
            self.data = self.original_numba_data[i:i+1, :]
        return self
    
    def select_col(self, i: int|np.uint32) -> 'HybridArray':
        if self.original_numpy_data is not None:
            self.data = self.original_numpy_data[:,i:i+1]
        elif self.original_numba_data is not None:
            self.data = self.original_numba_data[:,i:i+1]
        return self
    
    def swap(self, other: 'HybridArray') -> None:
        self.original_numba_data, other.original_numba_data = other.original_numba_data, self.original_numba_data
        self.original_numpy_data, other.original_numpy_data = other.original_numpy_data, self.original_numpy_data
        self.data, other.data = other.data, self.data
        
    def gpu_grid_block2D_square_shapes(self,\
                                        registers_per_thread: int|None = None,\
                                        debug: int|None = None) -> tuple[tuple, tuple]:
        if not globals.cuda_available:
            raise_cuda_not_available()
        block_size = self.calc_block_size(registers_per_thread = registers_per_thread, debug = debug)
        # priority to reduce rows per block over columns
        block_shape_y = min(np.uint32(np.sqrt(block_size)),self.nrows())
        block_shape_x = block_size // block_shape_y
        return self.get_grid_from_2D_block(block_shape_y=block_shape_y, block_shape_x=block_shape_x, debug=debug)
    
    def gpu_grid_block2D_columns_shapes(self,\
                              registers_per_thread: int|None = None,\
                              debug: int|None = None) -> tuple[tuple, tuple]:
        if not globals.cuda_available:
            raise_cuda_not_available() 
        block_size = self.calc_block_size(registers_per_thread = registers_per_thread,\
                                          debug = debug)
        nrows = self.nrows()
        if block_size <= nrows:
            # small block less than a single column
            times = (nrows + block_size-1) // block_size
            block_shape_y = calc_block_size(nrows // times)
            block_shape_x = 1
        else:
            # each block contains several columns
            block_shape_y = nrows
            block_shape_x = block_size // block_shape_y
        return self.get_grid_from_2D_block(block_shape_y=block_shape_y, block_shape_x=block_shape_x, debug=debug)
    
    def gpu_grid_block1D_rows_shapes(self) -> tuple[np.uint32, np.uint32]:
        return simple_data_size_to_grid_block_1D(self.nrows())

    def gpu_grid_block1D_cols_shapes(self) -> tuple[np.uint32, np.uint32]:
        return simple_data_size_to_grid_block_1D(self.ncols())

    def calc_block_size(self,
                        registers_per_thread: int|None = None,\
                        debug: int|None = None) -> np.uint32:
        if not globals.cuda_available:
            raise_cuda_not_available()  
        if debug is None:
            debug = int(globals.grid_block_shape_debug)
        data_shape = self.shape()
        data_size = data_shape[0]*data_shape[1]
        block_size = calc_block_size(data_size=data_size, registers_per_thread=registers_per_thread)
        if debug > 0:
            print(f'{data_shape=} --> {block_size=}')
        return block_size
    
    def get_grid_from_2D_block(self, block_shape_y, block_shape_x,\
                               debug: int|None = None) -> tuple[tuple, tuple]:
        data_shape = self.shape()
        grid_shape_y = (data_shape[0] + block_shape_y - 1) // block_shape_y
        grid_shape_x = (data_shape[1] + block_shape_x - 1) // block_shape_x
        grid_shape = (grid_shape_y, grid_shape_x)
        block_shape = (block_shape_y, block_shape_x)
        if debug is None:
            debug = int(globals.grid_block_shape_debug)
        if debug > 0:
            print(f'grid block shapes: {data_shape=} --> {grid_shape=}  {block_shape=}', flush=True)
        return grid_shape, block_shape


def simple_data_size_to_grid_block_1D(\
        data_size: int|np.uint64|np.uint32,\
        registers_per_thread: int|None = None,\
        debug: int|None = None) -> tuple[np.uint32, np.uint32]:
    if not globals.cuda_available:
        raise_cuda_not_available()  
    if debug is None:
        debug = int(globals.grid_block_shape_debug)
    block_size = calc_block_size(data_size=data_size, registers_per_thread=registers_per_thread)
    grid_size = np.uint32((data_size + block_size - 1) // block_size)
    if debug > 0:
        print(f'simple_data_size_to_grid_block_1D({data_size=}) --> {grid_size=} {block_size=}')
    return grid_size, block_size

def calc_block_size(data_size: int|np.uint64|np.uint32,\
                    registers_per_thread: int|np.uint64|np.uint32|None = None) -> np.uint32:
    if not globals.cuda_available:
        raise_cuda_not_available()    
    assert data_size > 0
    max_threads_per_block = [globals.max_threads_per_block, data_size]
    if registers_per_thread is not None:
        assert registers_per_thread > 0
        max_threads_per_block.append(globals.max_registers_per_block // registers_per_thread)
    block_size = min(max_threads_per_block)
    if block_size > globals.warp_size:
        block_size -= block_size % globals.warp_size
    assert block_size > 0
    return np.uint32(block_size)



############################################################3
globals.cpu_njit_num_threads = init_njit()

def use_njit(**kwargs) -> bool:
    if globals.cpu_njit_num_threads < 1:
        return False
    val = kwargs.get('use_njit', None)
    return val is None or val

globals.cuda_available = init_cuda()

if not globals.cuda_available:
    print("Numba or CUDA is not available. GPU operations will be disabled.")
else:    
    print_cuda_device_attributes()
