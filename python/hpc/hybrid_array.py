import warnings
import numpy as np

from hpc import globals, raise_cuda_not_available, cuda_garbage_collect, calc_block_size, simple_data_size_to_grid_block_1D

if not globals.cuda_available:
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
else:
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray


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
        # before clear state
        is_gpu = self.is_gpu()
        self._clear_state()
        # after clear state
        if garbage_collect and is_gpu:
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

