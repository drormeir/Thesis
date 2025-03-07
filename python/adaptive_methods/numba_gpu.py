from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def higher_criticism_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def higher_criticism_unstable_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def berk_jones_legacy_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def berk_jones_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def calc_lgamma_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import os
    import math
    import numpy as np
    import cupy
    import numba
    import numba.cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray
    from python.hpc import HybridArray

    @numba.cuda.jit(device=False)
    def higher_criticism_gpu(\
        sorted_p_values_input_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        sqrtN = math.sqrt(np.float64(N))
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                mean_base_line = (ind_col+1)/np.float64(N+1)
                stdev_base_line = math.sqrt(mean_base_line)*math.sqrt(1-mean_base_line)
                row[ind_col] = sqrtN*((row[ind_col]-mean_base_line)/stdev_base_line)

    @numba.cuda.jit(device=False)
    def higher_criticism_unstable_gpu(\
        sorted_p_values_input_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        sqrtN = math.sqrt(np.float64(N))
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                mean_base_line = (ind_col+1)/np.float64(N+1)
                stdev_base_line = math.sqrt(row[ind_col])*math.sqrt(1-row[ind_col])
                row[ind_col] = sqrtN*((row[ind_col]-mean_base_line)/stdev_base_line)

    @numba.cuda.jit(device=False)
    def berk_jones_legacy_gpu_max_iter(\
        sorted_p_values_input_output: DeviceNDArray,\
        lgamma_cache: DeviceNDArray,\
        max_iter_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            max_iter_output_row = max_iter_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                a = np.uint32(ind_col+1)
                b = np.uint32(N-ind_col)
                row[ind_col] = beta_cdf_legacy_gpu_max_iter(lgamma_cache,row[ind_col],a,b,\
                                            np.uint32(2000), np.float64(1e-20), np.float64(1e-30),\
                                            max_iter_output_row)

    @numba.cuda.jit(device=True)
    def beta_cdf_legacy_gpu_max_iter(\
        lgamma_cache: DeviceNDArray,\
        x: np.float64, int_a: np.uint32, int_b: np.uint32,\
        max_iter: np.uint32,\
        eps: np.float64, tiny: np.float64,\
        max_iter_output: DeviceNDArray) -> np.float64:
        """
        Returns the continued-fraction part of I_x(a,b), i.e. the factor
        that multiplies the 'front' = (x^a * (1-x)^b) / [a * B(a,b)].
        For large x (or after symmetry), the incomplete Beta integral is:
            I_x(a,b) = front * beta_cdf(a,b,x).
        Lentz's method is used to evaluate the CF.
        """
        # reference: https://github.com/codeplea/incbeta/blob/master/incbeta.c
        # The continued fraction converges nicely for x < (a+1)/(a+b+2)
        zero = np.float64(0.0)
        one = np.float64(1.0)
        two = np.float64(2.0)
        # Use the fact that beta is symmetrical.
        symmetry_flag = bool(x > (int_a+one)/(int_a+int_b+two))
        symmetry_flag1 = np.float64(symmetry_flag)
        symmetry_flag0 = np.float64(not symmetry_flag)
        a = int_a * symmetry_flag0 + int_b * symmetry_flag1
        b = int_b * symmetry_flag0 + int_a * symmetry_flag1
        x = x * symmetry_flag0 + (one-x)*symmetry_flag1
        # Find the first part before the continued fraction.
        lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b]-lgamma_cache[int_a+int_b]
        front = math.exp(math.log(x)*a+math.log(one-x)*b-lbeta_ab) / a
        # Use Lentz's algorithm to evaluate the continued fraction.
        f = one
        c = one
        d = zero
        iter_stop = max_iter+1
        for i in range(max_iter+1):
            m = np.float64(i // 2)
            a2m = a + two * m
            if i == 0:
                numerator = one # First numerator is 1.0
            elif i % 2 == 0:
                numerator = (m*(b-m)*x)/((a2m-one)*a2m) # Even term
            else:
                numerator = -((a+m)*(a+b+m)*x)/(a2m*(a2m+one)) # Odd term
            # Do an iteration of Lentz's algorithm
            d = one + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= zero else -tiny
            d = one/d
            if math.fabs(c) < tiny:
                c = tiny if c >= zero else -tiny
            c = one + numerator / c
            cd = c*d
            f *= cd
            # Check for stop
            if abs(one-cd) < eps:
                iter_stop = np.uint32(i)
                break
            pass

        if iter_stop <= max_iter:
            ret = front * (f-one)
            # Use the fact that beta is symmetrical.
            ret = ret*symmetry_flag0 + (one-ret)*symmetry_flag1
        else:
            ret = np.float64(-1.0) # did not converge!!!
        
        # output max iter. recall that: int_a == ind_col+1
        max_iter_output[int_a-1] = iter_stop
        return ret

    @numba.cuda.jit(device=False)
    def berk_jones_gpu_max_iter(\
        sorted_p_values_input_output: DeviceNDArray,\
        lgamma_cache: DeviceNDArray,\
        max_iter_output: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            max_iter_output_row = max_iter_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                a = np.uint32(ind_col+1)
                b = np.uint32(N-ind_col)
                row[ind_col] = beta_cdf_gpu_max_iter(lgamma_cache,row[ind_col],a,b,\
                                            np.uint32(2000), np.float64(1e-20), np.float64(1e-30),\
                                            max_iter_output_row)

    @numba.cuda.jit(device=True)
    def beta_cdf_gpu_max_iter(\
        lgamma_cache: DeviceNDArray,\
        x: np.float64, int_a: np.uint32, int_b: np.uint32,\
        max_iter: np.uint32,\
        eps: np.float64, tiny: np.float64,\
        max_iter_output: DeviceNDArray) -> np.float64:
        """
        Returns the continued-fraction part of I_x(a,b), i.e. the factor
        that multiplies the 'front' = (x^a * (1-x)^b) / [a * B(a,b)].
        For large x (or after symmetry), the incomplete Beta integral is:
            I_x(a,b) = front * beta_cdf(a,b,x).
        Lentz's method is used to evaluate the CF.
        """
        # reference: https://github.com/codeplea/incbeta/blob/master/incbeta.c
        # The continued fraction converges nicely for x < (a+1)/(a+b+2)
        one = np.float64(1.0)
        two = np.float64(2.0)
        # Use the fact that beta is symmetrical.
        symmetry_flag = bool(x > (int_a+one)/(int_a+int_b+two))
        symmetry_flag1 = np.float64(symmetry_flag)
        symmetry_flag0 = np.float64(not symmetry_flag)
        a = int_a * symmetry_flag0 + int_b * symmetry_flag1
        b = int_b * symmetry_flag0 + int_a * symmetry_flag1
        x = x * symmetry_flag0 + (one-x)*symmetry_flag1
        # Find the first part before the continued fraction.
        lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b]-lgamma_cache[int_a+int_b]
        front = math.exp(math.log(x)*a+math.log(one-x)*b-lbeta_ab) / a
        # Use Lentz's algorithm to evaluate the continued fraction.
        # iter 0
        d = np.float64(1.0)
        c = np.float64(2.0)
        f = np.float64(2.0)
        iter_stop = max_iter+1
        for m in range(max_iter+1):
            # Odd term
            a2m = a + 2 * m
            numerator = -((a+m)*(a+b+m)*x)/(a2m*(a2m+1))
            # Do an iteration of Lentz's algorithm
            d = 1 + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= 0.0 else -tiny
            d = 1/d
            if math.fabs(c) < tiny:
                c = tiny if c >= 0.0 else -tiny
            c = 1 + numerator / c
            f = c*d
            # Even term
            numerator = ((m+1)*(b-m-1)*x)/((a2m+1)*(a2m+2))
            # Do an iteration of Lentz's algorithm
            d = 1 + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= 0.0 else -tiny
            d = 1/d
            if math.fabs(c) < tiny:
                c = tiny if c >= 0.0 else -tiny
            c = 1 + numerator / c
            cd = c*d
            # Check for stop
            f *= cd
            if abs(1-cd) < eps:
                iter_stop = np.uint32(m)
                break
            pass

        if iter_stop <= max_iter:
            ret = front * (f-one)
            # Use the fact that beta is symmetrical.
            ret = ret*symmetry_flag0 + (one-ret)*symmetry_flag1
        else:
            ret = np.float64(-1.0) # did not converge!!!
        
        # output max iter. recall that: int_a == ind_col+1
        max_iter_output[int_a-1] = iter_stop
        return ret


    @numba.cuda.jit(device=False)
    def berk_jones_legacy_gpu(\
        sorted_p_values_input_output: DeviceNDArray,\
        lgamma_cache: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                a = np.uint32(ind_col+1)
                b = np.uint32(N-ind_col)
                row[ind_col] = beta_cdf_legacy_gpu(lgamma_cache,row[ind_col],a,b,\
                                            np.uint32(2000), np.float64(1e-20), np.float64(1e-30))

    @numba.cuda.jit(device=True)
    def beta_cdf_legacy_gpu(\
        lgamma_cache: DeviceNDArray,\
        x: np.float64, int_a: np.uint32, int_b: np.uint32,\
        max_iter: np.uint32,\
        eps: np.float64, tiny: np.float64) -> np.float64:
        """
        Returns the continued-fraction part of I_x(a,b), i.e. the factor
        that multiplies the 'front' = (x^a * (1-x)^b) / [a * B(a,b)].
        For large x (or after symmetry), the incomplete Beta integral is:
            I_x(a,b) = front * beta_cdf(a,b,x).
        Lentz's method is used to evaluate the CF.
        """
        # reference: https://github.com/codeplea/incbeta/blob/master/incbeta.c
        # The continued fraction converges nicely for x < (a+1)/(a+b+2)
        zero = np.float64(0.0)
        one = np.float64(1.0)
        two = np.float64(2.0)
        # Use the fact that beta is symmetrical.
        symmetry_flag = bool(x > (int_a+one)/(int_a+int_b+two))
        symmetry_flag1 = np.float64(symmetry_flag)
        symmetry_flag0 = np.float64(not symmetry_flag)
        a = int_a * symmetry_flag0 + int_b * symmetry_flag1
        b = int_b * symmetry_flag0 + int_a * symmetry_flag1
        x = x * symmetry_flag0 + (one-x)*symmetry_flag1
        # Find the first part before the continued fraction.
        lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b]-lgamma_cache[int_a+int_b]
        front = math.exp(math.log(x)*a+math.log(one-x)*b-lbeta_ab) / a
        # Use Lentz's algorithm to evaluate the continued fraction.
        f = one
        c = one
        d = zero
        iter_stop = max_iter+1
        for i in range(max_iter+1):
            m = np.float64(i // 2)
            a2m = a + two * m
            if i == 0:
                numerator = one # First numerator is 1.0
            elif i % 2 == 0:
                numerator = (m*(b-m)*x)/((a2m-one)*a2m) # Even term
            else:
                numerator = -((a+m)*(a+b+m)*x)/(a2m*(a2m+one)) # Odd term
            # Do an iteration of Lentz's algorithm
            d = one + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= zero else -tiny
            d = one/d
            if math.fabs(c) < tiny:
                c = tiny if c >= zero else -tiny
            c = one + numerator / c
            cd = c*d
            f *= cd
            # Check for stop
            if abs(one-cd) < eps:
                iter_stop = i
                break
            pass

        if iter_stop <= max_iter:
            ret = front * (f-one)
            # Use the fact that beta is symmetrical.
            ret = ret*symmetry_flag0 + (one-ret)*symmetry_flag1
        else:
            ret = np.float64(-1.0) # did not converge!!!
        return ret
    
    def create_berk_jones_kernel(berk_jones_kernel_code_path: str = os.path.join('c','berk_jones.cu'), include_path: bool = False):
        with open(berk_jones_kernel_code_path, 'r') as f:
            berk_jones_kernel_code = f.read()
        # Read the INCLUDE environment variable (Windows typically uses semicolons).
        if include_path:
            include_env = os.environ.get('INCLUDE', '')
            include_paths = [os.path.normpath(p.strip()) for p in include_env.split(';') if p.strip()]

            # Convert each path to a normalized OS-specific form, then build a -I argument.
            include_options = [f'-I"{path}"' for path in include_paths]
            print(f'Adding include paths:\n{include_options}')
        else:
            include_options = []
        # Create RawModule, adding the normalized include paths to the options.
        berk_jones_module = cupy.RawModule(
            code=berk_jones_kernel_code,
            name_expressions=['berk_jones_kernel'],
            options=tuple(['--std=c++11'] + include_options)    # or c++14/17
        )
        # Retrieve the kernel
        berk_jones_kernel = berk_jones_module.get_function('berk_jones_kernel')
        return berk_jones_kernel

    try:
        print('Compiling Berk Jones for CUDA...')
        berk_jones_kernel = create_berk_jones_kernel()
        print('Compiling Berk Jones for CUDA... Done!')
    except Exception as e:
        print(f'Could not compile Berk Jones in CUDA:\n{e}')
        berk_jones_kernel = None


    # need HybridArray for calculating block size
    def berk_jones_gpu(\
            sorted_p_values_input_output: HybridArray,\
            lgamma_cache: HybridArray|None = None,\
            **kwargs) -> None:
        N = sorted_p_values_input_output.ncols()
        lgamma_len = N+2 # because I want to calc from lgamma(1) to lgamma(N+1) inclusive and put them in the same indexes
        local_lgamma = lgamma_cache is None
        if local_lgamma:
            lgamma_cache = HybridArray().realloc(shape=(lgamma_len,), dtype=np.float64, use_gpu=True)
            recalc_lgamma = True
        else:
            recalc_lgamma = lgamma_len <= lgamma_cache.size()
            lgamma_cache.realloc(shape=(lgamma_len,))
        if recalc_lgamma:
            calc_lgamma_gpu[1, 1](lgamma_cache.gpu_data()) # type: ignore
        debug = kwargs.get('debug_berk_jones_gpu_block_size', None)
        if berk_jones_kernel is None:
            grid_shape, block_shape =\
                sorted_p_values_input_output.gpu_grid_block2D_columns_shapes(registers_per_thread=100, debug=debug)
            berk_jones_gpu_numba[grid_shape, block_shape](sorted_p_values_input_output.gpu_data(), lgamma_cache.gpu_data()) # type: ignore
        else:
            grid_shape, block_shape =\
                sorted_p_values_input_output.gpu_grid_block2D_columns_shapes(registers_per_thread=64, debug=debug)
            berk_jones_gpu_execute(grid_shape, block_shape, sorted_p_values_input_output.gpu_data(), lgamma_cache.gpu_data())
        if local_lgamma:
            lgamma_cache.close()

    @numba.cuda.jit(device=False)
    def berk_jones_gpu_numba(\
        sorted_p_values_input_output: DeviceNDArray,\
        lgamma_cache: DeviceNDArray) -> None:
        # Get the 2D indices of the current thread within the grid
        ind_row0, ind_col0 = numba.cuda.grid(2) # type: ignore
        # Calculate the strides
        row_stride, col_stride = numba.cuda.gridsize(2) # type: ignore
        num_monte, N = sorted_p_values_input_output.shape
        for ind_row in range(ind_row0, num_monte, row_stride):
            row = sorted_p_values_input_output[ind_row]
            for ind_col in range(ind_col0, N, col_stride):
                a = np.uint32(ind_col+1)
                b = np.uint32(N-ind_col)
                row[ind_col] = beta_cdf_gpu(lgamma_cache,row[ind_col],a,b, np.uint32(1000), np.float64(1e-20), np.float64(1e-30))

    @numba.cuda.jit(device=True)
    def beta_cdf_gpu(\
        lgamma_cache: DeviceNDArray,\
        x: np.float64, int_a: np.uint32, int_b: np.uint32,\
        max_iter: np.uint32,\
        eps: np.float64, tiny: np.float64) -> np.float64:
        """
        Returns the continued-fraction part of I_x(a,b), i.e. the factor
        that multiplies the 'front' = (x^a * (1-x)^b) / [a * B(a,b)].
        For large x (or after symmetry), the incomplete Beta integral is:
            I_x(a,b) = front * beta_cdf(a,b,x).
        Lentz's method is used to evaluate the CF.
        """
        # reference: https://github.com/codeplea/incbeta/blob/master/incbeta.c
        # The continued fraction converges nicely for x < (a+1)/(a+b+2)
        # Use the fact that beta is symmetrical.
        symmetry_flag = bool(x > (int_a+1)/np.float64(int_a+int_b+2))
        symmetry_flag1 = np.float64(symmetry_flag)
        symmetry_flag0 = np.float64(not symmetry_flag)
        a = int_a * symmetry_flag0 + int_b * symmetry_flag1
        b = int_b * symmetry_flag0 + int_a * symmetry_flag1
        x = x * symmetry_flag0 + (1-x)*symmetry_flag1
        # Find the first part before the continued fraction.
        lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b]-lgamma_cache[int_a+int_b]
        front = math.exp(math.log(x)*a+math.log(1-x)*b-lbeta_ab) / a
        # Use Lentz's algorithm to evaluate the continued fraction.
        # iter 0
        d = np.float64(1.0)
        c = np.float64(2.0)
        f = np.float64(2.0)
        for m in range(max_iter):
            # Odd term
            a2m = a + 2 * m
            numerator = -((a+m)*(a+b+m)*x)/(a2m*(a2m+1))
            # Do an iteration of Lentz's algorithm
            d = 1 + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= 0.0 else -tiny
            d = 1/d
            if math.fabs(c) < tiny:
                c = tiny if c >= 0.0 else -tiny
            c = 1 + numerator / c
            f *= c*d
            # Even term
            numerator = ((m+1)*(b-m-1)*x)/((a2m+1)*(a2m+2))
            # Do an iteration of Lentz's algorithm
            d = 1 + numerator * d
            if math.fabs(d) < tiny:
                d = tiny if d >= 0.0 else -tiny
            d = 1/d
            if math.fabs(c) < tiny:
                c = tiny if c >= 0.0 else -tiny
            c = 1 + numerator / c
            cd = c*d
            # Check for stop
            f *= cd
            if abs(1-cd) < eps:
                ret = front * (f-1)
                # Use the fact that beta is symmetrical.
                return ret*symmetry_flag0 + (1-ret)*symmetry_flag1
        return np.float64(-1) # did not converge!!!


    @numba.cuda.jit(device=False)
    def calc_lgamma_gpu(lgamma_cache: DeviceNDArray) -> None:
        N = lgamma_cache.shape[0]
        lgamma_cache[0] = np.float64(0.0) # dummy
        lgamma_cache[1] = np.float64(0.0) # lgamma(1) = log(0!)
        for ind_col in range(2,N):
            lgamma_cache[ind_col] = lgamma_cache[ind_col-1] + math.log(np.float64(ind_col-1))




    def berk_jones_gpu_execute(grid_shape: tuple, block_shape: tuple,\
                               sorted_p_values_input_output: DeviceNDArray,\
                               lgamma_cache: DeviceNDArray) -> None:
        if berk_jones_kernel is None:
            raise RuntimeError ("berk_jones_gpu_execute: berk_jones_kernel does not exists.")
        cupy_sorted_p = cupy.asarray(sorted_p_values_input_output)
        cupy_lgamma   = cupy.asarray(lgamma_cache)        
        num_monte, N = sorted_p_values_input_output.shape
        max_iter = np.uint32(150)
        eps = np.float64(1e-20)
        tiny = np.float64(1e-30)
        berk_jones_kernel(grid_shape, block_shape, args=(cupy_sorted_p, cupy_lgamma, np.uint32(num_monte), np.uint32(N), max_iter, eps, tiny))

