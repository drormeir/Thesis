from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_cuda_not_available
    def higher_criticism_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def higher_criticism_unstable_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def berk_jones_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
    def calc_lgamma_gpu(**kwargs) -> None: # type: ignore
        raise_cuda_not_available()
else:
    import math
    import numpy as np
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
    def berk_jones_gpu(\
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
                row[ind_col] = beta_cdf_gpu(lgamma_cache,row[ind_col],a,b, np.uint32(2000), np.float64(1e-20), np.float64(1e-30))

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
        zero = np.float64(0.0)
        one = np.float64(1.0)
        two = np.float64(2.0)
        a = np.float64(int_a)
        b = np.float64(int_b)
        if x > (a+one)/(a+b+two):
            # Use the fact that beta is symmetrical.
            return one-beta_cdf_gpu(lgamma_cache,one-x,int_b,int_a,max_iter,eps,tiny) 
        # Find the first part before the continued fraction.
        lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b]-lgamma_cache[int_a+int_b]
        front = math.exp(math.log(x)*a+math.log(one-x)*b-lbeta_ab) / a
        # Use Lentz's algorithm to evaluate the continued fraction.
        f = one
        c = one
        d = zero
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
                return front * (f-one)
        return -one # did not converge!!!

    @numba.cuda.jit(device=False)
    def calc_lgamma_gpu(lgamma_cache: DeviceNDArray) -> None:
        N = lgamma_cache.shape[0]
        lgamma_cache[0] = np.float64(0.0) # dummy
        lgamma_cache[1] = np.float64(0.0) # lgamma(1) = log(0!)
        for ind_col in range(2,N):
            lgamma_cache[ind_col] = lgamma_cache[ind_col-1] + math.log(np.float64(ind_col-1))
