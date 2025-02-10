from python.hpc import globals

if not globals.cpu_njit_num_threads:
    # Mock API
    from python.hpc import raise_njit_not_available
    def higher_criticism_stable_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def higher_criticism_unstable_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
    def berk_jones_cpu_njit(**kwargs) -> None: # type: ignore
        raise_njit_not_available()
else:
    import math
    import numpy as np
    import numba

    @numba.njit(parallel=True)
    def higher_criticism_stable_cpu_njit(\
            sorted_p_values_input_output: np.ndarray) -> None:
        _, N = sorted_p_values_input_output.shape
        p_base_line = np.arange(1,N+1, dtype=np.float64).reshape(1,-1)/(N+1)
        stdev = np.sqrt(p_base_line)*np.sqrt(1-p_base_line)
        sqrtN = math.sqrt(np.float64(N))
        sorted_p_values_input_output[:] = sqrtN*((sorted_p_values_input_output - p_base_line)/stdev)
        
    @numba.njit(parallel=True)
    def higher_criticism_unstable_cpu_njit(\
            sorted_p_values_input_output: np.ndarray) -> None:
        _, N = sorted_p_values_input_output.shape
        p_base_line = np.arange(1,N+1, dtype=np.float64).reshape(1,-1)/(N+1)
        stdev = np.sqrt(sorted_p_values_input_output)*np.sqrt(1-sorted_p_values_input_output)
        sqrtN = math.sqrt(np.float64(N))
        sorted_p_values_input_output[:] = sqrtN*((sorted_p_values_input_output - p_base_line)/stdev)

    @numba.njit(parallel=True)
    def berk_jones_cpu_njit(\
            sorted_p_values_input_output: np.ndarray) -> None:
        num_monte, N = sorted_p_values_input_output.shape
        for row in numba.prange(num_monte):
            data_row = sorted_p_values_input_output[row,:]
            for col in range(N):
                a = np.float64(col+1)
                b = np.float64(N-col)
                data_row[col] = beta_cdf_cpu_njit(data_row[col],a,b, np.uint32(2000), np.float64(1e-20), np.float64(1e-30))

    @numba.njit(parallel=False)
    def beta_cdf_cpu_njit(\
        x: np.float64, a: np.float64, b: np.float64,\
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
        if x > (a+one)/(a+b+two):
            # Use the fact that beta is symmetrical.
            return one-beta_cdf_cpu_njit(one-x,b,a,max_iter,eps,tiny) 
        # Find the first part before the continued fraction.
        lbeta_ab = math.lgamma(a)+math.lgamma(b)-math.lgamma(a+b)
        front = math.exp(math.log(x)*a+math.log(one-x)*b-lbeta_ab) / a
        
        # Use Lentz's algorithm to evaluate the continued fraction.
        f = one
        c = one
        d = zero

        for i in range(max_iter+1):
            m = i // 2
            a2m = a + two * m
            if i == 0:
                numerator = one # First numerator is 1.0
            elif i % 2 == 0:
                numerator = (m*(b-m)*x)/((a2m-one)*a2m) # Even term
            else:
                numerator = -((a+m)*(a+b+m)*x)/(a2m*(a2m+one)) # Odd term
            # Do an iteration of Lentz's algorithm
            d = one + numerator * d
            if -tiny < d < zero:
                d = -tiny
            elif zero <= d < tiny:
                d = tiny
            d = one/d
            if -tiny < c < zero:
                c = -tiny
            elif zero <= c < tiny:
                c = tiny
            c = one + numerator / c
            cd = c*d
            f *= cd

            # Check for stop
            if abs(one-cd) < eps:
                return front * (f-one)
        return -one # did not converge!!!

