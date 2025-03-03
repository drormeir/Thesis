//#include <cmath>
#include <cuda_runtime.h>
#include <device_functions.h>

// -------------------------------------------------
// Device function implementing your Lentz-based beta_cdf
// -------------------------------------------------
__device__ double beta_cdf_gpu(
    const double* __restrict__ lgamma_cache,
    double x,
    unsigned int int_a,
    unsigned int int_b,
    unsigned int max_iter,
    double eps,
    double tiny
){
    // Use symmetry to reduce domain
    double a_d = (double)int_a;
    double b_d = (double)int_b;
    double threshold = (a_d + 1.0) / (a_d + b_d + 2.0);
    bool symmetry_flag = (x > threshold);
    double symmetry_flag1 = symmetry_flag ? 1.0 : 0.0;
    double symmetry_flag0 = 1.0 - symmetry_flag1;

    double a = a_d * symmetry_flag0 + b_d * symmetry_flag1;
    double b = b_d * symmetry_flag0 + a_d * symmetry_flag1;
    x = x * symmetry_flag0 + (1.0 - x) * symmetry_flag1;

    // Compute log Beta(a,b) from cache
    double lbeta_ab = lgamma_cache[int_a] + lgamma_cache[int_b] - lgamma_cache[int_a + int_b];
    double front = exp(log(x)*a + log(1.0 - x)*b - lbeta_ab) / a;

    // Lentz's algorithm
    // iter: 0
    double d = 1.0;
    double c = 2.0;
    double f = 2.0;

    for(unsigned int m = 0; m < max_iter; m++){
        // Odd term
        double a2m = a + 2.0 * (double)m;
        double numerator = -((a + (double)m)*(a + b + (double)m)*x) / (a2m*(a2m+1.0));

        d = 1.0 + numerator * d;
        if(fabs(d) < tiny){
            if(d >= 0.0) d =  tiny;
            else         d = -tiny;
        }
        d = 1.0 / d;

        if(fabs(c) < tiny){
            if(c >= 0.0) c =  tiny;
            else         c = -tiny;
        }
        c = 1.0 + numerator / c;
        double cd = c * d;
        f *= cd;

        // Even term
        numerator = ((double)(m+1) * (b - (double)(m+1)) * x) / ((a2m+1.0)*(a2m+2.0));
        d = 1.0 + numerator * d;
        if(fabs(d) < tiny){
            d = d >= 0.0 ? tiny : -tiny;
        }
        d = 1.0 / d;

        if(fabs(c) < tiny){
            c = c >= 0.0 ? tiny : -tiny;
        }
        c = 1.0 + numerator / c;
        cd = c * d;
        f *= cd;

        // Check for convergence
        if(fabs(1.0 - cd) < eps){
            double ret = front * (f - 1.0);
            // Apply symmetry
            return ret * symmetry_flag0 + (1.0 - ret) * symmetry_flag1;
        }
    }

    // Did not converge
    return -1.0;
}

// -------------------------------------------------
// Global kernel that iterates over the 2D grid
// -------------------------------------------------
extern "C" __global__
void berk_jones_kernel(
    double* __restrict__ sorted_p_values,
    const double* __restrict__ lgamma_cache,
    unsigned int num_monte,
    unsigned int N,
    unsigned int max_iter,
    double eps,
    double tiny
){
    // 2D thread indexing similar to numba.cuda.grid(2)
    unsigned int ind_row0 = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int ind_col0 = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row_stride = blockDim.y * gridDim.y;
    unsigned int col_stride = blockDim.x * gridDim.x;

    for(unsigned int row = ind_row0; row < num_monte; row += row_stride){
        double* __restrict__ values = sorted_p_values + (row * (long)N);
        for(unsigned int col = ind_col0; col < N; col += col_stride){
            unsigned int a = col + 1;
            unsigned int b = (unsigned int)(N - col);
            // Compute the beta cdf
            values[col] = beta_cdf_gpu(lgamma_cache, values[col], a, b, max_iter, eps, tiny);
        }
    }
}
