"""CUDA kernels for CuPy

Description:
    This module contains CUDA kernel versions of internal routines for performing optimized mathematical
    operations. It includes functions for calculating position-position Hessian matrices and perturbation
    matrices derived from coordinate and covariance data, accelerated using CUDA. The computations are
    implemented as CUDA kernels and are intended for internal use within the reForge workflow.

Usage Example:
    >>> import cupy as cp
    >>> import numpy as np
    >>> from rcmath_cuda import calculate_hessian_cuda, hessian_cuda, perturbation_matrix_cuda, td_perturbation_matrix_cuda
    >>> n = 10
    >>> # Create random coordinate data on the GPU:
    >>> x = cp.asarray(np.random.rand(n))
    >>> y = cp.asarray(np.random.rand(n))
    >>> z = cp.asarray(np.random.rand(n))
    >>> hess = calculate_hessian_cuda(n, x, y, z, 1.2, 1000.0, 0)
    >>>
    >>> # Alternatively, if the coordinates are stored in an (n x 3) array:
    >>> vec = cp.asarray(np.random.rand(n, 3))
    >>> hess2 = hessian_cuda(vec, 1.2, 1000.0, 0)
    >>>
    >>> # Compute a perturbation matrix from a covariance matrix:
    >>> cov = cp.asarray(np.random.rand(3*n, 3*n))
    >>> pert = perturbation_matrix_cuda(cov)
    >>>
    >>> # Compute a block-wise perturbation matrix:
    >>> td_pert = td_perturbation_matrix_cuda(cov)

Requirements:
    - Python 3.x
    - NumPy
    - CuPy
    - reForge utilities (timeit, memprofit)

Author: Your Name
Date: YYYY-MM-DD
"""

import cupy as cp
import numpy as np
from math import ceil


# ---------------------------------------------------------------------
# CUDA Kernel for calculating Hessian from a coordinate matrix.
# (Each residue’s coordinates are stored consecutively in a (n x 3) array.)
# ---------------------------------------------------------------------
hessian_kernel_code = r"""
extern "C" __global__
void hessian_kernel(const int n, const double cutoff, const double spring_constant, const int dd,
                    const double *vec, double *hessian, const int hessian_size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < n && j < n && i != j) {
         double dx = vec[i*3 + 0] - vec[j*3 + 0];
         double dy = vec[i*3 + 1] - vec[j*3 + 1];
         double dz = vec[i*3 + 2] - vec[j*3 + 2];
         double r = sqrt(dx*dx + dy*dy + dz*dz);
         if(r < cutoff) {
             double invr = 1.0 / r;
             double gamma = spring_constant * pow(invr, 2 + dd);
             int base_i = 3 * i;
             int base_j = 3 * j;
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 0)], gamma * dx * dx);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 1)], gamma * dy * dy);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 2)], gamma * dz * dz);
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 1)], gamma * dx * dy);
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 2)], gamma * dx * dz);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 0)], gamma * dy * dx);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 2)], gamma * dy * dz);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 0)], gamma * dx * dz);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 1)], gamma * dy * dz);
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_j + 0)], -gamma * dx * dx);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_j + 1)], -gamma * dy * dy);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_j + 2)], -gamma * dz * dz);
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_j + 1)], -gamma * dx * dy);
             atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_j + 2)], -gamma * dx * dz);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_j + 0)], -gamma * dy * dx);
             atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_j + 2)], -gamma * dy * dz);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_j + 0)], -gamma * dx * dz);
             atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_j + 1)], -gamma * dy * dz);
         }
    }
}
"""
hessian_kernel = cp.RawKernel(hessian_kernel_code, "hessian_kernel")


def hessian_cuda(vec, cutoff=1.2, spring_constant=1000.0, dd=0):
    """CUDA version of _hessian.

    Parameters
    ----------
    vec : cupy.ndarray
        A coordinate matrix of shape (n, 3) with type float64.
    cutoff : float, optional
        Distance cutoff.
    spring_constant : float, optional
        Base spring constant.
    dd : int, optional
        Exponent modifier.

    Returns
    -------
    cupy.ndarray
        Hessian matrix of shape (3*n, 3*n) as a cupy array.
    """
    n = vec.shape[0]
    hessian_size = 3 * n
    hess = cp.zeros((hessian_size, hessian_size), dtype=cp.float64)
    block = (16, 16)
    grid_x = int(ceil(n / block[0]))
    grid_y = int(ceil(n / block[1]))
    grid = (grid_x, grid_y)
    hessian_kernel(
        grid, block, (n, cutoff, spring_constant, dd, vec, hess, hessian_size)
    )
    return hess


# ---------------------------------------------------------------------
# CUDA Kernel for computing a perturbation matrix from a covariance matrix.
# This version uses directional projections (7 normalized directions).
# ---------------------------------------------------------------------
perturbation_matrix_kernel_code = r"""
extern "C" __global__
void perturbation_matrix_kernel(const int m, const int n, const double *covar, double *pert, const int M)
{
    // Each thread computes one element of the perturbation matrix (size: m x n)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < m && j < n) {
         double sum = 0.0;
         // Define 7 direction vectors
         double dirs[7][3] = {
             {1.0, 0.0, 0.0},
             {0.0, 1.0, 0.0},
             {0.0, 0.0, 1.0},
             {1.0, 1.0, 0.0},
             {1.0, 0.0, 1.0},
             {0.0, 1.0, 1.0},
             {1.0, 1.0, 1.0}
         };
         int base_i = 3 * i;
         int base_j = 3 * j;
         for (int k = 0; k < 7; k++) {
             double f0 = dirs[k][0];
             double f1 = dirs[k][1];
             double f2 = dirs[k][2];
             // Normalize the direction vector
             double norm = sqrt(f0*f0 + f1*f1 + f2*f2);
             f0 /= norm; f1 /= norm; f2 /= norm;
             double delta0 = covar[(base_i + 0)*M + (base_j + 0)] * f0 +
                             covar[(base_i + 0)*M + (base_j + 1)] * f1 +
                             covar[(base_i + 0)*M + (base_j + 2)] * f2;
             double delta1 = covar[(base_i + 1)*M + (base_j + 0)] * f0 +
                             covar[(base_i + 1)*M + (base_j + 1)] * f1 +
                             covar[(base_i + 1)*M + (base_j + 2)] * f2;
             double delta2 = covar[(base_i + 2)*M + (base_j + 0)] * f0 +
                             covar[(base_i + 2)*M + (base_j + 1)] * f1 +
                             covar[(base_i + 2)*M + (base_j + 2)] * f2;
             double s_val = sqrt(delta0*delta0 + delta1*delta1 + delta2*delta2);
             sum += s_val;
         }
         pert[i*n + j] = sum;
    }
}
"""
perturbation_matrix_kernel = cp.RawKernel(
    perturbation_matrix_kernel_code, "perturbation_matrix_kernel"
)


def perturbation_matrix_cuda(covar):
    """CUDA version of _perturbation_matrix.

    Parameters
    ----------
    covar : cupy.ndarray
        Covariance matrix of shape (3*m, 3*n) with type float64.

    Returns
    -------
    cupy.ndarray
        Perturbation matrix of shape (m, n) (normalization should be applied separately if needed).
    """
    M = covar.shape[1]  # M = 3*n (number of columns)
    m = covar.shape[0] // 3
    n = M // 3
    pert = cp.zeros((m, n), dtype=cp.float64)
    block = (16, 16)
    grid_x = int(ceil(m / block[0]))
    grid_y = int(ceil(n / block[1]))
    grid = (grid_x, grid_y)
    perturbation_matrix_kernel(grid, block, (m, n, covar, pert, M))
    return pert


# ---------------------------------------------------------------------
# CUDA Kernel for computing block-wise perturbation matrix (td version).
# ---------------------------------------------------------------------
td_perturbation_matrix_kernel_code = r"""
extern "C" __global__
void td_perturbation_matrix_kernel(const int m, const int n, const double *ccf, double *pert, const int M)
{
    // Each thread computes the norm of a 3x3 block corresponding to residues (i,j)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i < m && j < n) {
         double temp = 0.0;
         int base_i = 3 * i;
         int base_j = 3 * j;
         for (int a = 0; a < 3; a++) {
             for (int b = 0; b < 3; b++) {
                 double val = ccf[(base_i + a)*M + (base_j + b)];
                 temp += val * val;
             }
         }
         pert[i*n + j] = sqrt(temp);
    }
}
"""
td_perturbation_matrix_kernel = cp.RawKernel(
    td_perturbation_matrix_kernel_code, "td_perturbation_matrix_kernel"
)


def td_perturbation_matrix_cuda(ccf, normalize=True):
    """CUDA version of _td_perturbation_matrix.

    Parameters
    ----------
    ccf : cupy.ndarray
        Covariance (or Hessian) matrix of shape (3*m, 3*n) with type float64.
    normalize : bool, optional
        If True, the output matrix is normalized so that the total sum equals 1.
        (Normalization is performed on the CPU after kernel execution.)

    Returns
    -------
    cupy.ndarray
        Perturbation matrix of shape (m, n).
    """
    M = ccf.shape[1]
    m = ccf.shape[0] // 3
    n = M // 3
    pert = cp.zeros((m, n), dtype=cp.float64)
    block = (16, 16)
    grid_x = int(ceil(m / block[0]))
    grid_y = int(ceil(n / block[1]))
    grid = (grid_x, grid_y)
    td_perturbation_matrix_kernel(grid, block, (m, n, ccf, pert, M))
    if normalize:
        total = cp.sum(pert)
        if total != 0:
            pert /= total
    return pert


# DFI KERNEL
dfi_kernel_code = r"""
extern "C" __global__ void dfi_kernel(const float* cov, const float* forces, const int resnum, float *result) {
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int twx = blockDim.x, twy = blockDim.y;
    
    __shared__ float f[3];
    
    // Load forces array into shared memory
    if (tx < 3) {
        f[tx] = forces[tx];
    }
    __syncthreads();
    

    if (bx < resnum && by < resnum){
        float sum_ij = 0;
        // Compute partial sum of this tile
        for (int i = 0; i < twy; i++){
            float partial_sum = 0;
            for (int j = 0; j < twx; j++){
                int row = by * twy + i;
                int col = bx * twx + j;
                int index = row * 3 * resnum + col;
                partial_sum += cov[index] * forces[j] * cov[index] * forces[j];
            }
            sum_ij += partial_sum;
        }
        sum_ij = sqrtf(sum_ij);
        __syncthreads();
        result[by*resnum + bx] = sum_ij;
    }
    
};
"""
dfi_kernel = cp.RawKernel(dfi_kernel_code, "dfi_kernel")
