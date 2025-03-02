/*
File: rcmath_cuda_kernels.cu
Description:
    This file contains CUDA kernel implementations for calculating position-position Hessian 
    matrices and perturbation matrices derived from coordinate and covariance data. These 
    kernels are intended for high-performance GPU acceleration within the reForge workflow.

Usage:
    Compile this file with nvcc to generate a CUDA object or library, then call the kernels 
    from your host code (e.g. via CUDA runtime API or using a higher-level interface such as CuPy).

Requirements:
    - CUDA Toolkit
    - A CUDA-capable GPU

Author: Your Name
Date: YYYY-MM-DD
*/

#include <cuda_runtime.h>
#include <math.h>

// Expose the kernels with C linkage for easier integration.
extern "C" {

    // ---------------------------------------------------------------------
    // Kernel: hessian_kernel
    // Description:
    //   Computes the contributions to the Hessian matrix from a coordinate matrix.
    //   The input 'vec' is expected to be a flattened (n x 3) array where each residue's
    //   coordinates are stored consecutively.
    // ---------------------------------------------------------------------
    __global__
    void hessian_kernel(const int n, const double cutoff, const double spring_constant, const int dd,
                        const double *vec, double *hessian, const int hessian_size)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < n && j < n && i != j) {
            double dx = vec[i*3 + 0] - vec[j*3 + 0];
            double dy = vec[i*3 + 1] - vec[j*3 + 1];
            double dz = vec[i*3 + 2] - vec[j*3 + 2];
            double r = sqrt(dx*dx + dy*dy + dz*dz);
            if (r < cutoff) {
                double invr = 1.0 / r;
                double gamma = spring_constant * pow(invr, 2 + dd);
                int base_i = 3 * i;
                int base_j = 3 * j;
                // Update diagonal block (Hii)
                atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 0)], gamma * dx * dx);
                atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 1)], gamma * dy * dy);
                atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 2)], gamma * dz * dz);
                atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 1)], gamma * dx * dy);
                atomicAdd(&hessian[(base_i + 0) * hessian_size + (base_i + 2)], gamma * dx * dz);
                atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 0)], gamma * dy * dx);
                atomicAdd(&hessian[(base_i + 1) * hessian_size + (base_i + 2)], gamma * dy * dz);
                atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 0)], gamma * dx * dz);
                atomicAdd(&hessian[(base_i + 2) * hessian_size + (base_i + 1)], gamma * dy * dz);
                // Update off-diagonal block (Hij)
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

    // ---------------------------------------------------------------------
    // Kernel: perturbation_matrix_kernel
    // Description:
    //   Computes a perturbation matrix from a covariance matrix using directional 
    //   projections. For each block corresponding to residues (i, j), it computes the 
    //   sum of the norms of projections onto 7 normalized direction vectors.
    // ---------------------------------------------------------------------
    __global__
    void perturbation_matrix_kernel(const int m, const int n, const double *covar, double *pert, const int M)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < m && j < n) {
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

    // ---------------------------------------------------------------------
    // Kernel: td_perturbation_matrix_kernel
    // Description:
    //   Computes a block-wise perturbation matrix (td version) from a covariance (or Hessian) 
    //   matrix. For each 3x3 block corresponding to residues (i, j), it computes the Frobenius 
    //   norm.
    // ---------------------------------------------------------------------
    __global__
    void td_perturbation_matrix_kernel(const int m, const int n, const double *ccf, double *pert, const int M)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;
        if (i < m && j < n) {
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

} // end extern "C"



