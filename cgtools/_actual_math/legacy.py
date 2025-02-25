import os
import sys
import time
import numpy as np
import cupy as cp
import scipy.sparse.linalg
import cupy.linalg
import cupyx.scipy.sparse.linalg
from cupyx.profiler import benchmark
from cgtools.lrt import timeit, memprofit


@timeit
@memprofit
def calculate_hessian(resnum, x, y, z, cutoff=12, spring_constant=1000, dd=0, dtype=np.float64):
    hessian = np.zeros((3 * resnum, 3 * resnum), dtype)
    for i in range(resnum):
        for j in range(resnum):
            if j == i:
                continue
            x_ij = x[i] - x[j]
            y_ij = y[i] - y[j]
            z_ij = z[i] - z[j]
            r = np.sqrt(x_ij**2 + y_ij**2 + z_ij**2)
            invr = r**-1            
            if r < cutoff: 
                gamma = spring_constant * invr**(2+dd)
            else:
                continue
            # creating Hii
            hessian[3 * i, 3 * i] += gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * i + 1] += gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * i + 2] += gamma * z_ij * z_ij
            hessian[3 * i, 3 * i + 1] += gamma * x_ij * y_ij
            hessian[3 * i, 3 * i + 2] += gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * i] += gamma * y_ij * x_ij
            hessian[3 * i + 1, 3 * i + 2] += gamma * y_ij * z_ij
            hessian[3 * i + 2, 3 * i] += gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * i + 1] += gamma * y_ij * z_ij
            hessian[3 * i, 3 * j] -= gamma * x_ij * x_ij
            hessian[3 * i + 1, 3 * j + 1] -= gamma * y_ij * y_ij
            hessian[3 * i + 2, 3 * j + 2] -= gamma * z_ij * z_ij           
            hessian[3 * i, 3 * j + 1] -= gamma * x_ij * y_ij
            hessian[3 * i, 3 * j + 2] -= gamma * x_ij * z_ij
            hessian[3 * i + 1, 3 * j] -= gamma * y_ij * x_ij           
            hessian[3 * i + 1, 3 * j + 2] -= gamma * y_ij * z_ij 
            hessian[3 * i + 2, 3 * j] -= gamma * x_ij * z_ij
            hessian[3 * i + 2, 3 * j + 1] -= gamma * y_ij * z_ij
    # hessian = hessian + hessian.T
    return hessian


@timeit
@memprofit   
def _perturbation_matrix_old(covariance_matrix, resnum, dtype=np.float64):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=dtype)
    directions = directions.T / np.sqrt(np.sum(directions, axis=1)).T # normalizing directions
    directions = directions.T
    perturbation_matrix = np.zeros((resnum, resnum), dtype=dtype)
    n = resnum
    for k in range(len(directions)):
        f = np.ascontiguousarray(directions[k, :])
        for j in range(n):
            for i in range(n):
                cov_ij = np.ascontiguousarray(covariance_matrix[3*i:3*i+3, 3*j:3*j+3])
                delta = np.dot(cov_ij, f)
                perturbation_matrix[i,j] += np.sqrt(np.sum(delta * delta))
    perturbation_matrix /= np.sum(perturbation_matrix)
    return perturbation_matrix


@timeit
@memprofit    
def _perturbation_matrix_cpu(covariance_matrix, dtype=np.float32):
    """
    Calculates perturbation matrix from a covariance matrix or a hessian on CPU
    The result is normalized such that the total sum of the matrix elements is equal to 1
    """
    n = covariance_matrix.shape[0] // 3
    perturbation_matrix = np.zeros((n, n), dtype=dtype)
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=dtype)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for k in range(len(directions)):
        f = directions[k, :]
        cov_blocks = covariance_matrix.reshape(n, 3, n, 3).swapaxes(1, 2)  # Shape: (n, n, 3, 3)
        # Compute delta for all i, j in one step
        delta = np.einsum('ijkl,l->ijk', cov_blocks, f)  # Shape: (n, n, 3)
        abs_delta = np.sqrt(np.sum(delta ** 2, axis=2))  # Shape: (n, n)
        perturbation_matrix += abs_delta
    perturbation_matrix /= np.sum(perturbation_matrix)
    return perturbation_matrix


@memprofit
@timeit
def _td_perturbation_matrix_cpu(ccf, normalize=True, dtype=np.float64):
    """
    Calculates perturbation matrix from a covariance matrix or a hessian on CPU
    The result is normalized such that the total sum of the matrix elements is equal to 1
    """
    m = ccf.shape[0] // 3
    n = ccf.shape[1] // 3
    blocks = ccf.reshape(m, 3, n, 3).swapaxes(1, 2)
    perturbation_matrix = np.sum(blocks**2, axis=(-2, -1))
    perturbation_matrix = np.sqrt(perturbation_matrix)
    if normalize:
        perturbation_matrix /= np.sum(perturbation_matrix)
    return perturbation_matrix    



def pad_matrix_if_odd(M):
    m = M.shape[0]
    if m % 2 != 0:
        M = np.pad(M, ((0, 3), (0, 3)), mode='constant', constant_values=0)
    return M


def split_matrix(M):
    # M = pad_matrix_if_odd(M)
    n = M.shape[0] // 2
    A = M[:n, :n]
    B = M[:n, n:]
    C = M[n:, :n]
    D = M[n:, n:]
    return A, B, C, D


@timeit
@memprofit
def invert_hessian(hessian, tol_conv, n_modes=20, v0=None):
    ei, evec = scipy.sparse.linalg.eigsh(hessian, k=n_modes, which='SM', tol=0, sigma=0)
    print("Inverting the Hessian using LAPACK")
    print(ei)
    tol = 1e-3
    singular = ei < tol
    invw = 1 / ei
    invw[singular] = 0.
    print(invw)
    invHrs = np.matmul(evec, np.matmul(np.diag(invw), evec.T))
    return invHrs


@timeit
@memprofit   
def invert_matrix_gpu(M, n_modes=20, k_singular=6, DENSE_NOT_SPARSE=True):
    M_gpu = cp.asarray(M, cp.float32)   
    # Diagonilizing M
    start_time = time.perf_counter() 
    if DENSE_NOT_SPARSE:
        evals_gpu, evecs_gpu = cupy.linalg.eigh(M_gpu)
    else:
        evals_gpu, evecs_gpu = cupyx.scipy.sparse.linalg.eigsh(M_gpu, k=n_modes, maxiter=None, which='SA', tol=0)      
    if n_modes != -1:
        evals_gpu = evals_gpu[:n_modes]
        evecs_gpu = evecs_gpu[:,:n_modes]       
    inv_evals_gpu = evals_gpu**-1
    inv_evals_gpu[:k_singular] = 0.0   
    print(evals_gpu[:100])
    end_time = time.perf_counter()
    print("GPU ALL modes inverse time: ", end_time - start_time) 
    # Finding Inverse
    invM_gpu = cp.matmul(evecs_gpu, cp.matmul(cp.diag(inv_evals_gpu), evecs_gpu.T))
    return invM_gpu

    
@timeit
@memprofit
def invert_symm_block_matrix_gpu(A, B, D):
    m = A.shape[0]
    n = D.shape[0]
    # print(m, n)
    start_time = time.perf_counter()  
    # Schur complement
    A_gpu = cp.asarray(A, cp.float32)
    B_gpu = cp.asarray(B, cp.float32)
    D_gpu = cp.asarray(D, cp.float32)
    invA_gpu = invert_matrix_gpu(A, -1, 0)
    S_gpu = cp.matmul(invA_gpu, B_gpu)
    compA_gpu = D_gpu - cp.matmul(B_gpu.T, S_gpu)
    invCompA_gpu = invert_matrix_gpu(compA_gpu, -1, 6)
    Q_gpu = cp.matmul(S_gpu, invCompA_gpu)
    invM_gpu = cp.zeros((n + m, n + m), cp.float32)
    M_11 = invA_gpu + cp.matmul(Q_gpu, S_gpu.T)
    invM_gpu[:m, :m] = M_11
    invM_gpu[:m, m:] = -Q_gpu
    invM_gpu[m:, :m] = -Q_gpu.T
    invM_gpu[m:, m:] = invCompA_gpu   
    end_time = time.perf_counter()
    print("Inverting block matrix on GPU: ", end_time - start_time)   
    return invM_gpu    


###########################################################################################
# GPU kernels
###########################################################################################

# DFI KERNEL
dfi_kernel_code = """
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


# INVERSE KERNEL
inverse_kernel_code = """
extern "C" __global__ void inverse_kernel(const float* cov) {

    
};
"""
inverse_kernel = cp.RawKernel(dfi_kernel_code, "inverse_kernel")    