import os
import sys
import time
import numpy as np
import cupy as cp
import scipy.sparse.linalg
import cupy.linalg
import cupyx.scipy.sparse.linalg
from cupyx.profiler import benchmark
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from numba import njit
from cgtools.lrt import timeit, memprofit


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

###########################################################################################
# Functions
###########################################################################################

def read_cif(_pdb_id):
    pdb_info = MMCIF2Dict(_pdb_id)
    atom_types = pdb_info["_atom_site.label_atom_id"]
    atom_ids = np.asarray(pdb_info["_atom_site.id"], np.intc)
    atom_comps = pdb_info["_atom_site.label_alt_id"]
    residues = pdb_info["_atom_site.label_comp_id"]
    atom_x_coordinates = np.asarray(pdb_info["_atom_site.Cartn_x"], np.float32)
    atom_y_coordinates = np.asarray(pdb_info["_atom_site.Cartn_y"], np.float32)
    atom_z_coordinates = np.asarray(pdb_info["_atom_site.Cartn_z"], np.float32)
    atom_occupancies = np.asarray(pdb_info["_atom_site.occupancy"], np.float32)
    atom_chain_ids = pdb_info["_atom_site.auth_asym_id"]
    atom_sequence_ids = pdb_info["_atom_site.label_seq_id"]

    p_mask = np.asarray([atom_type == "P" for atom_type in atom_types])
    c1p_mask = np.asarray([atom_type == "C1'" for atom_type in atom_types])
    n1_mask = np.asarray([atom_type == "N1" for atom_type in atom_types]) \
        * np.asarray([residue == "A" or residue == "G" for residue in residues])
    n3_mask = np.asarray([atom_type == "N3" for atom_type in atom_types]) \
        * np.asarray([residue == "C" or residue == "U" for residue in residues])
    ca_mask = np.asarray([atom_type == "CA" for atom_type in atom_types])
    mg_mask = np.asarray([atom_type == "MG" for atom_type in atom_types])
    zn_mask = np.asarray([atom_type == "ZN" for atom_type in atom_types])
    comp_mask = np.asarray([comp == "." or comp == "A" for comp in atom_comps])
    chain_mask  = np.asarray([chain == "DT" or chain == "DT1" or chain == "DT2" for chain in atom_chain_ids]) # DJ - L11 for 5it8
    mask = ca_mask + c1p_mask # p_mask + 
    mask *= comp_mask
    # mask *= np.invert(chain_mask)

    x_coordinates = atom_x_coordinates[mask]
    y_coordinates = atom_y_coordinates[mask]
    z_coordinates = atom_z_coordinates[mask]
    atypes = []
    occupancies = atom_occupancies[mask]
    sel_atom_ids = atom_ids[mask]
    chain_ids = [item for item, m in zip(atom_chain_ids, mask) if m]
    sequence_ids = [item for item, m in zip(atom_sequence_ids, mask) if m]
    return x_coordinates, y_coordinates, z_coordinates, occupancies, chain_ids, sequence_ids


def read_pdb(pdb):
    f=open(pdb,'r')
    pdb=f.readlines()
    f.close()
    x=[]
    y=[]
    z=[]
    chain_ids=[]
    res_names=[]
    res_number=[]
    for line in pdb:
        if line[:4] == "ATOM" or line[:6] == "HETATM":
            atomnumber=int(line[6:12].strip())
            atomname = line[12:16].strip()
            residuename = line[17:20].strip()
            chain_id = line[21:22].strip()
            resnumber = int(line[22:26].strip())
            orthogonalcoordinatesforx = float(line[30:38])
            orthogonalcoordinatesfory = float(line[38:46])
            orthogonalcoordinatesforz = float(line[46:54])
            occupancy = float(line[56:60].strip())
            if atomname == "CA" or atomname == "C1'": # or atomname == "ZN" or atomname == "MG":
                if occupancy != 1.00:
                    print("SHOUT")
                    print(atomnumber,atomname,residuename,chain_id,resnumber,occupancy)
                    return
                x.append(orthogonalcoordinatesforx)
                y.append(orthogonalcoordinatesfory)
                z.append(orthogonalcoordinatesforz)
                chain_ids.append(chain_id)
                res_names.append(residuename)
                res_number.append(resnumber)

    return x,y,z,chain_ids,res_names,res_number    


@njit(parallel=False)
def calculate_hessian(resnum, x, y, z, cutoff, spring_constant, dtype=np.float64):
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
                gamma = spring_constant * invr**2
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
def compute_hessian(resnum, x, y, z,  cutoff, spring_constant,dtype=float):
    # Convert coordinates into a single matrix for vectorized operations
    coords = np.vstack((x, y, z)).T  # Shape: (resnum, 3)    
    # Compute pairwise distance vectors
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # Shape: (resnum, resnum, 3)    
    # Compute pairwise distances
    r = np.linalg.norm(diff, axis=2)  # Shape: (resnum, resnum)    
    # Avoid division by zero
    np.fill_diagonal(r, np.inf)    
    # Compute inverse distance where r < cutoff
    mask = r < cutoff
    invr6 = np.where(mask, (1 / r**6) * spring_constant, 0)  # Shape: (resnum, resnum)   
    # Compute outer products for Hessian entries
    gamma = invr6[:, :, np.newaxis]  # Shape: (resnum, resnum, 1)
    outer_prod = gamma * (diff[:, :, :, np.newaxis] * diff[:, :, np.newaxis, :])  # Shape: (resnum, resnum, 3, 3)  
    # Initialize Hessian
    hessian = np.zeros((3 * resnum, 3 * resnum), dtype=dtype)
    # Create index arrays
    idx = np.arange(resnum)
    idx_i = np.repeat(3 * idx, 3).reshape(resnum, 3)
    idx_j = idx_i.T  # Shape for broadcasting
    # Vectorized Hessian updates
    hessian[np.ix_(idx_i.ravel(), idx_i.ravel())] += outer_prod.reshape(resnum * 3, resnum * 3)  # Hii
    hessian[np.ix_(idx_i.ravel(), idx_j.ravel())] -= outer_prod.reshape(resnum * 3, resnum * 3)  # Hij
    return hessian


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
    
@timeit
@memprofit
def calc_perturbation_matrix_cpu(covariance_matrix, dtype=np.float32):
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


def calc_perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = calc_perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat
    

def calc_dfi(perturbation_matrix):
    """
    Calculates DFI matrix from the pertubation matrix
    Normalized such that the total sum is equal to 1
    """
    dfi = np.sum(perturbation_matrix, axis=-1)
    return dfi
    

def calc_full_dci(perturbation_matrix, asym=False):
    """
    Calculates DCI matrix from the pertubation matrix
    Normalized such that the total sum of the matrix elements is equal to the number of residues
    """
    dci = perturbation_matrix * perturbation_matrix.shape[0] / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci = dci - dci.T
    return dci
    
    
def calculate_dfi_gpu(covarince_matrix, resn, dtype=np.float64):
    print("Calculating DFI")
    cov = covarince_matrix
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=dtype)
    norm = np.sqrt(np.sum(directions, axis=1))
    directions = (directions.T / norm).T
    dfi = np.zeros(resn)
    for f in directions:
        f = np.tile(f, resn)
        m = covarince_matrix * f
        m = np.reshape(m, (resn, resn, 3, 3))
        m = np.sum(m, axis=-1)
        pert_mat = np.sqrt(np.sum(m * m, axis=-1))
        dfi += np.sum(pert_mat, axis=-1)
    dfi /= np.sum(dfi)
    return dfi    
    
    
def get_dfi_gpu_split(cov_gpu, resnum, split_matrix=True):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float32)
    norm = np.sqrt(np.sum(directions**2, axis=1))
    directions = (directions.T / norm).T
    factor = 1
    if split_matrix:
        factor = 2
        cov = cov_gpu.get()
        if resnum % factor != 0:
            cov = np.pad(cov, ((0, 3), (0, 3)), mode='constant', constant_values=0)
    N = resnum // factor + resnum % factor
    dfi = np.zeros(resnum + resnum % factor)
    for i in range(factor):
        for j in range(factor):
            if split_matrix:
                cov_gpu = cp.asarray(cov[i*3*N:(i+1)*3*N, j*3*N:(j+1)*3*N])
            for k in range(7):
                f_gpu = cp.asarray(directions[k,:], cp.float32)
                result_gpu = cp.zeros((N, N), cp.float32)
                grid_dim = (N, N)
                block_dim = (3, 3)
                dfi_kernel(grid_dim, block_dim, (cov_gpu, f_gpu, cp.int32(N), result_gpu))
                result = result_gpu.get()
                dfi_k = np.sum(result, axis=1)
                dfi[i*N: (i+1)*N] += dfi_k
    if resnum % factor != 0:            
        dfi = dfi[:-1]
    return dfi
    
    
if __name__ == "__main__":
    exit()

