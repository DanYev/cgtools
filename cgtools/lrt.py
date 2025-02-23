#!/bin/python
import os
import sys
import MDAnalysis as mda
import numpy as np
import cupy as cp
import pandas as pd
from joblib import Parallel, delayed
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
from scipy import linalg as LA
from scipy.stats import pearsonr
from cgtools.utils import timeit, memprofit, logger


##############################################################
## To calculate correlations ##
############################################################## 

def _sfft_corr(x, y, ntmax=None, center=False, loop=True, dtype=np.float64):
    """
    Compute the correlation function <x(t)y(0)> using FFT.
    Parameters:
    - x: np.ndarray, first input signal. Shape - (n_coords, n_samples).
    - y: np.ndarray, second input signal. Shape - (n_coords, n_samples).
    - ntmax: positive int, number of time samples to save
    - center: bool,  whether to mean-center the signals
    - loop: bool, whether to calculate Cross-Power Spectral Density (CPSD) in a for loop.
        It's way more memory efficient for large arrays but may be slower
    Returns:
    - corr: np.ndarray, computed correlation function.
    """
    logger.info("Doing FFTs serially.")
    # Helper functions
    def compute_correlation(*args,):
        i, j, x_f, y_f, ntmax, counts = args
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2: # Extract only the valid part
        ntmax = (nt+1)//2   
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    y_f = fft(y, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    counts = np.arange(nt,  nt-ntmax , -1)**-1 # Normalize correctly over valid indices
    # Compute the FFT-based correlation via CPSD
    if loop:
        corr = np.zeros((nx, ny, ntmax), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                corr[i, j] = compute_correlation(i, j, x_f, y_f, ntmax, counts)
    else:
        corr = np.einsum('it,jt->ijt', x_f, np.conj(y_f))
        corr = ifft(corr, axis=-1).real / nt
    return corr


def _pfft_corr(x, y, ntmax=None, center=False, dtype=np.float64):
    """
    Compute the correlation function using FFT with parallelizing the cross-correlation loop.
    Looks like it starts getting faster for Nt >~ 10000
    Needs more memory than the serial version
    Takes 7-8 minutes and ~28Gb for 8 cores to process two (nx, nt)=(1000, 100000) arrays outputting 
    ~11Gb (nx, ny, nt=ntmax)=(1000, 1000, 1000) correlation array
    Parameters:
    - x: np.ndarray, first input signal. Shape - (n_coords, n_samples).
    - y: np.ndarray, second input signal. Shape - (n_coords, n_samples).
    - ntmax: positive int, number of time samples to save
    - center: bool,  whether to mean-center the signals.
    Returns:
    - corr: np.ndarray, computed correlation function.
    """
    logger.info("Doing FFTs in parallel.")
    # Helper functions for parrallelizing
    def compute_correlation(*args,):
        i, j, x_f, y_f, ntmax, counts = args
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts

    def parallel_fft_correlation(x_f, y_f, ntmax, nt, n_jobs=-1):
        nx, ny = x_f.shape[0], y_f.shape[0]
        corr = np.zeros((nx, ny, ntmax), dtype=np.float64)
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_correlation)(i, j, x_f, y_f, ntmax, nt) 
            for i in range(nx) for j in range(ny)
        )
        # Reshape results back to (nx, ny, ntmax)
        corr = np.array(results).reshape(nx, ny, ntmax)
        return corr        
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2: # Extract only the valid part
        ntmax = (nt+1)//2   
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    y_f = fft(y, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    counts = np.arange(nt,  nt-ntmax , -1)**-1 # Normalize correctly over valid indices
    # Compute the FFT-based correlation via CPSD
    corr = parallel_fft_correlation(x_f, y_f, ntmax, counts)
    return corr


def _gfft_corr(x, y, ntmax=None, center=True, dtype=cp.float32):
    """
    Another version for GPU
    WORKS LIKE WHOOOOOOOOOOOOOOOOOOOSHHHHHHHHHHH!
    """
    logger.info("Doing FFTs on GPU.")
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2: # Extract only the valid part
        ntmax = (nt+1)//2   
    corr = np.zeros((nx, ny, ntmax), dtype=np.float32)
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Convert NumPy arrays to CuPy arrays
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    # Compute FFT along the last axis
    x_f = cp.fft.fft(x, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    y_f = cp.fft.fft(y, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    counts = cp.arange(nt,  nt-ntmax , -1, dtype=dtype)**-1 # Normalize correctly over valid indices
    counts = counts[None, :]  # Reshape for broadcasting
    # Row-wise FFT-based correlation
    for i in range(nx):
        corr_row = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts
        corr[i, :, :] = corr_row.get()   
    return corr


@memprofit
@timeit
def fft_corr(*args, mode='parallel', **kwargs):
    if mode == 'ser':
        return _sfft_corr(*args, **kwargs)
    if mode == 'par':
        return _pfft_corr(*args, **kwargs)
    if mode == 'gpu':
        return _gfft_corr(*args, **kwargs)
    raise ValueError("Currently 'mode' should be 'ser', 'par' or 'gpu'.")


def gfft_conv(x, y, loop=False, dtype=cp.float32):
    """
    Compute element-wise convolution between two arrays of the same shape <x(t)y(0)> 
    using FFT along the last axis.
    Parameters:
    - x: np.ndarray, first input signal.
    - y: np.ndarray, second input signal. 
    - ntmax: positive int, number of time samples to save
    - center: bool,  whether to mean-center the signals
    - loop: bool, whether to calculate Cross-Power Spectral Density (CPSD) in a for loop.
        It's way more memory efficient for large arrays but may be slower
    Returns:
    - conv: np.ndarray, computed convolution.
    """
    print("Doing FFTs on GPU.", file=sys.stderr)
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = x.shape[1]
    # Convert NumPy arrays to CuPy arrays
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    # Compute FFT along the last axis
    x_f = cp.fft.fft(x, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    y_f = cp.fft.fft(y, n=2*nt, axis=-1) # Zero-pad to avoid circular effects
    counts = cp.arange(nt,  0, -1, dtype=dtype)**-1 # Normalize correctly over valid indices
    if loop:  
        conv = np.zeros((nx, ny, nt), dtype=np.float32)
        counts = counts[None, :]  # Reshape for broadcasting
        # Row-wise FFT-based correlation
        for i in range(nx):
            conv_row = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts
            conv[i, :, :] = conv_row.get()   
    else:
        counts = counts[None, None, :]  # Reshape for broadcasting
        conv = cp.fft.ifft(x_f * cp.conj(y_f), axis=-1).real[:, :, :nt] * counts
        conv = conv.get()
    return conv


def sfft_cpsd(x, y, ntmax=None, center=True, loop=True, dtype=np.float64):
    """
    Compute the Cross-Power Spectral Density (CPSD) using FFT.
    Parameters:
    - x: np.ndarray, first input signal. Shape - (n_coords, n_samples).
    - y: np.ndarray, second input signal. Shape - (n_coords, n_samples).
    - ntmax: positive int, number of time samples to save
    - center: bool,  whether to mean-center the signals
    - loop: bool, whether to calculate Cross-Power Spectral Density (CPSD) in a for loop.
        It's way more memory efficient for large arrays but may be slower
    Returns:
    - corr: np.ndarray, computed correlation function.
    """
    # Helper functions
    def compute_cpsd(*args):
        i, j, x_f, y_f, ntmax, nt = args
        cpsd_ij = x_f[i] * np.conj(y_f[j])
        cpsd_ij = np.abs(cpsd_ij) / nt
        # cpsd_ij[cpsd_ij<1] = 0
        cpsd_ij = np.average(cpsd_ij)
        return cpsd_ij
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    ntmax = nt if not ntmax else ntmax
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
    # Compute the CPSD
    if loop:
        cpsd = np.zeros((nx, ny), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                cpsd[i, j] = compute_cpsd(i, j, x_f, y_f, ntmax, nt)
    else:
        cpsd = np.einsum('it,jt->ijt', x_f, np.conj(y_f))
        cpsd = cpsd[:, :, :ntmax]    
    cpsd = np.abs(cpsd) 
    return cpsd


def covmat(positions, dtype=np.float32):
    """
    Calculate the position-position covariance matrix
    Parameters:
        positions (ndarray): arrays of positions
        b (float): Time of first frame to read from trajectory (default unit ps)
    """
    mean = positions.mean(axis=-1, keepdims=True) # (n_coords, 1) 
    centered_positions = positions - mean # Center the data by removing mean
    covmat = np.cov(centered_positions, rowvar=True, dtype=dtype) # Compute covariance matrix (n_coords x n_coords)
    return covmat


def calc_and_save_covmats(positions, n=1, dtype=np.float32):
    """
    Calculate and save the position-position covariance matrices by splitting trajectory into n segments
    
    Parameters:
        positions (ndarray): arrays of positions
        n (int): number of segments
    """
    trajs = np.array_split(positions, n, axis=-1)  # Split into `n` segments along frames (axis=1)
    for idx, traj in enumerate(trajs, start=1): # Process each segment
        logger.info(f"Processing matrix {idx}")
        covmat = covmat(traj), # Compute covariance matrix (n_coords x n_coords)
        np.save(f'covmat_{idx}.npy', covmat)
        logger.info(f"Covariance matrix {idx} saved to 'covmat_{idx}.npy'")
        
        
def ccf(xs, ys, ntmax=None, n=1, mode='parallel', center=True, dtype=np.float32):
    """
    Calculate the average cross-correlation function of x and y by splitting them into n segments
    """
    print(f"Calculating cross-correlation.", file=sys.stderr)
    # Split trajectories into `n` segments along frames (axis=1)
    xs = np.array_split(xs, n, axis=-1)
    ys = np.array_split(ys, n, axis=-1)
    nx = xs[0].shape[0]
    ny = ys[0].shape[0]
    nt = xs[-1].shape[1]
    print(f'Splitting trajectory into {n} parts', file=sys.stderr)
    if not ntmax or ntmax > (nt+1)//2: # Extract only the valid part
        ntmax = (nt+1)//2   
    corr = np.zeros((nx, ny, ntmax), dtype=np.float32)
    # Compute correlation for each segment
    for x, y in zip(xs, ys):
        corr_n = fft_corr(x, y, ntmax=ntmax, mode=mode, center=center, dtype=dtype)
        print(corr_n.shape, file=sys.stderr)
        corr += corr_n
    corr = corr / n
    print(np.sqrt(np.average(corr**2)), file=sys.stderr)
    # np.save('corr_pp.npy', corr)
    print(f"Finished calculating cross-correlation.", file=sys.stderr)
    return corr


##############################################################
## DCI DFI ##
############################################################## 

def _perturbation_matrix_old(covariance_matrix, resnum, dtype=np.float32):
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


def perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = _perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat


def _td_perturbation_matrix_cpu(ccf, normalize=False, dtype=np.float32):
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


def td_perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = calc_td_perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat    


def dfi(perturbation_matrix):
    """
    Calculates DFI matrix from the pertubation matrix
    Normalized such that the total sum is equal to 1
    """
    dfi = np.sum(perturbation_matrix, axis=-1)
    return dfi
    

def dci(perturbation_matrix, asym=False):
    """
    Calculates DCI matrix from the pertubation matrix
    Normalized such that the total sum of the matrix elements is equal to the number of residues
    """
    dci = perturbation_matrix * perturbation_matrix.shape[0] / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci = dci - dci.T
    return dci    
    

def group_molecule_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Calculates DCI between a group of atoms in 'groups' and the rest of the molecule
    """
    dcis = []
    dci_tot = perturbation_matrix / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for group in groups:
        ids = [idx - 1 for idx in group]
        top = np.sum(dci_tot[:, ids], axis=-1) * perturbation_matrix.shape[0]
        bot = len(ids)  
        dci = top / bot
        dcis.append(dci)
    return dcis 
    
    
def group_group_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Calculates DCI matrix between the groups of atoms in 'groups'
    """
    dcis = []
    dci_tot = perturbation_matrix / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for ch1 in groups:
        temp = []
        for ch2 in groups:
            ids1 = [idx - 1 for idx in ch1]
            ids2 = [idx - 1 for idx in ch2]
            idx1, idx2 = np.meshgrid(ids1, ids2, indexing='ij')
            top = np.sum(dci_tot[idx1, idx2]) * perturbation_matrix.shape[0]
            bot = len(ids1) * len(ids2)
            dci = top / bot
            temp.append(dci)
        dcis.append(temp)
    return dcis 
    

##############################################################
## MISC ##
############################################################## 
 
def percentile(x):
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px
    
if __name__ == '__main__':
    pass



