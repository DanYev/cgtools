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
from cgtools._actual_math import mycmath, mypymath


@memprofit
@timeit
def fft_corr(*args, mode='serial', **kwargs):
    if mode == 'serial':
        return mypymath.sfft_corr(*args, **kwargs)
    if mode == 'parallel':
        return mypymath.pfft_corr(*args, **kwargs)
    if mode == 'gpu':
        return mypymath.gfft_corr(*args, **kwargs)
    raise ValueError("Currently 'mode' should be 'serial', 'parallel' or 'gpu'.")


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


def covariance_matrix(positions, dtype=np.float32):
    """
    Calculate the position-position covariance matrix
    Parameters:
        positions (ndarray): arrays of positions
        b (float): Time of first frame to read from trajectory (default unit ps)
    """
    mean = positions.mean(axis=-1, keepdims=True) # (n_coords, 1) 
    centered_positions = positions - mean # Center the data by removing mean
    covmat = np.cov(centered_positions, rowvar=True, dtype=dtype) # Compute covariance matrix (n_coords x n_coords)
    return np.array(covmat)


def calc_and_save_covmats(positions, outdir, n=1, outtag='covmat', dtype=np.float32):
    """
    Calculate and save the position-position covariance matrices by splitting trajectory into n segments
    
    Parameters:
        positions (ndarray): arrays of positions
        n (int): number of segments
    """
    trajs = np.array_split(positions, n, axis=-1)  # Split into `n` segments along frames (axis=1)
    for idx, traj in enumerate(trajs, start=1): # Process each segment
        logger.info(f"Processing matrix {idx}")
        covmat = covariance_matrix(traj, dtype=dtype) # Compute covariance matrix (n_coords x n_coords)
        outfile = os.path.join(outdir, f'{outtag}_{idx}.npy')
        np.save(outfile, covmat)
        logger.info(f"Saved covariance matrix to {outfile}")
        
        
def ccf(xs, ys, ntmax=None, n=1, mode='parallel', center=True, dtype=np.float32):
    """
    Calculate the average cross-correlation function of x and y by splitting them into n segments
    """
    logger.info(f"Calculating cross-correlation.")
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


def perturbation_matrix(covariance_matrix, dtype=np.float64):
    pertmat = _perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat



def td_perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = _td_perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat    


def dfi(perturbation_matrix):
    """
    Calculates DFI matrix from the pertubation matrix
    Normalized such that the total sum is equal to 1
    """
    dfi = np.sum(perturbation_matrix, axis=-1)
    dfi *= len(dfi)
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
    for ids in groups:
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
    for ids1 in groups:
        temp = []
        for ids2 in groups:
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



