"""
File: mypymath.py
Description:
    This module contains internal routines for performing various mathematical 
    and signal processing operations required in our workflow. It includes FFT‐based 
    correlation computations (serial, parallel, and GPU versions), covariance matrix 
    calculation, dynamic coupling and flexibility index evaluations, sparse matrix 
    inversion on both CPU and GPU, and additional helper functions such as percentile 
    computation and FFT-based convolution.

    Note: This module is intended for internal use only.

Usage Example:
    >>> from mypymath import _sfft_ccf, fft_ccf
    >>> import numpy as np
    >>> # Generate random signals
    >>> x = np.random.rand(10, 256)
    >>> y = np.random.rand(10, 256)
    >>> # Compute serial FFT-based correlation
    >>> corr = _sfft_ccf(x, y, ntmax=64, center=True, loop=True)
    >>> # Or use the unified FFT correlation wrapper
    >>> corr = fft_ccf(x, y, mode='serial', ntmax=64, center=True)

Requirements:
    - Python 3.x
    - NumPy
    - SciPy
    - CuPy (for GPU-based functions)
    - joblib (for parallel processing)
    - MDAnalysis (if required elsewhere)

Author: DY
Date: YYYY-MM-DD
"""

import os
import sys
import MDAnalysis as mda
import numpy as np
import cupy as cp
import pandas as pd
import scipy.sparse.linalg
import cupy.linalg
import cupyx.scipy.sparse.linalg
from cupyx.profiler import benchmark
from reforge.utils import timeit, memprofit, logger
from joblib import Parallel, delayed
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
from scipy.stats import pearsonr

##############################################################
## For time dependent analysis ##
##############################################################

@memprofit
@timeit
def _sfft_ccf(x, y, ntmax=None, center=False, loop=True, dtype=None):
    """
    Compute the correlation function between two signals using a serial FFT-based method.

    This internal function calculates the correlation function <x(t)y(0)> by applying
    the Fast Fourier Transform (FFT) to the input signals. It optionally mean-centers the 
    signals and computes the inverse FFT of their product. When loop=True, it iterates over 
    coordinate pairs for improved memory efficiency on large arrays.

    Parameters:
        x (np.ndarray): First input signal of shape (n_coords, n_samples).
        y (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Maximum number of time samples to retain; defaults to (nt+1)//2.
        center (bool, optional): If True, subtract the mean from each signal along the time axis.
        loop (bool, optional): If True, compute the correlation in a loop.
        dtype (data-type, optional): Desired data type for computation (default: np.float64).

    Returns:
        np.ndarray: Correlation function array of shape (n_coords, n_coords, ntmax).
    """
    logger.info("Doing FFTs serially.")
    if not dtype:
        dtype = x.dtype
    def compute_correlation(*args):
        i, j, x_f, y_f, ntmax, counts = args
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2:
        ntmax = (nt+1)//2   
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x_f = fft(x, n=2*nt, axis=-1)
    y_f = fft(y, n=2*nt, axis=-1)
    counts = np.arange(nt, nt-ntmax, -1).astype(dtype)**-1
    if loop:
        corr = np.zeros((nx, ny, ntmax), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                corr[i, j] = compute_correlation(i, j, x_f, y_f, ntmax, counts)
    else:
        corr = np.einsum('it,jt->ijt', x_f, np.conj(y_f))
        corr = ifft(corr, axis=-1).real / nt
    return corr

@memprofit
@timeit
def _pfft_ccf(x, y, ntmax=None, center=False, dtype=None):
    """
    Compute the correlation function using a parallel FFT-based method.

    This function parallelizes the cross-correlation computation across all coordinate pairs
    using joblib, which can lead to performance gains on large arrays at the expense of memory usage.

    Parameters:
        x (np.ndarray): First input signal of shape (n_coords, n_samples).
        y (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Maximum number of time samples to retain; defaults to (nt+1)//2.
        center (bool, optional): If True, subtract the mean along the time axis.
        dtype (data-type, optional): Data type for computation (default: np.float64).

    Returns:
        np.ndarray: Correlation function array with shape (n_coords, n_coords, ntmax).
    """
    logger.info("Doing FFTs in parallel.")
    if not dtype:
        dtype = x.dtype
    def compute_correlation(*args):
        i, j, x_f, y_f, ntmax, counts = args
        corr = ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real
        return corr * counts
    def parallel_fft_correlation(x_f, y_f, ntmax, nt, n_jobs=-1):
        nx, ny = x_f.shape[0], y_f.shape[0]
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_correlation)(i, j, x_f, y_f, ntmax, nt) 
            for i in range(nx) for j in range(ny)
        )
        return np.array(results).reshape(nx, ny, ntmax)   
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2:
        ntmax = (nt+1)//2   
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x_f = fft(x, n=2*nt, axis=-1)
    y_f = fft(y, n=2*nt, axis=-1)
    counts = np.arange(nt, nt-ntmax, -1).astype(dtype)**-1
    corr = parallel_fft_correlation(x_f, y_f, ntmax, counts)
    return corr

@memprofit
@timeit
def _gfft_ccf(x, y, ntmax=None, center=True, dtype=None):
    """
    Compute the correlation function on the GPU using FFT.

    This function leverages CuPy to compute the FFT-based correlation of the input 
    signals on the GPU. It converts the input arrays to CuPy arrays, performs zero-padding 
    to avoid circular effects, and returns the real component of the computed inverse FFT.

    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        ntmax (int, optional): Maximum number of time samples to retain; defaults to (nt+1)//2.
        center (bool, optional): If True, subtract the mean along the time axis.
        dtype (data-type, optional): Desired CuPy data type (default: cp.float32).

    Returns:
        cp.ndarray: The computed correlation function as a CuPy array.
    """
    logger.info("Doing FFTs on GPU.")
    if not dtype:
        dtype = x.dtype
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if not ntmax or ntmax > (nt+1)//2:
        ntmax = (nt+1)//2   
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    x_f = cp.fft.fft(x, n=2*nt, axis=-1)
    y_f = cp.fft.fft(y, n=2*nt, axis=-1)
    counts = cp.arange(nt, nt-ntmax, -1, dtype=dtype)**-1
    counts = counts[None, :]
    corr = cp.zeros((nx, ny, ntmax), dtype=dtype)
    for i in range(nx):
        corr_row = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] * counts
        corr[i, :, :] = corr_row
    return corr

def _fft_ccf(*args, mode='serial', **kwargs):
    """
    Unified wrapper for FFT-based correlation functions.

    This function dispatches to one of the internal FFT correlation routines based on 
    the specified 'mode': 'serial' for _sfft_ccf, 'parallel' for _pfft_ccf, or 'gpu' 
    for _gfft_ccf.

    Parameters:
        *args: Positional arguments for the chosen correlation function.
        mode (str): Mode to use ('serial', 'parallel', or 'gpu').
        **kwargs: Additional keyword arguments for the correlation function.

    Returns:
        np.ndarray or cp.ndarray: The computed correlation function.

    Raises:
        ValueError: If an unsupported mode is specified.
    """
    if mode == 'serial':
        return _sfft_ccf(*args, **kwargs)
    if mode == 'parallel':
        return _pfft_ccf(*args, **kwargs)
    if mode == 'gpu':
        return _gfft_ccf(*args, **kwargs)
    raise ValueError("Currently 'mode' should be 'serial', 'parallel' or 'gpu'.")

@memprofit
@timeit
def _ccf(xs, ys, ntmax=None, n=1, mode='parallel', center=True, dtype=np.float32):
    """
    Compute the average cross-correlation function of two signals by segmenting them.

    The function splits the input signals into 'n' segments, computes the correlation 
    for each segment using the specified mode ('parallel', 'serial', or 'gpu'), and 
    returns their average.

    Parameters:
        xs (np.ndarray): First input signal of shape (n_coords, n_samples).
        ys (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Maximum number of time samples to retain per segment.
        n (int, optional): Number of segments to split the signals into.
        mode (str, optional): Mode for correlation computation ('parallel', 'serial', or 'gpu').
        center (bool, optional): If True, mean-center the signals along the time axis.
        dtype (data-type, optional): Desired data type (default: np.float32).

    Returns:
        np.ndarray: The averaged cross-correlation function.
    """
    logger.info(f"Calculating cross-correlation.")
    xs = np.array_split(xs, n, axis=-1)
    ys = np.array_split(ys, n, axis=-1)
    nx = xs[0].shape[0]
    ny = ys[0].shape[0]
    nt = xs[-1].shape[1]
    logger.info(f'Splitting trajectory into {n} parts')
    if not ntmax or ntmax > (nt+1)//2:
        ntmax = (nt+1)//2   
    corr = np.zeros((nx, ny, ntmax), dtype=dtype)
    for x_seg, y_seg in zip(xs, ys):
        corr_n = _fft_ccf(x_seg, y_seg, ntmax=ntmax, mode=mode, center=center, dtype=dtype)
        logger.debug(corr_n.shape)
        corr += corr_n
    corr = corr / n
    logger.debug(np.sqrt(np.average(corr**2)))
    logger.info(f"Finished calculating cross-correlation.")
    return corr

@memprofit
@timeit
def _gfft_conv(x, y, loop=False, dtype=cp.float32):
    """
    Compute element-wise convolution between two signals on the GPU using FFT.

    This function performs convolution via FFT by converting the inputs to CuPy arrays,
    applying FFT with zero-padding, and then computing the inverse FFT of their product.
    An optional loop-based implementation is available for memory efficiency.

    Parameters:
        x (np.ndarray): First input signal.
        y (np.ndarray): Second input signal.
        loop (bool, optional): If True, compute convolution using a loop; otherwise, vectorized.
        dtype (data-type, optional): Desired CuPy data type (default: cp.float32).

    Returns:
        np.ndarray: Convolution result as a NumPy array.
    """
    print("Doing FFTs on GPU.", file=sys.stderr)
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = x.shape[1]
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    x_f = cp.fft.fft(x, n=2*nt, axis=-1)
    y_f = cp.fft.fft(y, n=2*nt, axis=-1)
    counts = cp.arange(nt, 0, -1, dtype=dtype)**-1
    if loop:
        conv = np.zeros((nx, ny, nt), dtype=np.float32)
        counts = counts[None, :]
        for i in range(nx):
            conv_row = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :nt] * counts
            conv[i, :, :] = conv_row.get()
    else:
        counts = counts[None, None, :]
        conv = cp.fft.ifft(x_f * cp.conj(y_f), axis=-1).real[:, :, :nt] * counts
        conv = conv.get()
    return conv

@memprofit
@timeit
def _sfft_cpsd(x, y, ntmax=None, center=True, loop=True, dtype=np.float64):
    """
    Compute the Cross-Power Spectral Density (CPSD) between two signals using FFT.

    This function calculates the CPSD by applying FFT to the input signals and then 
    computing the product of one FFT with the complex conjugate of the other. Depending 
    on the 'loop' parameter, the computation can be done iteratively or in a vectorized manner.

    Parameters:
        x (np.ndarray): First input signal of shape (n_coords, n_samples).
        y (np.ndarray): Second input signal of shape (n_coords, n_samples).
        ntmax (int, optional): Number of frequency bins to retain; defaults to nt.
        center (bool, optional): If True, mean-center the signals along the time axis.
        loop (bool, optional): If True, use a loop-based computation; otherwise, vectorized.
        dtype (data-type, optional): Desired data type (default: np.float64).

    Returns:
        np.ndarray: The computed CPSD.
    """
    def compute_cpsd(*args):
        i, j, x_f, y_f, ntmax, nt = args
        cpsd_ij = x_f[i] * np.conj(y_f[j])
        cpsd_ij = np.abs(cpsd_ij) / nt
        cpsd_ij = np.average(cpsd_ij)
        return cpsd_ij
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    ntmax = nt if not ntmax else ntmax
    if center:
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
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

@memprofit
@timeit
def _covariance_matrix(positions, dtype=np.float32):
    """
    Calculate the position-position covariance matrix from a set of positions.

    The function centers the input positions by subtracting their mean and then 
    computes the covariance matrix using NumPy's covariance function.

    Parameters:
        positions (np.ndarray): Array of positions with shape (n_coords, n_samples).
        dtype (data-type, optional): Data type for the covariance matrix (default: np.float32).

    Returns:
        np.ndarray: The computed covariance matrix.
    """
    mean = positions.mean(axis=-1, keepdims=True)
    centered_positions = positions - mean
    covmat = np.cov(centered_positions, rowvar=True, dtype=dtype)
    return np.array(covmat)

##############################################################
## DCI and DFI Calculations ##
##############################################################

def _dci(perturbation_matrix, asym=False):
    """
    Calculate the Dynamic Coupling Index (DCI) matrix from a perturbation matrix.

    The DCI is normalized such that the sum of matrix elements equals the number of residues.
    If asym=True, the function returns the asymmetric DCI matrix (difference between DCI and its transpose).

    Parameters:
        perturbation_matrix (np.ndarray): Input perturbation matrix.
        asym (bool, optional): If True, return the asymmetric DCI matrix.

    Returns:
        np.ndarray: The computed DCI matrix.
    """
    dci_val = perturbation_matrix * perturbation_matrix.shape[0] / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci_val = dci_val - dci_val.T
    return dci_val

def _group_molecule_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Compute the DCI for specified groups of atoms relative to the entire molecule.

    For each group in 'groups', the function computes a DCI value by averaging the normalized 
    entries in the perturbation matrix corresponding to that group.

    Parameters:
        perturbation_matrix (np.ndarray): The perturbation matrix.
        groups (list of list): List of groups (each group is a list of atom indices).
        asym (bool, optional): If True, adjust the DCI for asymmetry.

    Returns:
        list: A list of DCI values for each group.
    """
    dcis = []
    dci_tot = perturbation_matrix / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci_tot = dci_tot - dci_tot.T
    for ids in groups:
        top = np.sum(dci_tot[:, ids], axis=-1) * perturbation_matrix.shape[0]
        bot = len(ids)
        dci_val = top / bot
        dcis.append(dci_val)
    return dcis

def _group_group_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Calculate the DCI matrix between different groups of atoms.

    For each pair of groups, the function computes the average normalized perturbation 
    over all atom pairs between the groups.

    Parameters:
        perturbation_matrix (np.ndarray): The perturbation matrix.
        groups (list of list): List of groups (each a list of atom indices).
        asym (bool, optional): If True, compute an asymmetric DCI.

    Returns:
        list: A nested list representing the DCI matrix between groups.
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
            dci_val = top / bot
            temp.append(dci_val)
        dcis.append(temp)
    return dcis

##############################################################
## Elastic Network Model (ENM) Functions ##
##############################################################

@timeit
@memprofit
def _inverse_sparse_matrix_cpu(matrix, k_singular=6, n_modes=20, dtype=None, **kwargs):
    """
    Compute the inverse of a sparse matrix on the CPU using eigen-decomposition.

    This function computes a truncated inverse of the input matrix by calculating its 
    eigenvalues and eigenvectors, inverting the eigenvalues (with the smallest 'k_singular' 
    set to zero), and reconstructing the inverse matrix.

    Parameters:
        matrix (np.ndarray): The input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to compute.
        dtype: Desired data type (default: input matrix.dtype).
        **kwargs: Additional arguments for the eigensolver.

    Returns:
        np.ndarray: The computed inverse matrix.
    """
    kwargs.setdefault('k', n_modes)
    kwargs.setdefault('which', 'SA')
    kwargs.setdefault('tol', 0)
    kwargs.setdefault('maxiter', None)
    if dtype == None:
        dtype = matrix.dtype
    matrix = np.asarray(matrix, dtype=dtype)
    evals, evecs = scipy.sparse.linalg.eigsh(matrix, **kwargs)
    inv_evals = evals**-1
    inv_evals[:k_singular] = 0.0 
    print(evals[:20])
    inv_matrix = np.matmul(evecs, np.matmul(np.diag(inv_evals), evecs.T))
    return inv_matrix

@timeit
@memprofit
def _inverse_matrix_cpu(matrix, k_singular=6, n_modes=100, dtype=None, **kwargs):
    """
    Compute the inverse of a matrix on the CPU using dense eigen-decomposition.

    This function uses NumPy's dense eigenvalue solver to compute the eigen-decomposition 
    of the input matrix, then inverts the eigenvalues (with the smallest 'k_singular' set to zero)
    and reconstructs the inverse matrix.

    Parameters
    ----------
    matrix (np.ndarray): The input matrix.
    k_singular (int, optional): Number of smallest eigenvalues to zero out.
    n_modes (int, optional): Number of eigenmodes to consider.
    dtype: Desired data type (default: input matrix.dtype).
    **kwargs: Additional arguments for the eigenvalue solver.

    Returns
    -------
    np.ndarray: The inverse matrix computed on the CPU.
    """
    if dtype == None:
        dtype = matrix.dtype
    matrix = np.asarray(matrix, dtype=dtype)
    evals, evecs = np.linalg.eigh(matrix, **kwargs)
    evals = evals[:n_modes]
    evecs = evecs[:, :n_modes]
    inv_evals = evals**-1
    inv_evals[:k_singular] = 0.0
    print(evals[:20])
    inv_matrix = np.matmul(evecs, np.matmul(np.diag(inv_evals), evecs.T))
    return inv_matrix

@timeit
@memprofit
def _inverse_sparse_matrix_gpu(matrix, k_singular=6, n_modes=20, dtype=None, **kwargs):
    """
    Compute the inverse of a sparse matrix on the GPU using eigen-decomposition.

    The input matrix is transferred to the GPU and decomposed using CuPy's sparse 
    eigensolver. The eigenvalues are inverted (with the smallest 'k_singular' set to zero) 
    and the inverse matrix is reconstructed.

    Parameters:
        matrix (np.ndarray): The input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to compute.
        dtype: Desired CuPy data type (default: input matrix.dtype).
        **kwargs: Additional arguments for the GPU eigensolver.

    Returns:
        cp.ndarray: The inverse matrix computed on the GPU.
    """
    kwargs.setdefault('k', n_modes)
    kwargs.setdefault('which', 'SA')
    kwargs.setdefault('tol', 0)
    kwargs.setdefault('maxiter', None)
    if dtype == None:
        dtype = matrix.dtype
    matrix_gpu = cp.asarray(matrix, dtype)
    evals_gpu, evecs_gpu = cupyx.scipy.sparse.linalg.eigsh(matrix_gpu, **kwargs)
    inv_evals_gpu = evals_gpu**-1
    inv_evals_gpu[:k_singular] = 0.0  
    print(evals_gpu[:20])
    inv_matrix_gpu = cp.matmul(evecs_gpu, cp.matmul(cp.diag(inv_evals_gpu), evecs_gpu.T))
    return inv_matrix_gpu

@timeit
@memprofit   
def _inverse_matrix_gpu(matrix, k_singular=6, n_modes=100, dtype=None, **kwargs):
    """
    Compute the inverse of a matrix on the GPU using dense eigen-decomposition.

    This function uses CuPy's dense eigenvalue solver to compute the eigen-decomposition 
    of the input matrix, then inverts the eigenvalues (with the smallest 'k_singular' set to zero)
    and reconstructs the inverse matrix.

    Parameters:
        matrix (np.ndarray): The input matrix.
        k_singular (int, optional): Number of smallest eigenvalues to zero out.
        n_modes (int, optional): Number of eigenmodes to consider.
        dtype: Desired CuPy data type (default: input matrix.dtype).
        **kwargs: Additional arguments for the eigenvalue solver.

    Returns:
        cp.ndarray: The inverse matrix computed on the GPU.
    """
    if dtype == None:
        dtype = matrix.dtype
    matrix_gpu = cp.asarray(matrix, dtype)
    evals_gpu, evecs_gpu = cupy.linalg.eigh(matrix_gpu, **kwargs)
    evals_gpu = evals_gpu[:n_modes]
    evecs_gpu = evecs_gpu[:,:n_modes]
    inv_evals_gpu = evals_gpu**-1
    inv_evals_gpu[:k_singular] = 0.0   
    print(evals_gpu[:20])
    inv_matrix_gpu = cp.matmul(evecs_gpu, cp.matmul(cp.diag(inv_evals_gpu), evecs_gpu.T))
    return inv_matrix_gpu

##############################################################
## Miscellaneous Functions ##
##############################################################

def percentile(x):
    """
    Compute the empirical percentile rank for each element in an array.

    The function returns an array where each element is replaced by its percentile 
    rank based on the sorted order of the input.

    Parameters:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: Array of percentile ranks.
    """
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px
    

if __name__ == '__main__':
    pass
