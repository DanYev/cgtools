#!/usr/bin/env python
"""
File: mdm.py
Description:
    This module provides a unified interface for molecular dynamics (MD) and 
    structural analysis routines within the reForge package. It wraps a variety 
    of operations including FFT-based cross-correlation, covariance matrix 
    computation, perturbation matrix calculations (for DFI/DCI metrics), and 
    elastic network model (ENM) Hessian evaluations. Both CPU and GPU implementations 
    are supported, with fallbacks to CPU methods if CUDA is not available.

Usage Example:
    >>> import numpy as np
    >>> from mdm import fft_ccf, calc_and_save_covmats, inverse_matrix
    >>> 
    >>> # Compute FFT-based cross-correlation function in serial mode
    >>> ccf = fft_ccf(signal1, signal2, mode='serial')
    >>> 
    >>> # Calculate and save covariance matrices from trajectory positions
    >>> calc_and_save_covmats(positions, outdir='./covmats', n=5)
    >>> 
    >>> # Compute the inverse of a matrix using the unified inversion wrapper
    >>> inv_mat = inverse_matrix(matrix, device='cpu_sparse')

Requirements:
    - Python 3.x
    - MDAnalysis
    - NumPy
    - CuPy (if GPU routines are used)
    - Pandas
    - reForge utilities (timeit, memprofit, logger, cuda_detected)
    - reForge actual_math modules (mycmath, mypymath)

Author: DY
Date: YYYY-MM-DD
"""

import os
import sys
import MDAnalysis as mda
import numpy as np
import cupy as cp
import pandas as pd
from reforge.utils import timeit, memprofit, logger, cuda_detected
from reforge.actual_math import mycmath, mypymath


def fft_ccf(*args, mode='serial', **kwargs):
    """
    Unified wrapper for FFT-based correlation functions.

    This function dispatches to one of the internal FFT correlation routines 
    based on the specified mode:
      - 'serial' for _sfft_ccf,
      - 'parallel' for _pfft_ccf, or
      - 'gpu' for _gfft_ccf.

    Parameters
    ----------
    *args : 
        Positional arguments for the chosen correlation function.
    mode : str, optional
        Mode to use ('serial', 'parallel', or 'gpu'). Default is 'serial'.
    **kwargs :
        Additional keyword arguments for the internal routines.

    Returns
    -------
    np.ndarray
        The computed correlation function.

    Raises
    ------
    ValueError
        If an unsupported mode is specified.
    """
    if mode == 'serial':
        return mypymath._sfft_ccf(*args, **kwargs)
    if mode == 'parallel':
        return mypymath._pfft_ccf(*args, **kwargs)
    if mode == 'gpu':
        return mypymath._gfft_ccf(*args, **kwargs).get()
    raise ValueError("Mode must be 'serial', 'parallel' or 'gpu'.")


def covariance_matrix(positions, dtype=np.float64):
    """
    Compute the covariance matrix from trajectory positions.

    This function is a wrapper for mypymath's covariance matrix routine.

    Parameters
    ----------
    positions : np.ndarray
        Array of position coordinates.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed covariance matrix.
    """
    return mypymath._covariance_matrix(positions, dtype=dtype)
    

def calc_and_save_covmats(positions, outdir, n=1, outtag='covmat', dtype=np.float32):
    """
    Calculate and save covariance matrices by splitting a trajectory into segments.

    The trajectory positions are split into 'n' segments along the frame axis. For each 
    segment, the position-position covariance matrix is computed and saved as a .npy file.

    Parameters
    ----------
    positions : np.ndarray
        Array of positions with shape (n_coords, n_frames).
    outdir : str
        Directory where the covariance matrices will be saved.
    n : int, optional
        Number of segments to split the trajectory into (default is 1).
    outtag : str, optional
        Base tag for output file names (default is 'covmat').
    dtype : data-type, optional
        Desired data type for covariance computation (default is np.float32).

    Returns
    -------
    None
    """
    trajs = np.array_split(positions, n, axis=-1)  # Split into n segments along the frame axis
    for idx, traj in enumerate(trajs, start=1):
        logger.info(f"Processing covariance matrix {idx}")
        covmat = covariance_matrix(traj, dtype=dtype)
        outfile = os.path.join(outdir, f'{outtag}_{idx}.npy')
        np.save(outfile, covmat)
        logger.info(f"Saved covariance matrix to {outfile}")
        

##############################################################
## DFI / DCI Calculations
##############################################################

def perturbation_matrix(covariance_matrix, dtype=np.float64):
    """
    Compute the perturbation matrix from a covariance matrix.

    This wrapper calls the appropriate function from mycmath for the given data type.
    TODO: Improve type handling and add GPU support.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed perturbation matrix.
    """
    covariance_matrix = covariance_matrix.astype(np.float64)
    if dtype == np.float64:
        pertmat = mycmath._perturbation_matrix(covariance_matrix)
    return pertmat


def td_perturbation_matrix(covariance_matrix, dtype=np.float64):
    """
    Compute the block-wise (td) perturbation matrix from a covariance matrix.

    This wrapper calls the corresponding function from mycmath.
    TODO: Improve type handling and add GPU support.

    Parameters
    ----------
    covariance_matrix : np.ndarray
        The covariance matrix.
    dtype : data-type, optional
        Desired data type (default is np.float64).

    Returns
    -------
    np.ndarray
        The computed block-wise perturbation matrix.
    """
    covariance_matrix = covariance_matrix.astype(np.float64)
    if dtype == np.float64:
        pertmat = mycmath._td_perturbation_matrix(covariance_matrix)
    return pertmat    


def dfi(perturbation_matrix):
    """
    Calculate the Dynamic Flexibility Index (DFI) from a perturbation matrix.

    The DFI is computed by summing the rows of the perturbation matrix and then 
    normalizing such that the total sum equals the number of residues.

    Parameters
    ----------
    perturbation_matrix : np.ndarray
        The perturbation matrix.

    Returns
    -------
    np.ndarray
        The DFI values.
    """
    dfi_val = np.sum(perturbation_matrix, axis=-1)
    dfi_val *= len(dfi_val)
    return dfi_val
    

def dci(perturbation_matrix, asym=False):
    """
    Calculate the Dynamic Coupling Index (DCI) from a perturbation matrix.

    The DCI is computed by scaling the perturbation matrix such that the total sum of its 
    elements equals the number of residues. Optionally, an asymmetric DCI (difference between 
    DCI and its transpose) can be returned.

    Parameters
    ----------
    perturbation_matrix : np.ndarray
        The perturbation matrix.
    asym : bool, optional
        If True, return an asymmetric version (default is False).

    Returns
    -------
    np.ndarray
        The DCI matrix.
    """
    dci_val = perturbation_matrix * perturbation_matrix.shape[0] / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    if asym:
        dci_val = dci_val - dci_val.T
    return dci_val    
    

def group_molecule_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Calculate the DCI between groups of atoms and the remainder of the molecule.

    For each group in 'groups', the DCI is computed by summing the coupling values between 
    the group and all residues, then normalizing by the number of atoms in the group.

    Parameters
    ----------
    perturbation_matrix : np.ndarray
        The perturbation matrix.
    groups : list of lists, optional
        A list of groups, each containing indices of atoms (default is a list with an empty list).
    asym : bool, optional
        If True, use the asymmetric DCI (default is False).

    Returns
    -------
    list of np.ndarray
        A list of DCI values for each group.
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
    
    
def group_group_dci(perturbation_matrix, groups=[[]], asym=False):
    """
    Calculate the inter-group DCI matrix.

    For each pair of groups specified in 'groups', compute the average DCI between atoms 
    in the first group and atoms in the second group. Optionally, an asymmetric DCI can be computed.

    Parameters
    ----------
    perturbation_matrix : np.ndarray
        The perturbation matrix.
    groups : list of lists, optional
        A list of groups, each containing indices of atoms (default is a list with an empty list).
    asym : bool, optional
        If True, compute the asymmetric DCI (default is False).

    Returns
    -------
    list of lists
        A 2D list containing the DCI values between each pair of groups.
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
## Elastic Network Model (ENM)
##############################################################

def hessian(vecs, cutoff, spring_constant, dd):
    """
    Compute the Hessian matrix using an elastic network model.

    This function is a simple wrapper for the internal _hessian routine from mycmath.
    TODO: Improve type handling and add GPU support.

    Parameters
    ----------
    vecs : np.ndarray
        Coordinate matrix of shape (n, 3) where each row corresponds to a residue.
    cutoff : float
        Distance cutoff threshold.
    spring_constant : float
        Base spring constant.
    dd : int
        Exponent modifier for the inverse distance.

    Returns
    -------
    np.ndarray
        The computed Hessian matrix.
    """
    return mycmath._hessian(vecs, cutoff, spring_constant, dd)


def inverse_matrix(matrix, device='cpu_sparse', k_singular=6, n_modes=100, dtype=None, **kwargs):
    """
    Unified wrapper for computing the inverse of a matrix via eigen-decomposition.

    Depending on the 'device' parameter, the function selects an appropriate routine:
    
      - 'cpu_sparse' (default): Uses a sparse eigensolver on the CPU.
      - 'cpu_dense': Uses a dense CPU eigen-decomposition.
      - 'gpu_sparse': Uses a sparse eigensolver on the GPU.
      - 'gpu_dense': Uses a dense eigen-decomposition on the GPU.

    If a GPU routine is requested but CUDA is unavailable or an error occurs,
    the function falls back to the 'cpu_sparse' routine.

    Parameters
    ----------
    matrix : np.ndarray
        The input matrix.
    device : str, optional
        Inversion method to use ('cpu_sparse', 'cpu_dense', 'gpu_sparse', or 'gpu_dense').
        Default is 'cpu_sparse'.
    k_singular : int, optional
        Number of smallest eigenvalues to set to zero (default is 6).
    n_modes : int, optional
        Number of eigenmodes to compute/consider.
    dtype : data-type, optional
        Desired data type for computations (default: uses matrix.dtype).
    **kwargs :
        Additional keyword arguments for the eigensolver.

    Returns
    -------
    np.ndarray or cp.ndarray
        The computed inverse matrix. A NumPy array is returned for CPU routines and a 
        CuPy array for GPU routines.
    """
    if dtype is None:
        dtype = matrix.dtype

    if device.lower().startswith('gpu'):
        try:
            import cupy as cp
            if not cp.cuda.is_available():
                raise RuntimeError("CUDA not available.")
            if device.lower() == 'gpu_sparse':
                return _inverse_sparse_matrix_gpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)
            elif device.lower() == 'gpu_dense':
                return _inverse_matrix_gpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)
            else:
                logger.info("Unknown GPU method; falling back to CPU sparse inversion.")
                return _inverse_sparse_matrix_cpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)
        except Exception as e:
            logger.info(f"GPU inversion failed with error '{e}'. Falling back to CPU sparse inversion.")
            return _inverse_sparse_matrix_cpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)

    elif device.lower() == 'cpu_dense':
        return _inverse_matrix_cpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)
    
    else:
        return _inverse_sparse_matrix_cpu(matrix, k_singular=k_singular, n_modes=n_modes, dtype=dtype, **kwargs)


##############################################################
## Miscellaneous Functions
##############################################################

def percentile(x):
    """
    Compute the percentile ranking for each element in an array.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        An array containing the percentile (from 0 to 1) of each element in x.
    """
    sorted_idx = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        # Find the rank (as a fraction)
        px[n] = np.where(sorted_idx == n)[0][0] / len(x)
    return px
    
    
if __name__ == '__main__':
    pass
