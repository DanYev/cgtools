#!/bin/python
import os
import sys
import time
import tracemalloc
import MDAnalysis as mda
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
from scipy import linalg as LA
from scipy.stats import pearsonr
from functools import wraps


def timeit(func):
    """
    A decorator to measure the execution time of a function.
    Parameters:
        func (callable): The function to be timed.
    Returns:
        callable: A wrapped function that prints its execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start the timer
        result = func(*args, **kwargs)  # Call the original function
        end_time = time.perf_counter()  # End the timer
        execution_time = end_time - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.6f} seconds.", file=sys.stderr)
        return result
    return wrapper


def memprofit(func):
    """
    A decorator to memory profile a function.
    Parameters:
        func (callable): The function to be timed.
    Returns:
        callable: A wrapped function that prints its execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()             # Start the profiles
        result = func(*args, **kwargs)  # Call the original function
        current, peak = tracemalloc.get_traced_memory()  # Get the current and peak memory usage
        print(f"Current memory usage after executing '{func.__name__}': {current/1024**2:.2f} MB", file=sys.stderr)
        print(f"Peak memory usage: {peak/1024**2:.2f} MB", file=sys.stderr)
        tracemalloc.stop()
        return result
    return wrapper 


def sfft_corr(x, y, ntmax=1000, center=False, loop=True, dtype=np.float64):
    """
    Compute the correlation function using FFT.
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
    def compute_correlation(*args):
        i, j, x_f, y_f, ntmax, nt = args
        return ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real / nt    

    tracemalloc.start()
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
    # Compute the FFT-based correlation via CPSD
    if loop:
        corr = np.zeros((nx, ny, ntmax), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                corr[i, j] = compute_correlation(i, j, x_f, y_f, ntmax, nt)
    else:
        corr = np.einsum('it,jt->ijt', x_f, np.conj(y_f))
        corr = ifft(corr, axis=-1).real / nt
    current, peak = tracemalloc.get_traced_memory()
    return corr.real


def pfft_corr(x, y, ntmax=1000, center=False, dtype=np.float64):
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
    # Helper functions for parrallelizing
    def compute_correlation(i, j, x_f, y_f, ntmax, nt):
        return ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real / nt

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
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
    # Compute the FFT-based correlation via CPSD
    corr = parallel_fft_correlation(x_f, y_f, ntmax, nt)
    return corr.real


@memprofit
@timeit
def fft_corr(*args, mode='parallel', **kwargs):
    if mode == 'serial':
        return sfft_corr(*args, **kwargs)
    if mode == 'parallel':
        return pfft_corr(*args, **kwargs)
    raise ValueError("Currently 'mode' should be 'serial' or 'parallel'.")



def calc_covmats(f='../traj.trr', s='../traj.pdb', n=1, b=000000, dtype=np.float32):
    """
    Calculate the position-position covariance matrix from a GROMACS trajectory file.
    
    Parameters:
        f (str): Path to the GROMACS trajectory file.
        s (str): Path to the corresponding topology file.
        b (float): Time of first frame to read from trajectory (default unit ps)
    """
    # Load trajectory
    u = mda.Universe(s, f)
    selection = u.atoms
    # Extract positions efficiently
    positions = np.array(
        [selection.positions.flatten() for ts in u.trajectory if ts.time > b], dtype=dtype
    )
    # Transpose for better memory access (n_coords, n_frames)
    positions = np.ascontiguousarray(positions.T)  # Ensures efficient memory layout
    # Compute the mean across frames
    # mean_positions = positions.mean(axis=1, keepdims=True)  # Shape: (n_coords, 1)
    # Split into `n` segments along frames (axis=1)
    trajs = np.array_split(positions, n, axis=-1)
    # Process each segment
    for idx, traj in enumerate(trajs, start=1):
        print(f"Processing matrix {idx}", file=sys.stderr)
        # Center the data by removing mean
        mean_traj = traj.mean(axis=-1, keepdims=True)  # Shape: (n_coords, 1)
        centered_positions = traj - mean_traj  # Broadcasting subtraction
        # Compute covariance matrix (n_coords x n_coords)
        covariance_matrix = np.cov(centered_positions, rowvar=True, dtype=dtype)
        # Save covariance matrix
        np.save(f'covmat_{idx}.npy', covariance_matrix)
        print(f"Covariance matrix {idx} saved to 'covmat_{idx}.npy'", file=sys.stderr)
        
        
def calc_power_spectrum_xv(resp_ids, pert_ids,  f='../traj.trr', s='../traj.pdb', n=1, b=000000, dtype=np.float32):
    """
    Calculate the position-velocity power spectrum from a GROMACS trajectory file.
    
    Parameters:
        f (str): Path to the GROMACS trajectory file.
        s (str): Path to the corresponding topology file.
        b (float): Time of first frame to read from trajectory (default unit ps)
    """
    # Load trajectory
    u = mda.Universe(s, f)
    # If IDs are not given the use all atoms
    if resp_ids:
        resp_selection = u.atoms[resp_ids]
    else:
        resp_selection = u.atoms
    if pert_ids:
        pert_selection = u.atoms[pert_ids]
    else:
        pert_selection = u.atoms
    # Extract positions and velocities efficiently
    skip_step = 1
    positions = np.array(
        [resp_selection.positions.flatten() for ts in u.trajectory[::skip_step] if ts.time > b], dtype=dtype
    )
    velocities = np.array(
        [pert_selection.velocities.flatten() for ts in u.trajectory[::skip_step] if ts.time > b], dtype=dtype
    )
    # Transpose for memory efficiency (shape: (n_coords, n_frames))
    positions = np.ascontiguousarray(positions.T)
    print(positions.shape)
    velocities = np.ascontiguousarray(velocities.T)
    # Center data (mean subtraction)
    positions -= positions.mean(axis=-1, keepdims=True)  # Shape: (n_coords, n_frames)
    velocities -= velocities.mean(axis=-1, keepdims=True)
    # Split trajectories into `n` segments along frames (axis=1)
    trajs_pos = np.array_split(positions, n, axis=-1)
    trajs_vel = np.array_split(velocities, n, axis=-1)
    # Compute power spectra for each segment
    cpsd_list = []
    for pos, vel in zip(trajs_pos, trajs_vel):
        cpsd = fftcorr(pos, pos)
        cpsd_list.append(cpsd)
    # Average CPSD across segments
    cpsd_avg = np.mean(cpsd_list, axis=0)  # Take magnitude for power spectrum
    # corr_xv = np.average(cpsd_avg, axis=-1)
    # corr_xv = N * np.abs(ifft(cpsd, n=2*N, axis=-1))
    corr_xv = cpsd_avg[:,:,0]
    np.save(f'corr_xv.npy', corr_xv)
    return corr_xv


def parse_covar_dat(file, dtype=np.float32):
    """
    Reads covar.dat filed generated by GROMACS
    """
    df = pd.read_csv(file, sep='\\s+', header=None)
    covariance_matrix = np.asarray(df, dtype=dtype)
    resnum = int(np.sqrt(len(covariance_matrix) / 3))
    covariance_matrix = np.reshape(covariance_matrix, (3*resnum, 3*resnum))
    return covariance_matrix
    
    
@timeit    
@njit(parallel=False)
def get_perturbation_matrix_old(covariance_matrix, resnum, dtype=np.float32):
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
    
@timeit
def calc_td_perturbation_matrix_cpu(corr_xv, dtype=np.float32):
    """
    Calculates perturbation matrix from a covariance matrix or a hessian on CPU
    The result is normalized such that the total sum of the matrix elements is equal to 1
    """
    n = corr_xv.shape[0] // 3
    blocks = corr_xv.reshape(n, 3, n, 3).swapaxes(1, 2)
    perturbation_matrix = np.sum(blocks**2, axis=(-2, -1))
    perturbation_matrix = np.sqrt(perturbation_matrix)
    perturbation_matrix /= np.sum(perturbation_matrix)
    return perturbation_matrix    


def calc_perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = calc_perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
    return pertmat


def calc_td_perturbation_matrix(covariance_matrix, dtype=np.float32):
    pertmat = calc_td_perturbation_matrix_cpu(covariance_matrix, dtype=dtype)
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
    

def calc_group_molecule_dci(perturbation_matrix, groups=[[]], asym=False):
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
    
    
def calc_group_group_dci(perturbation_matrix, groups=[[]], asym=False):
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
    

def calc_dci_asymmetry(dci):
    dci_asymmetry = dci - dci.T
    return dci_asymmetry    
    
 
def percentile(x):
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px
    
    
def save_1d_data(data, ids=[], fpath='dfi.xvg'): 
    """ 
    Saves 1d data like DFI in the GROMACS's .xvg format
    Input: 
    data: list or numpy array
        y-column
    ids: list or numpy array
        x-column
    fpath: string
        Path to the file to save
    ------
    """
    ids = list(ids)
    if not ids:
        ids = np.arange(1, len(data)+1).astype(int)
    df = pd.DataFrame({'ids': ids, 'data': data})
    df.to_csv(fpath, index=False, header=None, float_format='%.3E', sep=' ')
    
    
def save_2d_data(data, ids=[], fpath='dfi.xvg'): 
    """ 
    Saves 2d data like group-group DCI in the GROMACS's .xvg format
    Input: 
    data: list or numpy array
    fpath: string
        Path to the file to save
    ------
    """
    df = pd.DataFrame(data)
    df.to_csv(fpath, index=False, header=None, float_format='%.3E', sep=' ')
    

if __name__ == '__main__':
    pass



