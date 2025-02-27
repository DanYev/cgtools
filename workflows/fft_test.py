import os
import psutil
import sys
import time
import tracemalloc
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import MDAnalysis as mda
import cupy as cp
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.fft import fft, ifft, rfft, irfft, fftfreq, fftshift, ifftshift
from scipy import linalg as LA
from scipy.stats import pearsonr
from functools import wraps
from reforge import gmxmd, lrt
from reforge.plotting import plot_mean_sem, plot_heatmaps


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


def sfft_corr(x, y, ntmax=None, center=False, loop=True, dtype=np.float64):
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
    print("Doing FFTs in serial", file=sys.stderr)
    # Helper functions
    def compute_correlation(*args):
        i, j, x_f, y_f, ntmax, nt = args
        return ifft(x_f[i] * np.conj(y_f[j]), axis=-1)[:ntmax].real / nt    

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
    # Compute the FFT-based correlation via CPSD
    if loop:
        corr = np.zeros((nx, ny, ntmax), dtype=dtype)
        for i in range(nx):
            for j in range(ny):
                corr[i, j] = compute_correlation(i, j, x_f, y_f, ntmax, nt)
    else:
        corr = np.einsum('it,jt->ijt', x_f, np.conj(y_f))
        corr = ifft(corr, axis=-1).real / nt
    return corr


def pfft_corr(x, y, ntmax=None, center=False, dtype=np.float64):
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
    print("Doing FFTs in parallel", file=sys.stderr)
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
    ntmax = nt if not ntmax else ntmax
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Compute FFT along the last axis as an (nx, nt) array
    x_f = fft(x, axis=-1)
    y_f = fft(y, axis=-1)
    # Compute the FFT-based correlation via CPSD
    corr = parallel_fft_correlation(x_f, y_f, ntmax, nt)
    return corr
  

def gfft_corr(x, y, ntmax=None, center=True, dtype=cp.float64):
    print("Doing FFTs on GPU", file=sys.stderr)
    def compute_correlation_gpu(*args):
        i, j, x_f, y_f, ntmax, nt = args
        return cp.fft.ifft(x_f[i] * cp.conj(y_f[j]), axis=-1)[:ntmax].real / nt  
    nt = x.shape[-1]
    nx = x.shape[0]
    ny = y.shape[0]
    ntmax = nt if ntmax is None else ntmax
    corr = np.zeros((nx, ny, ntmax), dtype=np.float32)
    if center:  # Mean-center the signals
        x = x - np.mean(x, axis=-1, keepdims=True)
        y = y - np.mean(y, axis=-1, keepdims=True)
    # Convert NumPy arrays to CuPy arrays
    x = cp.asarray(x, dtype=dtype)
    y = cp.asarray(y, dtype=dtype)
    # Compute FFT along the last axis
    x_f = cp.fft.fft(x, axis=-1)
    y_f = cp.fft.fft(y, axis=-1)
    # Compute the FFT-based correlation via CPSD 
    for i in range(nx):
        corr_row = cp.fft.ifft(x_f[i, None, :] * cp.conj(y_f), axis=-1).real[:, :ntmax] / nt 
        corr[i, :, :] = corr_row.get()
    return corr


@memprofit
@timeit
def fft_corr(*args, mode='parallel', **kwargs):
    if mode == 'ser':
        return sfft_corr(*args, **kwargs)
    if mode == 'par':
        return pfft_corr(*args, **kwargs)
    if mode == 'gpu':
        return gfft_corr(*args, **kwargs)
    raise ValueError("Currently 'mode' should be 'ser', 'par' or 'gpu'.")


@memprofit
@timeit
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


@memprofit
@timeit
def read_trajectory(resp_ids, pert_ids, f='../traj.trr', s='../traj.pdb',  b=0, e=10000000, skip_rate=1, dtype=np.float32):
    print(f"Reading trajectory.", file=sys.stderr)
    def in_range(ts, b, e): # Check if ts.time is within the range (b, e)
        return b < ts.time < e
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
    positions = np.array(
        [resp_selection.positions.flatten() for ts in u.trajectory[::skip_rate] if in_range(ts, b, e)], dtype=dtype
    )
    velocities = np.array(
        [pert_selection.velocities.flatten() for ts in u.trajectory[::skip_rate] if in_range(ts, b, e)], dtype=dtype
    )  
    # Transpose for memory efficiency (shape: (n_coords, n_frames))
    positions = np.ascontiguousarray(positions.T)
    velocities = np.ascontiguousarray(velocities.T)
    print(f"Finished reading trajectory.", file=sys.stderr)
    return positions, velocities


@memprofit
@timeit
def calc_ccf(x, y, ntmax=None, n=1, mode='parallel', center=True, dtype=np.float32):
    """
    Calculate the average cross-correlation function of x and y by splitting them into n segments
    
    Parameters:
        f (str): Path to the GROMACS trajectory file.
        s (str): Path to the corresponding topology file.
        b (float): Time of first frame to read from trajectory (default unit ps)
    """
    print(f"Calculating cross-correlation.", file=sys.stderr)
    # Split trajectories into `n` segments along frames (axis=1)
    trajs_pos = np.array_split(x, n, axis=-1)
    trajs_vel = np.array_split(y, n, axis=-1)
    # Compute correlation for each segment
    corr_list = []
    for pos, vel in zip(trajs_pos, trajs_vel):
        corr = fft_corr(pos, vel, ntmax=ntmax, mode=mode, center=center, dtype=dtype)
        corr_list.append(corr)
    # Average corr across segments
    corr_avg = np.mean(corr_list, axis=0)  # Take magnitude for power spectrum
    corr = corr_avg
    np.save(f'corr.npy', corr)
    print(f"Finished calculating cross-correlation.", file=sys.stderr)
    return corr


def test_synt_data():
    # Example 2D data
    T = 1
    N = 100000
    t = np.linspace(0, T, N)
    a = np.cos(10000 * 2*np.pi * t) + 1.0 * np.random.randn(N) 
    b = np.cos(10300 * 2*np.pi * t) + 1.0 * np.random.randn(N) 
    n = 10
    at = np.tile(a, (n, 1))
    bt = np.tile(b, (n, 1))
    # CALC CORR
    # corr_ftt = fft_corr(at, bt, mode='ser')
    corr_ftt = sfft_cpsd(at, bt, ntmax=50000, loop=False)
    # Get available CPU cores for this process
    available_cores = os.sched_getaffinity(0)
    print(f"Process can run on CPU cores: {available_cores}")
    # Get the current process
    process = psutil.Process()
    # Get the CPU core the process is currently running on
    cpu_core = process.cpu_num()
    print(f"Process is running on CPU core: {cpu_core}")
    # SANITY CHECK
    av1 = np.average(a*b)
    corr_av = np.average(corr_ftt, axis=(0, 1))
    av2 = corr_av[0]
    rel_err = np.abs(av1 - av2) / np.average([av1, av2])
    # PLOTTING
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(corr_av)), corr_av)
    plt.grid()
    fig.savefig('png/test.png')
    plt.close()
    print("Relative error:", f"{rel_err:.2e}")
    exit()


def makefig(datas):
    fig = plt.figure(figsize=(10, 5))
    for data in datas:
        # data -= np.average(data)
        # data[np.abs(data) < 0.2] = 0
        plt.plot(np.arange(len(data)), data)
    plt.grid()
    fig.savefig('png/plot.png')
    plt.close()


def make_animation(infile, nframes, outfile="data/ani.mp4"):
    print("Working on animation")
    matrices = []
    mat_t = np.load(infile)
    nframes = 10
    for i in range(nframes):
        mat = mat_t[:,:,i]
        pertmat = lrt.calc_perturbation_matrix(mat)
        dci = lrt.calc_dci(pertmat)
        # dci = pertmat / np.sum(pertmat) * pertmat.shape[0] * pertmat.shape[1]
        matrices.append(dci)
    fig, ax = plt.subplots()
    xmax = matrices[0].shape[0]
    ymax = matrices[0].shape[1]
    line_positions = [30, 60]
    img = ax.imshow(matrices[0], cmap='bwr', vmin=0, vmax=2,)
    ax.vlines(line_positions, ymin=0, ymax=xmax, colors='black', linestyles='dashed', linewidth=1)
    ax.hlines(line_positions, xmin=0, xmax=ymax, colors='black', linestyles='dashed', linewidth=1)   
    def update(frame):
        img.set_array(matrices[frame])
        # ax.set_title(f"Matrix {frame + 1}/{N}")
        return img,
    ani = animation.FuncAnimation(fig, update, frames=nframes, interval=50, blit=False)  
    ani.save(outfile, writer="ffmpeg") # Save as mp4


def test_blac():   
    bdir = os.getcwd()
    os.chdir(run.rundir) 
    nskip = 1
    ntmax = 500
    corr_file = os.path.join(run.rundir, 'corr.npy') 
    # # CALC CCF   
    # pos, vel = read_trajectory(resp_ids=[], pert_ids=[], f='traj.trr', s='traj.pdb', b=0, e=10000, skip_rate=nskip, dtype=np.float32)
    # corr = calc_ccf(vel, vel, ntmax=ntmax, n=1, mode='gpu', center=True)
    # np.save(corr_file, corr)
    # CALC DCI DFI
    if 1:
        make_animation(infile=corr_file, nframes=ntmax, outfile=f'{bdir}/data/vv_ccf.mp4')
    if 0:
        corr_t = np.load(corr_file)
        for i in range(ntmax):
            corr = corr_t[:,:,i]
            pertmat = lrt.calc_td_perturbation_matrix(corr)
            dci = pertmat / np.sum(pertmat) * pertmat.shape[0] * pertmat.shape[1]
            # dci = lrt.calc_dci(pertmat)
            lrt.save_2d_data(dci, fpath=f'dci_{i}.csv', sep=',')
            dci_file = os.path.join(run.rundir, f'dci_{i}.csv')
            plot_heatmaps([dci_file], f'{bdir}/png/dci_{i}.png', vmin=0, vmax=2, cmap='bwr')
    os.chdir(bdir)
    # dfi = lrt.calc_dfi(pertmat) 
    # lrt.save_1d_data(dfi, fpath='dfi.csv', sep=',') 
    # # PLOTTING
    # dfi_file = os.path.join(run.rundir, 'dfi.csv')
    # plot_mean_sem([dfi_file], f'png/dfi_skip_{nskip}.png', ) 
    # # CORR
    # corr_file = os.path.join(run.rundir, 'corr.npy') 
    # corr = np.load(corr_file)
    # datas = []
    # n = 256
    # dn = 1
    # for i in range(n, n+dn):
    #     datas.append(corr[i, :, 0])
    # # print(datas)
    # makefig(datas)



#############################################################################
system = gmxmd.gmxSystem('systems', 'test_1')
run = system.initmd('mdrun_1')
test_blac()  
# test_synt_data() 

