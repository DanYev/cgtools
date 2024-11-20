#!/bin/python
import os
import sys
import MDAnalysis as mda
import numpy as np
import pandas as pd
from numba import njit
from scipy import linalg as LA
from scipy.stats import pearsonr


def calculate_covariance_matrix(f, s, o="covmat.npy"):
    """
    Calculate the covariance matrix from a GROMACS trajectory file.
    
    Parameters:
        f (str): Path to the GROMACS .trr file.
        s (str): Path to the corresponding topology file.
        o (str): Path to save the covariance matrix (default: 'covariance_matrix.npy').
    """
    # Load the trajectory and topology
    u = mda.Universe(s, f)
    
    # Select atoms to analyze (e.g., backbone atoms)
    selection = u.select_atoms('name BB or name BB1') 
    selection = u.select_atoms('segid 0 and name BB') 
    print(selection.positions)
    exit()
    n_atoms = selection.n_atoms
    n_coords = n_atoms * 3  # Each atom contributes 3 coordinates (x, y, z)
    
    # Collect all frames into a single array
    positions = []
    for ts in u.trajectory:
        # Flatten the positions for the selected atoms
        positions.append(selection.positions.flatten())
    
    positions = np.array(positions)  # Shape: (n_frames, n_coords)

    # Center the data by subtracting the mean for each coordinate
    mean_positions = positions.mean(axis=0)
    centered_positions = positions - mean_positions
    
    # Calculate the covariance matrix
    covariance_matrix = np.cov(centered_positions, rowvar=False)  # Shape: (n_coords, n_coords)
    
    # Save the covariance matrix
    np.save(o, covariance_matrix)
    print(f"Covariance matrix saved to {output_file}")

    return covariance_matrix


def parse_covar_dat(file):
    df = pd.read_csv(file, sep='\\s+', header=None)
    covariance_matrix = np.asarray(df, dtype=np.float64)
    resnum = int(np.sqrt(len(covariance_matrix) / 3))
    covariance_matrix = np.reshape(covariance_matrix, (3*resnum, 3*resnum))
    return covariance_matrix, resnum
    
    
@njit(parallel=False)
def get_perturbation_matrix(covariance_matrix, resnum):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float64)
    directions = directions.T / np.sqrt(np.sum(directions, axis=1)).T # normalizing directions
    directions = directions.T
    perturbation_matrix = np.zeros((resnum, resnum))
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
    

@njit(parallel=False)
def get_just_dfi(cov, resnum):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float64)
    norm = np.sqrt(np.sum(directions, axis=1))
    directions = (directions.T / norm).T
    dfi = np.zeros(resnum)
    for k in range(7):
        f = np.ascontiguousarray(directions[k, :])
        for j in range(resnum):
            pm_ij = 0
            for i in range(resnum):
                cov_ij = np.ascontiguousarray(cov[3*i:3*i+3, 3*j:3*j+3])
                delta = np.dot(cov_ij, f)
                pm_ij += np.sqrt(np.sum(delta * delta))
            dfi[j] += pm_ij
    return dfi    
    

def get_dfi(perturbation_matrix):
    dfi = np.sum(perturbation_matrix, axis=-1)
    return dfi
    

def get_full_dci(perturbation_matrix):
    dci = perturbation_matrix / np.sum(perturbation_matrix, axis=-1, keepdims=True)
    return dci    
    

def get_dci_asymmetry(dci):
    dci_asymmetry = dci - dci.T
    return dci_asymmetry    
    
 
def percentile(x):
    sorted_x = np.argsort(x)
    px = np.zeros(len(x))
    for n in range(len(x)):
        px[n] = np.where(sorted_x == n)[0][0] / len(x)
    return px
    
    
def get_system_idx(run):
    names = run.split('_')
    system = '_'.join(names[0:3])
    idx = names[3]
    return system, idx


if __name__ == '__main__':
    pass


    

    
    

    



