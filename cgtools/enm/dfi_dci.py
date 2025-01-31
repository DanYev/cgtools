#!/bin/python

import numpy as np
import pandas as pd
import time
import sys
from numba import njit

# Usage: dfi_dci.py <cov> <fname> <job_id> <work_dir>
pdb = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]
# job_id = sys.argv[3]
# work_dir = sys.argv[4]


# Inputs amber covariance matrices. Use mass-weighted covariance matrix
def parse_covar(files):
    f = open(files,'r')
    data = f.read()
    data = data.split("\n")
    del data[-1]
    f.close()
    covar = []
    for f in data:
        covar.append(np.asarray(f.split(), float))
    return np.asarray(covar, float)


# @jit(nopython=True, parallel=False)
def calcperturbMatBad(cov, resnum):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float32)
    normL = np.sqrt(np.sum(directions, axis=1))
    direct = (directions.T / normL).T
    perturbMat = np.zeros((resnum, resnum))
    for j in range(int(resnum)):
        for k in range(len(direct)):
            perturbDir = direct[k,:]
            delforce = np.zeros(3 * resnum, dtype=np.float32)
            delforce[3*j:3*j+3] = perturbDir
            delXperbVex = np.dot(cov, delforce)
            delXperbMat = delXperbVex.reshape((resnum, 3))
            delRperbVec = np.sqrt(np.sum(delXperbMat * delXperbMat, axis=1))
            perturbMat[:,j] += delRperbVec[:]
    perturbMat /= 7.0
    perturbMat /= np.sum(perturbMat)
    return perturbMat
    

@njit(parallel=False)
def calcperturbMat(cov, resnum):
    directions = np.array(([1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]), dtype=np.float32)
    normL = np.sqrt(np.sum(directions, axis=1))
    direct = (directions.T / normL).T
    
    perturbMat = np.zeros((resnum, resnum))
    n = resnum
    for k in range(len(direct)):
        f = np.ascontiguousarray(direct[k, :])
        for j in range(n):
            for i in range(n):
                cov_ij = np.ascontiguousarray(cov[3*i:3*i+3, 3*j:3*j+3])
                delta = np.dot(cov_ij, f)
                perturbMat[i,j] += np.sqrt(np.sum(delta * delta))

    # perturbMat /= 7.0
    # perturbMat /= np.sum(perturbMat)
    return perturbMat
    
    
@njit(parallel=False)
def calc_rmsf(cov, resnum):
    rmsf = np.zeros(resnum)
    for k in range(resnum):
        rmsf[k] = np.sqrt(cov[3*k, 3*k] + cov[3*k+1, 3*k+1] + cov[3*k+2, 3*k+2])
    return rmsf
    
    

@njit(parallel=False)
def get_dfi(nrmlperturbMat):
    dfi = np.sum(nrmlperturbMat, axis=1)
    return dfi
    

@njit(parallel=False)
def get_dci(pos, nrmlperturbMat):
    dci=[]
    for p in pos:
        dci.append(nrmlperturbMat[:,p])
    dci=(np.sum(dci,axis=0) / len(pos)) / ((np.sum(nrmlperturbMat, axis=1) / len(nrmlperturbMat)))
    return dci


cov = np.load(f"output/{pdb}/{input_file}.npy")
resnum = int(len(cov) / 3)
cov = np.reshape(cov, (3 * resnum, 3 * resnum))

rmsf = calc_rmsf(cov, resnum)


start_time = time.perf_counter()
pert_mat = calcperturbMat(cov, resnum)
end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(elapsed_time)

dfi = get_dfi(pert_mat)


data = pd.DataFrame()
data["Renum_Res"] = list(range(1, resnum+1))
data["dfi"] = dfi
data["rmsf"] = rmsf
data.to_csv(f"output/{pdb}/{output_file}.dat", index=False)

