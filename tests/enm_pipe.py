import os
import sys
import time
import logging
import numpy as np
import cupy as cp
import pandas as pd
import cupy.linalg
import cupyx.scipy.sparse.linalg
import scipy.sparse
import scipy.linalg
import MDAnalysis as mda
from cgtools.gmxmd import gmxSystem
from cgtools.utils import timeit, memprofit, logger, cuda_detected
from cgtools._actual_math import mycmath, mypymath, legacy
from cgtools import io, mdm
from cgtools.plotting import *

logger.setLevel(logging.DEBUG)
sysdir = 'systems'
sysname = '1btl' # 1btl # ribosome
mdsys = gmxSystem(sysdir, sysname)
# pdb = os.path.join(mdsys.wdir, 'ref.pdb')
pdb = mdsys.inpdb
# ref_pertmat_path = os.path.join(mdsys.datdir, 'pertmat_av.npy')
# ref_covmat_path = os.path.join(mdsys.datdir, 'covmat_av.npy')
# ref_pertmat = np.load(ref_pertmat_path).astype('double')
# ref_covmat = np.load(ref_covmat_path).astype('double')

atoms = io.pdb2atomlist(pdb)
mask = ["CA", "C1'"] 
bb = atoms.mask(mask, mode='name')
xs = np.array(bb.xs)
ys = np.array(bb.ys)
zs = np.array(bb.zs)
n = len(bb)

nt = 1
# covmat = np.tile(ref_covmat, (nt, nt))

vecs = np.array(bb.vecs)
hess = mycmath._hessian(vecs, cutoff=18, spring_constant=1e3, dd=0,) 
exit()
covmat = mypymath._inverse_sparse_matrix_cpu(hess, k_singular=6, n_modes=20)
# covmat_gpu = mypymath._inverse_matrix_gpu(hess, k_singular=6, n_modes=20, gpu_dtype=cp.float32)
# covmat_gpu = mypymath._inverse_sparse_matrix_gpu(hess, k_singular=6, n_modes=20, gpu_dtype=cp.float32)
# covmat = covmat_gpu.get()
# covmat = mypymath._inverse_sparse_matrix_gpu(hess)
pertmat =  mdm.td_perturbation_matrix(covmat)

# Plotting
dfi = mdm.dfi(pertmat)
dfi = np.split(dfi, nt)
dfi = np.sum(dfi, axis=0)
fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
ax.plot(np.arange(len(dfi)), dfi, label='dfi')
set_ax_parameters(ax, xlabel='Residue', ylabel='DFI')
plot_figure(fig, ax, figpath='png/dfi.png',)
dasdasdjjjllllllllllllllllllllll