import os
import sys
import time
import numpy as np
import cupy as cp
import pandas as pd
import cupy.linalg
import cupyx.scipy.sparse.linalg
import scipy.sparse
import scipy.linalg
import MDAnalysis as mda
from cgtools.gmxmd import gmxSystem
from cgtools.utils import timeit, memprofit
from cgtools.mycmath import mycmath, legacy
from cgtools import io, mdm
from cgtools.plotting import *


def plot_dfi(system):
    # Pulling data
    datas, errs = pull_data('dfi*')
    param = {'lw':2}
    xs = [np.arange(len(data)) for data in datas]
    datas = [data*len(data) for data in datas]
    errs = [err*len(err) for err in errs]
    params = [param for data in datas]
    # Plotting
    fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
    make_errorbar(ax, xs, datas, errs, params, alpha=0.7)
    set_ax_parameters(ax, xlabel='Residue', ylabel='DFI')
    plot_figure(fig, ax, figname=system.sysname, figpath='png/dfi.png',)


sysdir = 'systems'
sysname = '1btl'
mdsys = gmxSystem(sysdir, sysname)
pdb = mdsys.inpdb
# pdb = os.path.join(mdsys.wdir, 'ref.pdb')
ref_pertmat_path = os.path.join(mdsys.datdir, 'pertmat_av.npy')
ref_covmat_path = os.path.join(mdsys.datdir, 'covmat_av.npy')

N_MODES = 20
CUTOFF = 1700000
DENSE_NOT_SPARSE = False

atoms = io.pdb2atomlist(pdb)
mask = ["CA", ] # "P", "C1'"
bb = atoms.mask(mask, mode='name')

xs = np.array(bb.xs)
ys = np.array(bb.ys)
zs = np.array(bb.zs)
n = len(bb)

ref_pertmat = np.load(ref_pertmat_path).astype('double')
ref_covmat = np.load(ref_covmat_path).astype('double')

nt = 50
covmat = np.tile(ref_covmat, (nt, nt))

# hess = mycmath.hessian(n, xs, ys, zs, cutoff=25, spring_constant=1e3, dd=0, )
pertmat =  mdm._td_perturbation_matrix_cpu(covmat, dtype=np.float64)
# pertmat = mycmath._perturbation_matrix_old(covmat, covmat.shape[0] // 3)
# pertmat = mycmath._perturbation_matrix(covmat)
pertmat = mycmath._td_perturbation_matrix(covmat)


# Plotting
dfi = mdm.dfi(pertmat)
dfi = np.split(dfi, nt)
dfi = np.sum(dfi, axis=0)
fig, ax = init_figure(grid=(1, 1), axsize=(12, 5))
ax.plot(np.arange(len(dfi)), dfi, label='dfi')
set_ax_parameters(ax, xlabel='Residue', ylabel='DFI')
plot_figure(fig, ax, figpath='png/dfi.png',)