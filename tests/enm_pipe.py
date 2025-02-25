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
from cgtools import io, lrt


sysdir = 'systems'
sysname = 'ribosome'
mdsys = gmxSystem(sysdir, sysname)
pdb = os.path.join(mdsys.wdir, 'ref.pdb')
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
hess = mycmath.hessian(n, xs, ys, zs, cutoff=25, spring_constant=1e3, dd=0, )
petmat =  lrt._perturbation_matrix_cpu(hess, dtype=np.float32)
print(np.sum(petmat))
petmat = mycmath._perturbation_matrix_old(hess, n)
print(np.sum(petmat))

# hess = enm_tools.calculate_hessian(len(bb), xs, ys, zs, cutoff=25, spring_constant=1e3, dd=0, )
# print(np.sum(hess))
