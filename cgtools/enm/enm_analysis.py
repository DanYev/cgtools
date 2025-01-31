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
import enm_tools


PDB = sys.argv[1]
N_MODES = 1000
CUTOFF = 17
DENSE_NOT_SPARSE = True

if not os.path.exists(f"output/{PDB}"):
    os.makedirs(f"output/{PDB}")


# CIF
start_time = time.perf_counter()
x, y, z, occupancies, chain_ids, sequence_ids = enm_tools.read_cif(PDB)
resnum = len(x)
end_time = time.perf_counter()
print("Reading CIF: ", end_time - start_time)


# Calculating Hessian
start_time = time.perf_counter()
hessian = enm_tools.calculate_hessian(resnum, x, y, z, cutoff=CUTOFF, spring_constant=1e3, dtype=np.float64)
hessian_gpu = cp.asarray(hessian) # Moving to GPU
end_time = time.perf_counter()
print("Calculating Hessian: ", end_time - start_time)


# Inverting hessian 
cov_gpu = enm_tools.invert_matrix_gpu(hessian_gpu, n_modes=N_MODES, DENSE_NOT_SPARSE=DENSE_NOT_SPARSE)
# cov_gpu = 0.5 * (cov_gpu + cov_gpu.T)

# RMSF
start_time = time.perf_counter()
rmsf = np.zeros(resnum, np.float64)
rmsf_gpu = cp.asarray(rmsf)
for k in range(resnum):
    rmsf_gpu[k] = cp.sqrt(cov_gpu[3*k, 3*k] + cov_gpu[3*k+1, 3*k+1] + cov_gpu[3*k+2, 3*k+2])
rmsf = rmsf_gpu.get()
end_time = time.perf_counter()
print("RMSF: ", end_time - start_time)

# DFI DCI
start_time = time.perf_counter()
cov = cov_gpu.get()
pert_mat = enm_tools.calc_perturbation_matrix(cov)
dfi = enm_tools.calc_dfi(pert_mat)
dci = enm_tools.calc_full_dci(pert_mat)
end_time = time.perf_counter()
print("DFI: ", (end_time - start_time))

data = pd.DataFrame()
data["resi"] = list(range(1, resnum+1))
data["dfi"] = dfi * len(dfi)
data["err"] = 0.01 * dfi * len(dfi)
data.to_csv(f"output/{PDB}/dfi.csv", index=False, header=None, float_format='%.3E', sep=',')

df = pd.DataFrame(dci)
df.to_csv(f"output/{PDB}/dci.csv", index=False, header=None, float_format='%.3E', sep=',')
df = pd.DataFrame(0.01 * dci)
df.to_csv(f"output/{PDB}/dci_err.csv", index=False, header=None, float_format='%.3E', sep=',')



    


