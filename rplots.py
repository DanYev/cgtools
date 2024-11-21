import os
import numpy as np
import pandas as pd
import sys
sys.path.append('cgtools')
from plotting import plot_mean_sem
from set_bfactors import update_bfactors
from system import CGSystem
from cli import sbatch, run
                
sysdir = 'ribosomes_old' 
sysnames = ['ribosome_test', 'ribosome', 'ribosome_k'] 
# sysnames = ['ribosome_aa']


def pull_files(sysnames, fdir, fname):
    systems = [CGSystem(sysdir, sysname) for sysname in sysnames]
    fpaths = [os.path.join(system.wdir, fdir, fname) for system in systems]
    fpaths = [f for f in fpaths if os.path.exists(f)]
    return fpaths
        
        
def plot_csvs(fnames, figdir):
    for fname in fnames:
        figpath = os.path.join(figdir, fname.replace('csv', 'png'))
        files = pull_files(sysnames, 'data', fname)
        plot_mean_sem(files, figpath, )
    
    
def png(): 
    fnames = [f for f in os.listdir(os.path.join(sysdir, 'ribosome', 'data')) if f.startswith('r')]
    figdir = os.path.join(sysdir, 'png')    
    plot_csvs(fnames, figdir)
    
    
def pdb():
    ribosome = 'ribosome'
    sysnames = [ribosome, 'ribosome']
    x = []
    for sysname in sysnames:
        system = CGSystem(sysdir, sysname)
        fdir =  os.path.join(sysdir, sysname, 'data')
        fnames = [f for f in os.listdir(fdir) if f.startswith('rmsf.')]
        datas = [pd.read_csv(os.path.join(fdir, fname), header=None) for fname in fnames]
        data = datas[0]
        x.append((data[1], data[2]))
    
    b_factors = (x[0][0] - x[1][0]) * 10
    b_factors = (x[0][0]) * 10
    errs = np.sqrt(x[0][1]**2 + x[1][1]**2) * 10
    sysname = ribosome
    inpdb = os.path.join(sysdir, 'pdb', f'{sysname}_tmp.pdb')
    inpdb = os.path.join(system.mdcpdb)
    # CGSystem.mask_pdb(system.syspdb, inpdb, mask=['BB', 'BB2'])
    outpdb = os.path.join(sysdir, 'pdb', f'{sysname}.pdb')
    update_bfactors(inpdb, b_factors, outpdb)
    # errpdb = os.path.join(sysdir, 'pdb', f'{sysname}_err.pdb')
    # update_bfactors(inpdb, errs, errpdb)
   
    
if __name__ == '__main__':
    # pdb()
    png()
    # 


